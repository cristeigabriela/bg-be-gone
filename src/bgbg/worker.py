#!/usr/bin/env python3
"""Background-removal + segmentation worker for bg-be-gone.

Runs inside the bundled virtualenv. Speaks line-delimited JSON on stdin/stdout
so the GTK frontend can keep a model resident on the GPU across requests. Heavy
work runs on a background thread so a ``seg_cancel`` message can interrupt a slow
"segment everything" pass while it is still running.

Background removal needs ``rembg`` (imported lazily); segmentation needs only
``onnxruntime`` + ``numpy`` + ``pillow`` (via :mod:`segmentation`). Either half
can be absent — a segmentation-only install ships without rembg.

Requests (one JSON object per line on stdin):
  {"op":"single","input":P,"output":P,"model":M,"alpha":bool,"bg":BG,"id":N}
  {"op":"batch","input_dir":D,"output_dir":D,"model":M,"alpha":bool,"bg":BG,
   "pattern":"{name}_nobg","id":N}
  {"op":"gif","input":P,"output":P,"model":M,"alpha":bool,"bg":BG,"blur":int,
   "rot":0..3,"fh":bool,"fv":bool,"id":N}   # per-frame background removal;
                                             # GIF or WebP (animated) by output
  {"op":"convert","input":P,"output":P,"id":N}   # re-encode as output's format
  {"op":"frames","input":P,"id":N}           # count an image's frames
  {"op":"cancel"}                            # stop single/batch/gif in progress
  {"op":"models"}
  {"op":"seg_load","input":P,"model":"auto|large|base_plus|small|tiny|mobile","id":N}
  {"op":"seg_everything","points_per_side":int?,"id":N}
  {"op":"seg_point","points":[[x,y,label],...],"use_prev":bool,"id":N}
  {"op":"seg_extract","ids":[...]|"mask":P,"bg":BG,"blur":int,"output":P,"id":N}
  {"op":"seg_cancel"}
  {"op":"engine_status","id":N}              # waits for PyTorch's start-up, if any
  {"op":"shutdown"}

BG is "transparent", "blur" or "#rrggbb" to flatten onto.

Responses (one JSON object per line on stdout):
  {"type":"ready","models":[...],"providers":[...],"seg":bool,"bgremove":bool,
   "seg_models":[{"rung":R,"label":L}],"engines":["ort"]|["ort","torch"],
   "settings":P}                             # the settings.json this worker reads
  {"type":"engine_status","engine":"ort"|"torch","state":"ready"|"failed"|...,
   "reason":str|null,"device":str|null,"id":N}
  {"type":"loading","model":M,"id":N}
  {"type":"device","provider":P,"label":L,"gpu":bool,"id":N}
  {"type":"progress","done":i,"total":n,"name":str,"id":N}
  {"type":"done_single","output":P,"preview":P,"seconds":f,"id":N}
  {"type":"done_batch","count":n,"seconds":f,"outdir":P,"id":N}
  {"type":"gif_progress","done":i,"total":n,"id":N}
  {"type":"gif_done","output":P,"preview":P,"frames":n,"seconds":f,"id":N}
  {"type":"converted","output":P,"frames":n,"id":N}
  {"type":"frames","frames":n,"id":N}
  {"type":"canceled","scope":"single|batch|gif","id":N}
  {"type":"seg_download","done":i,"total":n,"id":N}
  {"type":"seg_step","rung":R,"message":str,"id":N}
  {"type":"seg_ready","rung":R,"model":L,"provider":P,"label":L,"gpu":bool,
   "mode":auto|manual|fallback,"family":str,"id":N}
  {"type":"seg_progress","done":i,"total":n,"id":N}
  {"type":"seg_objects","label_map":P,"count":n,"objects":[...],"id":N}
  {"type":"seg_mask","mask":P,"score":f,"bbox":[x,y,w,h],"id":N}
  {"type":"seg_extracted","output":P,"seconds":f,"id":N}
  {"type":"seg_canceled","id":N}
  {"type":"notice","message":str,"id":N|null}   # shown to the user; null at start-up
  {"type":"error","message":str,"id":N}
"""
import os
import sys
import glob

import plat

# Before onnxruntime is imported anywhere: the GPU stack reads its settings
# only at start-up, and a provider with no code for this card can reset the GPU.
plat.prepare_process()

import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import queue  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

from PIL import Image, ImageFilter, UnidentifiedImageError  # noqa: E402
# Imports onnxruntime, then rembg when it is installed.
from engines import ort as ort_engine  # noqa: E402
from engines import selection  # noqa: E402
from engines import torch_birefnet  # noqa: E402  (imports no torch itself)
import formats  # noqa: E402

# HEIC/HEIF (phone photos) needs libheif; pi-heif is its decode-only build.
try:
    from pi_heif import register_heif_opener
    register_heif_opener()
    _HAVE_HEIF = True
except ImportError:
    _HAVE_HEIF = False

# Segmentation is optional too (needs the extra numpy dependency + weights).
try:
    import segmentation as seg
    _HAVE_SEG = True
except Exception:  # noqa: BLE001
    _HAVE_SEG = False

_sessions = {}
_cpu_sessions = {}          # model -> CPU-only fallback session (low-VRAM path)
_fallback_notified = set()  # models we've already warned about this session
_torch_models = set()       # models whose session in _sessions is PyTorch's

# The engine BiRefNet runs on, from engines/selection.py; see _apply_engine_choice.
_engine = selection.Engine.ORT
_torch = torch_birefnet.TorchEngine()
_torch_notified = False     # the user heard why PyTorch isn't used, this choice


def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _notice(message, req_id=None):
    sys.stderr.write(message + "\n")
    out({"type": "notice", "message": message, "id": req_id})


def _torch_session(model, req_id):
    """A PyTorch session for `model`, or None to run it on ONNX Runtime.

    None when the engine isn't asked for or the model isn't BiRefNet, and
    when PyTorch failed to start or this model failed to load: the user
    hears why once, and the job carries on on ONNX Runtime.
    """
    global _torch_notified
    if _engine is not selection.Engine.TORCH or model not in torch_birefnet.MODELS:
        return None
    if _torch.wait() is torch_birefnet.EngineState.READY:
        try:
            out({"type": "loading", "model": model, "id": req_id})
            session = _torch.new_session(model)
        except Exception as err:  # noqa: BLE001  (out of VRAM, a bad download, ...)
            _torch.release_memory()
            reason = f"{model} didn't load ({type(err).__name__}: {err})"
        else:
            out({"type": "device", "provider": "PyTorch", "label": _torch.device,
                 "gpu": True, "id": req_id})
            return session
    else:
        reason = _torch.reason
    if not _torch_notified:
        _torch_notified = True
        _notice(f"The PyTorch engine can't run this: {reason}. Using ONNX Runtime.", req_id)
    return None


def get_session(model, req_id):
    if model not in _sessions:
        session = _torch_session(model, req_id)
        if session is not None:
            _sessions[model] = session
            _torch_models.add(model)
            return session
        out({"type": "loading", "model": model, "id": req_id})
        sess = ort_engine.new_session(model, ort_engine.preferred_providers())
        active = sess.inner_session.get_providers()
        prov = active[0] if active else "CPUExecutionProvider"
        _sessions[model] = sess
        out({"type": "device", "provider": prov,
             "label": ort_engine.PROVIDER_LABELS.get(prov, prov),
             "gpu": prov in ort_engine.GPU_PROVIDERS, "id": req_id})
    return _sessions[model]


def get_cpu_session(model, req_id):
    """A CPU-only session for `model`, built once and cached. Used to complete a
    job when the GPU can't (e.g. not enough VRAM)."""
    if model not in _cpu_sessions:
        out({"type": "loading", "model": model, "id": req_id})
        _cpu_sessions[model] = ort_engine.new_session(model, ["CPUExecutionProvider"])
    # Tell the UI the run moved to the CPU. Without this the status bar keeps
    # claiming the GPU through every fallback, so a wrong-but-fast result and a
    # correct-but-slow one look identical from the outside.
    out({"type": "device", "provider": "CPUExecutionProvider",
         "label": ort_engine.PROVIDER_LABELS["CPUExecutionProvider"],
         "gpu": False, "id": req_id})
    return _cpu_sessions[model]


def _remove_resilient(model, req_id, img, alpha):
    """Run rembg on the preferred (GPU) session; if it fails with a runtime error
    — typically low VRAM — retry on CPU so the user still gets a result. The GPU
    session is kept, so a later run recovers automatically once VRAM frees up."""
    session = get_session(model, req_id)
    try:
        with warnings.catch_warnings():
            # A NaN mask does not raise: rembg scales the prediction to uint8,
            # numpy warns "invalid value encountered in cast", and the user gets
            # a silently garbage cutout that still reports success. Promote that
            # one warning to an error so it lands in the CPU-retry path below.
            warnings.filterwarnings("error", category=RuntimeWarning,
                                    message="invalid value encountered")
            res = ort_engine.remove(img, session=session, alpha_matting=alpha)
        _fallback_notified.discard(model)      # GPU healthy again
        return res
    except Exception as e:
        # On PyTorch any RuntimeError is a run failure: out of VRAM, a HIP/CUDA
        # error, or the non-finite mask TorchSession refuses to return.
        on_torch = model in _torch_models
        if not (ort_engine.is_runtime_error(e) or (on_torch and isinstance(e, RuntimeError))):
            raise
        if on_torch:
            _torch.release_memory()
        if model not in _fallback_notified:
            _fallback_notified.add(model)
            out({"type": "notice",
                 "message": "The GPU couldn't run this (usually low VRAM) — "
                            "running on the CPU (slower). Close other GPU apps "
                            "to free memory, then try again.",
                 "id": req_id})
        session = get_cpu_session(model, req_id)
        return ort_engine.remove(img, session=session, alpha_matting=alpha)


def _hex_to_rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _checkerboard(size, cell=24):
    w, h = size
    board = Image.new("RGB", (w, h), (245, 245, 245))
    dark = (200, 200, 200)
    px = board.load()
    for y in range(h):
        yodd = (y // cell) & 1
        for x in range(w):
            if ((x // cell) & 1) ^ yodd:
                px[x, y] = dark
    return board


class UnreadableImageError(OSError):
    def __init__(self, path, reason):
        super().__init__(f"can't read {os.path.basename(path)}: {reason}")
        self.path = path
        self.reason = reason


class UnsupportedOutputError(ValueError):
    def __init__(self, path):
        super().__init__(f"can't save as {os.path.basename(path)}: "
                         "bg-be-gone writes PNG, WebP or JPEG")
        self.path = path


def open_image(path):
    """Open and decode an image, or raise UnreadableImageError naming the file.

    Decoding here (Image.open alone only reads the header) makes a truncated
    file fail with its name, not later in the middle of a cut-out. Failures go
    to the worker log too, with the full path.
    """
    try:
        im = Image.open(path)
        im.load()
        return im
    except UnidentifiedImageError as e:
        if formats.format_of(path) is formats.HEIF and not _HAVE_HEIF:
            reason = "HEIC/HEIF support (pi-heif) isn't installed; re-run the installer"
        else:
            reason = "not an image format bg-be-gone reads"
        err = UnreadableImageError(path, reason)
        cause = e
    except OSError as e:
        err = UnreadableImageError(path, str(e))
        cause = e
    print(f"decode failed: {path}: {cause}", file=sys.stderr, flush=True)
    raise err from cause


# Encoder settings per format we write. WebP keeps its alpha lossless by default
# (alpha_quality=100), so only the colour is lossy.
_SAVE_OPTIONS = {
    formats.PNG: {},
    formats.WEBP: {"quality": 90},
    formats.JPEG: {"quality": 92},
}


def _plain(im):
    """The image as 8-bit RGB, or RGBA when it has transparency.

    Sources come in many modes (CMYK TIFF, palette GIF and BMP, greyscale);
    the cut-out and every encoder take these two.

    16-bit greyscale is scaled to 8 bits first. Pillow's own conversion clips
    it, so mid grey (32768) and white (65535) would both come out white.
    """
    if im.mode == "I" or im.mode.startswith("I;16"):
        im = im.convert("I").point(lambda v: v * (1 / 256)).convert("L")
    has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    mode = "RGBA" if has_alpha else "RGB"
    return im if im.mode == mode else im.convert(mode)


def save_image(im, dst):
    """Save the current frame of `im` in the format `dst`'s suffix names."""
    fmt = formats.format_of(dst)
    if fmt not in _SAVE_OPTIONS:
        raise UnsupportedOutputError(dst)
    im = _plain(im)
    if fmt is formats.JPEG:
        im = im.convert("RGB")
    im.save(dst, **_SAVE_OPTIONS[fmt])


def handle_convert(req):
    """Re-encode an image into the format its output path names.

    The frontend uses it to preview a source GdkPixbuf can't decode (no loader
    for HEIC or AVIF, say) and to save a result as WebP or JPEG. Only the first
    frame is written; "frames" tells the frontend whether the source animates.
    """
    im = open_image(req["input"])
    save_image(im, req["output"])
    out({"type": "converted", "output": req["output"],
         "frames": getattr(im, "n_frames", 1), "id": req.get("id")})


def handle_frames(req):
    """Count a source's frames, so the frontend can tell an animated WebP from
    a still. GdkPixbuf can't be asked: a WebP loader that decodes only stills
    reports an animation as one frame."""
    im = open_image(req["input"])
    out({"type": "frames", "frames": getattr(im, "n_frames", 1), "id": req.get("id")})


def process_one(model, req_id, src, dst, alpha, bg, blur=20, want_preview=False):
    img = _plain(open_image(src))
    res = _remove_resilient(model, req_id, img, alpha).convert("RGBA")

    preview_path = None
    if bg == "transparent":
        res.save(dst)
        if want_preview:
            board = _checkerboard(res.size).convert("RGBA")
            board.alpha_composite(res)
            preview_path = dst + ".preview.png"
            board.convert("RGB").save(preview_path)
    elif bg == "blur":
        # Keep the picture, blur only the background: sharp foreground (the
        # model's cutout) composited over a Gaussian-blurred copy of the source.
        base = img.convert("RGB").filter(ImageFilter.GaussianBlur(max(1, blur)))
        base = base.convert("RGBA")
        base.alpha_composite(res)
        base.convert("RGB").save(dst)
        if want_preview:
            preview_path = dst
    else:
        flat = Image.new("RGBA", res.size, _hex_to_rgb(bg) + (255,))
        flat.alpha_composite(res)
        flat.convert("RGB").save(dst)
        if want_preview:
            preview_path = dst
    return preview_path


def handle_single(req):
    rid = req.get("id")
    get_session(req["model"], rid)         # load + emit the device message
    t = time.time()
    preview = process_one(req["model"], rid, req["input"], req["output"],
                          req.get("alpha", False), req.get("bg", "transparent"),
                          req.get("blur", 20), want_preview=True)
    if _cancel.is_set():                    # a single inference can't be stopped
        out({"type": "canceled", "scope": "single", "id": rid})   # mid-run; drop it
        return
    out({"type": "done_single", "output": req["output"],
         "preview": preview or req["output"],
         "seconds": round(time.time() - t, 2), "id": rid})


def _webp_frames(path):
    """How many frames a WebP holds, from its header; 1 for any other format.

    A file that won't open counts as one frame, so process_one reports it with
    its name like any other unreadable image.
    """
    if formats.format_of(path) is not formats.WEBP:
        return 1
    try:
        with Image.open(path) as im:
            return getattr(im, "n_frames", 1)
    except (UnidentifiedImageError, OSError):
        return 1


def handle_batch(req):
    rid = req.get("id")
    indir = req["input_dir"]
    outdir = req["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(indir, "*"))
                   if f.lower().endswith(formats.BATCH_SUFFIXES)
                   and os.path.isfile(f))
    # Batch would keep only an animation's first frame, as with GIFs.
    animated = [f for f in files if _webp_frames(f) > 1]
    files = [f for f in files if f not in animated]
    if animated:
        out({"type": "notice", "id": rid, "message":
             "Skipped %d animated WebP (%s): open them on the Single page."
             % (len(animated), ", ".join(os.path.basename(f) for f in animated))})
    total = len(files)
    if total == 0:
        out({"type": "error", "message": "No images found in input folder.",
             "id": rid})
        return
    pattern = req.get("pattern") or "{name}_nobg"
    get_session(req["model"], rid)         # load + emit the device message
    alpha = req.get("alpha", False)
    bg = req.get("bg", "transparent")
    blur = req.get("blur", 20)
    t = time.time()
    for i, f in enumerate(files, 1):
        if _cancel.is_set():               # stop before starting the next file
            out({"type": "canceled", "scope": "batch", "done": i - 1,
                 "total": total, "id": rid})
            return
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            stem = pattern.format(name=name, n=i)
        except Exception:
            stem = name + "_nobg"
        dst = os.path.join(outdir, stem + ".png")
        out({"type": "progress", "done": i - 1, "total": total,
             "name": os.path.basename(f), "id": rid})
        process_one(req["model"], rid, f, dst, alpha, bg, blur,
                    want_preview=False)
    out({"type": "progress", "done": total, "total": total, "name": "",
         "id": rid})
    out({"type": "done_batch", "count": total,
         "seconds": round(time.time() - t, 1), "outdir": outdir, "id": rid})


def _apply_transform(im, rot, fh, fv):
    """Bake the view's rotate/flip into a frame, matching ImageView.export_pixbuf
    (flip horizontal, flip vertical, then rotate clockwise) — so a rotated GIF
    comes out exactly like a rotated still would."""
    if fh:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if fv:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    rot %= 4
    if rot == 1:
        im = im.transpose(Image.ROTATE_270)    # 90 clockwise
    elif rot == 2:
        im = im.transpose(Image.ROTATE_180)
    elif rot == 3:
        im = im.transpose(Image.ROTATE_90)     # 90 counter-clockwise
    return im


def _compose_frame(orig_rgba, cut_rgba, bg, blur):
    """Apply the chosen background to one processed frame; returns RGBA."""
    if bg == "transparent":
        return cut_rgba
    if bg == "blur":
        base = orig_rgba.convert("RGB").filter(
            ImageFilter.GaussianBlur(max(1, blur))).convert("RGBA")
    else:
        base = Image.new("RGBA", cut_rgba.size, _hex_to_rgb(bg) + (255,))
    base.alpha_composite(cut_rgba)
    return base


def _gif_palette_frame(path, transparent):
    """Load one processed frame from disk as a GIF-ready palette image. GIF only
    supports 1-bit alpha, so for a transparent background we reserve a palette
    index for the cut-out pixels (hard edges are inherent to the format)."""
    fr = Image.open(path)
    if not transparent:
        return fr.convert("RGB")
    p = fr.convert("RGB").quantize(colors=255)         # leave index 255 free
    mask = fr.getchannel("A").point(lambda a: 255 if a < 128 else 0)
    p.paste(255, mask)
    return p


def _frames_to_gif(paths, dst, durations, loop, transparent):
    """Assemble frame PNGs into an animated GIF, loading them one at a time
    (streamed via a generator) so the rebuild stays memory-bounded."""
    first = _gif_palette_frame(paths[0], transparent)
    rest = (_gif_palette_frame(p, transparent) for p in paths[1:])
    kw = dict(save_all=True, append_images=rest, duration=durations,
              loop=loop, optimize=False)
    if transparent:
        kw.update(transparency=255, disposal=2)
    first.save(dst, **kw)


def _frames_to_webp(paths, dst, durations, loop):
    """Assemble frame PNGs into an animated WebP, streamed like _frames_to_gif.
    WebP keeps the full alpha channel, so cut-out edges stay soft."""
    first = Image.open(paths[0])
    rest = (Image.open(p) for p in paths[1:])
    first.save(dst, save_all=True, append_images=rest, duration=durations,
               loop=loop, **_SAVE_OPTIONS[formats.WEBP])


def handle_gif(req):
    """Split an animated GIF or WebP into frames, remove each frame's
    background, and reassemble into the format the output path names. The
    model is loaded once and reused for every frame. Each finished frame is
    written to disk (not held in RAM) so a long animation stays memory-bounded
    during the slow per-frame inference. Progress is per frame; cancellable
    between frames."""
    rid = req.get("id")
    src, dst, model = req["input"], req["output"], req["model"]
    alpha = req.get("alpha", False)
    bg = req.get("bg", "transparent")
    blur = int(req.get("blur", 20))
    rot = int(req.get("rot", 0))
    fh, fv = bool(req.get("fh")), bool(req.get("fv"))
    im = open_image(src)
    n = getattr(im, "n_frames", 1)
    loop = im.info.get("loop", 0)
    # Report 0/N up front so the progress bar appears immediately — the model
    # load + first-frame warmup (slow on BiRefNet) happen before frame 1 lands.
    out({"type": "gif_progress", "done": 0, "total": n, "id": rid})
    get_session(model, rid)                # load once; reused for every frame
    framedir = tempfile.mkdtemp(prefix="gifframes-", dir=os.path.dirname(dst) or None)
    paths, durations = [], []
    t = time.time()
    try:
        for i in range(n):
            if _cancel.is_set():
                out({"type": "canceled", "scope": "gif", "id": rid})
                return
            im.seek(i)
            frame = _apply_transform(im.convert("RGBA"), rot, fh, fv)
            # After decoding: a WebP frame's duration is set only once it loads.
            durations.append(im.info.get("duration", 100))
            cut = _remove_resilient(model, rid, frame, alpha).convert("RGBA")
            fp = os.path.join(framedir, "f%05d.png" % i)
            _compose_frame(frame, cut, bg, blur).save(fp)
            paths.append(fp)
            out({"type": "gif_progress", "done": i + 1, "total": n, "id": rid})
        if _cancel.is_set():
            out({"type": "canceled", "scope": "gif", "id": rid})
            return
        if formats.format_of(dst) is formats.WEBP:
            _frames_to_webp(paths, dst, durations, loop)
        else:
            _frames_to_gif(paths, dst, durations, loop, bg == "transparent")
        # For a frontend whose GdkPixbuf has no loader for the animation.
        preview = dst + ".preview.png"
        shutil.copyfile(paths[0], preview)
        out({"type": "gif_done", "output": dst, "preview": preview, "frames": n,
             "seconds": round(time.time() - t, 2), "id": rid})
    finally:
        shutil.rmtree(framedir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
_cancel = threading.Event()
_seg = None  # {session, image, rung, size, maskdir, objects, prev_low}


class _Canceled(Exception):
    pass


def _gpu_hint():
    return any(ort_engine.provider_name(p) in ort_engine.GPU_PROVIDERS
               for p in ort_engine.preferred_providers())


def handle_seg_load(req):
    global _seg
    rid = req.get("id")
    path = req["input"]
    want = req.get("model", "auto")
    pil = open_image(path).convert("RGB")
    maskdir = os.path.join(os.path.dirname(os.path.abspath(path)), "segmasks")
    os.makedirs(maskdir, exist_ok=True)

    def on_step(rung, exc):
        out({"type": "seg_step", "rung": rung,
             "message": str(exc).splitlines()[0][:160], "id": rid})

    def on_progress(done, total):
        if _cancel.is_set():
            raise _Canceled()
        out({"type": "seg_download", "done": done, "total": total, "id": rid})

    try:
        session, rung, mode = seg.auto_select_session(
            ort_engine.preferred_providers(), _gpu_hint(), want=want, probe_image=pil,
            on_step=on_step, on_progress=on_progress)
    except _Canceled:
        out({"type": "seg_canceled", "id": rid})
        return
    prov = session.provider()
    _seg = {"session": session, "image": pil, "rung": rung, "size": pil.size,
            "maskdir": maskdir, "objects": [], "prev_low": None}
    out({"type": "seg_ready", "rung": rung, "model": seg.MODELS[rung]["label"],
         "provider": prov, "label": ort_engine.PROVIDER_LABELS.get(prov, prov),
         "gpu": prov in ort_engine.GPU_PROVIDERS, "mode": mode, "family": session.family,
         "id": rid})


def _require_seg(rid):
    if _seg is None:
        out({"type": "error", "message": "Load an image to segment first.",
             "id": rid})
        return False
    return True


def handle_seg_everything(req):
    rid = req.get("id")
    if not _require_seg(rid):
        return
    session = _seg["session"]
    gpu = session.provider() in ort_engine.GPU_PROVIDERS
    pps = int(req.get("points_per_side") or (32 if gpu else 16))

    def progress(done, total):
        out({"type": "seg_progress", "done": done, "total": total, "id": rid})

    t = time.time()
    masks = session.everything(points_per_side=pps, cancel=_cancel,
                               progress=progress)
    if _cancel.is_set():
        out({"type": "seg_canceled", "id": rid})
        return
    maps, objs = seg.save_objects(masks, _seg["size"], _seg["maskdir"], f"e{rid}")
    _seg["objects"] = objs
    out({"type": "seg_objects", "label_map": maps["label"],
         "general_map": maps["general"], "depth_map": maps["depth"],
         "count": len(objs), "objects": objs,
         "seconds": round(time.time() - t, 2), "id": rid})


def handle_seg_point(req):
    rid = req.get("id")
    if not _require_seg(rid):
        return
    session = _seg["session"]
    pts = req.get("points") or []
    if not pts:
        out({"type": "error", "message": "No points given.", "id": rid})
        return
    points = [(float(p[0]), float(p[1])) for p in pts]
    labels = [int(p[2]) for p in pts]
    prev = _seg.get("prev_low") if req.get("use_prev") else None
    mask, score, low = session.decode_points(points, labels, prev_low=prev)
    _seg["prev_low"] = low
    mp = os.path.join(_seg["maskdir"], f"p{rid}.png")
    bbox = seg.save_mask(mask, mp)
    out({"type": "seg_mask", "mask": mp, "score": round(score, 3),
         "bbox": bbox, "contour": seg.contour(mask), "id": rid})


def handle_seg_extract(req):
    rid = req.get("id")
    if not _require_seg(rid):
        return
    dst = req["output"]
    bg = req.get("bg", "transparent")
    blur = int(req.get("blur", 20))
    if req.get("mask"):
        paths = [req["mask"]]
    else:
        ids = set(req.get("ids") or [])
        paths = [o["mask"] for o in _seg["objects"] if o["id"] in ids]
    if not paths:
        out({"type": "error", "message": "Nothing selected to extract.",
             "id": rid})
        return
    t = time.time()
    alpha = seg.load_union(paths)
    seg.composite_extract(_seg["image"], alpha, bg, dst, blur=blur)
    out({"type": "seg_extracted", "output": dst,
         "seconds": round(time.time() - t, 2), "id": rid})


def handle_unload(req):
    """Release resident models to reclaim memory (RAM/VRAM). Runs on the worker
    thread, so it only fires when no task is in flight — never freeing a model
    out from under a running pass. Scope: "bg", "seg", or "all"."""
    global _seg
    rid = req.get("id")
    scope = req.get("scope", "all")
    freed = []
    if scope in ("bg", "all") and (_sessions or _cpu_sessions):
        _drop_bg_sessions()
        freed.append("bg")
    if scope in ("seg", "all") and _seg is not None:
        _seg = None
        freed.append("seg")
    if freed:
        gc.collect()
    out({"type": "unloaded", "scope": freed, "id": rid})


def _apply_engine_choice():
    """Settle which engine runs BiRefNet, from the engine settings, once at
    start-up: a change of the setting starts a new worker.

    Skipped layers go to the worker log. A request for PyTorch that can't be
    met (switched off, or not installed) reaches the user; otherwise PyTorch
    starts loading on its own thread, and BiRefNet jobs wait for it.
    """
    global _engine
    choice = selection.resolve()
    for problem in choice.problems:
        sys.stderr.write(f"engine setting ignored: {problem}\n")
    wanted = choice.engine
    if wanted is selection.Engine.TORCH:
        off = torch_birefnet.switched_off()
        if off or not torch_birefnet.installed():
            why = f"{off} is set" if off else "this install doesn't have PyTorch"
            _notice(f"The PyTorch engine was asked for by {choice.source.value}, but {why}. "
                    "Using ONNX Runtime.")
            wanted = selection.Engine.ORT
        else:
            _torch.start()
    _engine = wanted


def handle_engine_status(req):
    """Which engine BiRefNet runs on, once PyTorch's start-up (if any) is over.

    The install scripts ask this before they keep a ROCm worker.
    """
    status = {"state": "ready", "reason": None, "device": None}
    if _engine is selection.Engine.TORCH:
        _torch.wait()
        status = _torch.status()
    out({"type": "engine_status", "engine": _engine.value, **status, "id": req.get("id")})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_HANDLERS = {
    "single": handle_single,
    "batch": handle_batch,
    "gif": handle_gif,
    "convert": handle_convert,
    "frames": handle_frames,
    "seg_load": handle_seg_load,
    "seg_everything": handle_seg_everything,
    "seg_point": handle_seg_point,
    "seg_extract": handle_seg_extract,
    "unload": handle_unload,
    "engine_status": handle_engine_status,
}
_SEG_OPS = {"seg_load", "seg_everything", "seg_point", "seg_extract"}
_BG_OPS = {"single", "batch", "gif"}


def _drop_bg_sessions():
    _sessions.clear()
    _cpu_sessions.clear()
    _fallback_notified.clear()
    _torch_models.clear()
    _torch.release_memory()


def _engines_on_offer():
    """The engines this worker can run, for the frontend's Engine setting."""
    on_offer = ["ort"]
    if torch_birefnet.installed() and not torch_birefnet.switched_off():
        on_offer.append("torch")
    return on_offer


def _ready():
    out({"type": "ready",
         "engines": _engines_on_offer(),
         # Resolved here, after plat.prepare_process() may have moved it, so the
         # frontend's Engine row reads and writes the file this worker reads.
         "settings": selection.default_settings_path(),
         "models": list(ort_engine.sessions_names) if ort_engine.HAVE_REMBG else [],
         "providers": ort_engine.provider_names(),
         "bgremove": ort_engine.HAVE_REMBG,
         "seg": _HAVE_SEG,
         "seg_models": ([{"rung": r, "label": seg.MODELS[r]["label"],
                          "vram": seg.MODELS[r]["vram_gate"]}
                         for r in seg.LADDER] if _HAVE_SEG else [])})


def _dispatch(req):
    op = req.get("op")
    rid = req.get("id")
    try:
        if op == "models":
            _ready()
            return
        if op in _BG_OPS and not ort_engine.HAVE_REMBG:
            out({"type": "error", "message":
                 "Background removal is not installed in this build.", "id": rid})
            return
        if op in _SEG_OPS and not _HAVE_SEG:
            out({"type": "error", "message":
                 "Segmentation is not installed in this build.", "id": rid})
            return
        handler = _HANDLERS.get(op)
        if handler:
            handler(req)
    except Exception as e:  # noqa: BLE001
        out({"type": "error",
             "message": f"{e}\n{traceback.format_exc()}", "id": rid})


def _worker_loop(q):
    while True:
        req = q.get()
        if req is None:
            return
        _cancel.clear()
        _dispatch(req)


def main():
    _ready()
    _apply_engine_choice()
    q = queue.Queue()
    threading.Thread(target=_worker_loop, args=(q,), daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        op = req.get("op")
        if op == "shutdown":
            break
        # Cancel is handled on the stdin thread so it can interrupt a running
        # task (segmentation checks _cancel between grid points; gif between
        # frames). "seg_cancel" is kept for back-compat.
        if op in ("cancel", "seg_cancel"):
            _cancel.set()
            continue
        q.put(req)


if __name__ == "__main__":
    main()
