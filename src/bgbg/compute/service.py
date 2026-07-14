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
   "rot":0..3,"fh":bool,"fv":bool,"id":N}   # per-frame background removal
  {"op":"cancel"}                            # stop single/batch/gif in progress
  {"op":"models"}
  {"op":"seg_load","input":P,"model":"auto|large|base_plus|small|tiny|mobile","id":N}
  {"op":"seg_everything","points_per_side":int?,"id":N}
  {"op":"seg_point","points":[[x,y,label],...],"use_prev":bool,"id":N}
  {"op":"seg_extract","ids":[...]|"mask":P,"bg":BG,"blur":int,"output":P,"id":N}
  {"op":"seg_cancel"}
  {"op":"shutdown"}

BG is "transparent", "blur" or "#rrggbb" to flatten onto.

Responses (one JSON object per line on stdout):
  {"type":"ready","models":[...],"providers":[...],"seg":bool,"bgremove":bool,
   "seg_models":[{"rung":R,"label":L}]}
  {"type":"loading","model":M,"id":N}
  {"type":"device","provider":P,"label":L,"gpu":bool,"id":N}
  {"type":"progress","done":i,"total":n,"name":str,"id":N}
  {"type":"done_single","output":P,"preview":P,"seconds":f,"id":N}
  {"type":"done_batch","count":n,"seconds":f,"outdir":P,"id":N}
  {"type":"gif_progress","done":i,"total":n,"id":N}
  {"type":"gif_done","output":P,"frames":n,"seconds":f,"id":N}
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
  {"type":"error","message":str,"id":N}
"""
import os
import sys
import glob

# This lives in bgbg/compute/ but `segmentation` is still a sibling of the
# package (it becomes compute/sam.py in step 10), so put bgbg/ on the path.
# Getting this wrong would not raise: the `import segmentation` below is inside a
# try/except that degrades to "this worker has no segmentation", so it would
# silently lose half the app instead of failing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_gpu_libs():
    """Put any venv-bundled NVIDIA CUDA/cuDNN libs on LD_LIBRARY_PATH.

    onnxruntime-gpu links libcudart.so.* which ship inside the pip wheels; the
    dynamic loader needs them on the path before import, so we set the env and
    re-exec once if necessary. AMD/ROCm libs come from the system loader path,
    so nothing to do there.
    """
    venv = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    libdirs = glob.glob(os.path.join(
        venv, "lib", "python*", "site-packages", "nvidia", "*", "lib"))
    if not libdirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = cur.split(os.pathsep) if cur else []
    if all(d in parts for d in libdirs):
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(libdirs + parts)
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_gpu_libs()

import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import queue  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

from PIL import Image  # noqa: E402
import onnxruntime as ort  # noqa: E402

from compute import outputs_impl  # noqa: E402

# Background removal is optional (a segmentation-only install ships without it).
try:
    from rembg import new_session, remove
    from rembg.sessions import sessions_names
    _HAVE_REMBG = True
except Exception:  # noqa: BLE001
    _HAVE_REMBG = False
    sessions_names = []

# Segmentation is optional too (needs the extra numpy dependency + weights).
try:
    import segmentation as seg
    _HAVE_SEG = True
except Exception:  # noqa: BLE001
    _HAVE_SEG = False

# Preference order; the first one onnxruntime can actually initialise wins.
_PROVIDER_ORDER = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]
_PROVIDER_LABELS = {
    "CUDAExecutionProvider": "NVIDIA (CUDA)",
    "ROCMExecutionProvider": "AMD (ROCm)",
    "MIGraphXExecutionProvider": "AMD (MIGraphX)",
    "DmlExecutionProvider": "DirectML",
    "CPUExecutionProvider": "CPU",
}

_sessions = {}
_cpu_sessions = {}          # model -> CPU-only fallback session (low-VRAM path)
_fallback_notified = set()  # models we've already warned about this session


def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# HEURISTIC conv-algo search skips onnxruntime's default EXHAUSTIVE benchmark.
# EXHAUSTIVE re-runs on every process's first inference (no on-disk cache), so it
# would slow the first generation of *every* launch; HEURISTIC keeps it fast.
_PROVIDER_OPTIONS = {
    "CUDAExecutionProvider": {"cudnn_conv_algo_search": "HEURISTIC"},
}


def _name(p):
    return p[0] if isinstance(p, tuple) else p


def preferred_providers():
    avail = set(ort.get_available_providers())
    chosen = []
    for p in _PROVIDER_ORDER:
        if p not in avail:
            continue
        opts = _PROVIDER_OPTIONS.get(p)
        chosen.append((p, opts) if opts else p)
    if not any(_name(c) == "CPUExecutionProvider" for c in chosen):
        chosen.append("CPUExecutionProvider")
    return chosen


def provider_names():
    return [_name(p) for p in preferred_providers()]


# The first inference of a session still warms up (memory alloc, kernel load),
# so the first generation is a little slower; the UI mentions this.
_GPU_PROVIDERS = {"CUDAExecutionProvider", "ROCMExecutionProvider",
                  "MIGraphXExecutionProvider"}


def get_session(model, req_id):
    if model not in _sessions:
        out({"type": "loading", "model": model, "id": req_id})
        sess = new_session(model, providers=preferred_providers())
        active = sess.inner_session.get_providers()
        prov = active[0] if active else "CPUExecutionProvider"
        _sessions[model] = sess
        out({"type": "device", "provider": prov,
             "label": _PROVIDER_LABELS.get(prov, prov),
             "gpu": prov in _GPU_PROVIDERS, "id": req_id})
    return _sessions[model]


def get_cpu_session(model, req_id):
    """A CPU-only session for `model`, built once and cached. Used to complete a
    job when the GPU can't (e.g. not enough VRAM)."""
    if model not in _cpu_sessions:
        out({"type": "loading", "model": model, "id": req_id})
        _cpu_sessions[model] = new_session(
            model, providers=["CPUExecutionProvider"])
    return _cpu_sessions[model]


def _is_ort_runtime_error(e):
    """True for an onnxruntime run failure — most importantly a GPU out-of-memory,
    which BiRefNet surfaces as a failed Mul node deep in the ASPP decoder."""
    if type(e).__module__.startswith("onnxruntime"):
        return True
    m = str(e).lower()
    return ("onnxruntimeerror" in m or "non-zero status" in m
            or "out of memory" in m or "cuda" in m or "cudnn" in m
            or "failed to allocate" in m or "hiperroroutofmemory" in m)


def _remove_resilient(model, req_id, img, alpha):
    """Run rembg on the preferred (GPU) session; if it fails with a runtime error
    — typically low VRAM — retry on CPU so the user still gets a result. The GPU
    session is kept, so a later run recovers automatically once VRAM frees up."""
    session = get_session(model, req_id)
    try:
        res = remove(img, session=session, alpha_matting=alpha)
        _fallback_notified.discard(model)      # GPU healthy again
        return res
    except Exception as e:
        if not _is_ort_runtime_error(e):
            raise
        if model not in _fallback_notified:
            _fallback_notified.add(model)
            out({"type": "notice",
                 "message": "The GPU couldn't run this (usually low VRAM) — "
                            "running on the CPU (slower). Close other GPU apps "
                            "to free memory, then try again.",
                 "id": req_id})
        session = get_cpu_session(model, req_id)
        return remove(img, session=session, alpha_matting=alpha)


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


def process_one(model, req_id, src, dst, alpha, bg, blur=20, want_preview=False):
    img = Image.open(src)
    res = _remove_resilient(model, req_id, img, alpha).convert("RGBA")

    # One outputter, shared with the segmentation extract path (outputs_impl).
    outputs_impl.apply_bg(res, bg, source=img, blur=blur).save(dst)

    if not want_preview:
        return None
    if bg != "transparent":
        return dst
    # Transparent has nothing to show against, so the preview gets a
    # checkerboard baked in. (On the canvas the renderer draws one; this is for
    # the saved-preview path, which is a flat image.)
    board = _checkerboard(res.size).convert("RGBA")
    board.alpha_composite(res)
    preview_path = dst + ".preview.png"
    board.convert("RGB").save(preview_path)
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


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def handle_batch(req):
    rid = req.get("id")
    indir = req["input_dir"]
    outdir = req["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(indir, "*"))
                   if f.lower().endswith(IMG_EXTS) and os.path.isfile(f))
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
    """Apply the chosen background to one processed GIF frame; returns RGBA.

    The same outputter as the single and segment paths — the GIF encoder just
    wants RGBA back rather than a file.
    """
    img = outputs_impl.apply_bg(cut_rgba, bg, source=orig_rgba, blur=blur)
    return img.convert("RGBA")


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


def handle_gif(req):
    """Split an animated GIF into frames, remove each frame's background, and
    reassemble. The model is loaded once and reused for every frame. Each
    finished frame is written to disk (not held in RAM) so a long GIF stays
    memory-bounded during the slow per-frame inference. Progress is per frame;
    cancellable between frames."""
    rid = req.get("id")
    src, dst, model = req["input"], req["output"], req["model"]
    alpha = req.get("alpha", False)
    bg = req.get("bg", "transparent")
    blur = int(req.get("blur", 20))
    rot = int(req.get("rot", 0))
    fh, fv = bool(req.get("fh")), bool(req.get("fv"))
    im = Image.open(src)
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
            durations.append(im.info.get("duration", 100))
            frame = _apply_transform(im.convert("RGBA"), rot, fh, fv)
            cut = _remove_resilient(model, rid, frame, alpha).convert("RGBA")
            fp = os.path.join(framedir, "f%05d.png" % i)
            _compose_frame(frame, cut, bg, blur).save(fp)
            paths.append(fp)
            out({"type": "gif_progress", "done": i + 1, "total": n, "id": rid})
        if _cancel.is_set():
            out({"type": "canceled", "scope": "gif", "id": rid})
            return
        _frames_to_gif(paths, dst, durations, loop, bg == "transparent")
        out({"type": "gif_done", "output": dst, "frames": n,
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
    return any(_name(p) in _GPU_PROVIDERS for p in preferred_providers())


def handle_seg_load(req):
    global _seg
    rid = req.get("id")
    path = req["input"]
    want = req.get("model", "auto")
    pil = Image.open(path).convert("RGB")
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
            preferred_providers(), _gpu_hint(), want=want, probe_image=pil,
            on_step=on_step, on_progress=on_progress)
    except _Canceled:
        out({"type": "seg_canceled", "id": rid})
        return
    prov = session.provider()
    _seg = {"session": session, "image": pil, "rung": rung, "size": pil.size,
            "maskdir": maskdir, "objects": [], "prev_low": None}
    out({"type": "seg_ready", "rung": rung, "model": seg.MODELS[rung]["label"],
         "provider": prov, "label": _PROVIDER_LABELS.get(prov, prov),
         "gpu": prov in _GPU_PROVIDERS, "mode": mode, "family": session.family,
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
    gpu = session.provider() in _GPU_PROVIDERS
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
        _sessions.clear()
        _cpu_sessions.clear()
        _fallback_notified.clear()
        freed.append("bg")
    if scope in ("seg", "all") and _seg is not None:
        _seg = None
        freed.append("seg")
    if freed:
        gc.collect()
    out({"type": "unloaded", "scope": freed, "id": rid})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_HANDLERS = {
    "single": handle_single,
    "batch": handle_batch,
    "gif": handle_gif,
    "seg_load": handle_seg_load,
    "seg_everything": handle_seg_everything,
    "seg_point": handle_seg_point,
    "seg_extract": handle_seg_extract,
    "unload": handle_unload,
}
_SEG_OPS = {"seg_load", "seg_everything", "seg_point", "seg_extract"}
_BG_OPS = {"single", "batch", "gif"}


def _ready():
    out({"type": "ready",
         "models": list(sessions_names) if _HAVE_REMBG else [],
         "providers": provider_names(),
         "bgremove": _HAVE_REMBG,
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
        if op in _BG_OPS and not _HAVE_REMBG:
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
