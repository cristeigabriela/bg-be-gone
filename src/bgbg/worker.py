#!/usr/bin/env python3
"""Background-removal worker for bg-be-gone.

Runs inside the bundled virtualenv. Speaks line-delimited JSON on
stdin/stdout so the GTK frontend can keep a model resident on the GPU across
requests.

Requests (one JSON object per line on stdin):
  {"op":"single","input":P,"output":P,"model":M,"alpha":bool,"bg":BG,"id":N}
  {"op":"batch","input_dir":D,"output_dir":D,"model":M,"alpha":bool,"bg":BG,
   "pattern":"{name}_nobg","id":N}
  {"op":"models"}
  {"op":"shutdown"}

BG is "transparent" or "#rrggbb" to flatten onto.

Responses (one JSON object per line on stdout):
  {"type":"ready","models":[...],"providers":[...]}
  {"type":"loading","model":M,"id":N}
  {"type":"device","provider":P,"label":L,"id":N}
  {"type":"progress","done":i,"total":n,"name":str,"id":N}
  {"type":"done_single","output":P,"preview":P,"seconds":f,"id":N}
  {"type":"done_batch","count":n,"seconds":f,"outdir":P,"id":N}
  {"type":"error","message":str,"id":N}
"""
import os
import sys
import glob


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

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

from PIL import Image, ImageFilter  # noqa: E402
import onnxruntime as ort  # noqa: E402
from rembg import new_session, remove  # noqa: E402
from rembg.sessions import sessions_names  # noqa: E402

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


def process_one(session, src, dst, alpha, bg, blur=20, want_preview=False):
    img = Image.open(src)
    res = remove(img, session=session, alpha_matting=alpha).convert("RGBA")

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
    session = get_session(req["model"], rid)
    t = time.time()
    preview = process_one(session, req["input"], req["output"],
                          req.get("alpha", False), req.get("bg", "transparent"),
                          req.get("blur", 20), want_preview=True)
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
    session = get_session(req["model"], rid)
    alpha = req.get("alpha", False)
    bg = req.get("bg", "transparent")
    blur = req.get("blur", 20)
    t = time.time()
    for i, f in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            stem = pattern.format(name=name, n=i)
        except Exception:
            stem = name + "_nobg"
        dst = os.path.join(outdir, stem + ".png")
        out({"type": "progress", "done": i - 1, "total": total,
             "name": os.path.basename(f), "id": rid})
        process_one(session, f, dst, alpha, bg, blur, want_preview=False)
    out({"type": "progress", "done": total, "total": total, "name": "",
         "id": rid})
    out({"type": "done_batch", "count": total,
         "seconds": round(time.time() - t, 1), "outdir": outdir, "id": rid})


def main():
    out({"type": "ready", "models": list(sessions_names),
         "providers": provider_names()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        op = req.get("op")
        rid = req.get("id")
        try:
            if op == "single":
                handle_single(req)
            elif op == "batch":
                handle_batch(req)
            elif op == "models":
                out({"type": "ready", "models": list(sessions_names),
                     "providers": provider_names(), "id": rid})
            elif op == "shutdown":
                break
        except Exception as e:
            out({"type": "error",
                 "message": f"{e}\n{traceback.format_exc()}", "id": rid})


if __name__ == "__main__":
    main()
