#!/usr/bin/env python3
"""Freeze/check the CV and model_io goldens — the TypeScript port's worklist.

`compute/maskops.py` is the half of segmentation that needs no model and no GPU:
given a mask, produce contours, labelmaps, id maps, NMS decisions. All of it has
to be rewritten in TypeScript for the web build, and none of it is covered by the
render goldens (those start *after* the objects exist).

So: fixed synthetic masks in, every derived value out, frozen as JSON. `maskops.ts`
is correct when it reproduces this file. A diff points at the exact kernel that
drifted — a contour that wound the other way, an 8-vs-4-connected label, an NMS
tie broken differently — instead of at "the segmentation looks a bit off".

The model_io half pins `sam.preprocess`, which is the other silent-failure
surface: a mis-scaled or mis-normalised input tensor does not raise, it just
segments the wrong thing.

Needs numpy + PIL, so it re-execs into the venv.

    python spec/tools/cvgold.py --freeze
    python spec/tools/cvgold.py --check
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.dirname(HERE)
ROOT = os.path.dirname(SPEC)
SRC = os.path.join(ROOT, "src", "bgbg")

def _reexec_in_venv():
    """Run inside the worker's venv, with the CUDA libs on the loader path.

    Two traps, both of which cost real time if rediscovered:
      * the system python happens to HAVE numpy and PIL, so probing for those
        does not tell you whether you can import `sam` — onnxruntime is the one
        that is missing;
      * onnxruntime-gpu links libcudart from inside the pip wheels, so importing
        it outside the worker fails on the dynamic loader unless LD_LIBRARY_PATH
        points at the venv's bundled nvidia/*/lib. service.py does this for
        itself (_ensure_gpu_libs); a tool that imports `sam` must do it too.
    """
    import glob
    venv = (os.environ.get("BGBG_VENV_PYTHON")
            or os.path.expanduser("~/.local/share/bg-be-gone/venv/bin/python"))
    if not os.path.exists(venv) or os.environ.get("_BGBG_REEXEC"):
        print("SKIP: no venv with onnxruntime at %s" % venv)
        sys.exit(0)
    root = os.path.dirname(os.path.dirname(venv))
    libs = glob.glob(os.path.join(root, "lib", "python*", "site-packages",
                                  "nvidia", "*", "lib"))
    env = dict(os.environ, _BGBG_REEXEC="1")
    if libs:
        cur = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            libs + ([cur] if cur else []))
    os.execve(venv, [venv, os.path.abspath(__file__)] + sys.argv[1:], env)


try:
    import numpy as np
    from PIL import Image
    import onnxruntime  # noqa: F401  (only to prove we are in the venv)
except ImportError:
    _reexec_in_venv()

sys.path.insert(0, SRC)
from compute import maskops as M            # noqa: E402
from compute import sam                     # noqa: E402

GOLDEN = os.path.join(SPEC, "goldens", "cv.json")
W = H = 64


# ---------------------------------------------------------------- the masks ---
def disc(cx, cy, r):
    y, x = np.ogrid[:H, :W]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= r * r


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def masks():
    """Deterministic shapes chosen to exercise the awkward cases."""
    big = disc(28, 30, 20)                      # a plain blob
    small = disc(24, 26, 7)                     # nested inside `big` (a stack)
    ring = disc(30, 30, 16) & ~disc(30, 30, 9)  # a HOLE: two contours, one object
    two = disc(8, 8, 5) | disc(56, 56, 5)       # DISJOINT: one mask, two components
    # two squares touching only at a corner — 8-connected says one object,
    # 4-connected says two. Getting this wrong silently changes the object count.
    diag = rect(4, 40, 12, 48) | rect(12, 48, 20, 56)
    return {"big": big, "small": small, "ring": ring,
            "disjoint": two, "diagonal_touch": diag}


# ---------------------------------------------------------------- the dumps ---
def _f(x, n=4):
    return round(float(x), n)


def cv_golden():
    ms = masks()
    out = {}

    for name, m in sorted(ms.items()):
        _, n8 = M._label(m, connectivity=8)     # (labels, count)
        _, n4 = M._label(m, connectivity=4)
        polys = M.contour(m)
        out[name] = {
            "area": int(m.sum()),
            "bbox": [int(v) for v in M._bbox(m)],
            # The kernel is 8-connected. For `diagonal_touch` these two DISAGREE
            # (1 vs 2) — which is the whole point of pinning both: a TS port that
            # reaches for 4-connectivity changes the object count, silently.
            "components_8con": int(n8),
            "components_4con": int(n4),
            "contours": [[[_f(x, 2), _f(y, 2)] for x, y in poly]
                         for poly in polys],
            "contour_points": [len(p) for p in polys],
        }

    # IoU / overlap / stability — the scoring the AMG pass filters on
    a, b = ms["big"], ms["small"]
    out["_scoring"] = {
        "iou_big_small": _f(M._iou(a, b)),
        "iou_self": _f(M._iou(a, a)),
        "bbox_overlap": _f(M._bbox_overlap(M._bbox(a), M._bbox(b))),
        # _stability keys off the logit margin, so feed it a real-ish field
        "stability_sharp": _f(M._stability(
            np.where(a, 6.0, -6.0).astype(np.float32))),
        "stability_soft": _f(M._stability(
            np.where(a, 0.4, -0.4).astype(np.float32))),
    }

    # NMS: records are (score, mask). Feed it the same object twice at different
    # scores (the higher must win and suppress the lower), a nested one that
    # overlaps but not enough to be suppressed, and a disjoint one. The winner
    # and every suppression must be identical in both cores.
    records = [
        (0.90, ms["big"]),
        (0.95, ms["big"]),        # same object, better score -> this one wins
        (0.80, ms["small"]),      # nested: high overlap, low IoU -> survives
        (0.70, ms["disjoint"]),
    ]
    kept = M._nms(records, iou_thresh=0.7)
    out["_nms"] = {
        "kept_scores": [_f(s) for s, _ in kept],
        "kept_areas": [int(m.sum()) for _, m in kept],
        "n_in": len(records), "n_kept": len(kept),
    }

    # The id maps the engine hit-tests against. `general` paints largest-last so
    # hovering a stack gives you the WHOLE; `label` paints smallest-last so
    # dwelling gives you the PART. Swapping them inverts the whole hover UX.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        maps, objs = M.save_objects(
            [ms["big"], ms["small"], ms["disjoint"]], (W, H), td, "g")
        ids = {}
        for key in ("label", "general", "depth"):
            im = Image.open(maps[key])
            px = np.asarray(im.convert("RGB")).astype(np.int32)

            # label/general pack an object id as R + G*256; depth is an L count
            # map. Decoding depth as an id would give 1 + 1*256 = 257 for a depth
            # of one — stable, and completely wrong.
            def at(y, x, p=px, is_id=(key != "depth")):
                return int(p[y, x][0] + p[y, x][1] * 256) if is_id else int(p[y, x][0])

            ids[key] = {
                # inside the nested small disc: specific=small(2), general=big(1),
                # depth=2. If label and general were swapped, hovering a stack
                # would give you the part and dwelling the whole — inside out.
                "at_nested": at(26, 24),
                # in `big` but outside `small`: both maps say 1, depth 1
                "at_big_only": at(42, 28),
                "at_empty": at(2, 40),
            }
        out["_idmaps"] = ids
        out["_objects"] = [
            {"id": o["id"], "color": o["color"], "area": int(o["area"]),
             "bbox": [int(v) for v in o["bbox"]],
             "n_contours": len(o["contour"] or [])}
            for o in objs]
        # masks must be ALPHA-COVERAGE RGBA, not L — an L mask is opaque
        # everywhere, so stacking them unions nothing. (The step-5 bug.)
        out["_mask_format"] = sorted(
            {Image.open(o["mask"]).mode for o in objs})
    return out


def model_io_golden():
    """sam.preprocess — the tensor that goes into the encoder."""
    out = {}
    for name, size in (("landscape", (200, 120)), ("portrait", (90, 160)),
                       ("square", (128, 128))):
        # a deterministic gradient, so the normalisation is observable
        w, h = size
        a = np.zeros((h, w, 3), np.uint8)
        a[..., 0] = (np.arange(w)[None, :] * 255 // max(1, w - 1))
        a[..., 1] = (np.arange(h)[:, None] * 255 // max(1, h - 1))
        a[..., 2] = 128
        pil = Image.fromarray(a, "RGB")

        for family in ("sam2", "sam1"):
            t, meta = sam.preprocess(pil, family)
            out["%s/%s" % (name, family)] = {
                "shape": list(t.shape),
                "letterbox": {"scale": _f(meta["scale"], 6),
                              "nw": meta["nw"], "nh": meta["nh"],
                              "size": list(meta["size"])},
                "min": _f(t.min(), 4), "max": _f(t.max(), 4),
                "mean": _f(t.mean(), 4),
                "sum": _f(t.sum(), 1),
                # The letterbox is TOP-LEFT, so for sam2 the far corner is pad
                # and must be exactly 0.0 — a centred letterbox, or a resize that
                # fills the canvas, would put content there and quietly shift
                # every coordinate the decoder maps back.
                "pad_corner": (_f(t[0, 0, 1023, 1023], 4) if family == "sam2"
                               else None),
                "first_px": _f(t.reshape(-1)[0], 4),
            }
    out["_notes"] = {
        "sam2": "NCHW, ImageNet mean/std, letterboxed TOP-LEFT into 1024x1024",
        "sam1": "HWC 0..255; the encoder normalises and pads itself, but does "
                "NOT resize — the longest side must already be 1024",
        "imagenet_mean": [_f(v, 4) for v in sam._MEAN],
        "imagenet_std": [_f(v, 4) for v in sam._STD],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    got = json.dumps({"cv": cv_golden(), "model_io": model_io_golden()},
                     indent=1, sort_keys=True)

    if args.freeze:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            f.write(got + "\n")
        print("froze %s (%d B)" % (GOLDEN, len(got)))
        return 0

    if not os.path.exists(GOLDEN):
        print("MISSING GOLDEN %s" % GOLDEN)
        return 1
    with open(GOLDEN) as f:
        want = f.read().rstrip("\n")
    if want == got:
        d = json.loads(got)
        print("cv + model_io match (%d mask cases, %d tensor cases)"
              % (sum(1 for k in d["cv"] if not k.startswith("_")),
                 sum(1 for k in d["model_io"] if not k.startswith("_"))))
        return 0
    with open(GOLDEN + ".actual", "w") as f:
        f.write(got + "\n")
    print("CV/MODEL_IO DIFF — wrote %s.actual" % GOLDEN)
    a, b = json.loads(want), json.loads(got)
    for half in ("cv", "model_io"):
        for k in sorted(set(a[half]) | set(b[half])):
            if a[half].get(k) != b[half].get(k):
                print("  %s.%s differs" % (half, k))
    return 1


if __name__ == "__main__":
    sys.exit(main())
