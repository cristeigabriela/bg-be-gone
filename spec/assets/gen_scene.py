#!/usr/bin/env python3
"""Generate the deterministic render-golden scene.

Runs in the worker venv (numpy + pillow + segmentation). The object masks and the
three lookup maps are produced by the REAL `segmentation.save_objects`, so the
fixture encoding (soft L masks, R+G*256 id maps, depth map, contours, hash
colours) is byte-for-byte what the worker emits in production.

The scene is chosen to exercise every visual the renderer has:
  - obj 1 "field"  : the largest blob                      -> general object of a stack
  - obj 2 "island" : two disconnected squares              -> multi-polygon contour + holes
  - obj 3 "core"   : a disc nested inside obj 1            -> depth >= 2 (layered masks)
Selecting 1 + 2 together exercises the mask-union path (today's last-wins bug).

    <venv>/bin/python spec/assets/gen_scene.py
"""
import os
import sys
import json

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "bgbg"))
from compute import maskops as seg  # noqa: E402

W, H = 160, 120
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene")


def _disc(cx, cy, r):
    ys, xs = np.mgrid[0:H, 0:W]
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r


def _rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def source_image():
    """A deterministic synthetic photo: vertical gradient + two solid shapes."""
    ys, xs = np.mgrid[0:H, 0:W]
    r = (40 + ys * 0.9).astype(np.uint8)
    g = (70 + xs * 0.4).astype(np.uint8)
    b = np.full((H, W), 120, np.uint8)
    img = np.dstack([r, g, b])
    img[_disc(58, 62, 34)] = (210, 190, 120)      # under obj1
    img[_disc(46, 52, 13)] = (240, 110, 90)       # under obj3 (the nested core)
    img[_rect(112, 24, 136, 48)] = (90, 200, 170)  # under obj2 part A
    img[_rect(112, 70, 136, 94)] = (90, 200, 170)  # under obj2 part B
    return Image.fromarray(img, "RGB")


def main():
    os.makedirs(OUT, exist_ok=True)

    field = _disc(58, 62, 34)                              # large blob
    island = _rect(112, 24, 136, 48) | _rect(112, 70, 136, 94)   # 2 components
    # Nested inside `field`, but deliberately OFF-CENTRE: _obj_color hashes the
    # centroid bucket (cx//8, cy//8), so a concentric child would hash to the same
    # colour as its parent and the layered-mask visual (general vs specific in
    # distinct colours) would be untestable.
    core = _disc(46, 52, 13)

    # save_objects expects masks LARGEST-FIRST (it paints smallest last, so the
    # smallest ends up on top in the `label` map).
    masks = sorted([field, island, core], key=lambda m: -int(m.sum()))
    names = {int(field.sum()): "field", int(island.sum()): "island",
             int(core.sum()): "core"}

    src = source_image()
    src.save(os.path.join(OUT, "source.png"))

    maps, objs = seg.save_objects(masks, (W, H), OUT, "s")

    for o in objs:                                # paths -> repo-relative, stable
        o["name"] = names[int(masks[o["id"] - 1].sum())]
        o["mask"] = os.path.basename(o["mask"])
    meta = {
        "size": [W, H],
        "source": "source.png",
        "maps": {k: os.path.basename(v) for k, v in maps.items()},
        "objects": objs,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)

    print("scene -> %s" % OUT)
    for o in objs:
        print("  obj %d %-7s area=%-6d contour_polys=%d colour=%s"
              % (o["id"], o["name"], o["area"], len(o["contour"]), o["color"]))
    dep = np.array(Image.open(os.path.join(OUT, meta["maps"]["depth"])))
    print("  depth: max=%d  pixels_with_depth>=2=%d" % (dep.max(), int((dep >= 2).sum())))


if __name__ == "__main__":
    main()
