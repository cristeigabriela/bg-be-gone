#!/usr/bin/env python3
"""Selecting several objects must not dim any of them.

The regression this guards was subtle and shipped for a long time. Object masks
were written as `L` PNGs, which load back *opaque*. "Dim outside the selection"
builds its mask by stacking one texture per selected object inside a single mask
node — but stacking opaque textures makes each one REPLACE the last, so the mask
collapsed to whichever object happened to be iterated last. Every other selected
object got the 0.55 black dim painted over it, underneath its own tint.

It never looked obviously broken — it read as "the highlight goes muddy when I
select several objects" — which is exactly why it needs a number, not an eyeball.

Masks are now alpha-coverage (white RGB, alpha = coverage), so stacking them
alpha-composites into a real union. GSK keys the mask off ALPHA, and so does
Canvas2D's destination-in, which is the other half of why the change matters.

Run: python tests/test_mask_union.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "spec", "tools"))

import gi  # noqa: E402
gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, GdkPixbuf  # noqa: E402

Gtk.init()
import rasterize as R  # noqa: E402

# image (58, 80): well inside object 1 ("field"), clear of the nested core
PROBE = (58, 80)


def luma(selected):
    fx = {"name": "probe", "view": [320, 240], "seg_mode": "everything",
          "selected": selected, "anim": {"pulse": 0.6, "ant": 8.0}}
    meta, objs, maps = R.load_scene()
    png = R.render(fx, meta, objs, maps)
    tmp = "/tmp/_union_probe.png"
    with open(tmp, "wb") as f:
        f.write(png)
    pb = GdkPixbuf.Pixbuf.new_from_file(tmp)
    px, rs, nc = pb.get_pixels(), pb.get_rowstride(), pb.get_n_channels()
    # image px -> view px: 160x120 image in a 320x240 view at zoom 1 => scale 2
    i = (PROBE[1] * 2) * rs + (PROBE[0] * 2) * nc
    r, g, b = px[i], px[i + 1], px[i + 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def main():
    alone = luma([1])            # object 1 selected by itself
    with_2 = luma([1, 2])        # object 1 selected alongside another
    with_23 = luma([1, 2, 3])    # ...and another
    unselected = luma([2])       # object 1 NOT selected -> must be dimmed

    print("object 1, selected alone        luma=%.1f" % alone)
    print("object 1, selected with 2       luma=%.1f" % with_2)
    print("object 1, selected with 2 and 3 luma=%.1f" % with_23)
    print("object 1, NOT selected          luma=%.1f" % unselected)

    drift2 = abs(alone - with_2) / alone
    drift3 = abs(alone - with_23) / alone
    print()
    print("  drift when co-selected with 1 other : %.1f%%" % (100 * drift2))
    print("  drift when co-selected with 2 others: %.1f%%" % (100 * drift3))

    assert drift2 < 0.02, \
        "co-selecting dims object 1 by %.1f%% — the mask is not a union" % (100 * drift2)
    assert drift3 < 0.02, \
        "co-selecting dims object 1 by %.1f%% — the mask is not a union" % (100 * drift3)
    # and the dim must still WORK for things that aren't selected
    assert unselected < alone * 0.75, \
        "unselected objects must still be dimmed (got %.1f vs %.1f)" % (
            unselected, alone)

    print("\nMASK UNION OK")


if __name__ == "__main__":
    main()
