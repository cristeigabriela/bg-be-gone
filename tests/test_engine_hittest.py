#!/usr/bin/env python3
"""engine.hittest — id unpacking, bounds, and the real scene's maps.

Two halves:
  1. synthetic buffers (pure stdlib) — pin the R + G*256 encoding, the bounds
     behaviour, and that each map is bounded by its OWN dimensions;
  2. the actual maps emitted by segmentation.save_objects for spec/assets/scene —
     so the encoding contract is verified against production output, not against
     my idea of it.

Run: python tests/test_engine_hittest.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src", "bgbg"))
from engine.hittest import PixelMap, HitMaps  # noqa: E402

SCENE = os.path.join(ROOT, "spec", "assets", "scene")


def synthetic():
    w, h, nc = 4, 3, 3
    stride = w * nc
    data = bytearray(stride * h)

    def put(x, y, oid):
        i = y * stride + x * nc
        data[i] = oid & 0xFF
        data[i + 1] = (oid >> 8) & 0xFF

    put(0, 0, 1)
    put(1, 0, 255)          # boundary of the low byte
    put(2, 0, 256)          # pure high byte -> proves G is the *high* byte
    put(3, 0, 65535)        # max id
    put(1, 2, 300)

    m = PixelMap(bytes(data), stride, nc, w, h)
    assert m.id_at(0, 0) == 1
    assert m.id_at(1, 0) == 255
    assert m.id_at(2, 0) == 256, "G must be the high byte (R + G*256)"
    assert m.id_at(3, 0) == 65535
    assert m.id_at(1, 2) == 300
    assert m.id_at(0, 1) == 0, "unset pixel is background"

    for x, y in ((-1, 0), (0, -1), (w, 0), (0, h), (99, 99)):
        assert m.id_at(x, y) == 0, "out of bounds must be background, not a crash"
        assert m.value_at(x, y) == 0

    # a single-channel map (depth): value_at reads the count, no high byte
    d = PixelMap(bytes([0, 1, 2, 5]), 4, 1, 4, 1)
    assert [d.value_at(i, 0) for i in range(4)] == [0, 1, 2, 5]

    # each map is bounded by its OWN size (the pre-extraction code bounded the
    # general/depth maps by the *label* map's dimensions)
    big = PixelMap(bytes(3 * 3 * 4), 3 * 4, 3, 4, 3)
    small = PixelMap(bytes(3 * 1 * 1), 3 * 1, 3, 1, 1)
    maps = HitMaps(label=big, general=small, depth=small)
    assert maps.specific_at(3, 2) == 0          # inside label, unset
    assert maps.general_at(3, 2) == 0           # OUTSIDE general -> 0, not a read
    assert maps.depth_at(3, 2) == 0

    empty = HitMaps()                            # nothing loaded
    assert not empty.loaded
    assert empty.specific_at(0, 0) == 0 and empty.depth_at(0, 0) == 0
    print("  synthetic: encoding, bounds, per-map dims, empty  OK")


def real_scene():
    import gi
    gi.require_version("GdkPixbuf", "2.0")
    from gi.repository import GdkPixbuf

    with open(os.path.join(SCENE, "meta.json")) as f:
        meta = json.load(f)

    def load(name):
        pb = GdkPixbuf.Pixbuf.new_from_file(os.path.join(SCENE, name))
        return PixelMap(pb.get_pixels(), pb.get_rowstride(),
                        pb.get_n_channels(), pb.get_width(), pb.get_height())

    maps = HitMaps(label=load(meta["maps"]["label"]),
                   general=load(meta["maps"]["general"]),
                   depth=load(meta["maps"]["depth"]))
    W, H = meta["size"]

    # (46,52) is the centre of `core` (obj 3), nested inside `field` (obj 1)
    h = maps.hit(46, 52)
    assert h.specific == 3, "smallest object must win the label map, got %r" % h
    assert h.general == 1, "largest object must win the general map, got %r" % h
    assert h.depth == 2, "two objects overlap here, got %r" % h
    assert h.stacked, "this point is exactly the whole-vs-part case"

    # (58, 85) is inside `field` only (below the nested core)
    h = maps.hit(58, 85)
    assert h.specific == 1 and h.general == 1 and h.depth == 1, repr(h)
    assert not h.stacked

    # (120, 80) is inside `island` (obj 2), a separate object
    h = maps.hit(120, 80)
    assert h.specific == 2 and h.general == 2 and h.depth == 1, repr(h)

    # a corner is background
    h = maps.hit(2, 2)
    assert h.specific == 0 and h.general == 0 and h.depth == 0, repr(h)

    # off the image entirely
    h = maps.hit(W + 10, H + 10)
    assert h.specific == 0 and h.depth == 0

    print("  real scene: nested(46,52)->spec=3 gen=1 depth=2; field; island; bg  OK")


def main():
    print("engine.hittest")
    synthetic()
    real_scene()
    print("HITTEST OK")


if __name__ == "__main__":
    main()
