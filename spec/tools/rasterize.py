#!/usr/bin/env python3
"""Deterministic offscreen render of ImageView at a fixed state + time.

This is the regression gate for the engine extraction: it freezes what today's
`viewer.py` draws, so every later refactor (display list, GSK interpreter,
alpha-coverage masks) can be proven pixel-identical — or its delta reviewed.

Runs on the SYSTEM python (gi), like the GTK shell — no numpy, no PIL.

The widget has no frame clock offscreen, so `ImageView._now()` returns 0. Every
animation timestamp in viewer.py is absolute frame-clock µs, so a fixture states
ages ("pressed 300 ms ago") and we write them as negative µs. That makes the
whole animation state addressable without a running main loop.

    python spec/tools/rasterize.py --freeze     # write goldens
    python spec/tools/rasterize.py --check      # compare against goldens
    python spec/tools/rasterize.py --list
"""
import os
import sys
import json
import argparse

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gsk", "4.0")
gi.require_version("Graphene", "1.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, Gdk, Gsk, Graphene, GdkPixbuf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.dirname(HERE)
ROOT = os.path.dirname(SPEC)
SCENE = os.path.join(SPEC, "assets", "scene")
FIXTURES = os.path.join(SPEC, "fixtures")
GOLDENS = os.path.join(SPEC, "goldens", "render")

sys.path.insert(0, os.path.join(ROOT, "src", "bgbg"))
from shell_gtk.canvas import ImageView  # noqa: E402
from engine.color import parse_color  # noqa: E402


def _ms_ago(ms):
    """A frame-clock timestamp `ms` milliseconds in the past (now == 0)."""
    return None if ms is None else -int(ms * 1000)


def load_scene():
    with open(os.path.join(SCENE, "meta.json")) as f:
        meta = json.load(f)
    objs = []
    for o in meta["objects"]:
        o = dict(o)
        o["mask"] = os.path.join(SCENE, o["mask"])
        objs.append(o)
    maps = {k: os.path.join(SCENE, v) for k, v in meta["maps"].items()}
    return meta, objs, maps


def build_view(fx, meta, objs, maps):
    """An ImageView with the scene loaded and the fixture's state applied.

    The widget is still the thing under test — it is what ships — but all the
    state now lives in the engine session behind it, so that is what a fixture
    addresses.
    """
    v = ImageView()
    vw, vh = fx["view"]
    v.get_width = lambda: vw          # offscreen: no allocation
    v.get_height = lambda: vh
    v.session.clock = lambda: 0       # pin the clock: fixtures state ages as
    #                                   negative µs relative to "now"
    anim, objects = v.session.anim, v.session.objects

    v.set_pixbuf(GdkPixbuf.Pixbuf.new_from_file(
        os.path.join(SCENE, meta["source"])))

    mode = fx.get("seg_mode")
    if mode:
        v.set_seg_mode(mode)
        v.set_seg_layers(objs, maps["label"], maps["general"], maps["depth"])

    p = fx.get("pane", {})
    v.zoom = float(p.get("zoom", 1.0))
    v.ox = float(p.get("ox", 0.0))
    v.oy = float(p.get("oy", 0.0))
    v.rot = int(p.get("rot", 0))
    v.fh = bool(p.get("fh", False))
    v.fv = bool(p.get("fv", False))

    a = fx.get("anim", {})
    anim.pulse = float(a.get("pulse", 0.5))
    anim.ant = float(a.get("ant", 0.0))
    anim.scan_phase = float(a.get("scan_phase", 0.0))
    anim.reveal = float(a.get("reveal", 1.0))   # set_seg_layers begins a reveal
    anim.reveal_t0 = None
    anim.scanning = bool(fx.get("scanning", False))

    v.set_seg_selection(fx.get("selected", []))

    h = fx.get("hover")
    if h:
        objects.hover_gen = int(h.get("gen", 0))
        objects.hover_spec = int(h.get("spec", 0))
        objects.hover_depth = int(h.get("depth", 0))
        anim.drilled = bool(h.get("drilled", False))
        objects.hover_id = (objects.hover_spec if anim.drilled
                            else objects.hover_gen)

    for oid, ms in (fx.get("pop") or {}).items():
        anim.pop[int(oid)] = _ms_ago(ms)

    pr = fx.get("press")
    if pr:
        anim.press_obj = int(pr["obj"])
        anim.press_pt = tuple(pr["pt"])
        anim.press_t0 = _ms_ago(pr["press_ms_ago"])
        anim.release_t0 = _ms_ago(pr.get("release_ms_ago"))

    mo = fx.get("morph")
    if mo:
        # the rings/colours the tween interpolates between
        objects.morph = objects.build_morph(int(mo["from"]), int(mo["to"]))
        anim.morph_t0 = _ms_ago(mo["ms_ago"])

    comp = fx.get("composite")
    if comp:
        # The outputter's LOCAL preview (engine/outputs.py): the source clipped
        # to a union of masks, over a solid fill or the bare checkerboard. This
        # is what the Segment result panel now draws instead of round-tripping a
        # PNG through the worker, so it needs to be pinned like everything else.
        masks = [v.textures.masks[i] for i in comp["ids"]
                 if i in v.textures.masks]
        bg = parse_color(comp["bg"]) if comp.get("bg") else None
        v.set_composite(v.pixbuf, masks, bg)

    pm = fx.get("point_mask")
    if pm:
        obj = next(o for o in objs if o["id"] == pm)
        v.set_point_mask(obj["mask"], obj["contour"])
        anim.reveal = float(a.get("reveal", 1.0))
    return v


def render(fx, meta, objs, maps):
    v = build_view(fx, meta, objs, maps)
    vw, vh = fx["view"]
    snap = Gtk.Snapshot()
    v.do_snapshot(snap)
    node = snap.to_node()
    r = Gsk.CairoRenderer()        # software: deterministic, GPU-independent
    r.realize(None)
    try:
        if node is None:           # nothing drawn (e.g. no image)
            node = Gtk.Snapshot().to_node()
        tex = r.render_texture(node, Graphene.Rect().init(0, 0, vw, vh))
        return tex.save_to_png_bytes().get_data()
    finally:
        r.unrealize()


def fixtures():
    out = []
    for fn in sorted(os.listdir(FIXTURES)):
        if fn.endswith(".json"):
            with open(os.path.join(FIXTURES, fn)) as f:
                fx = json.load(f)
            fx.setdefault("name", fn[:-5])
            out.append(fx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    Gtk.init()
    meta, objs, maps = load_scene()
    fxs = fixtures()

    if args.list:
        for fx in fxs:
            print(fx["name"])
        return 0

    os.makedirs(GOLDENS, exist_ok=True)
    bad = 0
    for fx in fxs:
        png = render(fx, meta, objs, maps)
        path = os.path.join(GOLDENS, fx["name"] + ".png")
        if args.freeze:
            # determinism: the same fixture must render byte-identically twice
            again = render(fx, meta, objs, maps)
            if again != png:
                print("NON-DETERMINISTIC %s" % fx["name"])
                bad += 1
                continue
            with open(path, "wb") as f:
                f.write(png)
            print("froze  %-28s %6d B" % (fx["name"], len(png)))
        elif args.check:
            if not os.path.exists(path):
                print("MISSING GOLDEN %s" % fx["name"])
                bad += 1
                continue
            with open(path, "rb") as f:
                want = f.read()
            if want == png:
                print("ok     %s" % fx["name"])
            else:
                print("DIFF   %s" % fx["name"])
                with open(path + ".actual", "wb") as f:
                    f.write(png)
                bad += 1
    if bad:
        print("\n%d problem(s)" % bad)
    else:
        print("\nall %d fixtures %s" % (len(fxs), "frozen" if args.freeze else "match"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
