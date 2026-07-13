#!/usr/bin/env python3
"""Freeze/check the DISPLAY-LIST goldens.

The render goldens (rasterize.py) prove the pixels. These prove the *decisions* —
the eased curves, dash offsets, pulse phase, morph lerp, wave schedule — as
structured JSON, before a backend touches them.

That is what makes the TypeScript core a mechanical port rather than an
archaeological one: it must emit byte-identical JSON for the same fixture, and
any disagreement points at the exact op that drifted instead of at a blurry pixel
diff.

    python spec/tools/displaylist.py --freeze
    python spec/tools/displaylist.py --check
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.dirname(HERE)
ROOT = os.path.dirname(SPEC)
sys.path.insert(0, os.path.join(ROOT, "src", "bgbg"))
sys.path.insert(0, HERE)

import gi  # noqa: E402
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

import rasterize as R  # noqa: E402
from engine.render.builder import build  # noqa: E402
from engine.render.codec import to_json, from_json, encode  # noqa: E402

GOLDENS = os.path.join(SPEC, "goldens", "display_list")


def build_for(fx, meta, objs, maps):
    v = R.build_view(fx, meta, objs, maps)
    pane = v._sync_pane()
    return build(v._scene(), pane, v.anim, v._now())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    Gtk.init()
    meta, objs, maps = R.load_scene()
    os.makedirs(GOLDENS, exist_ok=True)

    bad = 0
    fxs = R.fixtures()
    for fx in fxs:
        dl = build_for(fx, meta, objs, maps)
        js = to_json(dl)
        path = os.path.join(GOLDENS, fx["name"] + ".json")

        # a display list must survive a JSON round-trip unchanged, or the wire
        # format is lossy and the TS core would receive something different
        again = to_json(from_json(js))
        if again != js:
            print("LOSSY ROUND-TRIP %s" % fx["name"])
            bad += 1
            continue

        if args.freeze:
            with open(path, "w") as f:
                f.write(js + "\n")
            n = len(json.loads(js)["ops"])
            print("froze  %-28s %5d B  %d top-level ops" % (fx["name"], len(js), n))
        elif args.check:
            if not os.path.exists(path):
                print("MISSING %s" % fx["name"])
                bad += 1
                continue
            with open(path) as f:
                want = f.read().rstrip("\n")
            if want == js:
                print("ok     %s" % fx["name"])
            else:
                print("DIFF   %s" % fx["name"])
                with open(path + ".actual", "w") as f:
                    f.write(js + "\n")
                bad += 1

    print("\n%s" % ("%d problem(s)" % bad if bad else
                    "all %d display lists %s" %
                    (len(fxs), "frozen" if args.freeze else "match")))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
