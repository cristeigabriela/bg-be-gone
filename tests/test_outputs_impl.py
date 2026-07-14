#!/usr/bin/env python3
"""The outputter's pixels, and their parity with what the engine draws.

`engine/outputs.py` says what an effect *is*; `compute/outputs_impl.py` renders
it. If the two disagree, the Segment preview lies about the file you are about to
save — and that is not hypothetical: the display-list builder masked the solid
background to the subject, so it rendered a bare checkerboard where the worker
wrote green. The render goldens now pin the engine's half; this pins the worker's.

It also guards the dedup: "put a background behind the cutout" used to be written
out four times (single, GIF, segment-extract, and a spare copy of _hex_to_rgb).
They all call `apply()` now, so one wrong branch would break every path at once.

Needs PIL, which lives in the venv — so it re-execs itself there.
Run: python tests/test_outputs_impl.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "src", "bgbg")

try:
    from PIL import Image
except ImportError:                       # the system python has no PIL, by design
    venv = (os.environ.get("BGBG_VENV_PYTHON")
            or os.path.expanduser("~/.local/share/bg-be-gone/venv/bin/python"))
    if not os.path.exists(venv) or os.environ.get("_BGBG_REEXEC"):
        print("SKIP: no venv with PIL at %s" % venv)
        sys.exit(0)
    os.environ["_BGBG_REEXEC"] = "1"
    os.execv(venv, [venv, os.path.abspath(__file__)])

sys.path.insert(0, SRC)
from compute import outputs_impl as OI    # noqa: E402
from engine import outputs as OUT         # noqa: E402

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print("  %-54s %s" % (label, "PASS" if ok else "FAIL"))
    if not ok:
        print("      want: %r" % (want,))
        print("      got:  %r" % (got,))


# A 8x8 red source; the left half is the "subject" (alpha 255), the right half
# is background (alpha 0).
SRC_RGB = (200, 30, 40)
SUBJ = (1, 1)          # a pixel inside the subject
BACK = (6, 6)          # a pixel outside it


def scene():
    src = Image.new("RGB", (8, 8), SRC_RGB)
    alpha = Image.new("L", (8, 8), 0)
    for y in range(8):
        for x in range(4):
            alpha.putpixel((x, y), 255)
    return src, alpha


def test_transparent():
    src, alpha = scene()
    img = OI.apply(OI.cutout(src, alpha), "transparent")
    check("transparent keeps an alpha channel", img.mode, "RGBA")
    check("... the subject is opaque and unchanged",
          img.getpixel(SUBJ), SRC_RGB + (255,))
    check("... and the background is fully transparent",
          img.getpixel(BACK)[3], 0)


def test_solid():
    src, alpha = scene()
    img = OI.apply(OI.cutout(src, alpha), "solid", color="#00b140")
    check("a solid background drops alpha", img.mode, "RGB")
    check("... the subject survives untouched", img.getpixel(SUBJ), SRC_RGB)
    # The bug the golden caught: the fill must cover the WHOLE canvas, so a
    # background pixel is the colour — not the checkerboard, not the source.
    check("... and the background IS the colour, everywhere the subject is not",
          img.getpixel(BACK), (0, 177, 64))


def test_blur():
    src, alpha = scene()
    # a source with structure, so a blur is detectable
    src.putpixel(BACK, (0, 0, 255))
    img = OI.apply(OI.cutout(src, alpha), "blur", source=src, strength=4)
    check("blur drops alpha too", img.mode, "RGB")
    check("... the subject stays SHARP (it is composited on top, not blurred)",
          img.getpixel(SUBJ), SRC_RGB)
    check("... and the background is blurred (the lone blue pixel is smeared)",
          img.getpixel(BACK) != (0, 0, 255), True)

    try:
        OI.apply(OI.cutout(src, alpha), "blur")     # no source
    except ValueError:
        check("... and blur without the source pixels is an error, not garbage",
              True, True)
    else:
        check("... and blur without the source pixels is an error, not garbage",
              False, True)


def test_apply_bg_matches_the_engine():
    """apply_bg is keyed by the `bg` string the protocol carries; it must land on
    the same effect the engine resolved for the same string."""
    src, alpha = scene()
    cut = OI.cutout(src, alpha)
    for bg in ("transparent", "blur", "#00b140"):
        eff = OUT.resolve(bg)
        img = OI.apply_bg(cut, bg, source=src, blur=4)
        want_mode = "RGBA" if eff.id == "transparent" else "RGB"
        check("bg=%-12r -> the engine says %-11r and the worker renders %s"
              % (bg, eff.id, want_mode), img.mode, want_mode)

    # and the colour the engine would FILL with is the colour the worker paints
    img = OI.apply_bg(cut, "#00b140")
    fill = OUT.resolve("#00b140").fill
    check("the engine's display-list fill == the worker's background pixel",
          tuple(round(c * 255) for c in fill[:3]), img.getpixel(BACK))


def main():
    print("transparent")
    test_transparent()
    print("solid")
    test_solid()
    print("blur")
    test_blur()
    print("engine <-> worker parity")
    test_apply_bg_matches_the_engine()
    print()
    if FAILED:
        print("OUTPUTS_IMPL FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("OUTPUTS_IMPL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
