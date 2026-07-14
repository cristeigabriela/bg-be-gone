#!/usr/bin/env python3
"""The outputter: which effects exist, and which ones cost a round-trip.

The interesting property is the LOCAL/COMPUTE split. Transparent and solid are
display-list ops (a mask and a fill), so the renderer draws them itself and the
Segment preview is instant. Blur needs the actual pixels, so it — and only it —
still goes to the compute port.

The parity that matters is between `engine.outputs` (what the engine draws) and
`compute.outputs_impl` (what the worker saves). If they disagree, the preview
lies about the file you are about to save. They disagreed once already: the
builder masked the background fill to the *subject*, so a solid background
rendered as a bare checkerboard. That is now pinned by the render goldens
(23/24/25_output_*) and by the shape of this file.

Run: python tests/test_engine_outputs.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "bgbg"))

from engine import outputs as OUT                                # noqa: E402

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print("  %-52s %s" % (label, "PASS" if ok else "FAIL"))
    if not ok:
        print("      want: %r" % (want,))
        print("      got:  %r" % (got,))


def test_registry():
    ids = sorted(e.id for e in OUT.OUTPUTS.all())
    check("three effects are registered", ids, ["blur", "solid", "transparent"])
    check("blur is the only one that needs the pixels",
          sorted(e.id for e in OUT.OUTPUTS.all() if not e.local), ["blur"])
    check("... and it says so", OUT.BLUR.requires, ("source_pixels",))
    check("blur's strength is a bounded parameter",
          [(p.name, p.minimum, p.maximum, p.default) for p in OUT.BLUR.params],
          [("strength", 2, 80, 20)])


def test_resolve():
    """The `bg` setting -> an effect. The sidebar stores a colour AS the value,
    so anything that is not a named effect is a solid colour."""
    t = OUT.resolve("transparent")
    check("transparent -> the transparent effect", t.id, "transparent")
    check("... it is LOCAL — no round-trip", t.local, True)
    check("... and it paints no fill (the checkerboard shows through)",
          t.fill, None)

    s = OUT.resolve("#00b140")
    check("a colour -> the solid effect", s.id, "solid")
    check("... it is LOCAL too — this is what kills the 150ms debounce",
          s.local, True)
    check("... and it paints that colour, parsed for the display list",
          s.fill, (0.0, 177 / 255.0, 64 / 255.0, 1.0))

    b = OUT.resolve("blur", blur=35)
    check("blur -> the blur effect", b.id, "blur")
    check("... it is NOT local: a display list cannot Gaussian-blur",
          b.local, False)
    check("... it carries the strength through to the worker",
          b.params, {"strength": 35})
    check("... and it has no local fill", b.fill, None)

    check("the blur strength defaults to the spinner's default",
          OUT.resolve("blur").params, {"strength": 20})


def main():
    print("the registry")
    test_registry()
    print("resolve(): the bg setting -> an effect")
    test_resolve()
    # Parity with the worker's pixels is NOT checked here: this file is
    # stdlib-only (it is the engine's half), and an import-it-and-assert-a-
    # string-is-a-string "parity" check would prove nothing. The real one —
    # the engine's display-list fill vs the worker's background pixel — needs
    # PIL and lives in tests/test_outputs_impl.py.
    print()
    if FAILED:
        print("OUTPUTS FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("OUTPUTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
