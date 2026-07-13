#!/usr/bin/env python3
"""Differential test: engine.pane.Pane vs the original viewer.py transform maths.

The reference below is a verbatim copy of `ImageView.widget_to_image` as it was
before the extraction. If the engine ever disagrees with it, the cursor lands on
the wrong pixel — a silent, subtly-wrong bug — so this is checked exhaustively
over random states rather than a couple of hand-picked points.

Stdlib only; no GTK. Run: python tests/test_engine_pane.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "bgbg"))
from engine.pane import Pane  # noqa: E402


def reference_widget_to_image(w, h, iw, ih, zoom, ox, oy, rot, fh, fv, px, py):
    """The pre-extraction implementation, verbatim."""
    ew, eh = (ih, iw) if rot % 2 == 1 else (iw, ih)
    if not ew or not eh or not w or not h:
        return None
    scale = min(w / ew, h / eh) * zoom
    if scale == 0:
        return None
    ux = (px - (w / 2 + ox)) / scale
    uy = (py - (h / 2 + oy)) / scale
    for _ in range(rot % 4):
        ux, uy = uy, -ux
    if fh:
        ux = -ux
    if fv:
        uy = -uy
    return ux + iw / 2, uy + ih / 2


def make(rng):
    p = Pane()
    p.set_image_size(rng.randint(1, 4000), rng.randint(1, 4000))
    p.set_view_size(rng.randint(1, 3000), rng.randint(1, 3000))
    p.zoom = rng.uniform(0.05, 40.0)
    p.ox = rng.uniform(-2000, 2000)
    p.oy = rng.uniform(-2000, 2000)
    p.rot = rng.randint(0, 3)
    p.fh = rng.random() < 0.5
    p.fv = rng.random() < 0.5
    return p


def main():
    rng = random.Random(20260713)
    n = 10000
    worst_ref = 0.0
    worst_rt = 0.0

    for _ in range(n):
        p = make(rng)
        px = rng.uniform(-500, p.view_w + 500)
        py = rng.uniform(-500, p.view_h + 500)

        got = p.view_to_image(px, py)
        want = reference_widget_to_image(
            p.view_w, p.view_h, p.image_w, p.image_h,
            p.zoom, p.ox, p.oy, p.rot, p.fh, p.fv, px, py)
        assert (got is None) == (want is None), "None-ness disagrees"
        if got is not None:
            d = max(abs(got[0] - want[0]), abs(got[1] - want[1]))
            worst_ref = max(worst_ref, d)
            assert d < 1e-9, "engine disagrees with reference by %g" % d

            # round-trip: image -> view -> image must be the identity
            back = p.image_to_view(*got)
            again = p.view_to_image(*back)
            r = max(abs(again[0] - got[0]), abs(again[1] - got[1]))
            worst_rt = max(worst_rt, r)
            assert r < 1e-6, "round-trip drifted by %g" % r

    # a few structural invariants
    p = Pane()
    p.set_image_size(100, 50)
    p.set_view_size(400, 400)
    assert p.effective_size() == (100, 50)
    p.rot = 1
    assert p.effective_size() == (50, 100), "odd quarter-turn must swap axes"

    p = Pane()                       # no image -> no coords
    p.set_view_size(400, 400)
    assert p.view_to_image(10, 10) is None
    assert p.image_to_view(10, 10) is None

    p = Pane()                       # not laid out -> no coords (scale would be 0)
    p.set_image_size(100, 100)
    p.set_view_size(0, 0)
    assert p.view_to_image(10, 10) is None

    p = Pane()                       # zoom_at pins the anchor point
    p.set_image_size(200, 200)
    p.set_view_size(400, 400)
    before = p.view_to_image(120, 300)
    p.zoom_at(1.1, 120, 300)
    after = p.view_to_image(120, 300)
    assert max(abs(before[0] - after[0]), abs(before[1] - after[1])) < 1e-9, \
        "cursor-anchored zoom must keep the anchor pixel under the cursor"

    p.zoom = 39.9                    # zoom clamps
    p.zoom_at(10.0, 200, 200)
    assert p.zoom == 40.0
    p.zoom = 0.06
    p.zoom_at(0.01, 200, 200)
    assert p.zoom == 0.05

    p = Pane()                       # flip is relative to what the user SEES
    p.set_image_size(100, 50)
    p.set_view_size(400, 400)
    p.rot = 1                        # odd turn -> "horizontal" means the other axis
    p.flip(True)
    assert p.fv is True and p.fh is False

    print("engine.pane: %d random states OK" % n)
    print("  max delta vs reference : %.2e" % worst_ref)
    print("  max round-trip drift   : %.2e" % worst_rt)
    print("  invariants             : effective_size, no-image, unlaid-out,")
    print("                           zoom anchor, zoom clamp, rotated flip")
    print("PANE OK")


if __name__ == "__main__":
    main()
