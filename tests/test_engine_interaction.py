#!/usr/bin/env python3
"""The interaction state machine: scripted events in, expected effects out.

This is the gate for the interaction extraction. Before it, everything the canvas
did in response to a pointer was welded to GTK controllers and could only be
tested by driving a real widget in a real main loop. Now a whole Segment session
— hover, dwell, drill, press, click, select, pan, zoom — is a list of events and
a list of effects, and it runs here with no display, no GTK and no main loop.

The scene is synthetic (raw id-map bytes, no PNG decoding) precisely so that this
file, like the engine, needs nothing but the standard library.

Run: python tests/test_engine_interaction.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "bgbg"))

from engine.hittest import PixelMap                                # noqa: E402
from engine.session import EngineSession                           # noqa: E402
from engine.effects import (                                       # noqa: E402
    Redraw, RequestTick, StopTick, SetCursor, GrabFocus, ViewChanged,
    ContextMenu, HoverChanged, SelectionChanged, SegClick,
)
from engine.events import (                                        # noqa: E402
    PointerEnter, PointerMove, PointerLeave, PointerDown, PointerUp,
    Scroll, DragBegin, DragUpdate, KeyDown, KeyUp, SECONDARY, SPACE,
)

# ---------------------------------------------------------------- the scene ---
# A 100x100 image in a 200x200 view: fit scale is exactly 2, so view == 2 x image
# and a fixture can name a pixel without arithmetic.
IMG = 100
VIEW = 200

# Object 1 is a "field" (0..60); object 3 is a "core" nested inside it (20..40).
# So (30,30) is stacked: the specific object is 3, the general one is 1, depth 2.
FIELD = (0, 0, 60, 60)
CORE = (20, 20, 40, 40)

CORE_PT = (30, 30)          # image px inside the nested core
FIELD_PT = (10, 10)         # image px in the field only
EMPTY_PT = (90, 90)         # image px with nothing under it


def to_view(pt):
    return (pt[0] * 2.0, pt[1] * 2.0)


def _in(box, x, y):
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _map(fn, channels=3):
    """A PixelMap over freshly-built bytes; ids pack as R + G*256."""
    stride = IMG * channels
    buf = bytearray(stride * IMG)
    for y in range(IMG):
        for x in range(IMG):
            v = fn(x, y)
            i = y * stride + x * channels
            buf[i] = v & 0xFF
            if channels >= 2:
                buf[i + 1] = (v >> 8) & 0xFF
    return PixelMap(bytes(buf), stride, channels, IMG, IMG)


def _specific(x, y):
    return 3 if _in(CORE, x, y) else (1 if _in(FIELD, x, y) else 0)


def _general(x, y):
    return 1 if _in(FIELD, x, y) else 0


def _depth(x, y):
    return (2 if _in(CORE, x, y) else 1) if _in(FIELD, x, y) else 0


def _square(box):
    x0, y0, x1, y1 = box
    return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]


OBJECTS = [
    {"id": 1, "color": "#51e5ff", "bbox": (0, 0, 60, 60),
     "contour": _square(FIELD), "handle": 1},
    {"id": 3, "color": "#ff5178", "bbox": (20, 20, 20, 20),
     "contour": _square(CORE), "handle": 3},
]


class Clock:
    """A hand-cranked frame clock, in microseconds."""

    def __init__(self):
        self.t = 1_000_000        # not 0: a real clock never starts there

    def __call__(self):
        return self.t

    def advance_ms(self, ms):
        self.t += int(ms * 1000)
        return self.t


def new_session(mode="everything", dwell_ms=300):
    clock = Clock()
    s = EngineSession(clock=clock)
    s.resize(VIEW, VIEW)
    s.set_image("src", (IMG, IMG))
    s.set_seg_mode(mode)
    s.set_dwell_ms(dwell_ms)
    s.load_objects(OBJECTS, label=_map(_specific), general=_map(_general),
                   depth=_map(_depth))
    s.drain_effects()             # discard the load's reveal/redraw
    return s, clock


# ----------------------------------------------------------------- the gate ---
FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print("  %-46s %s" % (label, "PASS" if ok else "FAIL"))
    if not ok:
        print("      want: %r" % (want,))
        print("      got:  %r" % (got,))


def test_hover_focuses_general_then_drills():
    """Hovering a stack focuses the WHOLE; dwelling drills to the PART."""
    s, clock = new_session()

    got = s.feed(PointerMove(*to_view(CORE_PT)))
    check("hover a stack -> focus the general object",
          got, [RequestTick(), Redraw(), HoverChanged(2, False, 60.0, 60.0)])
    check("  ... and the focused object is the WHOLE (1), not the part",
          (s.objects.hover_id, s.objects.hover_gen, s.objects.hover_spec),
          (1, 1, 3))

    # not yet dwelled: a tick before the dwell time must not drill
    clock.advance_ms(100)
    s.tick(clock())
    s.drain_effects()
    check("  ... a tick before the dwell time does not drill",
          (s.anim.drilled, s.objects.hover_id), (False, 1))

    # dwell expires: the tick drills to the specific object and says so
    clock.advance_ms(300)
    s.tick(clock())
    check("dwell expires -> drill to the specific object",
          s.drain_effects(),
          [RequestTick(), HoverChanged(2, True, 60.0, 60.0), Redraw()])
    check("  ... the focused object is now the PART (3)",
          s.objects.hover_id, 3)


def test_hover_unstacked_and_leave():
    s, clock = new_session()
    s.feed(PointerMove(*to_view(FIELD_PT)))
    check("hover an unstacked object -> depth 1",
          (s.objects.hover_id, s.objects.hover_depth), (1, 1))

    # RequestTick because the outline does not vanish — it collapses toward the
    # object's centroid, and that tween needs the clock.
    got = s.feed(PointerLeave())
    check("leaving clears the focus and reports depth 0",
          got, [RequestTick(), Redraw(), HoverChanged(0, False, 20.0, 20.0)])
    check("  ... nothing is focused", s.objects.hover_id, 0)
    check("  ... and the outline is collapsing away", s.objects.morph is not None
          and s.objects.morph.to_ring is None, True)


def test_click_toggles_selection():
    s, clock = new_session()
    s.feed(PointerMove(*to_view(CORE_PT)))
    s.drain_effects()

    vx, vy = to_view(CORE_PT)
    check("press -> ripple on the focused object",
          s.feed(PointerDown(vx, vy)), [RequestTick(), Redraw()])
    check("  ... the press is held on the general object (not drilled yet)",
          (s.anim.press_obj, s.anim.held), (1, True))

    check("release -> select it, and report the click",
          s.feed(PointerUp(vx, vy)),
          [Redraw(), SelectionChanged((1,)), RequestTick(),
           SegClick(30, 30, 1, "toggle")])
    check("  ... it is selected", s.selection(), [1])
    check("  ... and it popped", 1 in s.anim.pop, True)

    # clicking it again deselects
    s.feed(PointerDown(vx, vy))
    got = s.feed(PointerUp(vx, vy))
    check("click again -> deselect",
          [e for e in got if isinstance(e, (SelectionChanged, SegClick))],
          [SelectionChanged(()), SegClick(30, 30, 1, "toggle")])
    check("  ... nothing is selected", s.selection(), [])


def test_drag_is_not_a_click():
    """A press that travels more than the slop is a pan, not a selection."""
    s, clock = new_session()
    vx, vy = to_view(CORE_PT)
    s.feed(PointerDown(vx, vy))
    got = s.feed(PointerUp(vx + 20, vy + 20))
    check("press-move-release does NOT select",
          [e for e in got if isinstance(e, (SegClick, SelectionChanged))], [])
    check("  ... nothing is selected", s.selection(), [])


def test_space_pans_and_suppresses_selection():
    s, clock = new_session()
    s.feed(PointerMove(*to_view(CORE_PT)))
    s.drain_effects()

    check("Space -> grab cursor, and the highlight drops",
          s.feed(KeyDown(SPACE)), [SetCursor("grabbing"), Redraw()])
    check("  ... nothing is focused while panning", s.objects.hover_id, 0)

    vx, vy = to_view(CORE_PT)
    s.feed(PointerDown(vx, vy))
    got = s.feed(PointerUp(vx, vy))
    check("a click while panning does NOT select",
          [e for e in got if isinstance(e, (SegClick, SelectionChanged))], [])

    # motion while panning must not retrack the hover
    s.feed(PointerMove(*to_view(FIELD_PT)))
    check("  ... and motion while panning does not retrack hover",
          s.objects.hover_id, 0)

    check("Space up -> default cursor", s.feed(KeyUp(SPACE)),
          [SetCursor(None)])


def test_drag_pans_the_pane():
    s, clock = new_session()
    s.feed(DragBegin(10, 10))
    s.feed(DragUpdate(25, -15))
    check("drag pans by the cumulative offset",
          (s.pane.ox, s.pane.oy), (25.0, -15.0))
    # a second drag starts from where the first left off
    s.feed(DragBegin(0, 0))
    s.feed(DragUpdate(5, 5))
    check("  ... and the next drag starts from there",
          (s.pane.ox, s.pane.oy), (30.0, -10.0))


def test_scroll_zooms_about_the_cursor():
    """The pixel under the cursor must stay under the cursor."""
    s, clock = new_session()
    vx, vy = to_view(CORE_PT)
    s.feed(PointerMove(vx, vy))
    before = s.pane.view_to_image(vx, vy)

    s.feed(Scroll(-1))            # wheel up == zoom in
    check("scroll up zooms in", round(s.pane.zoom, 6), 1.1)
    after = s.pane.view_to_image(vx, vy)
    check("  ... and the pixel under the cursor stays put",
          (round(after[0], 6), round(after[1], 6)),
          (round(before[0], 6), round(before[1], 6)))

    s.feed(Scroll(1))             # and back out
    check("scroll down zooms out", round(s.pane.zoom, 6), 1.0)


def test_right_click_opens_the_menu():
    s, clock = new_session(mode=None)
    check("right-click -> context menu",
          s.feed(PointerDown(30, 40, button=SECONDARY)), [ContextMenu(30, 40)])


def test_point_mode():
    """In point mode clicks are prompts, not selections — and right-click is a
    NEGATIVE point rather than the view menu."""
    s, clock = new_session(mode="point")
    vx, vy = to_view(CORE_PT)

    s.feed(PointerDown(vx, vy))
    check("point mode: click -> a positive point",
          [e for e in s.feed(PointerUp(vx, vy)) if isinstance(e, SegClick)],
          [SegClick(30, 30, 1, "point")])

    s.feed(PointerDown(vx, vy, ctrl=True))
    check("point mode: ctrl-click -> a negative point",
          [e for e in s.feed(PointerUp(vx, vy, ctrl=True))
           if isinstance(e, SegClick)],
          [SegClick(30, 30, 0, "point")])

    check("point mode: right-click -> a negative point, not the menu",
          s.feed(PointerDown(vx, vy, button=SECONDARY)),
          [SegClick(30, 30, 0, "point")])

    check("point mode: no object is ever selected", s.selection(), [])


def test_click_outside_the_image_is_ignored():
    s, clock = new_session()
    # view (250,250) -> image (125,125): outside a 100x100 image
    s.feed(PointerDown(250, 250))
    got = s.feed(PointerUp(250, 250))
    check("a click outside the image reports nothing",
          [e for e in got if isinstance(e, (SegClick, SelectionChanged))], [])

    # and empty space inside the image toggles nothing, but still reports
    vx, vy = to_view(EMPTY_PT)
    s.feed(PointerMove(vx, vy))
    s.feed(PointerDown(vx, vy))
    got = s.feed(PointerUp(vx, vy))
    check("a click on empty image space reports id 0",
          [e for e in got if isinstance(e, SegClick)],
          [SegClick(90, 90, 0, "toggle")])
    check("  ... and selects nothing", s.selection(), [])


def test_enter_takes_focus():
    s, clock = new_session()
    check("entering the canvas takes keyboard focus (for Space-to-pan)",
          s.feed(PointerEnter(10, 10)), [GrabFocus()])


def test_cursor_info():
    """The "what is under the cursor" query the sidebar/badge will build on."""
    s, clock = new_session()
    s.feed(PointerMove(*to_view(CORE_PT)))
    ci = s.cursor_info()
    check("cursor_info: image coords, ids and depth",
          (ci.image, ci.inside, ci.specific_id, ci.general_id, ci.depth),
          ((30.0, 30.0), True, 3, 1, 2))
    check("cursor_info: focused is what a click would act on (the whole)",
          ci.focused_id, 1)


def test_rotate_drops_overlays():
    """Overlays are registered to un-rotated pixels, so a rotate must drop them
    rather than let them desync."""
    s, clock = new_session()
    check("rotating clears the overlay and stops the clock",
          s.rotate(1) or s.drain_effects(),
          [StopTick(), Redraw(), ViewChanged()])
    check("  ... the overlay is gone", s.has_seg(), False)
    check("  ... but the mode is kept", s.objects.seg_mode, "everything")
    check("  ... and the rotation applied", s.pane.rot, 1)


def test_tick_stops_when_idle():
    s, clock = new_session()
    clock.advance_ms(1000)          # let the reveal finish
    t = s.tick(clock())
    check("the clock retires once nothing is animating", t.animating, False)


def main():
    for fn in (test_hover_focuses_general_then_drills,
               test_hover_unstacked_and_leave,
               test_click_toggles_selection,
               test_drag_is_not_a_click,
               test_space_pans_and_suppresses_selection,
               test_drag_pans_the_pane,
               test_scroll_zooms_about_the_cursor,
               test_right_click_opens_the_menu,
               test_point_mode,
               test_click_outside_the_image_is_ignored,
               test_enter_takes_focus,
               test_cursor_info,
               test_rotate_drops_overlays,
               test_tick_stops_when_idle):
        print(fn.__doc__.splitlines()[0] if fn.__doc__ else fn.__name__)
        fn()
    print()
    if FAILED:
        print("INTERACTION FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("INTERACTION OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
