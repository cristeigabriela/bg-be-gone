"""The interaction state machine: events in, effects out. Stdlib only.

Everything the canvas does in response to a pointer, a key or a wheel lives here:
space-to-pan, click-vs-drag, cursor-anchored zoom, the hover focus with its
dwell-to-drill, the press-and-hold ripple, and the click that toggles an object.

None of it touches GTK. A GTK controller and a DOM listener both just translate
their native event into an `events.Event` and hand it over; whatever should
happen to the outside world comes back as a list of `effects.Effect`. So the
whole behaviour is testable by script, with no display and no main loop — which
is the gate for this step.
"""
from . import effects as fx
from .events import (
    PointerEnter, PointerMove, PointerLeave, PointerDown, PointerUp,
    Scroll, DragBegin, DragUpdate, KeyDown, KeyUp, PRIMARY, SECONDARY, SPACE,
)

CLICK_SLOP = 4.0      # view px of movement that still counts as a click, not a drag
ZOOM_STEP = 1.1


class Interaction:
    def __init__(self, session):
        self.s = session
        self.px = 0.0             # last known cursor position, view px
        self.py = 0.0
        self.space_down = False   # Space held == pan mode
        self.press_xy = None      # where the primary button went down
        self._ox0 = 0.0           # pan origin, captured at drag-begin
        self._oy0 = 0.0

    def reset(self):
        self.space_down = False
        self.press_xy = None

    # ---------- dispatch ----------
    def feed(self, ev):
        h = self._HANDLERS.get(type(ev))
        if h is not None:
            h(self, ev)

    # ---------- pointer ----------
    def _enter(self, ev):
        # Take keyboard focus so the Space-to-pan key controller receives keys.
        if self.s.pane.has_image():
            self.s.emit(fx.GrabFocus())

    def _move(self, ev):
        s = self.s
        self.px, self.py = ev.x, ev.y
        # Panning, or holding an object down — either way, don't retrack hover.
        if self.space_down or s.anim.held:
            return
        if not (s.objects.seg_mode == "everything" and s.objects.masks):
            return
        s.emit(fx.RequestTick())          # hovering animates; re-arm an idle tick

        gen = spec = depth = 0
        pt = s.pane.view_to_image(ev.x, ev.y)
        if pt is not None:
            hit = s.objects.maps.hit(int(pt[0]), int(pt[1]))
            gen, spec, depth = hit.general, hit.specific, hit.depth

        o = s.objects
        o.hover_depth = depth
        if gen != o.hover_gen:
            # A new object region: focus the GENERAL object and start the dwell
            # timer. The tick drills to the specific one once it expires.
            o.hover_gen = gen
            o.hover_spec = spec
            if gen:
                s.anim.begin_dwell(s.now())
            else:
                s.anim.cancel_dwell()
            s.focus(gen)                  # morphs the outline into the new object
            s.emit(fx.Redraw(),
                   fx.HoverChanged(depth, False, self.px, self.py))
        else:
            o.hover_spec = spec
            if s.anim.drilled and o.hover_id != spec:
                s.focus(spec)             # track the specific one under the cursor
                s.emit(fx.Redraw())
            # keep the stacked-objects badge glued to the pointer as it moves
            s.emit(fx.HoverChanged(depth, s.anim.drilled, self.px, self.py))

    def _leave(self, ev):
        s = self.s
        o = s.objects
        o.hover_gen = o.hover_spec = 0
        s.anim.cancel_dwell()
        if o.hover_id:
            s.focus(0)                    # morph the outline away
            s.emit(fx.Redraw())
        s.emit(fx.HoverChanged(0, False, self.px, self.py))

    def _down(self, ev):
        s = self.s
        if ev.button == SECONDARY:
            # In point mode a right-click is a negative point, not the view menu.
            if s.objects.seg_mode == "point":
                self._seg_click(ev.x, ev.y, 0, "point")
            else:
                s.emit(fx.ContextMenu(ev.x, ev.y))
            return
        if ev.button != PRIMARY:
            return

        self.press_xy = (ev.x, ev.y)
        # Double-click zoom toggle stays on press (only when not segmenting).
        if not s.objects.seg_mode and ev.n_press == 2:
            if abs(s.pane.zoom - 1.0) < 1e-3:
                s.actual_size()
            else:
                s.reset_view()
            return
        # Press-and-hold: ripple + glow on the focused object (the "swizzle").
        if s.objects.seg_mode == "everything" and not self.space_down:
            pt = s.pane.view_to_image(ev.x, ev.y)
            if pt is None:
                return
            ix, iy = int(pt[0]), int(pt[1])
            oid = s.objects.hover_id or s.objects.maps.specific_at(ix, iy)
            if oid:
                s.anim.begin_press(oid, (ix, iy), s.now())
                s.emit(fx.RequestTick(), fx.Redraw())

    def _up(self, ev):
        s = self.s
        if ev.button != PRIMARY:
            return
        # Don't clear the press object — let the ripple and the spring-back play
        # out (the tick retires it once decayed). Only stamp the release.
        if s.anim.release_press(s.now()):
            s.emit(fx.Redraw())
        # Select on release, and only for a real click — never while panning
        # (Space held) or when the press turned into a drag.
        if not s.objects.seg_mode or self.space_down:
            return
        px, py = self.press_xy or (ev.x, ev.y)
        if abs(ev.x - px) > CLICK_SLOP or abs(ev.y - py) > CLICK_SLOP:
            return
        if s.objects.seg_mode == "point":
            self._seg_click(ev.x, ev.y, 0 if ev.ctrl else 1, "point")
        else:
            self._seg_click(ev.x, ev.y, None, "toggle")

    def _seg_click(self, x, y, value, kind):
        s = self.s
        pt = s.pane.view_to_image(x, y)
        if pt is None:
            return
        ix, iy = int(pt[0]), int(pt[1])
        if not s.pane.contains_image_point(ix, iy):
            return
        if kind == "toggle":
            oid = s.objects.hover_id or s.objects.maps.specific_at(ix, iy)
            if oid:
                s.toggle_object(oid)
            s.emit(fx.SegClick(ix, iy, oid, "toggle"))
        else:
            s.emit(fx.SegClick(ix, iy, value, "point"))

    # ---------- wheel ----------
    def _scroll(self, ev):
        s = self.s
        if not s.pane.has_image():
            return
        # Anchored on the last *motion* position: a wheel event carries no usable
        # pointer position on every backend, and this is what the user sees.
        factor = 1.0 / ZOOM_STEP if ev.dy > 0 else ZOOM_STEP
        s.pane.zoom_at(factor, self.px, self.py)
        s.emit(fx.Redraw())

    # ---------- drag (pan) ----------
    def _drag_begin(self, ev):
        self._ox0, self._oy0 = self.s.pane.ox, self.s.pane.oy

    def _drag_update(self, ev):
        self.s.pane.pan_to(self._ox0 + ev.dx, self._oy0 + ev.dy)
        self.s.emit(fx.Redraw())

    # ---------- keys ----------
    def _key_down(self, ev):
        if ev.key != SPACE or self.space_down:
            return
        s = self.s
        self.space_down = True
        s.emit(fx.SetCursor("grabbing"))
        # Drop the highlight while panning — but bluntly, with no morph: the
        # outline should not animate away under a pan.
        if s.objects.hover_id:
            s.objects.hover_id = 0
            s.emit(fx.Redraw())

    def _key_up(self, ev):
        if ev.key != SPACE:
            return
        self.space_down = False
        self.s.emit(fx.SetCursor(None))

    _HANDLERS = {
        PointerEnter: _enter,
        PointerMove: _move,
        PointerLeave: _leave,
        PointerDown: _down,
        PointerUp: _up,
        Scroll: _scroll,
        DragBegin: _drag_begin,
        DragUpdate: _drag_update,
        KeyDown: _key_down,
        KeyUp: _key_up,
    }
