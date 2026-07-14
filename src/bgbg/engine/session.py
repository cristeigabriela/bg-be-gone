"""The engine's end-to-end interface for one image pane. Stdlib only.

`EngineSession` is what a UI talks to. It owns the view state (`Pane`), every
animation clock (`AnimState`), the overlay model (`ObjectStore`) and the
interaction state machine, and it is the only thing the shell needs to hold.

Two rules make it portable:

* **No pixels.** Images are opaque *handles* the shell minted; the engine passes
  them back inside a display list and never dereferences one.
* **No callbacks out.** Side effects are *returned* (see `effects.py`), never
  invoked, so the engine cannot reach GTK or the DOM even by accident — and a
  whole interaction can be replayed in a test with no display.

Time is absolute microseconds, supplied by the host's frame clock via `clock`.
The offscreen rasteriser pins it to 0 and states animation ages as negative
timestamps, which is what makes any moment of any animation addressable.
"""
from . import effects as fx
from .anim import AnimState
from .interaction import Interaction
from .objects import ObjectStore
from .pane import Pane
from .render.builder import build

#: Idempotent flags: "repaint" and "start the clock" mean the same thing however
#: many times they are said, and a single event routinely says them twice (a move
#: re-arms the tick, and so does the focus change it triggers). Collapsing them
#: keeps an event's effect list a readable description of what happened, rather
#: than something every host has to filter.
_IDEMPOTENT = (fx.Redraw, fx.RequestTick)


class CursorInfo:
    """What is under the cursor, in image space (origin at the image's (0,0))."""

    __slots__ = ("image", "inside", "specific_id", "general_id", "depth",
                 "focused_id")

    def __init__(self, image=None, inside=False, specific_id=0, general_id=0,
                 depth=0, focused_id=0):
        self.image = image              # (ix, iy) float image px, or None
        self.inside = inside            # within the image bounds
        self.specific_id = specific_id  # smallest object here
        self.general_id = general_id    # largest object here
        self.depth = depth              # how many objects overlap here
        self.focused_id = focused_id    # what a click would act on (dwell-aware)

    def __repr__(self):
        return ("CursorInfo(image=%r, inside=%r, specific=%d, general=%d, "
                "depth=%d, focused=%d)" % (self.image, self.inside,
                                           self.specific_id, self.general_id,
                                           self.depth, self.focused_id))


class EngineSession:
    def __init__(self, clock=None):
        self.pane = Pane()
        self.anim = AnimState()
        self.objects = ObjectStore()
        self.interaction = Interaction(self)
        # µs from the host's frame clock. Offscreen there is none, so a fixture
        # pins this and writes animation ages as negative timestamps.
        self.clock = clock or (lambda: 0)

        self.image = None               # handle of the source image
        self.image_size = (0, 0)
        self._effects = []

    def now(self):
        return self.clock()

    # ---------- effects ----------
    def emit(self, *effects):
        for e in effects:
            if (type(e) in _IDEMPOTENT
                    and any(type(p) is type(e) for p in self._effects)):
                continue
            self._effects.append(e)

    def drain_effects(self):
        out = self._effects
        self._effects = []
        return out

    def feed(self, ev):
        """Push one input event; get back what the host should do about it."""
        self.interaction.feed(ev)
        return self.drain_effects()

    # ---------- image ----------
    def set_image(self, handle, size, keep_transform=False):
        self.image = handle
        self.image_size = (int(size[0]), int(size[1]))
        self.pane.set_image_size(*self.image_size)
        self.objects.clear_composite()
        if not keep_transform:
            self.pane.rot = 0
            self.pane.fh = self.pane.fv = False
            self.pane.reset_view()
        self.emit(fx.Redraw(), fx.ViewChanged())

    def clear_image(self):
        self.image = None
        self.image_size = (0, 0)
        self.pane.set_image_size(0, 0)
        self.objects.clear_composite()
        self.pane.reset_view()
        self.emit(fx.Redraw(), fx.ViewChanged())

    def has_image(self):
        return self.image is not None

    def resize(self, w, h):
        self.pane.set_view_size(w, h)

    # ---------- view commands ----------
    def reset_view(self):
        self.pane.reset_view()
        self.emit(fx.Redraw())

    def actual_size(self):
        if not self.pane.has_image():
            return
        self.pane.actual_size()
        self.emit(fx.Redraw())

    def zoom_at(self, factor, cx, cy):
        self.pane.zoom_at(factor, cx, cy)
        self.emit(fx.Redraw())

    # Rotate and flip are VIEW state. The overlays are registered to un-rotated
    # image pixels and stay that way: the builder draws them inside the same
    # Push(transform=...) as the image, so they ride the rotation for free, and
    # `view_to_image` un-rotates before indexing the id maps, so hit-testing keeps
    # working. Both used to throw the overlays away instead — you rotated, and
    # your selection was gone.
    #
    # The one thing that has to hold for this to be true: whatever gets segmented
    # must be the UN-transformed source (see app._ensure_seg_loaded), or the masks
    # would come back in the rotated frame and the two would desync.
    def rotate(self, delta):
        if not self.pane.has_image():
            return
        self.pane.rotate(delta)          # also resets zoom/pan
        self.emit(fx.Redraw(), fx.ViewChanged())

    def flip(self, horizontal):
        if not self.pane.has_image():
            return
        self.pane.flip(horizontal)
        self.emit(fx.Redraw(), fx.ViewChanged())

    # ---------- segmentation ----------
    def set_seg_mode(self, mode):
        self.objects.seg_mode = mode
        self.emit(fx.Redraw())

    def set_dwell_ms(self, ms):
        """How long to hover before drilling from the general object to the
        specific one (0 drills immediately)."""
        self.anim.dwell_ms = float(ms)

    def load_objects(self, objects, label=None, general=None, depth=None):
        """Replace the overlay. `objects` carry shell-minted mask handles; the
        maps are raw `PixelMap`s the shell decoded."""
        self._reset_seg(keep_mode=True)
        self.objects.load(objects)
        self.objects.maps.label = label
        self.objects.maps.general = general
        self.objects.maps.depth = depth
        self.anim.begin_reveal(self.now())
        self.emit(fx.RequestTick(), fx.Redraw())

    def set_point_mask(self, handle, contour=None):
        self.objects.set_point_mask(handle, contour)
        self.anim.begin_reveal(self.now())
        self.emit(fx.RequestTick(), fx.Redraw())

    def set_selection(self, ids):
        self.objects.set_selection(ids)
        self.emit(fx.SelectionChanged(self.objects.selection()), fx.Redraw())

    def toggle_object(self, oid):
        if self.objects.toggle(oid):
            self.anim.begin_pop(oid, self.now())   # tactile "pop" on select
        self.emit(fx.SelectionChanged(self.objects.selection()),
                  fx.RequestTick(), fx.Redraw())

    def selection(self):
        return self.objects.selection()

    def focus(self, oid):
        """Move the hover focus, tweening the outline from the old object."""
        o = self.objects
        if oid == o.hover_id:
            return False
        o.morph = o.build_morph(o.hover_id, oid)
        o.hover_id = oid
        if o.morph is None:
            self.anim.clear_morph()      # nothing to tween between
        else:
            self.anim.begin_morph(self.now())
            self.emit(fx.RequestTick())
        return True

    def set_scanning(self, on):
        """The scan shimmer over the source, while an everything-pass runs."""
        on = bool(on)
        if on == self.anim.scanning:
            return
        self.anim.scanning = on
        if on:
            self.emit(fx.RequestTick())
        self.emit(fx.Redraw())

    def set_composite(self, handle, size, masks, bg=None):
        """Result-panel cutout preview: the source clipped to a union of masks."""
        self.image = handle
        self.image_size = (int(size[0]), int(size[1]))
        self.pane.set_image_size(*self.image_size)
        self.objects.set_composite(masks, bg)
        self.emit(fx.Redraw(), fx.ViewChanged())

    def update_composite(self, masks, bg=None):
        self.objects.set_composite(masks, bg)
        self.emit(fx.Redraw())

    def _reset_seg(self, keep_mode=False):
        """Overlay state back to nothing — without touching the scan shimmer or
        the animation epoch, neither of which belongs to the overlay."""
        self.objects.clear(keep_mode=keep_mode)
        self.anim.pop = {}
        self.anim.reveal = 1.0
        self.anim.reveal_t0 = None
        self.anim.cancel_dwell()
        self.anim.clear_press()
        self.anim.clear_morph()

    def clear_seg(self, keep_mode=False):
        self._reset_seg(keep_mode=keep_mode)
        self.emit(fx.StopTick(), fx.Redraw())

    def has_seg(self):
        return self.objects.has_seg()

    # ---------- the frame ----------
    def needs_tick(self):
        return self.anim.needs_tick(self.objects.active())

    def tick(self, now):
        """Advance every clock. Returns the `Tick`; drain the effects after."""
        o = self.objects
        t = self.anim.advance(now, active=o.active())

        if t.dwell_fired and o.hover_gen:
            # Dwelled long enough: drill from the general object to the specific.
            if o.hover_spec:
                self.focus(o.hover_spec)
            self.emit(fx.HoverChanged(o.hover_depth, True,
                                      self.interaction.px, self.interaction.py))
        if t.animating or t.changed:
            self.emit(fx.Redraw())
        return t

    def scene(self):
        return self.objects.scene(self.image, self.image_size)

    def display_list(self, now=None):
        return build(self.scene(), self.pane, self.anim,
                     self.now() if now is None else now)

    # ---------- queries ----------
    def cursor_info(self):
        """What is under the cursor — the answer to "what would a click hit?"."""
        o = self.objects
        pt = self.pane.view_to_image(self.interaction.px, self.interaction.py)
        if pt is None:
            return CursorInfo()
        ix, iy = int(pt[0]), int(pt[1])
        inside = self.pane.contains_image_point(ix, iy)
        hit = o.maps.hit(ix, iy)
        return CursorInfo(image=pt, inside=inside,
                          specific_id=hit.specific, general_id=hit.general,
                          depth=hit.depth,
                          focused_id=o.hover_id or hit.specific)
