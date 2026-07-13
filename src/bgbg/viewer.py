"""Interactive image view: zoom, pan, rotate, flip, reset.

A Gtk.Widget that renders a GdkPixbuf with GSK (no cairo dependency). It keeps
a view transform (zoom/pan) and an image transform (rotate/flip). The image
transform is exportable via :meth:`export_pixbuf` so it can be baked into what
gets processed or saved. Zoom and pan are view-only and never exported.
"""
import math
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gsk", "4.0")
gi.require_version("Graphene", "1.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, Gdk, Gsk, Graphene, GdkPixbuf, Gio, GLib  # noqa: E402

from engine.pane import Pane  # noqa: E402
from engine.hittest import PixelMap, HitMaps  # noqa: E402
from engine import anim as anim_mod  # noqa: E402
from engine.anim import AnimState  # noqa: E402
from engine.render.scene import Scene, Morph, POINT_COLOR  # noqa: E402
from engine.render.builder import build as build_display_list  # noqa: E402
from gsk_renderer import render as gsk_render  # noqa: E402
from engine.geometry import (  # noqa: E402
    MIN_ZOOM, MAX_ZOOM,
    polygon_area_abs as _polygon_area_abs,
    resample_closed as _resample_closed,
    align_ring as _align_ring,
    ease_out as _ease_out,
    ease_out_back as _ease_out_back,
)

_CELL = 16
_MORPH_N = anim_mod.MORPH_N


def _rgba_tuple(spec):
    """Parse a colour into the engine's (r, g, b, a) tuple."""
    c = Gdk.RGBA()
    c.parse(spec)
    return (c.red, c.green, c.blue, c.alpha)


class ImageView(Gtk.Widget):
    def __init__(self, on_change=None):
        super().__init__()
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_focusable(True)
        self.set_size_request(160, 160)

        self.pixbuf = None
        self._texture = None
        self._anim = None        # GdkPixbuf.PixbufAnimation for animated GIFs
        self._anim_iter = None
        self._anim_timer = 0     # GLib timeout id driving frame advance
        # All view state (zoom / pan / rotate / flip) and the coordinate maths
        # live in the engine; the widget just feeds it sizes and events.
        # zoom/ox/oy/rot/fh/fv stay readable+writable as attributes (properties
        # below) so existing call sites are untouched.
        self.pane = Pane()
        self._px = 0.0
        self._py = 0.0
        self._ox0 = 0.0
        self._oy0 = 0.0
        self._on_change = on_change
        self.on_paint = None     # optional callback(zoom_percent:int)


        # ---- segmentation overlay state ----
        self._seg_mode = None            # None | "everything" | "point"
        self._seg_masks = {}             # id -> Gdk.Texture (per-object mask)
        self._seg_colors = {}            # id -> (r,g,b,a) tint
        self._seg_selected = set()       # ids kept for output
        self._seg_hover_id = 0           # object under the cursor (glow)
        self._seg_point_tex = None       # Gdk.Texture (point-mode object)
        self.maps = HitMaps()            # specific / general / depth lookup maps
        # composite/clip mode (result panel): show the source clipped to a union
        # of masks over the checkerboard — a live cutout preview.
        self._clip_active = False
        self._clip_masks = []            # list of Gdk.Texture (selected objects)
        self._clip_bg = None             # (r,g,b,a) solid background, or None
        # fired on a click while a seg mode is active:
        #   everything: (ix, iy, object_id, "toggle")
        #   point:      (ix, iy, label 1/0, "point")
        self.on_seg_click = None
        # fired as the hovered object stack changes: (depth:int, drilled:bool)
        self.on_seg_hover = None
        # hover-dwell disambiguation (general object first, drill to specific).
        # The dwell *timer* lives in the engine; which object to focus is the
        # widget's business (that moves to the engine in a later step).
        self._hover_gen = 0
        self._hover_spec = 0
        self._hover_depth = 0

        # ---- interaction: space-to-pan, click-vs-drag ----
        self._space_down = False
        self._press_xy = None            # press position, to tell click from drag
        self._grab_cursor = Gdk.Cursor.new_from_name("grabbing", None)

        # ---- animation state (tick-driven; see _on_tick / _snapshot_seg) ----
        self._seg_polys = {}             # id -> [poly]  (all contours)
        self._seg_centroids = {}         # id -> (cx, cy) image px, for "pop"
        self._seg_radius = {}            # id -> ripple reach (image px)
        self._seg_morph = {}             # id -> [N (x,y)] canonical outline ring
        self._seg_sec_polys = {}         # id -> [poly]  (non-largest contours)
        self._point_polys = ()           # point-mode object contours
        # outline morph between focused objects (hover switch / dwell drill).
        # The engine owns the *clock*; these are the rings/colours it tweens.
        self._morph = None               # engine.render.scene.Morph, or None
        self._anim_tick = None           # add_tick_callback handle

        # Every clock (breathing glow, marching ants, shimmer, reveal, pop,
        # hover-dwell, press ripple, morph) lives in the engine.
        self.anim = AnimState()

        motion = Gtk.EventControllerMotion()
        motion.connect("motion", self._on_motion)
        motion.connect("enter", self._on_enter)
        motion.connect("leave", self._on_leave)
        self.add_controller(motion)

        scroll = Gtk.EventControllerScroll(
            flags=Gtk.EventControllerScrollFlags.BOTH_AXES)
        scroll.connect("scroll", self._on_scroll)
        self.add_controller(scroll)

        keys = Gtk.EventControllerKey()
        keys.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        keys.connect("key-pressed", self._on_key_pressed)
        keys.connect("key-released", self._on_key_released)
        self.add_controller(keys)

        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        self.add_controller(drag)

        rclick = Gtk.GestureClick(button=Gdk.BUTTON_SECONDARY)
        rclick.connect("pressed", self._on_right_click)
        self.add_controller(rclick)

        pclick = Gtk.GestureClick(button=Gdk.BUTTON_PRIMARY)
        pclick.connect("pressed", self._on_primary_pressed)
        pclick.connect("released", self._on_primary_released)
        self.add_controller(pclick)

        # pause GIF playback while this panel is off-screen (page switch / close)
        self.connect("map", self._on_map)
        self.connect("unmap", lambda *_: self._pause_animation())

        self._build_menu()

    # ---------- view state (delegated to the engine's Pane) ----------
    def _sync_pane(self):
        """Feed the engine the sizes only the widget knows."""
        if self.pixbuf is not None:
            self.pane.set_image_size(self.pixbuf.get_width(),
                                     self.pixbuf.get_height())
        else:
            self.pane.set_image_size(0, 0)
        self.pane.set_view_size(self.get_width(), self.get_height())
        return self.pane

    zoom = property(lambda s: s.pane.zoom,
                    lambda s, v: setattr(s.pane, "zoom", v))
    ox = property(lambda s: s.pane.ox, lambda s, v: setattr(s.pane, "ox", v))
    oy = property(lambda s: s.pane.oy, lambda s, v: setattr(s.pane, "oy", v))
    rot = property(lambda s: s.pane.rot, lambda s, v: setattr(s.pane, "rot", v))
    fh = property(lambda s: s.pane.fh, lambda s, v: setattr(s.pane, "fh", v))
    fv = property(lambda s: s.pane.fv, lambda s, v: setattr(s.pane, "fv", v))

    # ---------- animation state (delegated to the engine's AnimState) ----------
    def _a(name):        # noqa: N805  (a property factory, not a method)
        return property(lambda s: getattr(s.anim, name),
                        lambda s, v: setattr(s.anim, name, v))

    _t0 = _a("t0")
    _pulse = _a("pulse")
    _ant = _a("ant")
    _scan_phase = _a("scan_phase")
    _reveal = _a("reveal")
    _reveal_t0 = _a("reveal_t0")
    _scanning = _a("scanning")
    _pop = _a("pop")
    _dwell_ms = _a("dwell_ms")
    _hover_dwell_t0 = _a("dwell_t0")
    _hover_drilled = _a("drilled")
    _press_obj = _a("press_obj")
    _press_pt = _a("press_pt")
    _press_t0 = _a("press_t0")
    _release_t0 = _a("release_t0")
    _morph_t0 = _a("morph_t0")
    del _a

    # ---------- public API ----------
    def load_file(self, path, keep_transform=False):
        self._stop_animation()
        anim = self._open_animation(path)
        if anim is not None:               # animated GIF: play it back
            self._anim = anim
            self._anim_iter = anim.get_iter(None)
            self.pixbuf = self._anim_iter.get_pixbuf()
        else:
            try:
                self.pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
            except GLib.Error:
                return False
        self._texture = None
        self._clip_active = False
        if keep_transform:                 # keep the user's flip/zoom on reload
            self.queue_draw()
        else:
            self.rot = 0
            self.fh = self.fv = False
            self.reset_view()
        self._changed()
        if self._anim is not None:
            self._schedule_frame()
        return True

    def _open_animation(self, path):
        """Return a PixbufAnimation if `path` is an animated GIF we should play
        (not on the Segment view, which needs a stable frame), else None."""
        if self._seg_mode is not None or not path.lower().endswith(".gif"):
            return None
        try:
            anim = GdkPixbuf.PixbufAnimation.new_from_file(path)
        except GLib.Error:
            return None
        return None if anim.is_static_image() else anim

    def _stop_animation(self):
        self._pause_animation()
        self._anim = None
        self._anim_iter = None

    def _pause_animation(self):
        if self._anim_timer:
            GLib.source_remove(self._anim_timer)
            self._anim_timer = 0

    def _resume_animation(self, *_):
        if self._anim is not None and not self._anim_timer:
            self._schedule_frame()

    def _schedule_frame(self):
        # Only run while visible — a hidden page (or a closed window) shouldn't
        # keep waking to swap frames.
        if self._anim_iter is None or self._anim_timer or not self.get_mapped():
            return
        delay = self._anim_iter.get_delay_time()   # ms for the current frame
        self._anim_timer = GLib.timeout_add(max(20, delay if delay > 0 else 100),
                                            self._advance_frame)

    def _advance_frame(self):
        self._anim_timer = 0
        if self._anim_iter is None:
            return False
        self._anim_iter.advance(None)              # advance to the current time
        pb = self._anim_iter.get_pixbuf()
        if pb is not None:
            self.pixbuf = pb
            self._texture = None
            self.queue_draw()
        self._schedule_frame()
        return False                               # one-shot; _schedule re-arms

    def set_pixbuf(self, pixbuf):
        self._stop_animation()
        self.pixbuf = pixbuf
        self._texture = None
        self.rot = 0
        self.fh = self.fv = False
        self.reset_view()
        self._changed()

    def clear(self):
        self._stop_animation()
        self.pixbuf = None
        self._texture = None
        self._clip_active = False
        self._clip_masks = []
        self.reset_view()
        self._changed()

    def has_image(self):
        return self.pixbuf is not None

    def reset_view(self, *_):
        self.pane.reset_view()
        self.queue_draw()

    def export_pixbuf(self):
        """Return the pixbuf with rotate/flip baked in (no zoom/pan)."""
        if self.pixbuf is None:
            return None
        pb = self.pixbuf
        if self.fh:
            pb = pb.flip(True)
        if self.fv:
            pb = pb.flip(False)
        angle = {
            0: GdkPixbuf.PixbufRotation.NONE,
            1: GdkPixbuf.PixbufRotation.CLOCKWISE,
            2: GdkPixbuf.PixbufRotation.UPSIDEDOWN,
            3: GdkPixbuf.PixbufRotation.COUNTERCLOCKWISE,
        }[self.rot % 4]
        if angle != GdkPixbuf.PixbufRotation.NONE:
            pb = pb.rotate_simple(angle)
        return pb

    def is_transformed(self):
        return self.rot != 0 or self.fh or self.fv

    # ---------- segmentation overlays ----------
    def set_seg_mode(self, mode):
        self._seg_mode = mode
        self.queue_draw()

    def set_dwell_ms(self, ms):
        """How long to hover before drilling from the general to the specific
        object (0 drills immediately)."""
        self._dwell_ms = float(ms)

    @staticmethod
    def _load_map(path):
        """Decode a lookup map into a raw buffer the engine can index.

        Decoding is the shell's job (GdkPixbuf here, an ImageBitmap on the web);
        the engine only indexes the bytes.
        """
        if not path:
            return None
        try:
            pb = GdkPixbuf.Pixbuf.new_from_file(path)
        except GLib.Error:
            return None
        return PixelMap(pb.get_pixels(), pb.get_rowstride(), pb.get_n_channels(),
                        pb.get_width(), pb.get_height())

    def set_seg_layers(self, objects, labelmap_path, general_path=None,
                       depth_path=None):
        """Load per-object mask textures + the lookup maps (specific/general id
        per pixel + overlap depth), build marching-ants paths, play the reveal."""
        self.clear_seg(keep_mode=True)
        for o in objects:
            oid = o["id"]
            try:
                tex = Gdk.Texture.new_from_filename(o["mask"])
            except GLib.Error:
                continue
            self._seg_masks[oid] = tex
            self._seg_colors[oid] = _rgba_tuple(o["color"])
            contours = [[tuple(p) for p in poly] for poly in (o.get("contour") or [])]
            self._seg_polys[oid] = contours
            bx, by, bw, bh = o.get("bbox", (0, 0, 0, 0))
            self._seg_centroids[oid] = (bx + bw / 2.0, by + bh / 2.0)
            self._seg_radius[oid] = 0.6 * max(bw, bh, 8)
            # canonical outline ring (largest polygon) for the focus morph; the
            # remaining polygons (holes / extra parts) crossfade during a switch.
            if contours:
                largest = max(contours, key=_polygon_area_abs)
                self._seg_morph[oid] = _align_ring(
                    _resample_closed(largest, _MORPH_N))
                self._seg_sec_polys[oid] = [p for p in contours if p is not largest]
        self.maps = HitMaps(label=self._load_map(labelmap_path),
                            general=self._load_map(general_path),
                            depth=self._load_map(depth_path))
        self._begin_reveal()
        self._start_anim()
        self.queue_draw()

    def set_seg_selection(self, ids):
        self._seg_selected = set(ids)
        self.queue_draw()

    def toggle_seg(self, oid):
        if oid in self._seg_selected:
            self._seg_selected.discard(oid)
        else:
            self._seg_selected.add(oid)
            self.anim.begin_pop(oid, self._now())   # tactile "pop" on select
        self._start_anim()
        self.queue_draw()

    def get_seg_selection(self):
        return sorted(self._seg_selected)

    def set_point_mask(self, mask_path, contour=None):
        try:
            self._seg_point_tex = Gdk.Texture.new_from_filename(mask_path)
        except GLib.Error:
            self._seg_point_tex = None
        self._point_polys = [[tuple(p) for p in poly]
                             for poly in (contour or [])]
        self._begin_reveal()
        self._start_anim()
        self.queue_draw()

    def seg_texture(self, oid):
        return self._seg_masks.get(oid)

    def point_texture(self):
        return self._seg_point_tex

    # composite/clip preview (result panel): source clipped to `masks`
    def set_composite(self, pixbuf, masks, bg=None):
        self._stop_animation()
        self.pixbuf = pixbuf
        self._texture = None
        self._clip_active = True
        self._clip_masks = list(masks)
        self._clip_bg = bg
        self._changed()
        self.queue_draw()

    def update_composite(self, masks, bg=None):
        self._clip_masks = list(masks)
        self._clip_bg = bg
        self.queue_draw()

    def clear_seg(self, keep_mode=False):
        self._seg_masks = {}
        self._seg_colors = {}
        self._seg_selected = set()
        self._seg_hover_id = 0
        self._seg_point_tex = None
        self._seg_polys = {}
        self._seg_centroids = {}
        self._seg_radius = {}
        self._seg_morph = {}
        self._seg_sec_polys = {}
        self._morph = None
        self._morph_t0 = None
        self._point_polys = ()
        self._pop = {}
        self._reveal = 1.0
        self._reveal_t0 = None
        self.maps.clear()
        self._hover_gen = self._hover_spec = 0
        self._hover_dwell_t0 = None
        self._hover_drilled = False
        self._press_obj = 0
        self._press_pt = None
        self._release_t0 = None
        if not keep_mode:
            self._seg_mode = None
        self._stop_anim()
        self.queue_draw()

    def has_seg(self):
        return bool(self._seg_masks) or self._seg_point_tex is not None

    def set_scanning(self, on):
        """Show/hide the animated scan shimmer over the source (while an
        everything-pass is running). Cheaper and calmer than a numeric label."""
        on = bool(on)
        if on == self._scanning:
            return
        self._scanning = on
        if on:
            self._start_anim()
        self.queue_draw()

    # ---------- animation ----------
    def _now(self):
        """Animation clock, in microseconds.

        GdkFrameClock.get_frame_time() is on the same monotonic timebase as
        GLib.get_monotonic_time(), so falling back to it keeps timestamps
        comparable when we have no frame clock yet (unrealized). Returning 0
        there instead would stamp an animation's epoch at 0, and the moment a
        real clock arrived the animation would be instantly "expired".

        Offscreen rendering pins this (see spec/tools/rasterize.py) so a fixture
        can address any moment of any animation.
        """
        fc = self.get_frame_clock()
        return fc.get_frame_time() if fc is not None else GLib.get_monotonic_time()

    def _begin_reveal(self):
        self.anim.begin_reveal(self._now())

    def _start_anim(self):
        if self._anim_tick is None and self.get_frame_clock() is not None:
            self._t0 = None
            self._anim_tick = self.add_tick_callback(self._on_tick)

    def _stop_anim(self):
        if self._anim_tick is not None:
            self.remove_tick_callback(self._anim_tick)
            self._anim_tick = None

    def _seg_active(self):
        """Overlay state that wants the clock running but the engine doesn't own
        yet (hover, selection, a point mask)."""
        return bool(self._seg_hover_id or self._seg_selected
                    or self._seg_point_tex is not None)

    def _on_map(self, *_):
        self._resume_animation()          # GIF playback
        # Re-arm the overlay tick. A clock can be started (a reveal, say) while
        # we are still unrealized, when add_tick_callback silently does nothing —
        # so the animation would never play. Now that we have a frame clock,
        # start it if anything is actually pending.
        if self.anim.needs_tick(self._seg_active()):
            self._start_anim()

    def _on_tick(self, widget, clock):
        """Thin adapter: the engine advances every clock, the widget reacts."""
        t = clock.get_frame_time()
        tick = self.anim.advance(t, active=self._seg_active())

        if tick.dwell_fired and self._hover_gen:
            # dwelled long enough: drill from the general object to the specific
            if self._hover_spec:
                self._focus(self._hover_spec)     # morphs general -> specific
            if self.on_seg_hover:
                self.on_seg_hover(self._hover_depth, True, self._px, self._py)

        if tick.animating or tick.changed:
            self.queue_draw()
        if tick.animating:
            return GLib.SOURCE_CONTINUE
        # nothing left to animate — retire the tick so we don't wake every frame
        # while idle (a hover/press/select/scan re-arms it via _start_anim).
        self._anim_tick = None
        return GLib.SOURCE_REMOVE

    def _pop_scale(self, oid):
        return self.anim.pop_scale(oid, self._now())

    def widget_to_image(self, px, py):
        """Widget px -> image px (or None). See engine.pane.Pane."""
        return self._sync_pane().view_to_image(px, py)

    def image_to_widget(self, ix, iy):
        """Image px -> widget px (or None). The exact inverse."""
        return self._sync_pane().image_to_view(ix, iy)

    def hit_test(self, ix, iy):
        """Most specific object id at (ix, iy) (smallest on top), else 0."""
        return self.maps.specific_at(ix, iy)

    def hit_test_general(self, ix, iy):
        """Most general object id at (ix, iy) (largest on top), else 0."""
        return self.maps.general_at(ix, iy)

    def depth_at(self, ix, iy):
        """How many objects overlap at (ix, iy)."""
        return self.maps.depth_at(ix, iy)

    def hit_at(self, ix, iy):
        """Specific + general + depth in one shot (engine.hittest.Hit)."""
        return self.maps.hit(ix, iy)

    # ---------- context menu ----------
    def _build_menu(self):
        group = Gio.SimpleActionGroup()
        for name, cb in (
            ("rotate-cw", lambda *_: self._rotate(1)),
            ("rotate-ccw", lambda *_: self._rotate(-1)),
            ("flip-h", lambda *_: self._flip(True)),
            ("flip-v", lambda *_: self._flip(False)),
            ("reset", self.reset_view),
            ("fit", self.reset_view),
            ("actual", self._actual_size),
        ):
            act = Gio.SimpleAction.new(name, None)
            act.connect("activate", cb)
            group.add_action(act)
        self.insert_action_group("view", group)

        menu = Gio.Menu()
        transform = Gio.Menu()
        transform.append("Rotate right", "view.rotate-cw")
        transform.append("Rotate left", "view.rotate-ccw")
        transform.append("Flip horizontal", "view.flip-h")
        transform.append("Flip vertical", "view.flip-v")
        menu.append_section(None, transform)
        view = Gio.Menu()
        view.append("Fit to window", "view.fit")
        view.append("Actual size", "view.actual")
        view.append("Reset view", "view.reset")
        menu.append_section(None, view)

        self.popover = Gtk.PopoverMenu.new_from_model(menu)
        self.popover.set_parent(self)
        self.popover.set_has_arrow(False)

    def _on_right_click(self, gesture, n_press, x, y):
        # In point mode a right-click is a negative point, not the view menu.
        if self._seg_mode == "point":
            self._emit_seg_click(x, y, 0, "point")
            return
        rect = Gdk.Rectangle()
        rect.x, rect.y, rect.width, rect.height = int(x), int(y), 1, 1
        self.popover.set_pointing_to(rect)
        self.popover.popup()

    def _on_primary_pressed(self, gesture, n_press, x, y):
        self._press_xy = (x, y)
        # Double-click zoom toggle stays on press (only when not segmenting).
        if not self._seg_mode and n_press == 2:
            self._actual_size() if abs(self.zoom - 1.0) < 1e-3 else self.reset_view()
            return
        # Press-and-hold: ripple + glow on the focused object (the "swizzle").
        if self._seg_mode == "everything" and not self._space_down:
            pt = self.widget_to_image(x, y)
            if pt is not None:
                ix, iy = int(pt[0]), int(pt[1])
                oid = self._seg_hover_id or self.hit_test(ix, iy)
                if oid:
                    self.anim.begin_press(oid, (ix, iy), self._now())
                    self._start_anim()
                    self.queue_draw()

    def _on_primary_released(self, gesture, n_press, x, y):
        # Don't clear the press object — let the ripple + spring-back play out
        # (the tick retires it once fully decayed). Only stamp the release time.
        if self.anim.release_press(self._now()):
            self.queue_draw()
        # Select on release, and only for a real click — never while panning
        # (Space held) or when the press turned into a drag.
        if not self._seg_mode or self._space_down:
            return
        px, py = self._press_xy or (x, y)
        if abs(x - px) > 4 or abs(y - py) > 4:
            return
        neg = bool(gesture.get_current_event_state()
                   & Gdk.ModifierType.CONTROL_MASK)
        if self._seg_mode == "point":
            self._emit_seg_click(x, y, 0 if neg else 1, "point")
        else:
            self._emit_seg_click(x, y, None, "toggle")

    def _on_enter(self, ctrl, x, y):
        # Take keyboard focus so the Space-to-pan key controller receives keys.
        if self.pixbuf is not None:
            self.grab_focus()

    def _on_key_pressed(self, ctrl, keyval, keycode, state):
        if keyval == Gdk.KEY_space and not self._space_down:
            self._space_down = True
            self.set_cursor(self._grab_cursor)
            if self._seg_hover_id:
                self._seg_hover_id = 0
                self.queue_draw()
            return True          # consume so a focused button isn't triggered
        return False

    def _on_key_released(self, ctrl, keyval, keycode, state):
        if keyval == Gdk.KEY_space:
            self._space_down = False
            self.set_cursor(None)
            return True
        return False

    def _emit_seg_click(self, x, y, value, kind):
        pt = self.widget_to_image(x, y)
        if pt is None:
            return
        ix, iy = int(pt[0]), int(pt[1])
        iw, ih = self.pixbuf.get_width(), self.pixbuf.get_height()
        if not (0 <= ix < iw and 0 <= iy < ih):
            return
        if kind == "toggle":
            oid = self._seg_hover_id or self.hit_test(ix, iy)   # the focused one
            if oid:
                self.toggle_seg(oid)
            if self.on_seg_click:
                self.on_seg_click(ix, iy, oid, "toggle")
        else:
            if self.on_seg_click:
                self.on_seg_click(ix, iy, value, "point")

    # ---------- transforms ----------
    def _rotate(self, delta):
        if self.pixbuf is None:
            return
        # Overlays are registered to un-rotated image pixels; drop them so they
        # can't desync (the Segment page hides these buttons anyway).
        if self.has_seg():
            self.clear_seg(keep_mode=True)
        self.pane.rotate(delta)      # also resets zoom/pan
        self.queue_draw()
        self._changed()

    def _flip(self, horizontal):
        if self.pixbuf is None:
            return
        if self.has_seg():
            self.clear_seg(keep_mode=True)
        self.pane.flip(horizontal)
        self.queue_draw()
        self._changed()

    def _actual_size(self, *_):
        if self.pixbuf is None:
            return
        self._sync_pane().actual_size()
        self.queue_draw()

    # ---------- pointer / gestures ----------
    def _on_motion(self, ctrl, x, y):
        self._px, self._py = x, y
        held = self._press_obj and self._release_t0 is None
        if self._space_down or held:
            return                       # panning / holding — don't retrack hover
        if not (self._seg_mode == "everything" and self._seg_masks):
            return
        self._start_anim()               # hovering animates; re-arm an idle tick
        pt = self.widget_to_image(x, y)
        gen = spec = depth = 0
        if pt is not None:
            ix, iy = int(pt[0]), int(pt[1])
            gen = self.hit_test_general(ix, iy)
            spec = self.hit_test(ix, iy)
            depth = self.depth_at(ix, iy)
        self._hover_depth = depth
        if gen != self._hover_gen:
            # Entered a new object region: focus the GENERAL object and start the
            # dwell timer (the tick drills to the specific object after a while).
            self._hover_gen = gen
            self._hover_spec = spec
            self._hover_drilled = False
            self._hover_dwell_t0 = self._now() if gen else None
            self._focus(gen)                  # morph the outline into the new object
            self.queue_draw()
            if self.on_seg_hover:
                self.on_seg_hover(depth, False, self._px, self._py)
        else:
            self._hover_spec = spec
            if self._hover_drilled and self._seg_hover_id != spec:
                self._focus(spec)             # track the specific under the cursor
                self.queue_draw()
            # keep the stacked-objects badge glued to the pointer as it moves
            if self.on_seg_hover:
                self.on_seg_hover(depth, self._hover_drilled,
                                  self._px, self._py)

    def _on_leave(self, ctrl):
        self._hover_gen = self._hover_spec = 0
        self._hover_dwell_t0 = None
        self._hover_drilled = False
        if self._seg_hover_id:
            self._focus(0)                    # morph the outline away
            self.queue_draw()
        if self.on_seg_hover:
            self.on_seg_hover(0, False, self._px, self._py)

    def _on_scroll(self, ctrl, dx, dy):
        if self.pixbuf is None:
            return False
        factor = 1.0 / 1.1 if dy > 0 else 1.1
        self._zoom_at(factor, self._px, self._py)
        return True

    def _zoom_at(self, factor, cx, cy):
        self._sync_pane().zoom_at(factor, cx, cy)
        self.queue_draw()

    def _on_drag_begin(self, gesture, x, y):
        self._ox0, self._oy0 = self.ox, self.oy

    def _on_drag_update(self, gesture, ox, oy):
        self.ox = self._ox0 + ox
        self.oy = self._oy0 + oy
        self.queue_draw()

    # ---------- rendering ----------
    # The widget no longer draws anything by hand. The engine builds a display
    # list (what to draw); gsk_renderer replays it (how). The same list is what a
    # Canvas2D backend in the browser will consume.
    def _effective_size(self):
        return self._sync_pane().effective_size()

    def _get_texture(self):
        if self._texture is None and self.pixbuf is not None:
            self._texture = Gdk.Texture.new_for_pixbuf(self.pixbuf)
        return self._texture

    def _resolve(self, handle):
        """Display-list image handle -> Gdk.Texture. The engine only ever passes
        handles; the shell owns the textures."""
        if handle == "src":
            return self._get_texture()
        if handle == "point":
            return self._seg_point_tex
        if isinstance(handle, tuple) and handle[0] == "clip":
            i = handle[1]
            return self._clip_masks[i] if i < len(self._clip_masks) else None
        return self._seg_masks.get(handle)

    def _scene(self):
        """Snapshot the widget's state as plain data for the engine."""
        sc = Scene()
        if self.pixbuf is not None:
            sc.image = "src"
            sc.image_size = (self.pixbuf.get_width(), self.pixbuf.get_height())
        sc.seg_mode = self._seg_mode
        sc.masks = {oid: oid for oid in self._seg_masks}   # handle == object id
        sc.colors = self._seg_colors
        sc.polys = self._seg_polys
        sc.sec_polys = self._seg_sec_polys
        sc.centroids = self._seg_centroids
        sc.radius = self._seg_radius
        sc.selected = frozenset(self._seg_selected)
        sc.hover_id = self._seg_hover_id
        sc.hover_gen = self._hover_gen
        sc.hover_spec = self._hover_spec
        sc.hover_depth = self._hover_depth
        if self._seg_point_tex is not None:
            sc.point_mask = "point"
            sc.point_polys = self._point_polys
        sc.morph = self._morph
        sc.clip_active = self._clip_active
        sc.clip_masks = tuple(("clip", i) for i in range(len(self._clip_masks)))
        sc.clip_bg = self._clip_bg
        return sc

    def do_snapshot(self, snapshot):
        pane = self._sync_pane()
        dl = build_display_list(self._scene(), pane, self.anim, self._now())
        gsk_render(snapshot, dl, self._resolve)
        if self.on_paint and len(dl.ops) > 1:      # >1 == the image was drawn
            self.on_paint(int(round(dl.scale * 100)))


    def _focus(self, oid):
        """Move the hover focus to `oid`, tweening the outline from the old one."""
        if oid == self._seg_hover_id:
            return
        old = self._seg_hover_id
        self._seg_hover_id = oid
        self._begin_morph(old, oid)

    def _begin_morph(self, old, new):
        fr = self._seg_morph.get(old)
        to = self._seg_morph.get(new)
        if fr is None and to is None:
            self.anim.clear_morph()
            self._morph = None
            return
        self._morph = Morph(
            from_ring=fr, to_ring=to,
            cf=(self._seg_colors.get(old) or self._seg_colors.get(new)
                or POINT_COLOR),
            ct=(self._seg_colors.get(new) or self._seg_colors.get(old)
                or POINT_COLOR),
            sec_from=self._seg_sec_polys.get(old),
            sec_to=self._seg_sec_polys.get(new),
            cen_from=self._seg_centroids.get(old),
            cen_to=self._seg_centroids.get(new))
        self.anim.begin_morph(self._now())
        self._start_anim()

    def _changed(self):
        if self._on_change:
            self._on_change()
