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
from engine.geometry import (  # noqa: E402
    MIN_ZOOM, MAX_ZOOM,
    polygon_area_abs as _polygon_area_abs,
    resample_closed as _resample_closed,
    align_ring as _align_ring,
    ease_out as _ease_out,
    ease_out_back as _ease_out_back,
)

_CELL = 16
_REVEAL_MS = 300.0
_POP_MS = 300.0
_DWELL_MS = 1600.0      # default: hover this long to drill general -> specific
# press-and-hold "swizzle": a wave spawns every _SPAWN ms while held, each wave
# lives _WAVE ms; on release the object springs back and the glow decays.
_PRESS_WAVE_MS = 720.0
_PRESS_SPAWN_MS = 450.0
_PRESS_DECAY_MS = 460.0
_PRESS_SPRING_MS = 360.0
_MORPH_MS = 240.0       # outline tween when the focused object changes
_MORPH_N = 64           # resampled outline vertices (correspondence for the lerp)


def _rgba(r, g, b):
    c = Gdk.RGBA()
    c.red, c.green, c.blue, c.alpha = r, g, b, 1.0
    return c


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

        self._c_light = _rgba(0.20, 0.20, 0.22)
        self._c_dark = _rgba(0.15, 0.15, 0.17)

        # ---- segmentation overlay state ----
        self._seg_mode = None            # None | "everything" | "point"
        self._seg_masks = {}             # id -> Gdk.Texture (per-object mask)
        self._seg_colors = {}            # id -> Gdk.RGBA (tint)
        self._seg_selected = set()       # ids kept for output
        self._seg_hover_id = 0           # object under the cursor (glow)
        self._seg_point_tex = None       # Gdk.Texture (point-mode object)
        self._seg_point_color = _rgba(0.20, 0.52, 0.90)
        self._dim = _rgba(0.0, 0.0, 0.0)
        self._lm_pixels = None           # specific id per pixel (smallest on top)
        self._lm_rs = self._lm_nc = 0
        self._lm_w = self._lm_h = 0
        self._lmg_pixels = None          # general id per pixel (largest on top)
        self._lmg_rs = self._lmg_nc = 0
        self._depth_pixels = None        # overlap count per pixel (L)
        self._depth_rs = self._depth_nc = 0
        # composite/clip mode (result panel): show the source clipped to a union
        # of masks over the checkerboard — a live cutout preview.
        self._clip_active = False
        self._clip_masks = []            # list of Gdk.Texture (selected objects)
        self._clip_bg = None             # Gdk.RGBA solid background, or None
        # fired on a click while a seg mode is active:
        #   everything: (ix, iy, object_id, "toggle")
        #   point:      (ix, iy, label 1/0, "point")
        self.on_seg_click = None
        # fired as the hovered object stack changes: (depth:int, drilled:bool)
        self.on_seg_hover = None
        # hover-dwell disambiguation (general object first, drill to specific)
        self._hover_gen = 0
        self._hover_spec = 0
        self._hover_depth = 0
        self._hover_dwell_t0 = None
        self._hover_drilled = False
        self._dwell_ms = _DWELL_MS
        # press-and-hold ripple. The animation is "alive" from press until it has
        # fully decayed after release; _release_t0 is None only while held.
        self._press_obj = 0
        self._press_pt = None            # (ix, iy) image px
        self._press_t0 = None
        self._release_t0 = None
        # scan shimmer shown while segmentation is running
        self._scanning = False
        self._scan_phase = 0.0

        # ---- interaction: space-to-pan, click-vs-drag ----
        self._space_down = False
        self._press_xy = None            # press position, to tell click from drag
        self._grab_cursor = Gdk.Cursor.new_from_name("grabbing", None)

        # ---- animation state (tick-driven; see _on_tick / _snapshot_seg) ----
        self._seg_paths = {}             # id -> Gsk.Path (marching-ants outline)
        self._seg_centroids = {}         # id -> (cx, cy) image px, for "pop"
        self._seg_radius = {}            # id -> ripple reach (image px)
        self._seg_morph = {}             # id -> [N (x,y)] canonical outline ring
        self._seg_sec_paths = {}         # id -> Gsk.Path of secondary polys, or None
        self._point_path = None          # Gsk.Path for point-mode object
        # outline morph between focused objects (hover switch / dwell drill)
        self._morph_from = None          # ring or None (None = entering)
        self._morph_to = None            # ring or None (None = leaving)
        self._morph_t0 = None            # µs, or None when idle
        self._morph_cf = None            # from colour
        self._morph_ct = None            # to colour
        self._morph_sec_from = None      # secondary polys of the old object
        self._morph_sec_to = None        # secondary polys of the new object
        self._morph_cen_from = None      # (cx, cy) to bloom the old ring toward
        self._morph_cen_to = None
        self._anim_tick = None           # add_tick_callback handle
        self._t0 = None                  # animation epoch (µs)
        self._pulse = 0.0                # 0..1 breathing
        self._ant = 0.0                  # marching-ants dash phase (px)
        self._reveal = 1.0               # 0..1 fade/scale-in of overlays
        self._reveal_t0 = None           # µs
        self._pop = {}                   # id -> select "pop" start time (µs)

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
        self.connect("map", self._resume_animation)
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
        if not path:
            return (None, 0, 0, 0, 0)
        try:
            pb = GdkPixbuf.Pixbuf.new_from_file(path)
        except GLib.Error:
            return (None, 0, 0, 0, 0)
        return (pb.get_pixels(), pb.get_rowstride(), pb.get_n_channels(),
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
            c = Gdk.RGBA()
            c.parse(o["color"])
            self._seg_colors[oid] = c
            contours = o.get("contour") or []
            self._seg_paths[oid] = self._path_from_contours(contours)
            bx, by, bw, bh = o.get("bbox", (0, 0, 0, 0))
            self._seg_centroids[oid] = (bx + bw / 2.0, by + bh / 2.0)
            self._seg_radius[oid] = 0.6 * max(bw, bh, 8)
            # canonical outline ring (largest polygon) for the focus morph; the
            # remaining polygons (holes / extra parts) crossfade during a switch.
            if contours:
                largest = max(contours, key=_polygon_area_abs)
                self._seg_morph[oid] = _align_ring(
                    _resample_closed(largest, _MORPH_N))
                rest = [poly for poly in contours if poly is not largest]
                self._seg_sec_paths[oid] = (
                    self._path_from_contours(rest) if rest else None)
        (self._lm_pixels, self._lm_rs, self._lm_nc,
         self._lm_w, self._lm_h) = self._load_map(labelmap_path)
        self._lmg_pixels, self._lmg_rs, self._lmg_nc, _, _ = \
            self._load_map(general_path)
        self._depth_pixels, self._depth_rs, self._depth_nc, _, _ = \
            self._load_map(depth_path)
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
            self._pop[oid] = self._now()      # tactile "pop" on select
        self._start_anim()
        self.queue_draw()

    def get_seg_selection(self):
        return sorted(self._seg_selected)

    def set_point_mask(self, mask_path, contour=None):
        try:
            self._seg_point_tex = Gdk.Texture.new_from_filename(mask_path)
        except GLib.Error:
            self._seg_point_tex = None
        self._point_path = self._path_from_contours(contour)
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
        self._seg_paths = {}
        self._seg_centroids = {}
        self._seg_radius = {}
        self._seg_morph = {}
        self._seg_sec_paths = {}
        self._morph_from = self._morph_to = None
        self._morph_t0 = None
        self._point_path = None
        self._pop = {}
        self._reveal = 1.0
        self._reveal_t0 = None
        self._lm_pixels = self._lmg_pixels = self._depth_pixels = None
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
        fc = self.get_frame_clock()
        return fc.get_frame_time() if fc is not None else 0

    def _path_from_contours(self, contours):
        if not contours:
            return None
        pb = Gsk.PathBuilder()
        added = False
        for poly in contours:
            if len(poly) < 2:
                continue
            pb.move_to(poly[0][0], poly[0][1])
            for x, y in poly[1:]:
                pb.line_to(x, y)
            pb.close()
            added = True
        return pb.to_path() if added else None

    def _begin_reveal(self):
        self._reveal = 0.0
        self._reveal_t0 = self._now()

    def _start_anim(self):
        if self._anim_tick is None and self.get_frame_clock() is not None:
            self._t0 = None
            self._anim_tick = self.add_tick_callback(self._on_tick)

    def _stop_anim(self):
        if self._anim_tick is not None:
            self.remove_tick_callback(self._anim_tick)
            self._anim_tick = None

    def _on_tick(self, widget, clock):
        t = clock.get_frame_time()
        if self._t0 is None:
            self._t0 = t
        el = (t - self._t0) / 1_000_000.0
        self._pulse = 0.5 * (1.0 + math.sin(el * 2.0 * math.pi * 1.1))
        self._ant = el * 24.0                      # dash travel, px/s
        if self._scanning:
            self._scan_phase = (el / 1.2) % 1.0    # sweep loop, ~1.2 s
        if self._reveal_t0 is not None:
            p = (t - self._reveal_t0) / 1000.0 / _REVEAL_MS
            if p >= 1.0:
                self._reveal, self._reveal_t0 = 1.0, None
            else:
                self._reveal = 1.0 - (1.0 - p) ** 3   # ease-out cubic
        for oid in [k for k, t0 in self._pop.items()
                    if (t - t0) / 1000.0 > _POP_MS]:
            del self._pop[oid]
        # hover-dwell: after dwelling on a general object, drill to the specific
        if (self._hover_gen and not self._hover_drilled
                and self._hover_dwell_t0 is not None
                and (t - self._hover_dwell_t0) / 1000.0 > self._dwell_ms):
            self._hover_drilled = True
            if self._hover_spec:
                self._focus(self._hover_spec)     # morph general -> specific
            self.queue_draw()
            if self.on_seg_hover:
                self.on_seg_hover(self._hover_depth, True, self._px, self._py)
        # retire the press animation once it has fully decayed after release
        if (self._press_obj and self._release_t0 is not None
                and (t - self._release_t0) / 1000.0 > _PRESS_WAVE_MS):
            self._press_obj = 0
            self._press_pt = None
            self._release_t0 = None
            self.queue_draw()
        # retire a finished outline morph
        if (self._morph_t0 is not None
                and (t - self._morph_t0) / 1000.0 >= _MORPH_MS):
            self._morph_t0 = None
            self.queue_draw()
        animating = (self._seg_hover_id or self._seg_selected or self._press_obj
                     or self._seg_point_tex is not None or self._scanning
                     or self._morph_t0 is not None
                     or self._reveal_t0 is not None or self._pop)
        if animating:
            self.queue_draw()
            return GLib.SOURCE_CONTINUE
        # nothing left to animate — retire the tick so we don't wake every frame
        # while idle (a hover/press/select/scan re-arms it via _start_anim).
        self._anim_tick = None
        return GLib.SOURCE_REMOVE

    def _pop_scale(self, oid):
        t0 = self._pop.get(oid)
        if t0 is None:
            return 1.0
        p = (self._now() - t0) / 1000.0 / _POP_MS
        if p <= 0.0 or p >= 1.0:
            return 1.0
        return 1.0 + 0.09 * math.sin(p * math.pi) * (1.0 - p)

    def widget_to_image(self, px, py):
        """Widget px -> image px (or None). See engine.pane.Pane."""
        return self._sync_pane().view_to_image(px, py)

    def image_to_widget(self, ix, iy):
        """Image px -> widget px (or None). The exact inverse."""
        return self._sync_pane().image_to_view(ix, iy)

    def _id_at(self, pixels, rs, nc, ix, iy):
        if pixels is None or not (0 <= ix < self._lm_w and 0 <= iy < self._lm_h):
            return 0
        i = iy * rs + ix * nc
        return pixels[i] + ((pixels[i + 1] << 8) if nc >= 2 else 0)

    def hit_test(self, ix, iy):
        """Most specific object id at (ix, iy) (smallest on top), else 0."""
        return self._id_at(self._lm_pixels, self._lm_rs, self._lm_nc, ix, iy)

    def hit_test_general(self, ix, iy):
        """Most general object id at (ix, iy) (largest on top), else 0."""
        return self._id_at(self._lmg_pixels, self._lmg_rs, self._lmg_nc, ix, iy)

    def depth_at(self, ix, iy):
        """How many objects overlap at (ix, iy)."""
        if (self._depth_pixels is None
                or not (0 <= ix < self._lm_w and 0 <= iy < self._lm_h)):
            return 0
        return self._depth_pixels[iy * self._depth_rs + ix * self._depth_nc]

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
                    self._press_obj = oid
                    self._press_pt = (ix, iy)
                    self._press_t0 = self._now()
                    self._release_t0 = None
                    self._start_anim()
                    self.queue_draw()

    def _on_primary_released(self, gesture, n_press, x, y):
        # Don't clear the press object — let the ripple + spring-back play out
        # (the tick retires it once fully decayed). Only stamp the release time.
        if self._press_obj and self._release_t0 is None:
            self._release_t0 = self._now()
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
    def _effective_size(self):
        return self._sync_pane().effective_size()

    def _get_texture(self):
        if self._texture is None and self.pixbuf is not None:
            self._texture = Gdk.Texture.new_for_pixbuf(self.pixbuf)
        return self._texture

    def _snapshot_checker(self, snapshot, w, h):
        snapshot.append_color(self._c_light, Graphene.Rect().init(0, 0, w, h))
        cols = int(w // _CELL) + 1
        rows = int(h // _CELL) + 1
        for j in range(rows):
            for i in range(cols):
                if (i + j) & 1:
                    snapshot.append_color(
                        self._c_dark,
                        Graphene.Rect().init(i * _CELL, j * _CELL, _CELL, _CELL))

    def do_snapshot(self, snapshot):
        w, h = self.get_width(), self.get_height()
        self._snapshot_checker(snapshot, w, h)

        tex = self._get_texture()
        if tex is None:
            return
        iw, ih = self.pixbuf.get_width(), self.pixbuf.get_height()
        ew, eh = self._effective_size()
        if ew == 0 or eh == 0 or w <= 0 or h <= 0:
            return                       # unallocated: scale would be 0 (stroke /0)
        fit = min(w / ew, h / eh)
        scale = fit * self.zoom

        snapshot.save()
        snapshot.translate(Graphene.Point().init(w / 2 + self.ox,
                                                 h / 2 + self.oy))
        snapshot.scale(scale, scale)
        if self.rot:
            snapshot.rotate(self.rot * 90)
        snapshot.scale(-1 if self.fh else 1, -1 if self.fv else 1)
        snapshot.translate(Graphene.Point().init(-iw / 2, -ih / 2))
        rect = Graphene.Rect().init(0, 0, iw, ih)
        if self._clip_active:
            self._snapshot_composite(snapshot, tex, rect)
        else:
            snapshot.append_scaled_texture(tex, Gsk.ScalingFilter.TRILINEAR, rect)
            if self._scanning:
                self._snapshot_shimmer(snapshot, rect, iw, ih)
            if self._seg_mode:
                if self._reveal < 1.0:      # fade + gentle scale-in of overlays
                    snapshot.push_opacity(max(0.0, self._reveal))
                    s = 0.97 + 0.03 * self._reveal
                    snapshot.translate(Graphene.Point().init(iw / 2, ih / 2))
                    snapshot.scale(s, s)
                    snapshot.translate(Graphene.Point().init(-iw / 2, -ih / 2))
                    self._snapshot_seg(snapshot, rect, scale)
                    snapshot.pop()
                else:
                    self._snapshot_seg(snapshot, rect, scale)
        snapshot.restore()

        if self.on_paint:
            self.on_paint(int(round(scale * 100)))

    def _snapshot_shimmer(self, snapshot, rect, iw, ih):
        """A soft diagonal highlight band sweeping over the image on a loop — the
        "scanning" cue. A moving bright stop in an otherwise-transparent linear
        gradient (top-left -> bottom-right), clipped to the image bounds."""
        # p sweeps from before the top-left to past the bottom-right so the band
        # enters and leaves rather than wrapping abruptly.
        p = -0.20 + 1.40 * self._scan_phase
        band = 0.13

        def stop(off, a):
            s = Gsk.ColorStop()
            c = Gdk.RGBA()
            c.red = c.green = c.blue = 1.0
            c.alpha = a
            s.offset = min(1.0, max(0.0, off))
            s.color = c
            return s

        stops = [stop(0.0, 0.0), stop(p - band, 0.0), stop(p, 0.18),
                 stop(p + band, 0.0), stop(1.0, 0.0)]
        snapshot.push_opacity(0.9)
        snapshot.append_linear_gradient(
            rect, Graphene.Point().init(0, 0),
            Graphene.Point().init(iw, ih), stops)
        snapshot.pop()

    def _tint(self, snapshot, tex, col, alpha, rect):
        """Draw `col` at `alpha` wherever `tex`'s luminance is set."""
        snapshot.push_mask(Gsk.MaskMode.LUMINANCE)
        snapshot.append_texture(tex, rect)              # mask (first)
        snapshot.pop()
        snapshot.push_opacity(alpha)
        snapshot.append_color(col, rect)                # content
        snapshot.pop()
        snapshot.pop()

    def _glow(self, snapshot, tex, col, alpha, radius, rect):
        """Soft blurred halo of `col` around the masked shape."""
        snapshot.push_blur(radius)
        self._tint(snapshot, tex, col, alpha, rect)
        snapshot.pop()

    def _ants(self, snapshot, path, col, scale):
        """Animated marching-ants stroke along `path` (image-px coords). Dash and
        width are divided by `scale` so they stay screen-constant under zoom."""
        if path is None:
            return
        st = Gsk.Stroke.new(max(0.6, 1.6 / scale))
        st.set_dash([6.0 / scale, 4.5 / scale])
        st.set_dash_offset(-self._ant / scale)
        snapshot.append_stroke(path, st, col)

    # ---------- outline morph (focus change) ----------
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
            self._morph_t0 = None
            return
        self._morph_from, self._morph_to = fr, to
        self._morph_cf = (self._seg_colors.get(old) or self._seg_colors.get(new)
                          or self._seg_point_color)
        self._morph_ct = (self._seg_colors.get(new) or self._seg_colors.get(old)
                          or self._seg_point_color)
        self._morph_sec_from = self._seg_sec_paths.get(old)
        self._morph_sec_to = self._seg_sec_paths.get(new)
        self._morph_cen_from = self._seg_centroids.get(old)
        self._morph_cen_to = self._seg_centroids.get(new)
        self._morph_t0 = self._now()
        self._start_anim()

    @staticmethod
    def _blend(a, b, e):
        c = Gdk.RGBA()
        c.red = a.red + (b.red - a.red) * e
        c.green = a.green + (b.green - a.green) * e
        c.blue = a.blue + (b.blue - a.blue) * e
        c.alpha = a.alpha + (b.alpha - a.alpha) * e
        return c

    @staticmethod
    def _alpha(col, a):
        c = Gdk.RGBA()
        c.red, c.green, c.blue = col.red, col.green, col.blue
        c.alpha = col.alpha * max(0.0, min(1.0, a))
        return c

    def _ring_path(self, ring):
        pb = Gsk.PathBuilder()
        pb.move_to(ring[0][0], ring[0][1])
        for x, y in ring[1:]:
            pb.line_to(x, y)
        pb.close()
        return pb.to_path()

    def _snapshot_focus_outline(self, snapshot, hover, sel, scale):
        """The hovered object's marching-ants outline. While the focus is
        changing, tween the largest ring old->new and crossfade the rest."""
        now = self._now()
        if (self._morph_t0 is not None
                and (now - self._morph_t0) / 1000.0 < _MORPH_MS):
            e = _ease_out((now - self._morph_t0) / 1000.0 / _MORPH_MS)
            self._draw_morph(snapshot, e, scale)
            return
        if hover and hover in self._seg_masks and hover not in sel:
            col = self._seg_colors.get(hover, self._seg_point_color)
            self._ants(snapshot, self._seg_paths.get(hover), col, scale)

    def _draw_morph(self, snapshot, e, scale):
        a, b = self._morph_from, self._morph_to
        if a is not None and b is not None:              # switch: lerp ring
            ring = [(pa[0] + (pb[0] - pa[0]) * e, pa[1] + (pb[1] - pa[1]) * e)
                    for pa, pb in zip(a, b)]
            col, env = self._blend(self._morph_cf, self._morph_ct, e), 1.0
        elif b is not None:                              # enter: bloom from centroid
            cx, cy = self._morph_cen_to or b[0]
            ring = [(cx + (pb[0] - cx) * e, cy + (pb[1] - cy) * e) for pb in b]
            col, env = self._morph_ct, e
        elif a is not None:                              # leave: collapse to centroid
            cx, cy = self._morph_cen_from or a[0]
            ring = [(pa[0] + (cx - pa[0]) * e, pa[1] + (cy - pa[1]) * e) for pa in a]
            col, env = self._morph_cf, 1.0 - e
        else:
            return
        self._ants(snapshot, self._ring_path(ring), self._alpha(col, env), scale)
        if self._morph_sec_from is not None:
            self._ants(snapshot, self._morph_sec_from,
                       self._alpha(self._morph_cf, 1.0 - e), scale)
        if self._morph_sec_to is not None:
            self._ants(snapshot, self._morph_sec_to,
                       self._alpha(self._morph_ct, e), scale)

    def _snapshot_seg(self, snapshot, rect, scale):
        """Faintly tint clickable objects; glow + march ants on the hovered and
        selected ones; dim everything outside the selection. GPU-side (mask
        textures as luminance masks; a Gsk.Path stroke for the ants)."""
        sel, hover = self._seg_selected, self._seg_hover_id
        gen, spec = self._hover_gen, self._hover_spec
        # over a stack, light up the whole (general) and the part (specific) as
        # distinct colour layers rather than one flat highlight.
        layered = (self._hover_depth >= 2 and gen and spec and gen != spec
                   and gen in self._seg_masks and spec in self._seg_masks
                   and gen not in sel and spec not in sel)
        pulse = 0.35 + 0.65 * self._pulse
        skip = {gen, spec} if layered else {hover}
        # 1. faint tint on every non-selected, non-focused object.
        for oid, tex in self._seg_masks.items():
            if oid in sel or oid in skip:
                continue
            col = self._seg_colors.get(oid)
            if col is not None:
                self._tint(snapshot, tex, col, 0.12, rect)
        # 2. dim everything outside the selection so kept objects stand out.
        if self._seg_masks and sel:
            snapshot.push_mask(Gsk.MaskMode.INVERTED_LUMINANCE)
            for oid in sel:
                snapshot.append_texture(self._seg_masks[oid], rect)
            snapshot.pop()
            snapshot.push_opacity(0.55)
            snapshot.append_color(self._dim, rect)
            snapshot.pop()
            snapshot.pop()
        # 3. selected objects — glow + fill + ants, with a "pop" scale.
        for oid in sel:
            tex = self._seg_masks.get(oid)
            col = self._seg_colors.get(oid)
            if tex is None or col is None:
                continue
            pop = self._pop_scale(oid)
            snapshot.save()
            if pop != 1.0:
                cx, cy = self._seg_centroids.get(oid) or (
                    self.pixbuf.get_width() / 2, self.pixbuf.get_height() / 2)
                snapshot.translate(Graphene.Point().init(cx, cy))
                snapshot.scale(pop, pop)
                snapshot.translate(Graphene.Point().init(-cx, -cy))
            self._glow(snapshot, tex, col, 0.22 + 0.20 * self._pulse, 22.0, rect)
            self._tint(snapshot, tex, col, 0.42, rect)
            self._ants(snapshot, self._seg_paths.get(oid), col, scale)
            snapshot.restore()
        # 4. hovered (not selected). Over a stack, dim outside the whole and show
        #    whole + part as distinct layers; otherwise the single hovered object.
        #    The marching-ants outline morphs on top (see _snapshot_focus_outline).
        if layered:
            gtex, gcol = self._seg_masks[gen], self._seg_colors[gen]
            stex, scol = self._seg_masks[spec], self._seg_colors[spec]
            snapshot.push_mask(Gsk.MaskMode.INVERTED_LUMINANCE)
            snapshot.append_texture(gtex, rect)
            snapshot.pop()
            snapshot.push_opacity(0.5)
            snapshot.append_color(self._dim, rect)
            snapshot.pop()
            snapshot.pop()
            self._glow(snapshot, gtex, gcol, 0.14 * pulse, 22.0, rect)
            self._tint(snapshot, gtex, gcol, 0.18, rect)     # whole, beneath
            self._glow(snapshot, stex, scol, 0.26 * pulse, 18.0, rect)
            self._tint(snapshot, stex, scol, 0.40, rect)     # part, on top
        elif hover and hover in self._seg_masks and hover not in sel:
            col = self._seg_colors.get(hover, self._seg_point_color)
            self._glow(snapshot, self._seg_masks[hover], col, 0.28 * pulse, 20.0, rect)
            self._tint(snapshot, self._seg_masks[hover], col, 0.34, rect)
        self._snapshot_focus_outline(snapshot, hover, sel, scale)

        # Point mode — dim outside the object, glow + tint + ants.
        if self._seg_point_tex is not None:
            snapshot.push_mask(Gsk.MaskMode.INVERTED_LUMINANCE)
            snapshot.append_texture(self._seg_point_tex, rect)
            snapshot.pop()
            snapshot.push_opacity(0.55)
            snapshot.append_color(self._dim, rect)
            snapshot.pop()
            snapshot.pop()
            col = self._seg_point_color
            self._glow(snapshot, self._seg_point_tex, col, 0.20 + 0.18 * self._pulse,
                       20.0, rect)
            self._tint(snapshot, self._seg_point_tex, col, 0.30, rect)
            self._ants(snapshot, self._point_path, col, scale)

        # Press-and-hold "swizzle": ripple inside the object + strong glow outline.
        if self._press_obj in self._seg_masks and self._press_pt:
            self._snapshot_press(snapshot, rect, scale)

    def _snapshot_press(self, snapshot, rect, scale):
        """The press "swizzle": a slight press-scale, an intensified glow that
        decays after release, and expanding waves clipped to the object. Waves
        spawn every _PRESS_SPAWN_MS while held; each lives _PRESS_WAVE_MS — so a
        quick tap still plays one full wave that outlives the click."""
        oid = self._press_obj
        tex = self._seg_masks[oid]
        col = self._seg_colors.get(oid, self._seg_point_color)
        now = self._now()
        held = self._release_t0 is None
        rel_ms = 0.0 if held else (now - self._release_t0) / 1000.0
        fade = 1.0 if held else max(0.0, 1.0 - rel_ms / _PRESS_DECAY_MS)
        # press-scale: 0.97 while held, springs back to 1.0 (with overshoot).
        if held:
            sc = 0.97
        else:
            sc = 0.97 + 0.03 * _ease_out_back(min(1.0, rel_ms / _PRESS_SPRING_MS))
        iw, ih = self.pixbuf.get_width(), self.pixbuf.get_height()
        cx, cy = self._seg_centroids.get(oid) or (iw / 2, ih / 2)
        snapshot.save()
        if abs(sc - 1.0) > 1e-4:
            snapshot.translate(Graphene.Point().init(cx, cy))
            snapshot.scale(sc, sc)
            snapshot.translate(Graphene.Point().init(-cx, -cy))
        # intensified glow + bright fill + ants (all decay after release)
        glow_a = (0.34 + 0.24 * self._pulse) if held else 0.40 * fade
        self._glow(snapshot, tex, col, glow_a, 26.0, rect)
        self._tint(snapshot, tex, col, 0.30 * (1.0 if held else fade), rect)
        ant_col = col
        if not held and fade < 1.0:                     # fade the outline out too
            ant_col = Gdk.RGBA()
            ant_col.red, ant_col.green, ant_col.blue = col.red, col.green, col.blue
            ant_col.alpha = col.alpha * fade
        self._ants(snapshot, self._seg_paths.get(oid), ant_col, scale)
        # expanding waves from the press point, clipped to the object
        px, py = self._press_pt
        maxr = self._seg_radius.get(oid, 160.0)
        age_total = (now - self._press_t0) / 1000.0
        last_spawn = age_total if held else (self._release_t0 - self._press_t0) / 1000.0
        n = int(last_spawn / _PRESS_SPAWN_MS) + 1
        first = max(0, int((age_total - _PRESS_WAVE_MS) / _PRESS_SPAWN_MS))
        snapshot.push_mask(Gsk.MaskMode.LUMINANCE)
        snapshot.append_texture(tex, rect)              # clip content to object
        snapshot.pop()
        for i in range(first, n):
            age = age_total - i * _PRESS_SPAWN_MS
            if age < 0.0 or age >= _PRESS_WAVE_MS:
                continue
            ph = age / _PRESS_WAVE_MS
            r = 8.0 + _ease_out(ph) * maxr
            a = (1.0 - ph) * 0.5 * (1.0 if held else fade)
            ring = Gsk.PathBuilder()
            ring.add_circle(Graphene.Point().init(px, py), r)
            st = Gsk.Stroke.new(max(1.0, 3.0 / scale))
            rc = Gdk.RGBA()
            rc.red, rc.green, rc.blue, rc.alpha = 1.0, 1.0, 1.0, a
            snapshot.append_stroke(ring.to_path(), st, rc)
        snapshot.pop()
        snapshot.restore()

    def _snapshot_composite(self, snapshot, tex, rect):
        """Result-panel preview: the source clipped to the union of selected
        object masks (over the checkerboard), optionally on a solid background."""
        if not self._clip_masks:
            return
        if self._clip_bg is not None:
            snapshot.push_mask(Gsk.MaskMode.LUMINANCE)
            for m in self._clip_masks:
                snapshot.append_texture(m, rect)
            snapshot.pop()
            snapshot.append_color(self._clip_bg, rect)
            snapshot.pop()
        snapshot.push_mask(Gsk.MaskMode.LUMINANCE)
        for m in self._clip_masks:
            snapshot.append_texture(m, rect)            # union mask
        snapshot.pop()
        snapshot.append_scaled_texture(tex, Gsk.ScalingFilter.TRILINEAR, rect)
        snapshot.pop()

    def _changed(self):
        if self._on_change:
            self._on_change()
