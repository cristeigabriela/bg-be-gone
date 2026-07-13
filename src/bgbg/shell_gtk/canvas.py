"""The image canvas: a Gtk.Widget wrapped around an EngineSession.

This widget makes no decisions. It:

  * translates GTK events into `engine.events` and feeds them to the session,
  * applies the `engine.effects` the session hands back,
  * owns the pixels (GdkPixbuf / Gdk.Texture) the engine names by handle,
  * drives GIF playback and the frame clock,
  * replays the display list through the GSK backend.

Everything else — zoom, pan, rotate, flip, coordinates, hit-testing, hover,
dwell, press, selection, animation, and what to draw — lives in `bgbg.engine`,
which is stdlib-only and therefore mirrorable into the browser.
"""
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, Gio, GLib  # noqa: E402

from engine import effects as fx  # noqa: E402
from engine import events as ev  # noqa: E402
from engine.session import EngineSession  # noqa: E402
from .gsk_renderer import render as gsk_render  # noqa: E402
from .textures import (  # noqa: E402
    TextureStore, load_texture, load_pixel_map, SRC, POINT, CLIP,
)


class ImageView(Gtk.Widget):
    def __init__(self, on_change=None):
        super().__init__()
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_focusable(True)
        self.set_size_request(160, 160)

        # The engine owns all the state; the widget owns the pixels and the clock.
        self.session = EngineSession(clock=self._now)
        self.textures = TextureStore()

        self.pixbuf = None
        self._anim = None        # GdkPixbuf.PixbufAnimation for animated GIFs
        self._anim_iter = None
        self._anim_timer = 0     # GLib timeout id driving frame advance
        self._anim_tick = None   # add_tick_callback handle

        self._on_change = on_change
        self.on_paint = None     # optional callback(zoom_percent:int)
        # fired on a click while a seg mode is active:
        #   everything: (ix, iy, object_id, "toggle")
        #   point:      (ix, iy, label 1/0, "point")
        self.on_seg_click = None
        # fired as the hovered object stack changes: (depth, drilled, wx, wy)
        self.on_seg_hover = None

        self._cursors = {}

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

    # ---------- the engine seam ----------
    def _now(self):
        """Animation clock, in microseconds.

        GdkFrameClock.get_frame_time() is on the same monotonic timebase as
        GLib.get_monotonic_time(), so falling back to it keeps timestamps
        comparable when we have no frame clock yet (unrealized). Returning 0
        there instead would stamp an animation's epoch at 0, and the moment a
        real clock arrived the animation would be instantly "expired".
        """
        fc = self.get_frame_clock()
        return fc.get_frame_time() if fc is not None else GLib.get_monotonic_time()

    def _cursor(self, name):
        if name is None:
            return None
        if name not in self._cursors:
            self._cursors[name] = Gdk.Cursor.new_from_name(name, None)
        return self._cursors[name]

    def _apply(self, effects):
        """Do what the engine asked for. This is the only place GTK is driven by
        engine state — everything upstream of here is pure data."""
        for e in effects:
            t = type(e)
            if t is fx.Redraw:
                self.queue_draw()
            elif t is fx.RequestTick:
                self._start_anim()
            elif t is fx.StopTick:
                self._stop_anim()
            elif t is fx.SetCursor:
                self.set_cursor(self._cursor(e.name))
            elif t is fx.GrabFocus:
                self.grab_focus()
            elif t is fx.ViewChanged:
                if self._on_change:
                    self._on_change()
            elif t is fx.ContextMenu:
                self._popup(e.x, e.y)
            elif t is fx.HoverChanged:
                if self.on_seg_hover:
                    self.on_seg_hover(e.depth, e.drilled, e.x, e.y)
            elif t is fx.SegClick:
                if self.on_seg_click:
                    self.on_seg_click(e.ix, e.iy, e.value, e.kind)
            # SelectionChanged is deliberately not forwarded: the app reads the
            # selection back from get_seg_selection() inside on_seg_click, and
            # forwarding both would update the sidebar twice per click.

    def _run(self, *_):
        """Apply whatever the last session call queued up."""
        self._apply(self.session.drain_effects())

    def _feed(self, event):
        self._sync()
        self._apply(self.session.feed(event))

    def _sync(self):
        """Feed the engine the one thing only the widget knows: its size."""
        self.session.resize(self.get_width(), self.get_height())
        return self.session.pane

    # ---------- view state (delegated to the engine's Pane) ----------
    pane = property(lambda s: s.session.pane)
    anim = property(lambda s: s.session.anim)
    maps = property(lambda s: s.session.objects.maps)

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
            pixbuf = self._anim_iter.get_pixbuf()
        else:
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
            except GLib.Error:
                return False
        self._set_source(pixbuf, keep_transform=keep_transform)
        if self._anim is not None:
            self._schedule_frame()
        return True

    def set_pixbuf(self, pixbuf):
        self._stop_animation()
        self._set_source(pixbuf)

    def _set_source(self, pixbuf, keep_transform=False):
        self.pixbuf = pixbuf
        self.textures.source = None        # re-uploaded lazily on the next frame
        self.session.set_image(SRC, (pixbuf.get_width(), pixbuf.get_height()),
                               keep_transform=keep_transform)
        self._run()

    def clear(self):
        self._stop_animation()
        self.pixbuf = None
        self.textures.source = None
        self.textures.clip = []
        self.session.clear_image()
        self._run()

    def has_image(self):
        return self.pixbuf is not None

    def reset_view(self, *_):
        self.session.reset_view()
        self._run()

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
        return self.pane.is_transformed()

    # ---------- GIF playback ----------
    def _open_animation(self, path):
        """Return a PixbufAnimation if `path` is an animated GIF we should play
        (not on the Segment view, which needs a stable frame), else None."""
        if (self.session.objects.seg_mode is not None
                or not path.lower().endswith(".gif")):
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
            # Swap the frame only — never the view state, or playback would
            # reset the user's zoom 20 times a second.
            self.pixbuf = pb
            self.textures.source = None
            self.queue_draw()
        self._schedule_frame()
        return False                               # one-shot; _schedule re-arms

    # ---------- segmentation overlays ----------
    def set_seg_mode(self, mode):
        self.session.set_seg_mode(mode)
        self._run()

    def set_dwell_ms(self, ms):
        """How long to hover before drilling from the general to the specific
        object (0 drills immediately)."""
        self.session.set_dwell_ms(ms)

    def set_seg_layers(self, objects, labelmap_path, general_path=None,
                       depth_path=None):
        """Decode the per-object masks + the three lookup maps, hand the engine
        the handles, and let it play the reveal."""
        self.textures.clear_seg()
        loaded = []
        for o in objects:
            tex = load_texture(o["mask"])
            if tex is None:                  # undecodable mask: drop the object
                continue
            self.textures.masks[o["id"]] = tex
            loaded.append(dict(o, handle=o["id"]))
        self.session.load_objects(
            loaded,
            label=load_pixel_map(labelmap_path),
            general=load_pixel_map(general_path),
            depth=load_pixel_map(depth_path))
        self._run()

    def set_seg_selection(self, ids):
        self.session.set_selection(ids)
        self._run()

    def toggle_seg(self, oid):
        self.session.toggle_object(oid)
        self._run()

    def get_seg_selection(self):
        return self.session.selection()

    def set_point_mask(self, mask_path, contour=None):
        self.textures.point = load_texture(mask_path)
        self.session.set_point_mask(
            POINT if self.textures.point is not None else None, contour)
        self._run()

    def seg_texture(self, oid):
        return self.textures.masks.get(oid)

    def point_texture(self):
        return self.textures.point

    def clear_seg(self, keep_mode=False):
        self.textures.clear_seg()
        self.session.clear_seg(keep_mode=keep_mode)
        self._run()

    def has_seg(self):
        return self.session.has_seg()

    def set_scanning(self, on):
        """Show/hide the animated scan shimmer over the source (while an
        everything-pass is running)."""
        self.session.set_scanning(on)
        self._run()

    # ---------- composite / clip preview (result panel) ----------
    def set_composite(self, pixbuf, masks, bg=None):
        self._stop_animation()
        self.pixbuf = pixbuf
        self.textures.source = None
        self._set_clip(masks)
        self.session.set_composite(
            SRC, (pixbuf.get_width(), pixbuf.get_height()),
            self._clip_handles(), bg)
        self._run()

    def update_composite(self, masks, bg=None):
        self._set_clip(masks)
        self.session.update_composite(self._clip_handles(), bg)
        self._run()

    def _set_clip(self, masks):
        self.textures.clip = list(masks)

    def _clip_handles(self):
        return tuple((CLIP, i) for i in range(len(self.textures.clip)))

    # ---------- coordinates / hit-testing ----------
    def widget_to_image(self, px, py):
        """Widget px -> image px (or None). See engine.pane.Pane."""
        return self._sync().view_to_image(px, py)

    def image_to_widget(self, ix, iy):
        """Image px -> widget px (or None). The exact inverse."""
        return self._sync().image_to_view(ix, iy)

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

    def cursor_info(self):
        """What is under the cursor right now (engine.session.CursorInfo)."""
        self._sync()
        return self.session.cursor_info()

    # ---------- animation clock ----------
    def _start_anim(self):
        if self._anim_tick is None and self.get_frame_clock() is not None:
            self.session.anim.t0 = None
            self._anim_tick = self.add_tick_callback(self._on_tick)

    def _stop_anim(self):
        if self._anim_tick is not None:
            self.remove_tick_callback(self._anim_tick)
            self._anim_tick = None

    def _on_map(self, *_):
        self._resume_animation()          # GIF playback
        # Re-arm the overlay tick. A clock can be started (a reveal, say) while
        # we are still unrealized, when add_tick_callback silently does nothing —
        # so the animation would never play. Now that we have a frame clock,
        # start it if anything is actually pending.
        if self.session.needs_tick():
            self._start_anim()

    def _on_tick(self, widget, clock):
        tick = self.session.tick(clock.get_frame_time())
        self._apply(self.session.drain_effects())
        # Re-ask the engine rather than trusting the pre-effect snapshot: the
        # tick itself can start an animation (the dwell drill begins a morph).
        if tick.animating or self.session.needs_tick():
            return GLib.SOURCE_CONTINUE
        # nothing left to animate — retire the tick so we don't wake every frame
        # while idle (a hover/press/select/scan re-arms it via _start_anim).
        self._anim_tick = None
        return GLib.SOURCE_REMOVE

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

    def _popup(self, x, y):
        rect = Gdk.Rectangle()
        rect.x, rect.y, rect.width, rect.height = int(x), int(y), 1, 1
        self.popover.set_pointing_to(rect)
        self.popover.popup()

    # ---------- transforms ----------
    def _rotate(self, delta):
        self.session.rotate(delta)
        self._run()

    def _flip(self, horizontal):
        self.session.flip(horizontal)
        self._run()

    def _actual_size(self, *_):
        self._sync()
        self.session.actual_size()
        self._run()

    # ---------- GTK events -> engine events ----------
    @staticmethod
    def _ctrl(gesture):
        return bool(gesture.get_current_event_state()
                    & Gdk.ModifierType.CONTROL_MASK)

    def _on_motion(self, ctrl, x, y):
        self._feed(ev.PointerMove(x, y))

    def _on_enter(self, ctrl, x, y):
        self._feed(ev.PointerEnter(x, y))

    def _on_leave(self, ctrl):
        self._feed(ev.PointerLeave())

    def _on_scroll(self, ctrl, dx, dy):
        if self.pixbuf is None:
            return False
        self._feed(ev.Scroll(dy))
        return True

    def _on_key_pressed(self, ctrl, keyval, keycode, state):
        if keyval != Gdk.KEY_space:
            return False
        # Consume only the first press, so a focused button isn't triggered.
        # Key *repeats* while Space is held propagate, exactly as before the
        # extraction — swallowing them too would be a behaviour change, and this
        # step is meant to move code, not alter it.
        held = self.session.interaction.space_down
        self._feed(ev.KeyDown(ev.SPACE))
        return not held

    def _on_key_released(self, ctrl, keyval, keycode, state):
        if keyval != Gdk.KEY_space:
            return False
        self._feed(ev.KeyUp(ev.SPACE))
        return True

    def _on_drag_begin(self, gesture, x, y):
        self._feed(ev.DragBegin(x, y))

    def _on_drag_update(self, gesture, ox, oy):
        self._feed(ev.DragUpdate(ox, oy))

    def _on_right_click(self, gesture, n_press, x, y):
        self._feed(ev.PointerDown(x, y, button=ev.SECONDARY, n_press=n_press))

    def _on_primary_pressed(self, gesture, n_press, x, y):
        self._feed(ev.PointerDown(x, y, button=ev.PRIMARY, n_press=n_press,
                                  ctrl=self._ctrl(gesture)))

    def _on_primary_released(self, gesture, n_press, x, y):
        self._feed(ev.PointerUp(x, y, button=ev.PRIMARY,
                                ctrl=self._ctrl(gesture)))

    # ---------- rendering ----------
    # The widget draws nothing by hand. The engine builds a display list (what to
    # draw); gsk_renderer replays it (how). The same list is what a Canvas2D
    # backend in the browser will consume.
    def _get_texture(self):
        if self.textures.source is None and self.pixbuf is not None:
            self.textures.source = Gdk.Texture.new_for_pixbuf(self.pixbuf)
        return self.textures.source

    def do_snapshot(self, snapshot):
        self._sync()
        self._get_texture()                # lazy upload of the current frame
        dl = self.session.display_list()
        gsk_render(snapshot, dl, self.textures.resolve)
        if self.on_paint and len(dl.ops) > 1:      # >1 == the image was drawn
            self.on_paint(int(round(dl.scale * 100)))
