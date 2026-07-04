"""Interactive image view: zoom, pan, rotate, flip, reset.

A Gtk.Widget that renders a GdkPixbuf with GSK (no cairo dependency). It keeps
a view transform (zoom/pan) and an image transform (rotate/flip). The image
transform is exportable via :meth:`export_pixbuf` so it can be baked into what
gets processed or saved. Zoom and pan are view-only and never exported.
"""
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gsk", "4.0")
gi.require_version("Graphene", "1.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, Gdk, Gsk, Graphene, GdkPixbuf, Gio, GLib  # noqa: E402

MIN_ZOOM = 0.05
MAX_ZOOM = 40.0
_CELL = 16


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
        self.zoom = 1.0          # 1.0 == fit to widget
        self.ox = 0.0            # pan offset from centre, in widget px
        self.oy = 0.0
        self.rot = 0             # 0..3, each +90deg clockwise
        self.fh = False
        self.fv = False
        self._px = 0.0
        self._py = 0.0
        self._ox0 = 0.0
        self._oy0 = 0.0
        self._on_change = on_change
        self.on_paint = None     # optional callback(zoom_percent:int)

        self._c_light = _rgba(0.20, 0.20, 0.22)
        self._c_dark = _rgba(0.15, 0.15, 0.17)

        motion = Gtk.EventControllerMotion()
        motion.connect("motion", self._on_motion)
        self.add_controller(motion)

        scroll = Gtk.EventControllerScroll(
            flags=Gtk.EventControllerScrollFlags.BOTH_AXES)
        scroll.connect("scroll", self._on_scroll)
        self.add_controller(scroll)

        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        self.add_controller(drag)

        rclick = Gtk.GestureClick(button=Gdk.BUTTON_SECONDARY)
        rclick.connect("pressed", self._on_right_click)
        self.add_controller(rclick)

        pclick = Gtk.GestureClick(button=Gdk.BUTTON_PRIMARY)
        pclick.connect("pressed", self._on_primary_click)
        self.add_controller(pclick)

        self._build_menu()

    # ---------- public API ----------
    def load_file(self, path):
        try:
            self.pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
        except GLib.Error:
            return False
        self._texture = None
        self.rot = 0
        self.fh = self.fv = False
        self.reset_view()
        self._changed()
        return True

    def set_pixbuf(self, pixbuf):
        self.pixbuf = pixbuf
        self._texture = None
        self.rot = 0
        self.fh = self.fv = False
        self.reset_view()
        self._changed()

    def clear(self):
        self.pixbuf = None
        self._texture = None
        self.reset_view()
        self._changed()

    def has_image(self):
        return self.pixbuf is not None

    def reset_view(self, *_):
        self.zoom = 1.0
        self.ox = 0.0
        self.oy = 0.0
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
        rect = Gdk.Rectangle()
        rect.x, rect.y, rect.width, rect.height = int(x), int(y), 1, 1
        self.popover.set_pointing_to(rect)
        self.popover.popup()

    def _on_primary_click(self, gesture, n_press, x, y):
        if n_press == 2:
            self._actual_size() if abs(self.zoom - 1.0) < 1e-3 else self.reset_view()

    # ---------- transforms ----------
    def _rotate(self, delta):
        if self.pixbuf is None:
            return
        self.rot = (self.rot + delta) % 4
        self.reset_view()
        self._changed()

    def _flip(self, horizontal):
        if self.pixbuf is None:
            return
        if self.rot % 2 == 1:
            horizontal = not horizontal
        if horizontal:
            self.fh = not self.fh
        else:
            self.fv = not self.fv
        self.queue_draw()
        self._changed()

    def _actual_size(self, *_):
        if self.pixbuf is None:
            return
        w, h = self.get_width(), self.get_height()
        iw, ih = self._effective_size()
        if iw and ih:
            fit = min(w / iw, h / ih)
            self.zoom = 1.0 / fit if fit else 1.0
        self.ox = self.oy = 0.0
        self.queue_draw()

    # ---------- pointer / gestures ----------
    def _on_motion(self, ctrl, x, y):
        self._px, self._py = x, y

    def _on_scroll(self, ctrl, dx, dy):
        if self.pixbuf is None:
            return False
        factor = 1.0 / 1.1 if dy > 0 else 1.1
        self._zoom_at(factor, self._px, self._py)
        return True

    def _zoom_at(self, factor, cx, cy):
        w, h = self.get_width(), self.get_height()
        new = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        f = new / self.zoom
        centre_x = w / 2 + self.ox
        centre_y = h / 2 + self.oy
        self.ox = (cx + f * (centre_x - cx)) - w / 2
        self.oy = (cy + f * (centre_y - cy)) - h / 2
        self.zoom = new
        self.queue_draw()

    def _on_drag_begin(self, gesture, x, y):
        self._ox0, self._oy0 = self.ox, self.oy

    def _on_drag_update(self, gesture, ox, oy):
        self.ox = self._ox0 + ox
        self.oy = self._oy0 + oy
        self.queue_draw()

    # ---------- rendering ----------
    def _effective_size(self):
        if self.pixbuf is None:
            return 0, 0
        iw, ih = self.pixbuf.get_width(), self.pixbuf.get_height()
        return (ih, iw) if self.rot % 2 == 1 else (iw, ih)

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
        if ew == 0 or eh == 0:
            return
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
        snapshot.append_scaled_texture(
            tex, Gsk.ScalingFilter.TRILINEAR,
            Graphene.Rect().init(0, 0, iw, ih))
        snapshot.restore()

        if self.on_paint:
            self.on_paint(int(round(scale * 100)))

    def _changed(self):
        if self._on_change:
            self._on_change()
