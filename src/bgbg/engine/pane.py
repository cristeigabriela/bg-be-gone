"""The view state of one image pane: zoom, pan, rotate, flip — and the coordinate
maths that goes with them. Stdlib only.

This owns everything the two panes (source / result) need: the image->view
transform, its inverse, cursor->image conversion ("what is at (0,0) of the
image"), cursor-anchored zoom, fit / actual-size, and the rotate/flip that get
baked into an export.

The transform reproduces the renderer exactly. Forward (image px -> view px):

    translate(view_w/2 + ox, view_h/2 + oy)
    scale(s, s)                     s = fit_scale * zoom
    rotate(rot * 90deg, clockwise)
    scale(-1 if fh else 1, -1 if fv else 1)
    translate(-image_w/2, -image_h/2)

so the inverse un-rotates *before* it un-flips. Getting that order wrong is a
silent, subtly-wrong-cursor bug, which is why it is pinned by a differential test
against the original implementation.
"""
from .geometry import MIN_ZOOM, MAX_ZOOM, clamp


class Pane:
    def __init__(self):
        self.zoom = 1.0          # 1.0 == fit to view
        self.ox = 0.0            # pan offset from centre, in view px
        self.oy = 0.0
        self.rot = 0             # 0..3, each +90deg clockwise
        self.fh = False
        self.fv = False
        self.image_w = 0
        self.image_h = 0
        self.view_w = 0
        self.view_h = 0

    # ---------- sizes ----------
    def set_image_size(self, w, h):
        self.image_w, self.image_h = int(w), int(h)

    def set_view_size(self, w, h):
        self.view_w, self.view_h = float(w), float(h)

    def has_image(self):
        return self.image_w > 0 and self.image_h > 0

    def effective_size(self):
        """Image size as the view sees it — axes swap on an odd quarter-turn."""
        if not self.has_image():
            return 0, 0
        if self.rot % 2 == 1:
            return self.image_h, self.image_w
        return self.image_w, self.image_h

    def fit_scale(self):
        ew, eh = self.effective_size()
        if not ew or not eh or not self.view_w or not self.view_h:
            return 0.0
        return min(self.view_w / ew, self.view_h / eh)

    def scale(self):
        """Image px -> view px."""
        return self.fit_scale() * self.zoom

    # ---------- coordinates ----------
    def view_to_image(self, px, py):
        """View px -> image px (origin at the image's top-left), or None if the
        pane isn't laid out yet."""
        if not self.has_image():
            return None
        s = self.scale()
        if not s:
            return None
        ux = (px - (self.view_w / 2 + self.ox)) / s
        uy = (py - (self.view_h / 2 + self.oy)) / s
        for _ in range(self.rot % 4):        # undo each +90deg clockwise step
            ux, uy = uy, -ux
        if self.fh:
            ux = -ux
        if self.fv:
            uy = -uy
        return ux + self.image_w / 2, uy + self.image_h / 2

    def image_to_view(self, ix, iy):
        """Image px -> view px. The exact inverse of `view_to_image`."""
        if not self.has_image():
            return None
        s = self.scale()
        if not s:
            return None
        ux = ix - self.image_w / 2
        uy = iy - self.image_h / 2
        if self.fh:
            ux = -ux
        if self.fv:
            uy = -uy
        for _ in range(self.rot % 4):        # redo each +90deg clockwise step
            ux, uy = -uy, ux
        return ux * s + (self.view_w / 2 + self.ox), uy * s + (self.view_h / 2 + self.oy)

    def contains_image_point(self, ix, iy):
        return 0 <= ix < self.image_w and 0 <= iy < self.image_h

    # ---------- view commands ----------
    def zoom_at(self, factor, cx, cy):
        """Zoom about a view-space anchor (the cursor), keeping it pinned."""
        new = clamp(self.zoom * factor, MIN_ZOOM, MAX_ZOOM)
        f = new / self.zoom if self.zoom else 1.0
        centre_x = self.view_w / 2 + self.ox
        centre_y = self.view_h / 2 + self.oy
        self.ox = (cx + f * (centre_x - cx)) - self.view_w / 2
        self.oy = (cy + f * (centre_y - cy)) - self.view_h / 2
        self.zoom = new

    def reset_view(self):
        self.zoom = 1.0
        self.ox = 0.0
        self.oy = 0.0

    def actual_size(self):
        """Zoom so one image pixel == one view pixel."""
        fit = self.fit_scale()
        self.zoom = 1.0 / fit if fit else 1.0
        self.ox = self.oy = 0.0

    def pan_to(self, ox, oy):
        self.ox, self.oy = float(ox), float(oy)

    # ---------- image transform (exported; zoom/pan never are) ----------
    def rotate(self, delta):
        self.rot = (self.rot + delta) % 4
        self.reset_view()

    def flip(self, horizontal):
        # On an odd quarter-turn the on-screen axes are swapped, so "flip
        # horizontal" must mean what the user sees, not what the buffer stores.
        if self.rot % 2 == 1:
            horizontal = not horizontal
        if horizontal:
            self.fh = not self.fh
        else:
            self.fv = not self.fv

    def is_transformed(self):
        return self.rot != 0 or self.fh or self.fv

    def export_transform(self):
        """What a save/generate must bake in (never zoom or pan)."""
        return {"rot": self.rot, "fh": self.fh, "fv": self.fv}
