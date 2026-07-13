"""The segmentation overlay model: objects, selection, hover focus. Stdlib only.

What the canvas shows on top of the image — the per-object masks, their colours
and contours, which are selected, which is focused, and the outline tween between
two focused objects.

The engine never holds a texture. Each object carries an opaque `handle` that the
shell minted when it decoded the mask (a Gdk.Texture on the desktop, an
ImageBitmap on the web); the engine only ever passes it back in a display list.
The handle is the object's id, which is what makes the display-list goldens
readable.

The parallel dicts (masks/colors/polys/...) are deliberate rather than a list of
object records: `scene()` runs every frame and this way it only rebinds
references, instead of rebuilding up to 256 objects at 60fps.
"""
from .anim import MORPH_N
from .color import parse_color
from .geometry import polygon_area_abs, resample_closed, align_ring
from .hittest import HitMaps
from .render.scene import Scene, Morph, POINT_COLOR


class ObjectStore:
    def __init__(self):
        self.seg_mode = None          # None | "everything" | "point"

        self.masks = {}               # oid -> handle   (insertion order = paint order)
        self.colors = {}              # oid -> (r,g,b,a)
        self.polys = {}               # oid -> [poly]   all contours
        self.sec_polys = {}           # oid -> [poly]   the non-largest contours
        self.centroids = {}           # oid -> (cx, cy) image px
        self.radius = {}              # oid -> ripple reach, image px
        self.rings = {}               # oid -> [N (x,y)] canonical outline ring

        self.selected = set()
        self.hover_id = 0             # the focused object (glow + outline)
        self.hover_gen = 0            # the general object under the cursor
        self.hover_spec = 0           # the specific one
        self.hover_depth = 0          # how many overlap here

        self.point_mask = None        # handle, point mode
        self.point_polys = ()

        self.morph = None             # Morph | None — an in-flight outline tween
        self.maps = HitMaps()

        # result-panel cutout preview: the source clipped to a union of masks
        self.clip_active = False
        self.clip_masks = ()
        self.clip_bg = None

    # ---------- loading ----------
    def load(self, objects):
        """Replace the objects. Each is `{id, color, contour, bbox, handle}` —
        the shell has already decoded the mask and minted `handle`, dropping any
        object whose mask failed to load.
        """
        self.clear_objects()
        for o in objects:
            oid = o["id"]
            self.masks[oid] = o.get("handle", oid)
            self.colors[oid] = parse_color(o["color"])
            contours = [[tuple(p) for p in poly] for poly in (o.get("contour") or [])]
            self.polys[oid] = contours
            bx, by, bw, bh = o.get("bbox", (0, 0, 0, 0))
            self.centroids[oid] = (bx + bw / 2.0, by + bh / 2.0)
            self.radius[oid] = 0.6 * max(bw, bh, 8)
            # The canonical ring (the largest polygon) is what the focus morph
            # lerps; the rest (holes, detached parts) crossfade during a switch.
            if contours:
                largest = max(contours, key=polygon_area_abs)
                self.rings[oid] = align_ring(resample_closed(largest, MORPH_N))
                self.sec_polys[oid] = [p for p in contours if p is not largest]

    def set_point_mask(self, handle, contour=None):
        self.point_mask = handle
        self.point_polys = [[tuple(p) for p in poly] for poly in (contour or [])]

    def set_composite(self, masks, bg=None):
        self.clip_active = True
        self.clip_masks = tuple(masks)
        self.clip_bg = bg

    def clear_composite(self):
        self.clip_active = False
        self.clip_masks = ()
        self.clip_bg = None

    def clear_objects(self):
        for d in (self.masks, self.colors, self.polys, self.sec_polys,
                  self.centroids, self.radius, self.rings):
            d.clear()

    def clear(self, keep_mode=False):
        self.clear_objects()
        self.selected.clear()
        self.hover_id = self.hover_gen = self.hover_spec = self.hover_depth = 0
        self.point_mask = None
        self.point_polys = ()
        self.morph = None
        self.maps.clear()
        if not keep_mode:
            self.seg_mode = None

    # ---------- selection ----------
    def selection(self):
        return sorted(self.selected)

    def set_selection(self, ids):
        self.selected = set(ids)

    def toggle(self, oid):
        """Returns True when the object was *added* (so the host can pop it)."""
        if oid in self.selected:
            self.selected.discard(oid)
            return False
        self.selected.add(oid)
        return True

    # ---------- state ----------
    def has_seg(self):
        return bool(self.masks) or self.point_mask is not None

    def active(self):
        """Overlay state that wants the frame clock running but that no clock
        owns — a hovered object, a selection, a point mask."""
        return bool(self.hover_id or self.selected
                    or self.point_mask is not None)

    def build_morph(self, old, new):
        """The tween between two focused objects, or None when there is nothing
        to tween (neither has an outline)."""
        fr = self.rings.get(old)
        to = self.rings.get(new)
        if fr is None and to is None:
            return None
        return Morph(
            from_ring=fr, to_ring=to,
            cf=self.colors.get(old) or self.colors.get(new) or POINT_COLOR,
            ct=self.colors.get(new) or self.colors.get(old) or POINT_COLOR,
            sec_from=self.sec_polys.get(old),
            sec_to=self.sec_polys.get(new),
            cen_from=self.centroids.get(old),
            cen_to=self.centroids.get(new))

    # ---------- the frame ----------
    def scene(self, image, image_size):
        sc = Scene()
        sc.image = image
        sc.image_size = image_size
        sc.seg_mode = self.seg_mode
        sc.masks = self.masks
        sc.colors = self.colors
        sc.polys = self.polys
        sc.sec_polys = self.sec_polys
        sc.centroids = self.centroids
        sc.radius = self.radius
        sc.selected = frozenset(self.selected)
        sc.hover_id = self.hover_id
        sc.hover_gen = self.hover_gen
        sc.hover_spec = self.hover_spec
        sc.hover_depth = self.hover_depth
        sc.point_mask = self.point_mask
        sc.point_polys = self.point_polys
        sc.morph = self.morph
        sc.clip_active = self.clip_active
        sc.clip_masks = self.clip_masks
        sc.clip_bg = self.clip_bg
        return sc
