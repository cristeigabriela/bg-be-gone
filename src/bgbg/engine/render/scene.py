"""What the builder needs to know to draw a frame. Stdlib only.

Plain data: image *handles* (never textures), colours as (r,g,b,a) tuples,
contours as lists of polylines in image pixels. No GTK type reaches this far, so
the same Scene can be handed to a Canvas2D backend unchanged.
"""

CELL = 16                                     # checkerboard cell, view px
CHECK_LIGHT = (0.20, 0.20, 0.22, 1.0)
CHECK_DARK = (0.15, 0.15, 0.17, 1.0)
DIM = (0.0, 0.0, 0.0, 1.0)                    # what "dim the rest" paints
POINT_COLOR = (0.20, 0.52, 0.90, 1.0)


class Morph:
    """An in-flight outline tween between two focused objects."""

    __slots__ = ("from_ring", "to_ring", "cf", "ct",
                 "sec_from", "sec_to", "cen_from", "cen_to")

    def __init__(self, from_ring=None, to_ring=None, cf=None, ct=None,
                 sec_from=None, sec_to=None, cen_from=None, cen_to=None):
        self.from_ring = from_ring        # [(x, y)] or None when entering
        self.to_ring = to_ring            # [(x, y)] or None when leaving
        self.cf = cf                      # colour we morph from
        self.ct = ct                      # colour we morph to
        self.sec_from = sec_from          # secondary polys (holes/extra parts)
        self.sec_to = sec_to
        self.cen_from = cen_from          # centroid to collapse toward
        self.cen_to = cen_to              # centroid to bloom from


class Scene:
    def __init__(self):
        self.image = None                 # handle of the source image
        self.image_size = (0, 0)

        self.seg_mode = None              # None | "everything" | "point"
        self.masks = {}                   # oid -> handle
        self.colors = {}                  # oid -> (r,g,b,a)
        self.polys = {}                   # oid -> [poly]  (all contours)
        self.sec_polys = {}               # oid -> [poly]  (non-largest contours)
        self.centroids = {}               # oid -> (cx, cy)
        self.radius = {}                  # oid -> ripple reach, image px

        self.selected = frozenset()
        self.hover_id = 0
        self.hover_gen = 0
        self.hover_spec = 0
        self.hover_depth = 0

        self.point_mask = None            # handle
        self.point_polys = ()

        self.morph = None                 # Morph | None

        # result-panel cutout preview: source clipped to a union of masks
        self.clip_active = False
        self.clip_masks = ()
        self.clip_bg = None               # (r,g,b,a) or None
