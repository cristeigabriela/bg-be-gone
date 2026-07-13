"""The display list: a serialisable description of one frame. Stdlib only.

This is the render protocol — the contract between the engine (which decides
*what* to draw) and a backend (which knows *how*). There are two backends:
GSK on the desktop and Canvas2D in the browser, and every op below maps 1:1 onto
both. Nothing here knows about either.

Coordinates are image pixels inside a `Push(transform=...)`, and view pixels
outside it. Stroke widths and dash lengths are *already divided by the view
scale* by the builder, so they stay screen-constant under zoom — both GSK's
Gsk.Stroke and Canvas2D's setLineDash take them in user space, so the same
numbers work unchanged on both.

Transforms are carried as a list of primitive steps (translate / scale / rotate)
rather than a collapsed matrix, so a backend replays exactly the operations the
original renderer performed and cannot drift by a float.
"""

# ---- transform steps -------------------------------------------------------
# ("translate", x, y) | ("scale", sx, sy) | ("rotate", degrees_clockwise)


def translate(x, y):
    return ("translate", float(x), float(y))


def scale(sx, sy=None):
    return ("scale", float(sx), float(sx if sy is None else sy))


def rotate(deg):
    return ("rotate", float(deg))


def scale_about(sx, sy, cx, cy):
    """Scale around a point — the select "pop" and the press spring."""
    return (translate(cx, cy), scale(sx, sy), translate(-cx, -cy))


class Path:
    """Polylines and circles. No curves — the renderer never needed any."""

    __slots__ = ("polys", "circles")

    def __init__(self, polys=(), circles=()):
        self.polys = tuple(tuple(tuple(p) for p in poly) for poly in polys)
        self.circles = tuple(circles)      # ((cx, cy), r)

    def __bool__(self):
        return bool(self.polys or self.circles)


class MaskSpec:
    """Clip the children to (or away from) a set of coverage layers.

    `mode="keep"` shows the children only inside the mask; `mode="cut"` only
    outside it. GSK: push_mask(LUMINANCE / INVERTED_LUMINANCE). Canvas2D: an
    offscreen canvas composited with destination-in / destination-out.

    NOTE: `layers` are drawn one over another. Today's mask textures are opaque
    (L PNGs), so a later layer *replaces* an earlier one rather than unioning
    with it — which is the multi-select dim bug the goldens currently freeze. The
    op is shaped to express a real union; step 5 changes the textures so it is.
    """

    __slots__ = ("layers", "rect", "mode")

    def __init__(self, layers, rect, mode="keep"):
        self.layers = tuple(layers)        # image handles
        self.rect = tuple(rect)
        self.mode = mode


class Push:
    """The only nesting construct: a group with an optional transform, opacity,
    blur and mask. Applied in that order, outermost first."""

    __slots__ = ("children", "transform", "opacity", "blur_sigma", "mask")

    def __init__(self, children=(), transform=None, opacity=1.0,
                 blur_sigma=0.0, mask=None):
        self.children = tuple(children)
        self.transform = tuple(transform) if transform else ()
        self.opacity = float(opacity)
        self.blur_sigma = float(blur_sigma)
        self.mask = mask


class FillRect:
    __slots__ = ("rect", "color")

    def __init__(self, rect, color):
        self.rect = tuple(rect)
        self.color = tuple(color)          # (r, g, b, a) 0..1


class Checker:
    """The transparency checkerboard, as ONE op (the original emitted a node per
    cell — ~2000 of them per frame at window size)."""

    __slots__ = ("rect", "cell", "light", "dark")

    def __init__(self, rect, cell, light, dark):
        self.rect = tuple(rect)
        self.cell = float(cell)
        self.light = tuple(light)
        self.dark = tuple(dark)


class DrawImage:
    __slots__ = ("image", "rect", "filter", "tint")

    def __init__(self, image, rect, filter="smooth", tint=None):
        self.image = image                 # an opaque handle the backend resolves
        self.rect = tuple(rect)
        self.filter = filter               # "smooth" (trilinear) | "nearest"
        self.tint = tuple(tint) if tint else None


class LinearGradient:
    __slots__ = ("rect", "p0", "p1", "stops")

    def __init__(self, rect, p0, p1, stops):
        self.rect = tuple(rect)
        self.p0 = tuple(p0)
        self.p1 = tuple(p1)
        self.stops = tuple((float(o), tuple(c)) for o, c in stops)


class StrokePath:
    """`width` and `dash` are already in view-scale-corrected units."""

    __slots__ = ("path", "color", "width", "dash", "dash_offset")

    def __init__(self, path, color, width, dash=(), dash_offset=0.0):
        self.path = path
        self.color = tuple(color)
        self.width = float(width)
        self.dash = tuple(dash)
        self.dash_offset = float(dash_offset)


class DisplayList:
    __slots__ = ("version", "ops", "scale", "view_size", "animating")

    def __init__(self, ops=(), scale=1.0, view_size=(0, 0), animating=False,
                 version=1):
        self.version = version
        self.ops = tuple(ops)
        self.scale = float(scale)          # image px -> view px (the zoom badge)
        self.view_size = tuple(view_size)
        self.animating = bool(animating)
