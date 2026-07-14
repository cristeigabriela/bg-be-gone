"""The outputter: what to put behind the cutout. Stdlib only.

An output effect is declared once — its id, its label, its parameters, and,
crucially, **where it can be drawn**:

  LOCAL    the engine can express it as display-list ops. Transparent is the
           checkerboard showing through the mask; a solid colour is a FillRect
           under the mask. Both are just ops, so the preview is *instant* and
           costs nothing — no round-trip, no temp file, no debounce.
  COMPUTE  it needs the actual pixels (Gaussian-blurring the background is not
           something a display list can do), so it goes to the compute port.

That distinction is the point of this file. Selecting an object used to fire a
150 ms-debounced job at the worker, which composited a PNG, wrote it to a temp
dir, and handed the path back to be decoded and uploaded — to show a preview the
renderer could have drawn itself. For transparent and solid backgrounds it now
does.

Adding an effect is one `register()` here plus one `apply()` in
`compute/outputs_impl.py`, and both sidebars grow the control because they are
built from the schema.
"""
from .color import parse_color

LOCAL = "local"        # drawable as display-list ops — instant, no compute
COMPUTE = "compute"    # needs the pixels — goes to the compute port

INT = "int"
COLOR = "color"


class ParamSpec:
    __slots__ = ("name", "kind", "minimum", "maximum", "default")

    def __init__(self, name, kind, minimum=None, maximum=None, default=None):
        self.name = name
        self.kind = kind
        self.minimum = minimum
        self.maximum = maximum
        self.default = default


class OutputEffect:
    __slots__ = ("id", "label", "params", "requires", "preview", "compute_id")

    def __init__(self, id, label, params=(), requires=(), preview=LOCAL,
                 compute_id=None):
        self.id = id
        self.label = label
        self.params = list(params)
        self.requires = tuple(requires)
        self.preview = preview
        self.compute_id = compute_id or id

    @property
    def local(self):
        return self.preview == LOCAL

    def __repr__(self):
        return "OutputEffect(%r, %s)" % (self.id, self.preview)


class Registry:
    def __init__(self):
        self._by_id = {}

    def register(self, effect):
        self._by_id[effect.id] = effect
        return effect

    def get(self, eid):
        return self._by_id[eid]

    def all(self):
        return list(self._by_id.values())


OUTPUTS = Registry()

TRANSPARENT = OUTPUTS.register(OutputEffect(
    "transparent", "Transparent", preview=LOCAL))

SOLID = OUTPUTS.register(OutputEffect(
    "solid", "Solid colour",
    params=(ParamSpec("color", COLOR, default="#ffffff"),),
    preview=LOCAL))

BLUR = OUTPUTS.register(OutputEffect(
    "blur", "Blur background",
    params=(ParamSpec("strength", INT, 2, 80, 20),),
    # The one thing a display list cannot do: it needs the source pixels, so it
    # is the only background that still costs a round-trip.
    requires=("source_pixels",), preview=COMPUTE, compute_id="blur"))


class Resolved:
    """A chosen effect plus its parameter values."""

    __slots__ = ("effect", "params")

    def __init__(self, effect, params):
        self.effect = effect
        self.params = params

    @property
    def id(self):
        return self.effect.id

    @property
    def local(self):
        """Can the engine draw this itself, right now, with no compute?"""
        return self.effect.local

    @property
    def fill(self):
        """The colour the display list paints under the mask — None when the
        background is transparent (the checkerboard shows through)."""
        if self.effect is SOLID:
            return parse_color(self.params["color"])
        return None

    def __repr__(self):
        return "Resolved(%s, %r)" % (self.effect.id, self.params)


def resolve(bg, blur=20):
    """The `bg` setting -> an effect. `bg` is "transparent", "blur", or #rrggbb.

    The setting stores a colour *as* the value (that is what the sidebar picks),
    so anything that is not a named effect is a solid colour.
    """
    if bg == "transparent":
        return Resolved(TRANSPARENT, {})
    if bg == "blur":
        return Resolved(BLUR, {"strength": int(blur)})
    return Resolved(SOLID, {"color": bg})
