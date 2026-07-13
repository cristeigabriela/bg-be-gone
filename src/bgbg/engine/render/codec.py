"""DisplayList <-> JSON. Stdlib only.

This is the cross-platform wire format and, more importantly, the anti-drift
spine: the TypeScript core must emit byte-identical JSON for the same fixture, so
a golden corpus of display lists pins every eased curve, dash offset, pulse phase
and morph lerp *before* a single pixel is drawn. A pixel diff tells you something
changed; a display-list diff tells you exactly what.

Floats are rounded so two implementations agree despite last-bit noise.
"""
import json

from .ops import (
    DisplayList, Push, FillRect, Checker, DrawImage, LinearGradient, StrokePath,
    Path, MaskSpec,
)

PRECISION = 4


def _n(x):
    """Round a float so the two cores can agree exactly."""
    r = round(float(x), PRECISION)
    return 0.0 + r          # normalise -0.0


def _nums(seq):
    return [_n(v) for v in seq]


def _handle(h):
    """Image handles are opaque; make them JSON-safe and stable."""
    if isinstance(h, (list, tuple)):
        return list(h)
    return h


def _path(p):
    out = {}
    if p.polys:
        out["polys"] = [[_nums(pt) for pt in poly] for poly in p.polys]
    if p.circles:
        out["circles"] = [[_nums(c), _n(r)] for c, r in p.circles]
    return out


def _mask(m):
    return {"layers": [_handle(h) for h in m.layers],
            "rect": _nums(m.rect), "mode": m.mode}


def encode_op(op):
    if isinstance(op, Push):
        d = {"op": "push"}
        if op.transform:
            d["transform"] = [[s[0]] + _nums(s[1:]) for s in op.transform]
        if op.opacity != 1.0:
            d["opacity"] = _n(op.opacity)
        if op.blur_sigma:
            d["blur_sigma"] = _n(op.blur_sigma)
        if op.mask is not None:
            d["mask"] = _mask(op.mask)
        d["children"] = [encode_op(c) for c in op.children]
        return d
    if isinstance(op, FillRect):
        return {"op": "fill_rect", "rect": _nums(op.rect), "color": _nums(op.color)}
    if isinstance(op, Checker):
        return {"op": "checker", "rect": _nums(op.rect), "cell": _n(op.cell),
                "light": _nums(op.light), "dark": _nums(op.dark)}
    if isinstance(op, DrawImage):
        d = {"op": "draw_image", "image": _handle(op.image),
             "rect": _nums(op.rect), "filter": op.filter}
        if op.tint:
            d["tint"] = _nums(op.tint)
        return d
    if isinstance(op, LinearGradient):
        return {"op": "linear_gradient", "rect": _nums(op.rect),
                "p0": _nums(op.p0), "p1": _nums(op.p1),
                "stops": [[_n(o), _nums(c)] for o, c in op.stops]}
    if isinstance(op, StrokePath):
        d = {"op": "stroke", "path": _path(op.path), "color": _nums(op.color),
             "width": _n(op.width)}
        if op.dash:
            d["dash"] = _nums(op.dash)
            d["dash_offset"] = _n(op.dash_offset)
        return d
    raise TypeError("cannot encode %r" % (op,))


def decode_op(d):
    k = d["op"]
    if k == "push":
        return Push(
            children=[decode_op(c) for c in d.get("children", ())],
            transform=[tuple([s[0]] + list(s[1:])) for s in d.get("transform", ())],
            opacity=d.get("opacity", 1.0),
            blur_sigma=d.get("blur_sigma", 0.0),
            mask=(MaskSpec(d["mask"]["layers"], d["mask"]["rect"],
                           d["mask"]["mode"]) if "mask" in d else None))
    if k == "fill_rect":
        return FillRect(d["rect"], d["color"])
    if k == "checker":
        return Checker(d["rect"], d["cell"], d["light"], d["dark"])
    if k == "draw_image":
        return DrawImage(d["image"], d["rect"], d.get("filter", "smooth"),
                         d.get("tint"))
    if k == "linear_gradient":
        return LinearGradient(d["rect"], d["p0"], d["p1"],
                              [(o, c) for o, c in d["stops"]])
    if k == "stroke":
        p = d.get("path", {})
        return StrokePath(
            Path(polys=p.get("polys", ()),
                 circles=[((c[0], c[1]), r) for c, r in p.get("circles", ())]),
            d["color"], d["width"], d.get("dash", ()), d.get("dash_offset", 0.0))
    raise ValueError("unknown op %r" % k)


def encode(dl):
    return {"version": dl.version,
            "scale": _n(dl.scale),
            "view_size": _nums(dl.view_size),
            "animating": bool(dl.animating),
            "ops": [encode_op(o) for o in dl.ops]}


def decode(d):
    return DisplayList(ops=[decode_op(o) for o in d["ops"]],
                       scale=d["scale"], view_size=d["view_size"],
                       animating=d.get("animating", False),
                       version=d.get("version", 1))


def to_json(dl, indent=1):
    return json.dumps(encode(dl), indent=indent, sort_keys=True)


def from_json(s):
    return decode(json.loads(s))
