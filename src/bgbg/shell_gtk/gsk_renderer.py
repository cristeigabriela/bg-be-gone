"""DisplayList -> Gtk.Snapshot. The desktop render backend.

Deliberately dumb: it knows nothing about objects, hovers or animations — it just
replays ops. All the thinking happened in engine/render/builder.py, which is why
a Canvas2D backend can be written against the same list.

`resolve(handle) -> Gdk.Texture` is supplied by the shell, which owns the
textures. The engine only ever passes handles around.
"""
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gsk", "4.0")
gi.require_version("Graphene", "1.0")
from gi.repository import Gdk, Gsk, Graphene  # noqa: E402

from engine.render import ops as O  # noqa: E402

# ALPHA, not LUMINANCE. Mask textures carry coverage in their alpha channel (see
# segmentation._save_soft), so stacking several inside one mask node alpha-
# composites them into a real UNION. Keyed off luminance they were opaque, and
# each layer replaced the one before it.
_MASK_MODE = {
    "keep": Gsk.MaskMode.ALPHA,
    "cut": Gsk.MaskMode.INVERTED_ALPHA,
}


def _rgba(c):
    x = Gdk.RGBA()
    x.red, x.green, x.blue, x.alpha = c[0], c[1], c[2], c[3]
    return x


def _rect(r):
    return Graphene.Rect().init(r[0], r[1], r[2], r[3])


def _point(x, y):
    return Graphene.Point().init(x, y)


def _apply_transform(snapshot, steps):
    for st in steps:
        kind = st[0]
        if kind == "translate":
            snapshot.translate(_point(st[1], st[2]))
        elif kind == "scale":
            snapshot.scale(st[1], st[2])
        elif kind == "rotate":
            snapshot.rotate(st[1])


def _build_path(path):
    pb = Gsk.PathBuilder()
    any_ = False
    for poly in path.polys:
        if len(poly) < 2:
            continue
        pb.move_to(poly[0][0], poly[0][1])
        for x, y in poly[1:]:
            pb.line_to(x, y)
        pb.close()
        any_ = True
    for (cx, cy), r in path.circles:
        pb.add_circle(_point(cx, cy), r)
        any_ = True
    return pb.to_path() if any_ else None


def _checker(snapshot, op):
    """One op here, but still a node per cell on the GSK side — kept identical to
    the original so the render goldens do not move. Step 5 replaces it with a
    repeat node."""
    x, y, w, h = op.rect
    snapshot.append_color(_rgba(op.light), _rect((x, y, w, h)))
    cell = op.cell
    cols = int(w // cell) + 1
    rows = int(h // cell) + 1
    dark = _rgba(op.dark)
    for j in range(rows):
        for i in range(cols):
            if (i + j) & 1:
                snapshot.append_color(
                    dark, _rect((x + i * cell, y + j * cell, cell, cell)))


def _draw_image(snapshot, op, resolve):
    tex = resolve(op.image)
    if tex is None:
        return
    if op.filter == "nearest":
        snapshot.append_texture(tex, _rect(op.rect))
    else:
        snapshot.append_scaled_texture(
            tex, Gsk.ScalingFilter.TRILINEAR, _rect(op.rect))


def _gradient(snapshot, op):
    stops = []
    for off, col in op.stops:
        s = Gsk.ColorStop()
        s.offset = off
        s.color = _rgba(col)
        stops.append(s)
    snapshot.append_linear_gradient(
        _rect(op.rect), _point(*op.p0), _point(*op.p1), stops)


def _stroke(snapshot, op):
    path = _build_path(op.path)
    if path is None:
        return
    st = Gsk.Stroke.new(op.width)
    if op.dash:
        st.set_dash(list(op.dash))
        st.set_dash_offset(op.dash_offset)
    snapshot.append_stroke(path, st, _rgba(op.color))


def _push(snapshot, op, resolve):
    # Order matters and is fixed: transform, then opacity, then blur, then mask.
    # The builder nests one effect per node, so this order is never ambiguous.
    snapshot.save()
    if op.transform:
        _apply_transform(snapshot, op.transform)

    pops = 0
    if op.opacity != 1.0:
        snapshot.push_opacity(op.opacity)
        pops += 1
    if op.blur_sigma:
        snapshot.push_blur(op.blur_sigma * 2.0)   # GSK takes a radius, not sigma
        pops += 1
    if op.mask is not None:
        snapshot.push_mask(_MASK_MODE[op.mask.mode])
        r = _rect(op.mask.rect)
        for h in op.mask.layers:                  # the mask child comes first
            tex = resolve(h)
            if tex is not None:
                snapshot.append_texture(tex, r)
        snapshot.pop()                            # end the mask child
        pops += 1                                 # the content's pop closes it

    for child in op.children:
        render_op(snapshot, child, resolve)

    for _ in range(pops):
        snapshot.pop()
    snapshot.restore()


def render_op(snapshot, op, resolve):
    if isinstance(op, O.Push):
        _push(snapshot, op, resolve)
    elif isinstance(op, O.FillRect):
        snapshot.append_color(_rgba(op.color), _rect(op.rect))
    elif isinstance(op, O.Checker):
        _checker(snapshot, op)
    elif isinstance(op, O.DrawImage):
        _draw_image(snapshot, op, resolve)
    elif isinstance(op, O.LinearGradient):
        _gradient(snapshot, op)
    elif isinstance(op, O.StrokePath):
        _stroke(snapshot, op)
    else:
        raise TypeError("unknown display-list op: %r" % (op,))


def render(snapshot, display_list, resolve):
    for op in display_list.ops:
        render_op(snapshot, op, resolve)
