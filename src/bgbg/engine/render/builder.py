"""Build a DisplayList for one frame. Stdlib only — this is the render logic.

Everything the canvas draws is decided here: the checkerboard, the image, the
faint object tints, the dim-outside, the glows, the marching ants, the layered
whole-vs-part highlight, the outline morph, the press ripple, the scan shimmer
and the cutout preview. A backend then just *replays* the list.

This file is the thing that must be mirrored in TypeScript, and the golden
corpus in `spec/` is what proves the mirror is faithful.

Each Push carries a single effect so the nesting order is explicit — a glow is a
blur *outside* a mask *outside* an opacity, and combining them into one node
would make that ambiguous.
"""
from ..geometry import ease_out
from .ops import (
    DisplayList, Push, FillRect, Checker, DrawImage, LinearGradient, StrokePath,
    Path, MaskSpec, translate, scale, rotate, scale_about,
)
from .scene import CELL, CHECK_LIGHT, CHECK_DARK, DIM, POINT_COLOR


# ---------- small composites (the vocabulary the original renderer used) -----
def _tint(mask, color, alpha, rect):
    """Paint `color` at `alpha` wherever the mask covers."""
    return Push(mask=MaskSpec([mask], rect, "keep"), children=[
        Push(opacity=alpha, children=[FillRect(rect, color)])])


def _glow(mask, color, alpha, gsk_radius, rect):
    """A soft halo: a blurred, masked colour fill.

    GSK's push_blur takes a radius; a Gaussian sigma is half of it. We carry the
    sigma because that is what CSS/Canvas2D speak, and the GSK backend doubles it
    back — so the two backends agree by construction rather than by luck.
    """
    return Push(blur_sigma=gsk_radius / 2.0,
                children=[_tint(mask, color, alpha, rect)])


def _dim_outside(masks, rect, alpha):
    """Darken everything the masks do NOT cover."""
    return Push(mask=MaskSpec(masks, rect, "cut"), children=[
        Push(opacity=alpha, children=[FillRect(rect, DIM)])])


def _ants(polys, color, view_scale, ant_phase):
    """Marching ants. Width and dash are divided by the view scale so they stay
    screen-constant as you zoom."""
    if not polys:
        return None
    return StrokePath(
        Path(polys=polys), color,
        width=max(0.6, 1.6 / view_scale),
        dash=(6.0 / view_scale, 4.5 / view_scale),
        dash_offset=-ant_phase / view_scale)


def _with_alpha(color, a):
    return (color[0], color[1], color[2], color[3] * max(0.0, min(1.0, a)))


def _blend(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


# ---------- the pieces -------------------------------------------------------
def _shimmer(scan_phase, rect, iw, ih):
    """A soft diagonal highlight sweeping over the image — the "scanning" cue.
    p is offset so the band enters and leaves rather than wrapping abruptly."""
    p = -0.20 + 1.40 * scan_phase
    band = 0.13

    def stop(off, a):
        return (min(1.0, max(0.0, off)), (1.0, 1.0, 1.0, a))

    stops = [stop(0.0, 0.0), stop(p - band, 0.0), stop(p, 0.18),
             stop(p + band, 0.0), stop(1.0, 0.0)]
    return Push(opacity=0.9, children=[
        LinearGradient(rect, (0, 0), (iw, ih), stops)])


def _composite(sc, rect):
    """Result-panel cutout: the source clipped to the union of selected masks."""
    if not sc.clip_masks:
        return []
    out = []
    if sc.clip_bg is not None:
        out.append(Push(mask=MaskSpec(sc.clip_masks, rect, "keep"),
                        children=[FillRect(rect, sc.clip_bg)]))
    out.append(Push(mask=MaskSpec(sc.clip_masks, rect, "keep"),
                    children=[DrawImage(sc.image, rect)]))
    return out


def _press(sc, anim, now, rect, view_scale):
    """The press "swizzle": press-scale, an intensified glow that decays after
    release, and expanding waves clipped to the object."""
    oid = anim.press_obj
    mask = sc.masks.get(oid)
    p = anim.press(now)
    if mask is None or p is None or not anim.press_pt:
        return []
    col = sc.colors.get(oid, POINT_COLOR)
    iw, ih = sc.image_size
    cx, cy = sc.centroids.get(oid) or (iw / 2.0, ih / 2.0)

    inner = [_glow(mask, col, p.glow, 26.0, rect),
             _tint(mask, col, p.tint, rect)]
    ant_col = col if (p.held or p.fade >= 1.0) else _with_alpha(col, p.fade)
    a = _ants(sc.polys.get(oid), ant_col, view_scale, anim.ant)
    if a is not None:
        inner.append(a)

    # rings, clipped to the object's silhouette
    px, py = anim.press_pt
    maxr = sc.radius.get(oid, 160.0)
    rings = []
    for ph, alpha in anim.waves(now):
        r = 8.0 + ease_out(ph) * maxr
        rings.append(StrokePath(Path(circles=[((px, py), r)]),
                                (1.0, 1.0, 1.0, alpha),
                                width=max(1.0, 3.0 / view_scale)))
    if rings:
        inner.append(Push(mask=MaskSpec([mask], rect, "keep"), children=rings))

    xform = scale_about(p.scale, p.scale, cx, cy) if abs(p.scale - 1.0) > 1e-4 else ()
    return [Push(transform=xform, children=inner)]


def _morph_outline(sc, e, view_scale, ant_phase):
    """The focused object's outline mid-tween: lerp the ring on a switch, bloom
    from the centroid on enter, collapse to it on leave; crossfade the rest."""
    m = sc.morph
    if m is None:
        return []
    a, b = m.from_ring, m.to_ring
    if a is not None and b is not None:                 # switch
        ring = [(a[i][0] + (b[i][0] - a[i][0]) * e,
                 a[i][1] + (b[i][1] - a[i][1]) * e) for i in range(len(a))]
        col, env = _blend(m.cf, m.ct, e), 1.0
    elif b is not None:                                 # enter: bloom
        cx, cy = m.cen_to or b[0]
        ring = [(cx + (p[0] - cx) * e, cy + (p[1] - cy) * e) for p in b]
        col, env = m.ct, e
    elif a is not None:                                 # leave: collapse
        cx, cy = m.cen_from or a[0]
        ring = [(p[0] + (cx - p[0]) * e, p[1] + (cy - p[1]) * e) for p in a]
        col, env = m.cf, 1.0 - e
    else:
        return []

    out = []
    s = _ants([ring], _with_alpha(col, env), view_scale, ant_phase)
    if s is not None:
        out.append(s)
    if m.sec_from:
        s = _ants(m.sec_from, _with_alpha(m.cf, 1.0 - e), view_scale, ant_phase)
        if s is not None:
            out.append(s)
    if m.sec_to:
        s = _ants(m.sec_to, _with_alpha(m.ct, e), view_scale, ant_phase)
        if s is not None:
            out.append(s)
    return out


def _focus_outline(sc, anim, now, view_scale, hover, sel):
    """The focused outline: morphing while the focus changes, else static."""
    e = anim.morph_progress(now)
    if e is not None:
        return _morph_outline(sc, e, view_scale, anim.ant)
    if hover and hover in sc.masks and hover not in sel:
        col = sc.colors.get(hover, POINT_COLOR)
        s = _ants(sc.polys.get(hover), col, view_scale, anim.ant)
        if s is not None:
            return [s]
    return []


def _seg(sc, anim, now, rect, view_scale):
    """The segmentation overlay, in draw order."""
    sel, hover = sc.selected, sc.hover_id
    gen, spec = sc.hover_gen, sc.hover_spec
    # Paint the selection in id order. `sel` is a set, and a set's iteration
    # order is its hash order in Python but its insertion order in JS — so
    # relying on it would make the two cores emit ops in different orders for
    # the same scene. `sel` stays the set, for O(1) membership.
    sel_order = sorted(sel)
    # Over a stack, light up the whole (general) and the part (specific) as
    # distinct colour layers rather than one flat highlight.
    layered = (sc.hover_depth >= 2 and gen and spec and gen != spec
               and gen in sc.masks and spec in sc.masks
               and gen not in sel and spec not in sel)
    pulse = 0.35 + 0.65 * anim.pulse
    skip = {gen, spec} if layered else {hover}
    out = []

    # 1. faint tint on every non-selected, non-focused object
    for oid, mask in sc.masks.items():
        if oid in sel or oid in skip:
            continue
        col = sc.colors.get(oid)
        if col is not None:
            out.append(_tint(mask, col, 0.12, rect))

    # 2. dim everything outside the selection
    if sc.masks and sel:
        out.append(_dim_outside([sc.masks[o] for o in sel_order], rect, 0.55))

    # 3. selected objects — glow + fill + ants, with a tactile "pop"
    iw, ih = sc.image_size
    for oid in sel_order:
        mask, col = sc.masks.get(oid), sc.colors.get(oid)
        if mask is None or col is None:
            continue
        pop = anim.pop_scale(oid, now)
        kids = [_glow(mask, col, 0.22 + 0.20 * anim.pulse, 22.0, rect),
                _tint(mask, col, 0.42, rect)]
        a = _ants(sc.polys.get(oid), col, view_scale, anim.ant)
        if a is not None:
            kids.append(a)
        if pop != 1.0:
            cx, cy = sc.centroids.get(oid) or (iw / 2.0, ih / 2.0)
            out.append(Push(transform=scale_about(pop, pop, cx, cy), children=kids))
        else:
            out.extend(kids)

    # 4. hovered (not selected): the whole-vs-part layers, or a single highlight
    if layered:
        gmask, gcol = sc.masks[gen], sc.colors[gen]
        smask, scol = sc.masks[spec], sc.colors[spec]
        out.append(_dim_outside([gmask], rect, 0.5))
        out.append(_glow(gmask, gcol, 0.14 * pulse, 22.0, rect))
        out.append(_tint(gmask, gcol, 0.18, rect))       # whole, beneath
        out.append(_glow(smask, scol, 0.26 * pulse, 18.0, rect))
        out.append(_tint(smask, scol, 0.40, rect))       # part, on top
    elif hover and hover in sc.masks and hover not in sel:
        col = sc.colors.get(hover, POINT_COLOR)
        out.append(_glow(sc.masks[hover], col, 0.28 * pulse, 20.0, rect))
        out.append(_tint(sc.masks[hover], col, 0.34, rect))

    out.extend(_focus_outline(sc, anim, now, view_scale, hover, sel))

    # 5. point mode — dim outside the object, glow + tint + ants
    if sc.point_mask is not None:
        col = POINT_COLOR
        out.append(_dim_outside([sc.point_mask], rect, 0.55))
        out.append(_glow(sc.point_mask, col, 0.20 + 0.18 * anim.pulse, 20.0, rect))
        out.append(_tint(sc.point_mask, col, 0.30, rect))
        a = _ants(sc.point_polys, col, view_scale, anim.ant)
        if a is not None:
            out.append(a)

    # 6. press-and-hold ripple
    if anim.press_obj in sc.masks and anim.press_pt:
        out.extend(_press(sc, anim, now, rect, view_scale))

    return out


# ---------- the frame --------------------------------------------------------
def build(sc, pane, anim, now):
    """The whole frame, as a DisplayList."""
    vw, vh = pane.view_w, pane.view_h
    ops = [Checker((0, 0, vw, vh), CELL, CHECK_LIGHT, CHECK_DARK)]

    ew, eh = pane.effective_size()
    if sc.image is None or not ew or not eh or vw <= 0 or vh <= 0:
        return DisplayList(ops, scale=1.0, view_size=(vw, vh))

    s = pane.scale()
    if not s:
        return DisplayList(ops, scale=1.0, view_size=(vw, vh))

    iw, ih = sc.image_size
    rect = (0, 0, iw, ih)

    xform = [translate(vw / 2 + pane.ox, vh / 2 + pane.oy), scale(s, s)]
    if pane.rot:
        xform.append(rotate(pane.rot * 90))
    xform.append(scale(-1 if pane.fh else 1, -1 if pane.fv else 1))
    xform.append(translate(-iw / 2, -ih / 2))

    inner = []
    if sc.clip_active:
        inner.extend(_composite(sc, rect))
    else:
        inner.append(DrawImage(sc.image, rect, "smooth"))
        if anim.scanning:
            inner.append(_shimmer(anim.scan_phase, rect, iw, ih))
        if sc.seg_mode:
            seg = _seg(sc, anim, now, rect, s)
            if anim.reveal < 1.0:            # fade + gentle scale-in of overlays
                z = 0.97 + 0.03 * anim.reveal
                inner.append(Push(opacity=max(0.0, anim.reveal), children=[
                    Push(transform=scale_about(z, z, iw / 2, ih / 2),
                         children=seg)]))
            else:
                inner.extend(seg)

    ops.append(Push(transform=xform, children=inner))
    return DisplayList(ops, scale=s, view_size=(vw, vh),
                       animating=anim.needs_tick())
