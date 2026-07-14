"""The pixel half of the outputter: one `apply()` per effect.

Needs PIL. The *declaration* of what effects exist, what they take and where they
can be drawn lives in `engine/outputs.py`, which is stdlib-only and therefore
mirrorable; this is only the part that touches pixels.

Before this file, "put a background behind the cutout" was implemented twice —
once in the worker's `process_one` (background removal) and once in
`segmentation.composite_extract` (object extraction) — with the same three
branches and two chances to drift. Both now call `apply()`.
"""
from PIL import Image, ImageFilter


def hex_to_rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def cutout(source, alpha):
    """An RGBA foreground: the source, keyed by an "L" coverage mask."""
    out = source.convert("RGBA")
    out.putalpha(alpha)
    return out


def apply(cut, effect, source=None, **params):
    """Put `effect` behind the RGBA cutout `cut`; return the image to save.

    `source` is the *original*, unmodified image, and is only needed by effects
    that reach for the pixels behind the subject (blur). Transparent returns
    RGBA; the others return RGB, because a background makes alpha meaningless.
    """
    if effect == "transparent":
        return cut

    if effect == "blur":
        if source is None:
            raise ValueError("the blur effect needs the source pixels")
        radius = max(1, int(params.get("strength", 20)))
        # Keep the picture, blur only the background: the sharp cutout over a
        # Gaussian-blurred copy of the original.
        base = source.convert("RGB").filter(
            ImageFilter.GaussianBlur(radius)).convert("RGBA")
        base.alpha_composite(cut)
        return base.convert("RGB")

    if effect == "solid":
        flat = Image.new("RGBA", cut.size,
                         hex_to_rgb(params["color"]) + (255,))
        flat.alpha_composite(cut)
        return flat.convert("RGB")

    raise ValueError("unknown output effect %r" % (effect,))


def apply_bg(cut, bg, source=None, blur=20):
    """The same thing, keyed by the `bg` *setting* ("transparent" / "blur" /
    "#rrggbb") rather than by an effect id — which is what the wire protocol
    carries. Mirrors engine.outputs.resolve().
    """
    if bg == "transparent":
        return apply(cut, "transparent")
    if bg == "blur":
        return apply(cut, "blur", source=source, strength=blur)
    return apply(cut, "solid", color=bg)


def apply_transform(im, rot=0, fh=False, fv=False):
    """Bake a view transform into an image: flip horizontal, flip vertical, then
    rotate clockwise — the same order as ImageView.export_pixbuf.

    Used by the GIF path (a rotated GIF must come out like a rotated still) and by
    the segmentation extract, whose masks live in un-rotated source space: the
    view's rotation is baked in at the end, so what you save is what you saw.
    """
    if fh:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if fv:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    rot %= 4
    if rot == 1:
        im = im.transpose(Image.ROTATE_270)    # 90 clockwise
    elif rot == 2:
        im = im.transpose(Image.ROTATE_180)
    elif rot == 3:
        im = im.transpose(Image.ROTATE_90)     # 90 counter-clockwise
    return im
