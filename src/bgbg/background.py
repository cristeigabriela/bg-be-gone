"""The blurred background: how strong, and blurred without the subject.

Blur strength is what the user sets; the Gaussian radius follows the
picture's size, so a strength blurs a phone photo and a 4000 px scan alike.
Every path that blurs a background goes through ``blur_behind``: Generate,
Batch, animations, Segment's export, and re-compositing a result.

The range, the default and ``blur_radius`` are standard library only, so
the frontend can read them; ``blur_behind`` imports numpy and Pillow itself.
"""

STRENGTH_MIN = 1        # the least blur; 0 isn't offered, a "blur" that doesn't blur
STRENGTH_MAX = 100
STRENGTH_DEFAULT = 20   # ~10 px on a 512 px photo, ~80 px on a 4000 px one

# Strength is in thousandths of the picture's shorter side: 20 blurs by 2% of it.
_PER_STRENGTH = 1 / 1000

# Below this share of background nearby, the masked average is mostly
# rounding noise, so the plain blur takes over: that's under the subject, unseen.
_MIN_SHARE = 2 / 255


def blur_radius(strength: float, size: tuple) -> float:
    """The Gaussian radius, in pixels, for `strength` on a picture of `size` (w, h).

    Never under 1 px, so the weakest strength on the smallest picture still
    blurs a little.
    """
    shorter = min(size)
    return max(1.0, shorter * strength * _PER_STRENGTH)


def blur_behind(picture, alpha, strength):
    """The picture's background, blurred without the subject's own pixels.

    A plain blur spreads the subject into the background around it: a rim of
    its colour hugging the outline. Here each pixel averages only background,
    each source pixel weighted by how much background it is (1 - alpha):
    blur(picture × (1 - α)) / blur(1 - α), per channel.

    Args:
        picture: the PIL image (any mode; its RGB is used).
        alpha: the subject's alpha, a PIL "L" image of the same size.
        strength: the blur strength (see blur_radius).

    Returns:
        An RGB PIL image. Where no background is near (deep inside the
        subject, which covers it anyway) it's the plain blur instead.
    """
    # Imported here: the frontend reads this module but has no numpy.
    import numpy as np
    from PIL import Image, ImageFilter

    blur = ImageFilter.GaussianBlur(blur_radius(strength, picture.size))
    rgb = picture.convert("RGB")
    weight = alpha.point(lambda a: 255 - a)
    w = np.asarray(weight, np.float32) / 255
    behind = np.asarray(rgb, np.float32) * w[..., None]
    behind = Image.fromarray((behind + 0.5).astype(np.uint8), "RGB")
    total = np.asarray(behind.filter(blur), np.float32)
    share = np.asarray(weight.filter(blur), np.float32)[..., None] / 255
    plain = np.asarray(rgb.filter(blur), np.float32)
    masked = total / np.maximum(share, _MIN_SHARE)
    out = np.where(share >= _MIN_SHARE, masked, plain)
    return Image.fromarray((np.clip(out, 0, 255) + 0.5).astype(np.uint8), "RGB")
