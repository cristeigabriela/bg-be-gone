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

# Below this share of background nearby there's next to nothing to average,
# so the plain blur takes over: that's under the subject, unseen.
_MIN_SHARE = 1 / 1024

# Box blurs stacked to approximate a Gaussian, as Pillow's own GaussianBlur does.
_BOX_PASSES = 3

# A wide blur keeps no detail finer than a few pixels of its sigma, so it's
# worked out on a copy shrunk to about this many pixels per sigma, then
# scaled back up: 25 times less work for a 4000 px photo's default blur.
_PIXELS_PER_SIGMA = 4


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
    from PIL import Image

    # In floating point throughout, rounded once at the end: Pillow blurs
    # only 8-bit images, and rounding the weighted colours first shifts them.
    sigma = blur_radius(strength, picture.size)
    full = picture.size
    step = max(1, int(sigma / _PIXELS_PER_SIGMA))
    small = (max(1, round(full[0] / step)), max(1, round(full[1] / step)))
    sigma /= full[0] / small[0]

    def down(plane):
        return _resize(np, Image, plane, small, Image.Resampling.BOX)

    rgb = np.asarray(picture.convert("RGB"), np.float32)
    weight = 1 - np.asarray(alpha, np.float32) / 255
    share = _gaussian(down(weight), sigma)
    near = share >= _MIN_SHARE
    channels = []
    for c in range(3):
        plane = rgb[..., c]
        masked = _gaussian(down(plane * weight), sigma) / np.maximum(share, _MIN_SHARE)
        if not near.all():
            masked = np.where(near, masked, _gaussian(down(plane), sigma))
        channels.append(_resize(np, Image, masked, full, Image.Resampling.BILINEAR))
    out = np.stack(channels, axis=-1)
    return Image.fromarray(np.floor(np.clip(out, 0, 255) + 0.5).astype(np.uint8), "RGB")


def _resize(np, Image, plane, size, resample):
    """A float plane resized to `size` (w, h), staying in floating point."""
    if plane.shape[::-1] == tuple(size):
        return plane
    image = Image.fromarray(np.ascontiguousarray(plane, np.float32), "F")
    return np.asarray(image.resize(size, resample), np.float32)


def _gaussian(plane, sigma):
    """`plane` (a 2-D float array) blurred by a Gaussian of `sigma` pixels.

    Three box blurs in each direction, sized to add up to that Gaussian
    (Kutskir's "fastest Gaussian blur"), with the edges extended. Each box is
    a difference of running sums, so the cost doesn't grow with sigma.
    """
    import numpy as np

    for width in _box_widths(sigma, _BOX_PASSES):
        for axis in (0, 1):
            plane = _box(np, plane, (width - 1) // 2, axis)
    return plane


def _box_widths(sigma, passes):
    """Odd box widths whose `passes` stacked box blurs approximate the Gaussian."""
    import math

    ideal = math.sqrt(12 * sigma * sigma / passes + 1)
    lower = int(ideal)
    if lower % 2 == 0:
        lower -= 1
    upper = lower + 2
    n_lower = round((12 * sigma * sigma - passes * lower * lower - 4 * passes * lower
                     - 3 * passes) / (-4 * lower - 4))
    return [lower if i < n_lower else upper for i in range(passes)]


def _box(np, plane, r, axis):
    """The mean of each 2r+1 window along `axis`, edges extended."""
    if r <= 0:
        return plane
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)                 # one extra in front: the running sum's zero
    padded = np.pad(plane, pad, mode="edge")
    sums = np.cumsum(padded, axis=axis, dtype=np.float64)
    n = plane.shape[axis]
    ahead = np.take(sums, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    behind = np.take(sums, np.arange(0, n), axis=axis)
    return ((ahead - behind) / (2 * r + 1)).astype(np.float32)
