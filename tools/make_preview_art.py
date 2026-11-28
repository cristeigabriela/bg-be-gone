"""Draw the Batch page's preview art: a flat avatar bust on a glossy sky.

    python tools/make_preview_art.py

Writes src/bgbg/assets/preview/picture.png and cutout.png. The batch preview
re-composites them in the worker, as Generate keeps a photo and its cut-out
(the names are the ones the worker's recomposite op reads):

- picture.png is the "photo": the bust standing in front of the sky.
- cutout.png is its cut-out: the bust alone, RGBA, with the exact coverage
  picture.png was made from, so a composite has no fringe a model wouldn't give.

The art is original, drawn here from circles, curves and gradients: a grey
default-avatar bust (round head over rounded shoulders) on a mid-2000s
"Frutiger Aero" scene (sky gradient, lens glow, clouds, bubbles, glossy hills).
Seeded and offline, so a run gives the same files every time. Needs numpy and
Pillow, which the worker's environment has.
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "src" / "bgbg" / "assets" / "preview"
WIDTH, HEIGHT = 720, 480
# Drawn this many times larger, then box-filtered down: smooth edges for free.
SUPERSAMPLE = 3
SEED = 2016

# The bust's one flat tone.
BUST_GREY = (154, 166, 178)


def rgb(hex_colour):
    """A "#rrggbb" colour as floats in 0..1."""
    h = hex_colour.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], dtype=np.float32)


class Canvas:
    """An RGB float image at supersampled size, painted with soft or hard shapes."""

    def __init__(self, width, height, scale):
        self.scale = scale
        self.w, self.h = width * scale, height * scale
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        # Coordinates in final-image pixels, so every shape below reads at 720x480.
        self.x = (xs + 0.5) / scale
        self.y = (ys + 0.5) / scale
        self.px = np.zeros((self.h, self.w, 3), dtype=np.float32)

    def over(self, colour, alpha):
        """Paint `colour` (one colour, or one per pixel) with coverage `alpha`."""
        a = np.clip(alpha, 0.0, 1.0)[..., None]
        self.px *= 1.0 - a
        self.px += np.asarray(colour, dtype=np.float32) * a

    def add(self, colour, amount):
        """Add light: glows and flares brighten what is under them."""
        self.px += np.asarray(colour, dtype=np.float32) * np.clip(amount, 0.0, None)[..., None]

    def blob(self, cx, cy, sx, sy):
        """A soft elliptical Gaussian, 1 at its centre."""
        return np.exp(-(((self.x - cx) / sx) ** 2 + ((self.y - cy) / sy) ** 2) / 2.0)

    def disc(self, cx, cy, r):
        """A hard disc's coverage (anti-aliased by the downsample)."""
        return ((self.x - cx) ** 2 + (self.y - cy) ** 2 <= r * r).astype(np.float32)

    def to_image(self):
        """The canvas at its final size, as 8-bit RGB."""
        full = Image.fromarray((np.clip(self.px, 0.0, 1.0) * 255 + 0.5).astype(np.uint8), "RGB")
        return full.reduce(self.scale)


def paint_sky(c):
    """A deep azure top fading to a pale cyan horizon."""
    t = (c.y / HEIGHT)[..., None]
    top, mid, horizon = rgb("#0b63c9"), rgb("#3fa9ef"), rgb("#d8f3ff")
    upper = top + (mid - top) * np.clip(t / 0.45, 0, 1)
    lower = mid + (horizon - mid) * np.clip((t - 0.45) / 0.35, 0, 1)
    c.px[:] = np.where(t < 0.45, upper, lower)


def paint_glow(c):
    """The sun in the top left: a wide glow, a hot core, and soft rays."""
    sun_x, sun_y = 0.16 * WIDTH, 0.10 * HEIGHT
    c.add(rgb("#ffffff"), 0.55 * c.blob(sun_x, sun_y, 170, 150))
    c.add(rgb("#fffbe8"), 0.9 * c.blob(sun_x, sun_y, 38, 38))
    # Rays: a slow fan of brighter angles, fading away from the sun.
    angle = np.arctan2(c.y - sun_y, c.x - sun_x)
    dist = np.hypot(c.x - sun_x, c.y - sun_y)
    fan = (0.5 + 0.5 * np.cos(angle * 14.0)) ** 6
    c.add(rgb("#ffffff"), 0.16 * fan * np.exp(-dist / 330.0))


def paint_ribbons(c):
    """Two translucent white ribbons sweeping across the sky."""
    for base, amp, phase, width, alpha in ((150, 34, 0.4, 16, 0.22), (205, 26, 1.9, 9, 0.16)):
        centre = base + amp * np.sin(c.x / WIDTH * 2.6 + phase) - 40 * (c.x / WIDTH)
        band = np.exp(-((c.y - centre) / width) ** 2)
        c.over(rgb("#ffffff"), alpha * band)


def paint_clouds(c, rng):
    """Soft cumulus clusters, with a faint blue underside."""
    for cx, cy, size in ((110, 230, 1.0), (300, 120, 0.8), (520, 90, 1.1),
                         (650, 215, 0.9), (420, 250, 0.7)):
        cover = np.zeros((c.h, c.w), dtype=np.float32)
        for _ in range(9):
            dx, dy = rng.normal(0, 46 * size), rng.normal(0, 11 * size)
            sx, sy = rng.uniform(26, 44) * size, rng.uniform(16, 24) * size
            cover = np.maximum(cover, c.blob(cx + dx, cy + dy, sx, sy))
        shade = np.clip((c.y - cy) / (30 * size), 0, 1)[..., None]
        colour = rgb("#ffffff") * (1 - 0.18 * shade) + rgb("#9fd2f2") * 0.18 * shade
        c.over(colour, 0.92 * np.clip(cover * 1.6, 0, 1))


def hill_mask(c, base, amp, centre, spread, tilt):
    """Coverage below a hill's crest, and the crest's height per pixel."""
    crest = base - amp * np.exp(-((c.x - centre) / spread) ** 2) + tilt * (c.x / WIDTH)
    return (c.y >= crest).astype(np.float32), crest


def paint_hills(c):
    """Two glossy green hills, the far one darker, each with a lit crest."""
    for base, amp, centre, spread, tilt, top, bottom in (
            # The far hill, darker and bluer.
            (415, 120, 560, 260, -30, "#2fae4a", "#167a3a"),
            # The near hill, bright and grassy.
            (455, 95, 170, 300, 20, "#7bdc3a", "#239b2a")):
        cover, crest = hill_mask(c, base, amp, centre, spread, tilt)
        depth = np.clip((c.y - crest) / 140.0, 0, 1)[..., None]
        colour = rgb(top) * (1 - depth) + rgb(bottom) * depth
        c.over(colour, cover)
        # Gloss: a white sheen just under the crest, the "wet plastic" look.
        sheen = np.exp(-np.clip(c.y - crest, 0, None) / 16.0) * cover
        c.over(rgb("#ffffff"), 0.45 * sheen)


def paint_bubbles(c, rng):
    """Glassy orbs: a faint tinted body, a bright rim and a sharp highlight."""
    for _ in range(15):
        r = float(rng.uniform(12, 46))
        cx, cy = float(rng.uniform(20, WIDTH - 20)), float(rng.uniform(30, HEIGHT - 60))
        d = np.hypot(c.x - cx, c.y - cy) / r
        inside = (d <= 1.0).astype(np.float32)
        tint = rgb("#e8fbff") if rng.uniform() < 0.6 else rgb("#b9f0ff")
        body = 0.10 + 0.55 * np.clip(d, 0, 1) ** 6
        c.over(tint, body * inside)
        # Light caught on the far side, as a lens would bend it.
        glow = c.blob(cx + 0.35 * r, cy + 0.4 * r, 0.35 * r, 0.3 * r)
        c.add(rgb("#ffffff"), 0.18 * glow * inside)
        # The window highlight, top left.
        spot = c.blob(cx - 0.38 * r, cy - 0.42 * r, 0.26 * r, 0.15 * r)
        c.over(rgb("#ffffff"), 0.9 * np.clip(spot * 1.4, 0, 1) * inside)


def paint_flare(c):
    """Faint lens ghosts along the line from the sun through the middle."""
    sun_x, sun_y = 0.16 * WIDTH, 0.10 * HEIGHT
    for t, r, colour, alpha in ((0.55, 26, "#bff4ff", 0.10), (0.8, 12, "#fff3c4", 0.14),
                                (1.25, 40, "#c9ffd9", 0.07)):
        x = sun_x + (WIDTH * 0.5 - sun_x) * t * 1.6
        y = sun_y + (HEIGHT * 0.5 - sun_y) * t * 1.6
        c.add(rgb(colour), alpha * c.disc(x, y, r))


def bust_coverage(c):
    """The avatar: a round head over rounded shoulders, with a gap at the neck."""
    head_x, head_y, head_r = WIDTH * 0.5, 0.52 * HEIGHT, 62
    head = c.disc(head_x, head_y, head_r)
    shoulders = (((c.x - head_x) / 150.0) ** 2 + ((c.y - 505) / 195.0) ** 2 <= 1.0)
    neck_gap = c.disc(head_x, head_y, head_r + 12) > 0
    body = (shoulders & ~neck_gap).astype(np.float32)
    return np.maximum(head, body)


def coverage_image(c, cover):
    """A supersampled coverage mask as the final-size 8-bit alpha channel."""
    full = Image.fromarray((cover * 255 + 0.5).astype(np.uint8), "L")
    return full.reduce(c.scale)


def main():
    rng = np.random.default_rng(SEED)
    c = Canvas(WIDTH, HEIGHT, SUPERSAMPLE)
    paint_sky(c)
    paint_glow(c)
    paint_ribbons(c)
    paint_clouds(c, rng)
    paint_hills(c)
    paint_bubbles(c, rng)
    paint_flare(c)
    background = c.to_image().convert("RGBA")

    cutout = Image.new("RGBA", (WIDTH, HEIGHT), BUST_GREY + (255,))
    cutout.putalpha(coverage_image(c, bust_coverage(c)))
    # The "photo" is the bust composited with the very alpha its cut-out keeps.
    picture = Image.alpha_composite(background, cutout).convert("RGB")

    OUT.mkdir(parents=True, exist_ok=True)
    cutout.save(OUT / "cutout.png", optimize=True)
    picture.save(OUT / "picture.png", optimize=True)
    print("wrote", OUT / "picture.png", "and", OUT / "cutout.png")


if __name__ == "__main__":
    main()
