"""Pure geometry and easing. Stdlib only — no gi, no numpy, no PIL.

This is the first slice of the headless engine. Everything here is a plain
function over plain numbers so it can be mirrored 1:1 in the TypeScript core and
pinned by the shared golden corpus (see ``spec/``).
"""
import math

MIN_ZOOM = 0.05
MAX_ZOOM = 40.0


# ---------- scalars ----------
def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out(p):
    """Cubic ease-out."""
    return 1.0 - (1.0 - p) ** 3


def ease_out_back(p):
    """Ease-out with a slight overshoot past 1.0 near the end (spring feel)."""
    c = 1.70158
    p -= 1.0
    return 1.0 + (c + 1.0) * p ** 3 + c * p ** 2


# ---------- polygons ----------
def polygon_area_abs(poly):
    """Absolute shoelace area of a closed polygon (list of (x, y))."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) * 0.5


def resample_closed(poly, n):
    """Resample a closed polygon to `n` points spaced uniformly by arc length."""
    m = len(poly)
    if m == 0:
        return [(0.0, 0.0)] * n
    pts = [(float(x), float(y)) for x, y in poly]
    if m < 2:
        return [pts[0]] * n
    seglen = []
    total = 0.0
    for i in range(m):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % m]
        d = math.hypot(x1 - x0, y1 - y0)
        seglen.append(d)
        total += d
    if total <= 0.0:
        return [pts[0]] * n
    out = []
    step = total / n
    i = 0
    acc = 0.0
    for k in range(n):
        target = k * step
        while i < m and acc + seglen[i] < target:
            acc += seglen[i]
            i += 1
        if i >= m:
            out.append(pts[0])
            continue
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % m]
        f = (target - acc) / seglen[i] if seglen[i] > 0 else 0.0
        out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    return out


def align_ring(ring):
    """Rotate the ring so index 0 is the topmost-then-leftmost point — a stable
    starting correspondence so two rings tween without gratuitous twisting."""
    if not ring:
        return ring
    best = min(range(len(ring)), key=lambda i: (ring[i][1], ring[i][0]))
    return ring[best:] + ring[:best]
