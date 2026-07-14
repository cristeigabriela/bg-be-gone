"""Colour parsing. Stdlib only.

The compute side names object colours as CSS hex (`compute/maskops.py` derives them
from a hue ramp), and the engine speaks (r, g, b, a) floats in 0..1. The shell
used to parse these with `Gdk.RGBA`, which is a GTK dependency the engine cannot
have — and which quantises to **float32**, so it returns 0.31764707 where exact
arithmetic gives 0.31764706.

That difference is ~1e-8: it vanishes at u8 pixel depth and at the display list's
4-decimal precision, so it moves neither golden. The double is the more faithful
value and it is what a JS `Number` will produce, so the two cores agree exactly.
"""


def parse_color(spec):
    """`"#rgb"` / `"#rrggbb"` / `"#rrggbbaa"` -> (r, g, b, a) floats in 0..1.

    Already-parsed tuples pass through, so callers can hand over either.
    """
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)):
        c = tuple(float(v) for v in spec)
        return c if len(c) == 4 else c + (1.0,)

    s = str(spec).strip().lstrip("#")
    if len(s) in (3, 4):                      # #rgb / #rgba shorthand
        s = "".join(ch * 2 for ch in s)
    if len(s) not in (6, 8):
        raise ValueError("cannot parse colour %r" % (spec,))
    v = [int(s[i:i + 2], 16) / 255.0 for i in range(0, len(s), 2)]
    if len(v) == 3:
        v.append(1.0)
    return tuple(v)
