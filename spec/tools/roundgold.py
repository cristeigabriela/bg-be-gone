#!/usr/bin/env python3
"""Freeze CPython's round(x, 4) answers — the golden for web/core/pyround.ts.

The display-list codec rounds every float, so the two cores must round the same
way first. They very nearly do not: Python rounds half-to-EVEN on the exact
binary value, while JS's toFixed rounds half-AWAY-from-zero and Math.round(x*1e4)
is inexact in a third way. They disagree on ~0.6% of real values (any dyadic
rational that lands on an exact tie, e.g. 0.03125).

So: hand CPython's own answers to the TypeScript side and make it match them.

    python spec/tools/roundgold.py --freeze
"""
import os
import sys
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(os.path.dirname(HERE), "goldens", "round_cases.json")

# the tricky ones by hand, then a deterministic spread
CASES = [0.03125, 0.0312, 0.5, 2.5, -0.03125, 0.12345, 1 / 3, 320.0, 16.0, 0.0,
         1.0, 0.00005, 1e-5, 0.09375, 0.15625, 1.03125, -1.03125, 0.15625 * 3,
         255 / 255.0, 0.3176470588235294, 0.4568359375]


def main():
    rng = random.Random(7)
    vals = list(CASES)
    for _ in range(4000):
        vals.append(rng.uniform(-1000, 1000))
    for _ in range(2000):
        vals.append(rng.uniform(-1, 1))
    for _ in range(2000):
        # dyadic rationals: this is where the exact ties live
        vals.append(rng.randint(-100000, 100000) / (2 ** rng.randint(1, 12)))

    out = [[v, round(v, 4)] for v in vals]
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with open(GOLDEN, "w") as f:
        json.dump(out, f)
    ties = sum(1 for v, _ in out if abs(v * 10000 - int(v * 10000)) == 0.5)
    print("froze %s (%d cases, %d exact ties)" % (GOLDEN, len(out), ties))
    return 0


if __name__ == "__main__":
    sys.exit(main())
