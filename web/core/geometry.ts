/**
 * Pure geometry and easing — the TypeScript mirror of engine/geometry.py.
 *
 * Every function here is a line-for-line port. The golden corpus in spec/ is what
 * proves it: same fixture in, byte-identical display list out.
 */

export const MIN_ZOOM = 0.05;
export const MAX_ZOOM = 40.0;

export type Vec2 = [number, number];
export type Poly = Vec2[];

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Cubic ease-out. */
export function easeOut(p: number): number {
  return 1.0 - Math.pow(1.0 - p, 3);
}

/** Ease-out with a slight overshoot past 1.0 near the end (spring feel). */
export function easeOutBack(p: number): number {
  const c = 1.70158;
  p -= 1.0;
  return 1.0 + (c + 1.0) * Math.pow(p, 3) + c * Math.pow(p, 2);
}

/** Absolute shoelace area of a closed polygon. */
export function polygonAreaAbs(poly: Poly): number {
  const n = poly.length;
  if (n < 3) return 0.0;
  let s = 0.0;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % n];
    s += x0 * y1 - x1 * y0;
  }
  return Math.abs(s) * 0.5;
}

/** Resample a closed polygon to `n` points spaced uniformly by arc length. */
export function resampleClosed(poly: Poly, n: number): Poly {
  const m = poly.length;
  if (m === 0) return Array.from({ length: n }, () => [0.0, 0.0] as Vec2);
  const pts: Poly = poly.map(([x, y]) => [x, y] as Vec2);
  if (m < 2) return Array.from({ length: n }, () => pts[0]);

  const seglen: number[] = [];
  let total = 0.0;
  for (let i = 0; i < m; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % m];
    const d = Math.hypot(x1 - x0, y1 - y0);
    seglen.push(d);
    total += d;
  }
  if (total <= 0.0) return Array.from({ length: n }, () => pts[0]);

  const out: Poly = [];
  const step = total / n;
  let i = 0;
  let acc = 0.0;
  for (let k = 0; k < n; k++) {
    const target = k * step;
    while (i < m && acc + seglen[i] < target) {
      acc += seglen[i];
      i++;
    }
    if (i >= m) {
      out.push(pts[0]);
      continue;
    }
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % m];
    const f = seglen[i] > 0 ? (target - acc) / seglen[i] : 0.0;
    out.push([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]);
  }
  return out;
}

/**
 * Rotate the ring so index 0 is the topmost-then-leftmost point — a stable
 * starting correspondence so two rings tween without gratuitous twisting.
 *
 * NOTE: Python's `min(range(n), key=...)` keeps the FIRST minimum on a tie.
 * Reproduce that exactly: a different starting index rotates the whole
 * correspondence and every morph frame drifts.
 */
export function alignRing(ring: Poly): Poly {
  if (ring.length === 0) return ring;
  let best = 0;
  for (let i = 1; i < ring.length; i++) {
    const a = ring[i];
    const b = ring[best];
    if (a[1] < b[1] || (a[1] === b[1] && a[0] < b[0])) best = i;
  }
  return ring.slice(best).concat(ring.slice(0, best));
}
