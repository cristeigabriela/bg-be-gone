/**
 * The CV kernels — the mirror of compute/maskops.py.
 *
 * Masks in, geometry out: connected components, boundary tracing, RDP
 * simplification, IoU/NMS, stability. No model, no GPU — which is exactly why
 * this half can live in the browser's main bundle while ORT-Web does the rest.
 *
 * Pinned by spec/goldens/cv.json. The traps it pins, all of which are silent:
 *   * components are 8-connected. Four-connectivity splits a diagonal touch into
 *     two objects and the object count quietly changes.
 *   * the boundary walk's start pixel and initial direction fix the winding, and
 *     the marching ants are dashed along it.
 *   * `depth` is a COUNT map, not an id map.
 *
 * A `Mask` is a flat Uint8Array of 0/1, row-major, with width and height.
 */

export interface Mask {
  data: Uint8Array;
  w: number;
  h: number;
}

export function maskAt(m: Mask, x: number, y: number): number {
  if (x < 0 || y < 0 || x >= m.w || y >= m.h) return 0;
  return m.data[y * m.w + x];
}

export function maskArea(m: Mask): number {
  let n = 0;
  for (let i = 0; i < m.data.length; i++) n += m.data[i] ? 1 : 0;
  return n;
}

/** Stability: how little the mask changes when the logit threshold is nudged. */
export function stability(logit: Float32Array, offset = 1.0): number {
  let hi = 0;
  let lo = 0;
  for (let i = 0; i < logit.length; i++) {
    if (logit[i] > offset) hi++;
    if (logit[i] > -offset) lo++;
  }
  return lo === 0 ? 0.0 : hi / lo;
}

/** [x, y, w, h] of the mask's true pixels; w == 0 when empty. */
export function bbox(m: Mask): [number, number, number, number] {
  let x0 = m.w;
  let y0 = m.h;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < m.h; y++) {
    for (let x = 0; x < m.w; x++) {
      if (m.data[y * m.w + x]) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return [0, 0, 0, 0];
  return [x0, y0, x1 - x0 + 1, y1 - y0 + 1];
}

export function iou(a: Mask, b: Mask): number {
  let inter = 0;
  let union = 0;
  for (let i = 0; i < a.data.length; i++) {
    const x = a.data[i];
    const y = b.data[i];
    if (x && y) inter++;
    if (x || y) union++;
  }
  return union === 0 ? 0.0 : inter / union;
}

export function bboxOverlap(
  a: [number, number, number, number],
  b: [number, number, number, number],
): boolean {
  return !(
    a[0] + a[2] <= b[0] ||
    b[0] + b[2] <= a[0] ||
    a[1] + a[3] <= b[1] ||
    b[1] + b[3] <= a[1]
  );
}

/** Greedy IoU-NMS. Records are [score, mask]; survivors come back score-desc. */
export function nms(
  records: [number, Mask][],
  iouThresh = 0.7,
  maxObjects = 256,
): [number, Mask][] {
  // Python's sorted() is STABLE, so equal scores keep their input order. A JS
  // sort is not required to be stable across engines for large arrays — but it
  // is in V8/JSC for all arrays, and we compare on score only, so ties keep
  // input order exactly as Python does.
  const sorted = [...records].sort((r1, r2) => r2[0] - r1[0]);
  const kept: [number, Mask][] = [];
  const keptBb: [number, number, number, number][] = [];

  for (const [score, mask] of sorted) {
    const bb = bbox(mask);
    if (bb[2] === 0) continue;
    let drop = false;
    for (let i = 0; i < kept.length; i++) {
      if (bboxOverlap(bb, keptBb[i]) && iou(mask, kept[i][1]) > iouThresh) {
        drop = true;
        break;
      }
    }
    if (!drop) {
      kept.push([score, mask]);
      keptBb.push(bb);
      if (kept.length >= maxObjects) break;
    }
  }
  return kept;
}

/** Label connected components (iterative BFS). Returns [labels, count]. */
export function label(m: Mask, connectivity = 8): [Int32Array, number] {
  const { w, h } = m;
  const lab = new Int32Array(w * h);
  const nbr =
    connectivity === 8
      ? [
          [-1, -1], [-1, 0], [-1, 1],
          [0, -1], [0, 1],
          [1, -1], [1, 0], [1, 1],
        ]
      : [
          [-1, 0], [1, 0], [0, -1], [0, 1],
        ];

  let cur = 0;
  const queue: number[] = [];
  for (let y0 = 0; y0 < h; y0++) {
    for (let x0 = 0; x0 < w; x0++) {
      const i0 = y0 * w + x0;
      if (!m.data[i0] || lab[i0]) continue;
      cur++;
      lab[i0] = cur;
      queue.length = 0;
      queue.push(i0);
      let head = 0;
      while (head < queue.length) {
        const i = queue[head++];
        const y = (i / w) | 0;
        const x = i - y * w;
        for (const [dy, dx] of nbr) {
          const ny = y + dy;
          const nx = x + dx;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const j = ny * w + nx;
          if (m.data[j] && !lab[j]) {
            lab[j] = cur;
            queue.push(j);
          }
        }
      }
    }
  }
  return [lab, cur];
}

/** The clockwise neighbourhood, as (dy, dx) — the winding depends on this order. */
const CW: [number, number][] = [
  [0, -1], [-1, -1], [-1, 0], [-1, 1],
  [0, 1], [1, 1], [1, 0], [1, -1],
];

function cwIndex(dy: number, dx: number): number {
  for (let i = 0; i < 8; i++) if (CW[i][0] === dy && CW[i][1] === dx) return i;
  return -1;
}

/**
 * Moore-neighbour boundary tracing: the ordered outer contour of the component
 * containing the topmost-then-leftmost pixel.
 *
 * The start pixel and the initial "approached from the West" direction fix the
 * winding, and the marching ants are dashed along it — so both must match.
 */
export function traceBoundary(m: Mask): [number, number][] {
  const W = m.w;
  const H = m.h;
  const pw = W + 2;
  const ph = H + 2;
  const p = new Uint8Array(pw * ph);
  let total = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (m.data[y * W + x]) {
        p[(y + 1) * pw + (x + 1)] = 1;
        total++;
      }
    }
  }
  if (total === 0) return [];

  // topmost, then leftmost (np.lexsort((xs, ys))[0])
  let start: [number, number] | null = null;
  outer: for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      if (p[y * pw + x]) {
        start = [y, x];
        break outer;
      }
    }
  }
  let cur = start!;
  let back: [number, number] = [start![0], start![1] - 1]; // from the West
  const out: [number, number][] = [cur];
  const limit = 8 * total + 16;

  for (let step = 0; step < limit; step++) {
    const bi = cwIndex(back[0] - cur[0], back[1] - cur[1]);
    let nxt: [number, number] | null = null;
    for (let j = 1; j < 9; j++) {
      const [dy, dx] = CW[(bi + j) % 8];
      const y = cur[0] + dy;
      const x = cur[1] + dx;
      if (y < 0 || x < 0 || y >= ph || x >= pw) continue;
      if (p[y * pw + x]) {
        const [pdy, pdx] = CW[(bi + j - 1 + 8) % 8];
        back = [cur[0] + pdy, cur[1] + pdx];
        nxt = [y, x];
        break;
      }
    }
    if (nxt === null || (nxt[0] === start![0] && nxt[1] === start![1])) break;
    cur = nxt;
    out.push(cur);
  }
  return out.map(([y, x]) => [x - 1, y - 1] as [number, number]); // -> (x, y), unpad
}

/**
 * Ramer-Douglas-Peucker — a literal port of maskops._rdp.
 *
 * Three details that a from-memory RDP gets wrong, and that change every contour:
 *   * the distance is to the SEGMENT (the projection parameter is clamped to
 *     [0, 1]), not to the infinite line through the endpoints;
 *   * it is iterative over a keep-mask, so the survivors come back in ORIGINAL
 *     order rather than as a concatenation of recursive halves;
 *   * np.argmax keeps the FIRST maximum on a tie.
 */
export function rdp(pts: [number, number][], eps: number): [number, number][] {
  const n = pts.length;
  if (n < 3) return [...pts];

  const keep = new Uint8Array(n);
  keep[0] = 1;
  keep[n - 1] = 1;

  const stack: [number, number][] = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    if (b <= a + 1) continue;

    const segx = pts[b][0] - pts[a][0];
    const segy = pts[b][1] - pts[a][1];
    const l2 = segx * segx + segy * segy;

    let dmax = -1;
    let mi = -1;
    for (let i = a + 1; i < b; i++) {
      const rx = pts[i][0] - pts[a][0];
      const ry = pts[i][1] - pts[a][1];
      let d: number;
      if (l2 === 0.0) {
        d = Math.hypot(rx, ry);
      } else {
        let t = (rx * segx + ry * segy) / l2;
        t = t < 0 ? 0 : t > 1 ? 1 : t; // clamp to the segment
        const px = pts[a][0] + t * segx;
        const py = pts[a][1] + t * segy;
        d = Math.hypot(pts[i][0] - px, pts[i][1] - py);
      }
      if (d > dmax) {
        // strict > keeps the FIRST maximum, as np.argmax does
        dmax = d;
        mi = i;
      }
    }
    if (mi < 0) continue;
    if (dmax > eps) {
      keep[mi] = 1;
      stack.push([a, mi]);
      stack.push([mi, b]);
    }
  }

  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(pts[i]);
  return out;
}

/** Python's round(): half-to-even. int(round(x)) in contour() uses it. */
function roundHalfEven(x: number): number {
  const fl = Math.floor(x);
  const diff = x - fl;
  if (diff > 0.5) return fl + 1;
  if (diff < 0.5) return fl;
  return fl % 2 === 0 ? fl : fl + 1; // exact tie -> even
}

function subMask(lab: Int32Array, k: number, w: number, h: number): Mask {
  const data = new Uint8Array(w * h);
  for (let i = 0; i < data.length; i++) data[i] = lab[i] === k ? 1 : 0;
  return { data, w, h };
}

function invert(m: Mask): Mask {
  const data = new Uint8Array(m.data.length);
  for (let i = 0; i < data.length; i++) data[i] = m.data[i] ? 0 : 1;
  return { data, w: m.w, h: m.h };
}

/**
 * All boundary polylines of `mask`: the outer contour of every connected
 * component PLUS its holes, so rings and branchy/disjoint objects outline
 * correctly.
 *
 * NOTE: the mask is NOT downscaled here — the Python side reduces resolution
 * above max_dim for speed on big images. Ported faithfully for the sizes the
 * golden covers; a full port must add the NEAREST downscale + `up` rescale.
 */
export function contour(
  m: Mask,
  epsilon = 1.8,
  minArea = 80,
  maxPts = 500,
  maxDim = 480,
): number[][][] {
  if (maskArea(m) < minArea) return [];
  const W = m.w;
  const H = m.h;
  if (Math.max(H, W) > maxDim) {
    throw new Error(
      "contour(): the downscale path is not ported yet — see maskops.py",
    );
  }
  const small = m;
  const sw = W;
  const sh = H;
  const up = W / sw;
  const smallMin = Math.max(8, Math.trunc((minArea * (sw * sh)) / (W * H)));

  const regions: Mask[] = [];
  const [lab, n] = label(small, 8);
  for (let k = 1; k <= n; k++) {
    const comp = subMask(lab, k, sw, sh);
    if (maskArea(comp) >= smallMin) regions.push(comp);
  }

  // holes = background components that do not touch the border
  const [blab, bn] = label(invert(small), 4);
  const border = new Set<number>();
  for (let x = 0; x < sw; x++) {
    border.add(blab[x]);
    border.add(blab[(sh - 1) * sw + x]);
  }
  for (let y = 0; y < sh; y++) {
    border.add(blab[y * sw]);
    border.add(blab[y * sw + (sw - 1)]);
  }
  for (let k = 1; k <= bn; k++) {
    if (border.has(k)) continue;
    const hole = subMask(blab, k, sw, sh);
    if (maskArea(hole) >= Math.max(6, Math.trunc(smallMin / 4))) {
      regions.push(hole);
    }
  }

  const polys: number[][][] = [];
  for (const reg of regions) {
    let poly = traceBoundary(reg);
    if (poly.length < 3) continue;
    poly = poly.map(([x, y]) => [x * up, y * up] as [number, number]);
    poly = rdp(poly, epsilon);
    if (poly.length > maxPts) {
      const step = Math.ceil(poly.length / maxPts);
      poly = poly.filter((_, i) => i % step === 0);
    }
    polys.push(poly.map(([x, y]) => [roundHalfEven(x), roundHalfEven(y)]));
  }
  return polys;
}
