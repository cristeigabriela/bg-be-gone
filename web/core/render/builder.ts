/**
 * Build a DisplayList for one frame — the mirror of engine/render/builder.py.
 *
 * This is THE file that has to be faithful: everything the canvas draws is
 * decided here, and spec/goldens/display_list/*.json is the proof. Op order
 * matters as much as op content, which is why the selection is painted in sorted
 * id order (a Python set iterates by hash, a JS Set by insertion — relying on
 * either would make the two cores disagree for the same scene).
 */
import { easeOut, type Poly, type Vec2 } from "../geometry";
import type { RGBA } from "../color";
import { AnimState } from "../anim";
import { Pane } from "../pane";
import {
  DisplayList,
  Push,
  FillRect,
  Checker,
  DrawImage,
  LinearGradient,
  StrokePath,
  Path,
  MaskSpec,
  translate,
  scale,
  rotate,
  scaleAbout,
  type Handle,
  type Op,
  type Rect,
  type Step,
} from "./ops";
import {
  Scene,
  CELL,
  CHECK_LIGHT,
  CHECK_DARK,
  DIM,
  POINT_COLOR,
} from "./scene";

// ---------- small composites ----------
function tint(mask: Handle, color: RGBA, alpha: number, rect: Rect): Push {
  return new Push([new Push([new FillRect(rect, color)], [], alpha)], [], 1.0, 0.0,
    new MaskSpec([mask], rect, "keep"));
}

function glow(
  mask: Handle,
  color: RGBA,
  alpha: number,
  gskRadius: number,
  rect: Rect,
): Push {
  // GSK's push_blur takes a radius; a Gaussian sigma is half of it. We carry the
  // sigma because that is what CSS/Canvas2D speak.
  return new Push([tint(mask, color, alpha, rect)], [], 1.0, gskRadius / 2.0);
}

function dimOutside(masks: Handle[], rect: Rect, alpha: number): Push {
  return new Push([new Push([new FillRect(rect, DIM)], [], alpha)], [], 1.0, 0.0,
    new MaskSpec(masks, rect, "cut"));
}

/** Marching ants: width and dash divided by the view scale to stay screen-constant. */
function ants(
  polys: Poly[] | undefined,
  color: RGBA,
  viewScale: number,
  antPhase: number,
): StrokePath | null {
  if (!polys || polys.length === 0) return null;
  return new StrokePath(
    new Path(polys),
    color,
    Math.max(0.6, 1.6 / viewScale),
    [6.0 / viewScale, 4.5 / viewScale],
    -antPhase / viewScale,
  );
}

function withAlpha(color: RGBA, a: number): RGBA {
  return [color[0], color[1], color[2], color[3] * Math.max(0.0, Math.min(1.0, a))];
}

function blend(a: RGBA, b: RGBA, t: number): RGBA {
  return [0, 1, 2, 3].map((i) => a[i] + (b[i] - a[i]) * t) as RGBA;
}

// ---------- the pieces ----------
function shimmer(scanPhase: number, rect: Rect, iw: number, ih: number): Push {
  const p = -0.2 + 1.4 * scanPhase;
  const band = 0.13;
  const stop = (off: number, a: number): [number, RGBA] => [
    Math.min(1.0, Math.max(0.0, off)),
    [1.0, 1.0, 1.0, a],
  ];
  const stops: [number, RGBA][] = [
    stop(0.0, 0.0),
    stop(p - band, 0.0),
    stop(p, 0.18),
    stop(p + band, 0.0),
    stop(1.0, 0.0),
  ];
  return new Push(
    [new LinearGradient(rect, [0, 0], [iw, ih], stops)],
    [],
    0.9,
  );
}

function composite(sc: Scene, rect: Rect): Op[] {
  if (!sc.clipMasks.length) return [];
  const out: Op[] = [];
  // The background fills the WHOLE rect and the cutout goes on top — it is a
  // background, not a backing for the subject.
  if (sc.clipBg !== null) out.push(new FillRect(rect, sc.clipBg));
  out.push(
    new Push([new DrawImage(sc.image!, rect)], [], 1.0, 0.0,
      new MaskSpec(sc.clipMasks, rect, "keep")),
  );
  return out;
}

function press(
  sc: Scene,
  anim: AnimState,
  now: number,
  rect: Rect,
  viewScale: number,
): Op[] {
  const oid = anim.pressObj;
  const mask = sc.masks.get(oid);
  const p = anim.press(now);
  if (mask === undefined || p === null || !anim.pressPt) return [];

  const col = sc.colors.get(oid) ?? POINT_COLOR;
  const [iw, ih] = sc.imageSize;
  const [cx, cy] = sc.centroids.get(oid) ?? [iw / 2.0, ih / 2.0];

  const inner: Op[] = [
    glow(mask, col, p.glow, 26.0, rect),
    tint(mask, col, p.tint, rect),
  ];
  const antCol = p.held || p.fade >= 1.0 ? col : withAlpha(col, p.fade);
  const a = ants(sc.polys.get(oid), antCol, viewScale, anim.ant);
  if (a !== null) inner.push(a);

  const [px, py] = anim.pressPt;
  const maxr = sc.radius.get(oid) ?? 160.0;
  const rings: Op[] = [];
  for (const [ph, alpha] of anim.waves(now)) {
    const r = 8.0 + easeOut(ph) * maxr;
    rings.push(
      new StrokePath(
        new Path([], [[[px, py], r]]),
        [1.0, 1.0, 1.0, alpha],
        Math.max(1.0, 3.0 / viewScale),
      ),
    );
  }
  if (rings.length) {
    inner.push(new Push(rings, [], 1.0, 0.0, new MaskSpec([mask], rect, "keep")));
  }

  const xform: Step[] =
    Math.abs(p.scale - 1.0) > 1e-4 ? scaleAbout(p.scale, p.scale, cx, cy) : [];
  return [new Push(inner, xform)];
}

function morphOutline(
  sc: Scene,
  e: number,
  viewScale: number,
  antPhase: number,
): Op[] {
  const m = sc.morph;
  if (m === null) return [];
  const a = m.fromRing;
  const b = m.toRing;
  let ring: Poly;
  let col: RGBA;
  let env: number;

  if (a !== null && b !== null) {
    // switch
    ring = a.map(
      (_, i) =>
        [
          a[i][0] + (b[i][0] - a[i][0]) * e,
          a[i][1] + (b[i][1] - a[i][1]) * e,
        ] as Vec2,
    );
    col = blend(m.cf!, m.ct!, e);
    env = 1.0;
  } else if (b !== null) {
    // enter: bloom
    const [cx, cy] = m.cenTo ?? b[0];
    ring = b.map((p) => [cx + (p[0] - cx) * e, cy + (p[1] - cy) * e] as Vec2);
    col = m.ct!;
    env = e;
  } else if (a !== null) {
    // leave: collapse
    const [cx, cy] = m.cenFrom ?? a[0];
    ring = a.map((p) => [p[0] + (cx - p[0]) * e, p[1] + (cy - p[1]) * e] as Vec2);
    col = m.cf!;
    env = 1.0 - e;
  } else {
    return [];
  }

  const out: Op[] = [];
  let s = ants([ring], withAlpha(col, env), viewScale, antPhase);
  if (s !== null) out.push(s);
  if (m.secFrom && m.secFrom.length) {
    s = ants(m.secFrom, withAlpha(m.cf!, 1.0 - e), viewScale, antPhase);
    if (s !== null) out.push(s);
  }
  if (m.secTo && m.secTo.length) {
    s = ants(m.secTo, withAlpha(m.ct!, e), viewScale, antPhase);
    if (s !== null) out.push(s);
  }
  return out;
}

function focusOutline(
  sc: Scene,
  anim: AnimState,
  now: number,
  viewScale: number,
  hover: number,
  sel: Set<number>,
): Op[] {
  const e = anim.morphProgress(now);
  if (e !== null) return morphOutline(sc, e, viewScale, anim.ant);
  if (hover && sc.masks.has(hover) && !sel.has(hover)) {
    const col = sc.colors.get(hover) ?? POINT_COLOR;
    const s = ants(sc.polys.get(hover), col, viewScale, anim.ant);
    if (s !== null) return [s];
  }
  return [];
}

function seg(
  sc: Scene,
  anim: AnimState,
  now: number,
  rect: Rect,
  viewScale: number,
): Op[] {
  const sel = sc.selected;
  const hover = sc.hoverId;
  const gen = sc.hoverGen;
  const spec = sc.hoverSpec;

  // Paint the selection in id order: a Python set iterates in hash order and a
  // JS Set in insertion order, so relying on either would make the two cores
  // emit ops in a different order for the same scene.
  const selOrder = [...sel].sort((a, b) => a - b);

  const layered =
    sc.hoverDepth >= 2 &&
    !!gen &&
    !!spec &&
    gen !== spec &&
    sc.masks.has(gen) &&
    sc.masks.has(spec) &&
    !sel.has(gen) &&
    !sel.has(spec);

  const pulse = 0.35 + 0.65 * anim.pulse;
  const skip = layered ? new Set([gen, spec]) : new Set([hover]);
  const out: Op[] = [];

  // 1. faint tint on every non-selected, non-focused object
  for (const [oid, mask] of sc.masks) {
    if (sel.has(oid) || skip.has(oid)) continue;
    const col = sc.colors.get(oid);
    if (col !== undefined) out.push(tint(mask, col, 0.12, rect));
  }

  // 2. dim everything outside the selection
  if (sc.masks.size && sel.size) {
    out.push(dimOutside(selOrder.map((o) => sc.masks.get(o)!), rect, 0.55));
  }

  // 3. selected objects — glow + fill + ants, with a tactile "pop"
  const [iw, ih] = sc.imageSize;
  for (const oid of selOrder) {
    const mask = sc.masks.get(oid);
    const col = sc.colors.get(oid);
    if (mask === undefined || col === undefined) continue;
    const pop = anim.popScale(oid, now);
    const kids: Op[] = [
      glow(mask, col, 0.22 + 0.2 * anim.pulse, 22.0, rect),
      tint(mask, col, 0.42, rect),
    ];
    const a = ants(sc.polys.get(oid), col, viewScale, anim.ant);
    if (a !== null) kids.push(a);
    if (pop !== 1.0) {
      const [cx, cy] = sc.centroids.get(oid) ?? [iw / 2.0, ih / 2.0];
      out.push(new Push(kids, scaleAbout(pop, pop, cx, cy)));
    } else {
      out.push(...kids);
    }
  }

  // 4. hovered (not selected): the whole-vs-part layers, or a single highlight
  if (layered) {
    const gmask = sc.masks.get(gen)!;
    const gcol = sc.colors.get(gen)!;
    const smask = sc.masks.get(spec)!;
    const scol = sc.colors.get(spec)!;
    out.push(dimOutside([gmask], rect, 0.5));
    out.push(glow(gmask, gcol, 0.14 * pulse, 22.0, rect));
    out.push(tint(gmask, gcol, 0.18, rect)); // whole, beneath
    out.push(glow(smask, scol, 0.26 * pulse, 18.0, rect));
    out.push(tint(smask, scol, 0.4, rect)); // part, on top
  } else if (hover && sc.masks.has(hover) && !sel.has(hover)) {
    const col = sc.colors.get(hover) ?? POINT_COLOR;
    out.push(glow(sc.masks.get(hover)!, col, 0.28 * pulse, 20.0, rect));
    out.push(tint(sc.masks.get(hover)!, col, 0.34, rect));
  }

  out.push(...focusOutline(sc, anim, now, viewScale, hover, sel));

  // 5. point mode
  if (sc.pointMask !== null) {
    const col = POINT_COLOR;
    out.push(dimOutside([sc.pointMask], rect, 0.55));
    out.push(glow(sc.pointMask, col, 0.2 + 0.18 * anim.pulse, 20.0, rect));
    out.push(tint(sc.pointMask, col, 0.3, rect));
    const a = ants(sc.pointPolys, col, viewScale, anim.ant);
    if (a !== null) out.push(a);
  }

  // 6. press-and-hold ripple
  if (sc.masks.has(anim.pressObj) && anim.pressPt) {
    out.push(...press(sc, anim, now, rect, viewScale));
  }

  return out;
}

// ---------- the frame ----------
export function build(
  sc: Scene,
  pane: Pane,
  anim: AnimState,
  now: number,
): DisplayList {
  const vw = pane.viewW;
  const vh = pane.viewH;
  const ops: Op[] = [
    new Checker([0, 0, vw, vh], CELL, CHECK_LIGHT, CHECK_DARK),
  ];

  const [ew, eh] = pane.effectiveSize();
  if (sc.image === null || !ew || !eh || vw <= 0 || vh <= 0) {
    return new DisplayList(ops, 1.0, [vw, vh]);
  }

  const s = pane.scale();
  if (!s) return new DisplayList(ops, 1.0, [vw, vh]);

  const [iw, ih] = sc.imageSize;
  const rect: Rect = [0, 0, iw, ih];

  const xform: Step[] = [
    translate(vw / 2 + pane.ox, vh / 2 + pane.oy),
    scale(s, s),
  ];
  if (pane.rot) xform.push(rotate(pane.rot * 90));
  xform.push(scale(pane.fh ? -1 : 1, pane.fv ? -1 : 1));
  xform.push(translate(-iw / 2, -ih / 2));

  const inner: Op[] = [];
  if (sc.clipActive) {
    inner.push(...composite(sc, rect));
  } else {
    inner.push(new DrawImage(sc.image, rect, "smooth"));
    if (anim.scanning) inner.push(shimmer(anim.scanPhase, rect, iw, ih));
    if (sc.segMode) {
      const s2 = seg(sc, anim, now, rect, s);
      if (anim.reveal < 1.0) {
        const z = 0.97 + 0.03 * anim.reveal;
        inner.push(
          new Push(
            [new Push(s2, scaleAbout(z, z, iw / 2, ih / 2))],
            [],
            Math.max(0.0, anim.reveal),
          ),
        );
      } else {
        inner.push(...s2);
      }
    }
  }

  ops.push(new Push(inner, xform));
  return new DisplayList(ops, s, [vw, vh], anim.needsTick());
}
