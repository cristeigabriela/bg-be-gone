/**
 * The display list — the mirror of engine/render/ops.py.
 *
 * The contract between the engine (what to draw) and a backend (how). GSK on the
 * desktop, Canvas2D here. Every op maps 1:1 onto both.
 */
import type { RGBA } from "../color";
import type { Vec2, Poly } from "../geometry";

/** An opaque image handle: an object id, "src"/"point", or ["clip", i]. */
export type Handle = number | string | (string | number)[];

export type Rect = [number, number, number, number];

export type Step =
  | ["translate", number, number]
  | ["scale", number, number]
  | ["rotate", number];

export function translate(x: number, y: number): Step {
  return ["translate", x, y];
}

export function scale(sx: number, sy?: number): Step {
  return ["scale", sx, sy === undefined ? sx : sy];
}

export function rotate(deg: number): Step {
  return ["rotate", deg];
}

/** Scale around a point — the select "pop" and the press spring. */
export function scaleAbout(
  sx: number,
  sy: number,
  cx: number,
  cy: number,
): Step[] {
  return [translate(cx, cy), scale(sx, sy), translate(-cx, -cy)];
}

export class Path {
  constructor(
    public polys: Poly[] = [],
    public circles: [Vec2, number][] = [],
  ) {}
}

export type MaskMode = "keep" | "cut";

export class MaskSpec {
  constructor(
    public layers: Handle[],
    public rect: Rect,
    public mode: MaskMode = "keep",
  ) {}
}

export type Op =
  | Push
  | FillRect
  | Checker
  | DrawImage
  | LinearGradient
  | StrokePath;

export class Push {
  constructor(
    public children: Op[] = [],
    public transform: Step[] = [],
    public opacity = 1.0,
    public blurSigma = 0.0,
    public mask: MaskSpec | null = null,
  ) {}
}

export class FillRect {
  constructor(
    public rect: Rect,
    public color: RGBA,
  ) {}
}

export class Checker {
  constructor(
    public rect: Rect,
    public cell: number,
    public light: RGBA,
    public dark: RGBA,
  ) {}
}

export class DrawImage {
  constructor(
    public image: Handle,
    public rect: Rect,
    public filter: "smooth" | "nearest" = "smooth",
    public tint: RGBA | null = null,
  ) {}
}

export class LinearGradient {
  constructor(
    public rect: Rect,
    public p0: Vec2,
    public p1: Vec2,
    public stops: [number, RGBA][],
  ) {}
}

/** `width` and `dash` are already view-scale-corrected by the builder. */
export class StrokePath {
  constructor(
    public path: Path,
    public color: RGBA,
    public width: number,
    public dash: number[] = [],
    public dashOffset = 0.0,
  ) {}
}

export class DisplayList {
  constructor(
    public ops: Op[] = [],
    public scale = 1.0,
    public viewSize: Vec2 = [0, 0],
    public animating = false,
    public version = 1,
  ) {}
}
