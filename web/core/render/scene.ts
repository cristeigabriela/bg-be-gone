/** What the builder needs to know to draw a frame — the mirror of scene.py. */
import type { RGBA } from "../color";
import type { Poly, Vec2 } from "../geometry";
import type { Handle } from "./ops";

export const CELL = 16;
export const CHECK_LIGHT: RGBA = [0.2, 0.2, 0.22, 1.0];
export const CHECK_DARK: RGBA = [0.15, 0.15, 0.17, 1.0];
export const DIM: RGBA = [0.0, 0.0, 0.0, 1.0];
export const POINT_COLOR: RGBA = [0.2, 0.52, 0.9, 1.0];

/** An in-flight outline tween between two focused objects. */
export class Morph {
  constructor(
    public fromRing: Poly | null = null,
    public toRing: Poly | null = null,
    public cf: RGBA | null = null,
    public ct: RGBA | null = null,
    public secFrom: Poly[] | null = null,
    public secTo: Poly[] | null = null,
    public cenFrom: Vec2 | null = null,
    public cenTo: Vec2 | null = null,
  ) {}
}

export class Scene {
  image: Handle | null = null;
  imageSize: Vec2 = [0, 0];

  segMode: string | null = null;
  masks = new Map<number, Handle>(); // insertion order == paint order
  colors = new Map<number, RGBA>();
  polys = new Map<number, Poly[]>();
  secPolys = new Map<number, Poly[]>();
  centroids = new Map<number, Vec2>();
  radius = new Map<number, number>();

  selected = new Set<number>();
  hoverId = 0;
  hoverGen = 0;
  hoverSpec = 0;
  hoverDepth = 0;

  pointMask: Handle | null = null;
  pointPolys: Poly[] = [];

  morph: Morph | null = null;

  clipActive = false;
  clipMasks: Handle[] = [];
  clipBg: RGBA | null = null;
}
