/** The segmentation overlay model — the mirror of engine/objects.py. */
import { MORPH_N } from "./anim";
import { parseColor, type RGBA } from "./color";
import {
  polygonAreaAbs,
  resampleClosed,
  alignRing,
  type Poly,
  type Vec2,
} from "./geometry";
import { HitMaps } from "./hittest";
import { Scene, Morph, POINT_COLOR } from "./render/scene";
import type { Handle } from "./render/ops";

export interface SegObjectSpec {
  id: number;
  color: string;
  contour?: number[][][];
  bbox?: [number, number, number, number];
  handle?: Handle;
}

export class ObjectStore {
  segMode: string | null = null;

  masks = new Map<number, Handle>();
  colors = new Map<number, RGBA>();
  polys = new Map<number, Poly[]>();
  secPolys = new Map<number, Poly[]>();
  centroids = new Map<number, Vec2>();
  radius = new Map<number, number>();
  rings = new Map<number, Poly>();

  selected = new Set<number>();
  hoverId = 0;
  hoverGen = 0;
  hoverSpec = 0;
  hoverDepth = 0;

  pointMask: Handle | null = null;
  pointPolys: Poly[] = [];

  morph: Morph | null = null;
  maps = new HitMaps();

  clipActive = false;
  clipMasks: Handle[] = [];
  clipBg: RGBA | null = null;

  load(objects: SegObjectSpec[]) {
    this.clearObjects();
    for (const o of objects) {
      const oid = o.id;
      this.masks.set(oid, o.handle ?? oid);
      this.colors.set(oid, parseColor(o.color)!);
      const contours: Poly[] = (o.contour ?? []).map(
        (poly) => poly.map((p) => [p[0], p[1]] as Vec2),
      );
      this.polys.set(oid, contours);
      const [bx, by, bw, bh] = o.bbox ?? [0, 0, 0, 0];
      this.centroids.set(oid, [bx + bw / 2.0, by + bh / 2.0]);
      this.radius.set(oid, 0.6 * Math.max(bw, bh, 8));
      if (contours.length) {
        // `max(..., key=area)` keeps the FIRST maximum on a tie in Python.
        let largest = contours[0];
        let best = polygonAreaAbs(largest);
        for (let i = 1; i < contours.length; i++) {
          const a = polygonAreaAbs(contours[i]);
          if (a > best) {
            best = a;
            largest = contours[i];
          }
        }
        this.rings.set(oid, alignRing(resampleClosed(largest, MORPH_N)));
        this.secPolys.set(
          oid,
          contours.filter((p) => p !== largest),
        );
      }
    }
  }

  setPointMask(handle: Handle | null, contour?: number[][][]) {
    this.pointMask = handle;
    this.pointPolys = (contour ?? []).map(
      (poly) => poly.map((p) => [p[0], p[1]] as Vec2),
    );
  }

  setComposite(masks: Handle[], bg: RGBA | null = null) {
    this.clipActive = true;
    this.clipMasks = [...masks];
    this.clipBg = bg;
  }

  clearComposite() {
    this.clipActive = false;
    this.clipMasks = [];
    this.clipBg = null;
  }

  clearObjects() {
    this.masks.clear();
    this.colors.clear();
    this.polys.clear();
    this.secPolys.clear();
    this.centroids.clear();
    this.radius.clear();
    this.rings.clear();
  }

  clear(keepMode = false) {
    this.clearObjects();
    this.selected.clear();
    this.hoverId = this.hoverGen = this.hoverSpec = this.hoverDepth = 0;
    this.pointMask = null;
    this.pointPolys = [];
    this.morph = null;
    this.maps.clear();
    if (!keepMode) this.segMode = null;
  }

  selection(): number[] {
    return [...this.selected].sort((a, b) => a - b);
  }

  setSelection(ids: Iterable<number>) {
    this.selected = new Set(ids);
  }

  /** Returns true when the object was ADDED (so the host can pop it). */
  toggle(oid: number): boolean {
    if (this.selected.has(oid)) {
      this.selected.delete(oid);
      return false;
    }
    this.selected.add(oid);
    return true;
  }

  hasSeg(): boolean {
    return this.masks.size > 0 || this.pointMask !== null;
  }

  active(): boolean {
    return !!(this.hoverId || this.selected.size || this.pointMask !== null);
  }

  buildMorph(oldId: number, newId: number): Morph | null {
    const fr = this.rings.get(oldId) ?? null;
    const to = this.rings.get(newId) ?? null;
    if (fr === null && to === null) return null;
    return new Morph(
      fr,
      to,
      this.colors.get(oldId) ?? this.colors.get(newId) ?? POINT_COLOR,
      this.colors.get(newId) ?? this.colors.get(oldId) ?? POINT_COLOR,
      this.secPolys.get(oldId) ?? null,
      this.secPolys.get(newId) ?? null,
      this.centroids.get(oldId) ?? null,
      this.centroids.get(newId) ?? null,
    );
  }

  scene(image: Handle | null, imageSize: Vec2): Scene {
    const sc = new Scene();
    sc.image = image;
    sc.imageSize = imageSize;
    sc.segMode = this.segMode;
    sc.masks = this.masks;
    sc.colors = this.colors;
    sc.polys = this.polys;
    sc.secPolys = this.secPolys;
    sc.centroids = this.centroids;
    sc.radius = this.radius;
    sc.selected = this.selected;
    sc.hoverId = this.hoverId;
    sc.hoverGen = this.hoverGen;
    sc.hoverSpec = this.hoverSpec;
    sc.hoverDepth = this.hoverDepth;
    sc.pointMask = this.pointMask;
    sc.pointPolys = this.pointPolys;
    sc.morph = this.morph;
    sc.clipActive = this.clipActive;
    sc.clipMasks = this.clipMasks;
    sc.clipBg = this.clipBg;
    return sc;
  }
}
