/**
 * The view state of one image pane — the mirror of engine/pane.py.
 *
 * The inverse un-rotates BEFORE it un-flips. Getting that order wrong is a
 * silent, subtly-wrong-cursor bug, which is why the Python side pins it with a
 * differential test and why this is a literal transcription.
 */
import { MIN_ZOOM, MAX_ZOOM, clamp, type Vec2 } from "./geometry";

export class Pane {
  zoom = 1.0;
  ox = 0.0;
  oy = 0.0;
  rot = 0; // 0..3, each +90deg clockwise
  fh = false;
  fv = false;
  imageW = 0;
  imageH = 0;
  viewW = 0;
  viewH = 0;

  setImageSize(w: number, h: number) {
    this.imageW = Math.trunc(w);
    this.imageH = Math.trunc(h);
  }

  setViewSize(w: number, h: number) {
    this.viewW = w;
    this.viewH = h;
  }

  hasImage(): boolean {
    return this.imageW > 0 && this.imageH > 0;
  }

  /** Image size as the view sees it — axes swap on an odd quarter-turn. */
  effectiveSize(): Vec2 {
    if (!this.hasImage()) return [0, 0];
    return this.rot % 2 === 1
      ? [this.imageH, this.imageW]
      : [this.imageW, this.imageH];
  }

  fitScale(): number {
    const [ew, eh] = this.effectiveSize();
    if (!ew || !eh || !this.viewW || !this.viewH) return 0.0;
    return Math.min(this.viewW / ew, this.viewH / eh);
  }

  /** Image px -> view px. */
  scale(): number {
    return this.fitScale() * this.zoom;
  }

  viewToImage(px: number, py: number): Vec2 | null {
    if (!this.hasImage()) return null;
    const s = this.scale();
    if (!s) return null;
    let ux = (px - (this.viewW / 2 + this.ox)) / s;
    let uy = (py - (this.viewH / 2 + this.oy)) / s;
    for (let i = 0; i < this.rot % 4; i++) {
      [ux, uy] = [uy, -ux]; // undo each +90deg clockwise step
    }
    if (this.fh) ux = -ux;
    if (this.fv) uy = -uy;
    return [ux + this.imageW / 2, uy + this.imageH / 2];
  }

  /** The exact inverse of viewToImage. */
  imageToView(ix: number, iy: number): Vec2 | null {
    if (!this.hasImage()) return null;
    const s = this.scale();
    if (!s) return null;
    let ux = ix - this.imageW / 2;
    let uy = iy - this.imageH / 2;
    if (this.fh) ux = -ux;
    if (this.fv) uy = -uy;
    for (let i = 0; i < this.rot % 4; i++) {
      [ux, uy] = [-uy, ux]; // redo each +90deg clockwise step
    }
    return [
      ux * s + (this.viewW / 2 + this.ox),
      uy * s + (this.viewH / 2 + this.oy),
    ];
  }

  containsImagePoint(ix: number, iy: number): boolean {
    return ix >= 0 && ix < this.imageW && iy >= 0 && iy < this.imageH;
  }

  /** Zoom about a view-space anchor (the cursor), keeping it pinned. */
  zoomAt(factor: number, cx: number, cy: number) {
    const next = clamp(this.zoom * factor, MIN_ZOOM, MAX_ZOOM);
    const f = this.zoom ? next / this.zoom : 1.0;
    const centreX = this.viewW / 2 + this.ox;
    const centreY = this.viewH / 2 + this.oy;
    this.ox = cx + f * (centreX - cx) - this.viewW / 2;
    this.oy = cy + f * (centreY - cy) - this.viewH / 2;
    this.zoom = next;
  }

  resetView() {
    this.zoom = 1.0;
    this.ox = 0.0;
    this.oy = 0.0;
  }

  /** Zoom so one image pixel == one view pixel. */
  actualSize() {
    const fit = this.fitScale();
    this.zoom = fit ? 1.0 / fit : 1.0;
    this.ox = 0.0;
    this.oy = 0.0;
  }

  panTo(ox: number, oy: number) {
    this.ox = ox;
    this.oy = oy;
  }

  rotate(delta: number) {
    this.rot = (((this.rot + delta) % 4) + 4) % 4; // JS % is not Python's %
    this.resetView();
  }

  flip(horizontal: boolean) {
    // On an odd quarter-turn the on-screen axes are swapped, so "flip
    // horizontal" must mean what the user sees, not what the buffer stores.
    if (this.rot % 2 === 1) horizontal = !horizontal;
    if (horizontal) this.fh = !this.fh;
    else this.fv = !this.fv;
  }

  isTransformed(): boolean {
    return this.rot !== 0 || this.fh || this.fv;
  }

  exportTransform() {
    return { rot: this.rot, fh: this.fh, fv: this.fv };
  }
}
