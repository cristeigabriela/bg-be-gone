/**
 * DisplayList -> Canvas2D. The web render backend.
 *
 * The twin of shell_gtk/gsk_renderer.py, and just as dumb: it knows nothing
 * about objects, hovers or animations — it replays ops. All the thinking already
 * happened in core/render/builder.ts, which is byte-for-byte the same as the
 * Python one (spec/goldens/display_list).
 *
 * The op -> Canvas2D mapping, and why each one is what it is:
 *
 *   Push(transform)  save() + setTransform steps + restore()
 *   Push(opacity)    globalAlpha (composed multiplicatively, like GSK nests)
 *   Push(mask,keep)  an offscreen layer, then destination-in with the mask
 *   Push(mask,cut)   ... destination-out
 *   Push(blur)       an offscreen layer + ctx.filter = blur(sigma * scale)
 *   Checker          createPattern, one fill (NOT ~2000 rects)
 *   DrawImage        drawImage
 *   LinearGradient   createLinearGradient
 *   StrokePath       setLineDash / lineDashOffset
 *
 * Two calibration notes that are easy to get wrong:
 *
 *  * **Masks key off ALPHA.** destination-in composites by the source's alpha, so
 *    the mask textures must carry coverage in their alpha channel. They do — that
 *    is what compute/maskops._save_soft writes (RGBA, A=coverage), and it is the
 *    same reason GSK uses MaskMode.ALPHA rather than LUMINANCE. A luminance mask
 *    would be opaque everywhere and every layer would replace the last instead of
 *    unioning with it.
 *  * **Blur is applied in device space.** `ctx.filter` is not affected by the CTM
 *    consistently across browsers, so we bake the transform and blur at
 *    `sigma * scale` on an offscreen layer rather than trusting the CTM.
 */
import {
  DisplayList,
  Push,
  FillRect,
  Checker,
  DrawImage,
  LinearGradient,
  StrokePath,
  type Handle,
  type Op,
  type Rect,
  type Step,
} from "../core/render/ops";
import type { RGBA } from "../core/color";

/** Anything drawable: an ImageBitmap, a canvas, an <img>. */
export type Img = CanvasImageSource & { width: number; height: number };

/** The shell owns the pixels; the engine only ever passes handles. */
export type Resolve = (h: Handle) => Img | null;

/** Just enough of a 2D context to be stubbed in a test. */
export type Ctx = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

export interface Surface {
  /** A fresh transparent layer the same size as the target. */
  layer(w: number, h: number): { ctx: Ctx; canvas: CanvasImageSource };
}

function css(c: RGBA): string {
  const to255 = (v: number) => Math.round(Math.max(0, Math.min(1, v)) * 255);
  return `rgba(${to255(c[0])}, ${to255(c[1])}, ${to255(c[2])}, ${c[3]})`;
}

function applyTransform(ctx: Ctx, steps: Step[]) {
  for (const s of steps) {
    if (s[0] === "translate") ctx.translate(s[1], s[2]);
    else if (s[0] === "scale") ctx.scale(s[1], s[2]);
    else ctx.rotate((s[1] * Math.PI) / 180); // degrees clockwise
  }
}

export class Canvas2DBackend {
  constructor(
    private surface: Surface,
    private resolve: Resolve,
  ) {}

  render(ctx: Ctx, dl: DisplayList) {
    const [w, h] = dl.viewSize;
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    for (const op of dl.ops) this.op(ctx, op, dl, w, h);
    ctx.restore();
  }

  private op(ctx: Ctx, op: Op, dl: DisplayList, w: number, h: number) {
    if (op instanceof Push) return this.push(ctx, op, dl, w, h);
    if (op instanceof FillRect) return this.fillRect(ctx, op);
    if (op instanceof Checker) return this.checker(ctx, op);
    if (op instanceof DrawImage) return this.drawImage(ctx, op);
    if (op instanceof LinearGradient) return this.gradient(ctx, op);
    if (op instanceof StrokePath) return this.stroke(ctx, op);
    throw new TypeError(`cannot render ${op}`);
  }

  private push(ctx: Ctx, op: Push, dl: DisplayList, w: number, h: number) {
    const needsLayer = op.mask !== null || op.blurSigma > 0;

    if (!needsLayer) {
      ctx.save();
      applyTransform(ctx, op.transform);
      if (op.opacity !== 1.0) ctx.globalAlpha *= op.opacity;
      for (const c of op.children) this.op(ctx, c, dl, w, h);
      ctx.restore();
      return;
    }

    // A mask or a blur needs its own layer: both are composited, not painted.
    const { ctx: lctx, canvas } = this.surface.layer(w, h);
    lctx.setTransform(ctx.getTransform());
    applyTransform(lctx, op.transform);
    for (const c of op.children) this.op(lctx, c, dl, w, h);

    if (op.mask !== null) {
      // destination-in/out key off the source's ALPHA — which is why the mask
      // textures carry coverage in their alpha channel.
      lctx.setTransform(ctx.getTransform());
      applyTransform(lctx, op.transform);
      lctx.globalCompositeOperation =
        op.mask.mode === "keep" ? "destination-in" : "destination-out";
      for (const layer of op.mask.layers) {
        const img = this.resolve(layer);
        if (img) this.blit(lctx, img, op.mask.rect);
      }
      lctx.globalCompositeOperation = "source-over";
    }

    ctx.save();
    if (op.opacity !== 1.0) ctx.globalAlpha *= op.opacity;
    if (op.blurSigma > 0) {
      // Bake the transform: ctx.filter's relationship to the CTM is not
      // consistent across browsers, so blur in device space at sigma * scale.
      ctx.filter = `blur(${op.blurSigma * dl.scale}px)`;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(canvas, 0, 0);
    ctx.restore();
  }

  private blit(ctx: Ctx, img: Img, rect: Rect) {
    ctx.drawImage(img, rect[0], rect[1], rect[2], rect[3]);
  }

  private fillRect(ctx: Ctx, op: FillRect) {
    ctx.fillStyle = css(op.color);
    ctx.fillRect(op.rect[0], op.rect[1], op.rect[2], op.rect[3]);
  }

  private checker(ctx: Ctx, op: Checker) {
    // ONE op: a 2x2-cell pattern tile, filled once. The original GTK renderer
    // emitted a node per cell — about 2000 of them per frame at window size.
    const cell = op.cell;
    const { ctx: tctx, canvas } = this.surface.layer(cell * 2, cell * 2);
    tctx.fillStyle = css(op.light);
    tctx.fillRect(0, 0, cell * 2, cell * 2);
    tctx.fillStyle = css(op.dark);
    tctx.fillRect(cell, 0, cell, cell);
    tctx.fillRect(0, cell, cell, cell);

    const pat = (ctx as CanvasRenderingContext2D).createPattern(
      canvas as CanvasImageSource,
      "repeat",
    );
    if (pat) {
      ctx.fillStyle = pat;
      ctx.fillRect(op.rect[0], op.rect[1], op.rect[2], op.rect[3]);
    }
  }

  private drawImage(ctx: Ctx, op: DrawImage) {
    const img = this.resolve(op.image);
    if (!img) return;
    ctx.imageSmoothingEnabled = op.filter === "smooth";
    this.blit(ctx, img, op.rect);
  }

  private gradient(ctx: Ctx, op: LinearGradient) {
    const g = ctx.createLinearGradient(op.p0[0], op.p0[1], op.p1[0], op.p1[1]);
    for (const [off, col] of op.stops) g.addColorStop(off, css(col));
    ctx.fillStyle = g;
    ctx.fillRect(op.rect[0], op.rect[1], op.rect[2], op.rect[3]);
  }

  private stroke(ctx: Ctx, op: StrokePath) {
    ctx.beginPath();
    for (const poly of op.path.polys) {
      if (!poly.length) continue;
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
      ctx.closePath();
    }
    for (const [[cx, cy], r] of op.path.circles) {
      ctx.moveTo(cx + r, cy);
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
    }
    ctx.strokeStyle = css(op.color);
    ctx.lineWidth = op.width;
    // The builder already divided width and dash by the view scale, so these are
    // in user space on both backends and need no further correction.
    ctx.setLineDash(op.dash.length ? [...op.dash] : []);
    ctx.lineDashOffset = op.dashOffset;
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
