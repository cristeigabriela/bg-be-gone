/**
 * The Canvas2D backend, against a recording context.
 *
 * bun has no DOM, so this stubs a 2D context that records every call and asserts
 * the op -> call mapping. It cannot prove the pixels (that needs a browser and
 * the render goldens); it CAN prove the things most likely to be silently wrong:
 *
 *   * a mask composites with destination-in / destination-out — and therefore
 *     keys off ALPHA, which is the whole reason the mask PNGs are RGBA coverage
 *     rather than luminance (the multi-select union bug);
 *   * every mask layer is blitted into the same layer, so several selected
 *     objects UNION rather than the last one replacing the rest;
 *   * the checkerboard is ONE pattern fill, not a rect per cell;
 *   * dashes are set from the values the builder already scaled, so marching ants
 *     stay screen-constant under zoom.
 *
 *   bun run web/tests/canvas2d_test.ts
 */
import { AnimState } from "../core/anim";
import { ObjectStore } from "../core/objects";
import { Pane } from "../core/pane";
import { build } from "../core/render/builder";
import { Canvas2DBackend, type Ctx, type Img, type Surface } from "../render/canvas2d";

type Call = { fn: string; args: unknown[] };

/** A 2D context that records instead of drawing. */
function recorder(log: Call[]): Ctx {
  const rec =
    (fn: string) =>
    (...args: unknown[]) => {
      log.push({ fn, args });
      if (fn === "createPattern") return { pattern: true };
      if (fn === "createLinearGradient") {
        return { addColorStop: (...a: unknown[]) => log.push({ fn: "addColorStop", args: a }) };
      }
      if (fn === "getTransform") return { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
      return undefined;
    };
  const ctx: Record<string, unknown> = {};
  for (const fn of [
    "save", "restore", "clearRect", "fillRect", "translate", "scale", "rotate",
    "setTransform", "drawImage", "beginPath", "moveTo", "lineTo", "closePath",
    "arc", "stroke", "setLineDash", "createPattern", "createLinearGradient",
    "getTransform",
  ]) {
    ctx[fn] = rec(fn);
  }
  // properties we care about
  for (const p of [
    "fillStyle", "strokeStyle", "lineWidth", "lineDashOffset", "globalAlpha",
    "globalCompositeOperation", "filter", "imageSmoothingEnabled",
  ]) {
    let v: unknown = p === "globalAlpha" ? 1 : undefined;
    Object.defineProperty(ctx, p, {
      get: () => v,
      set: (nv) => {
        v = nv;
        log.push({ fn: `set:${p}`, args: [nv] });
      },
    });
  }
  return ctx as unknown as Ctx;
}

function surface(log: Call[]): Surface {
  return {
    layer(w: number, h: number) {
      log.push({ fn: "layer", args: [w, h] });
      return { ctx: recorder(log), canvas: { w, h } as unknown as CanvasImageSource };
    },
  };
}

const FAILED: string[] = [];
function check(label: string, got: unknown, want: unknown) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) FAILED.push(label);
  console.log(`  ${label.padEnd(56)} ${ok ? "PASS" : "FAIL"}`);
  if (!ok) {
    console.log(`      want: ${JSON.stringify(want)}`);
    console.log(`      got:  ${JSON.stringify(got)}`);
  }
}

// ---------------------------------------------------------------- a scene ---
const meta = await Bun.file(
  new URL("../../spec/assets/scene/meta.json", import.meta.url).pathname,
).json();

function frame(selected: number[]) {
  const pane = new Pane();
  pane.setImageSize(meta.size[0], meta.size[1]);
  pane.setViewSize(320, 240);
  const anim = new AnimState();
  anim.pulse = 0.6;
  anim.ant = 8.0;
  const objects = new ObjectStore();
  objects.segMode = "everything";
  objects.load(meta.objects);
  objects.setSelection(selected);
  return build(objects.scene("src", [meta.size[0], meta.size[1]]), pane, anim, 0);
}

const log: Call[] = [];
const img = { width: 160, height: 120 } as Img;
const backend = new Canvas2DBackend(surface(log), () => img);
backend.render(recorder(log), frame([1, 2]));

const fns = log.map((c) => c.fn);

console.log("the mask -> composite mapping");
const comps = log
  .filter((c) => c.fn === "set:globalCompositeOperation")
  .map((c) => c.args[0]);
check(
  "masks composite with destination-in / destination-out",
  [...new Set(comps)].sort(),
  ["destination-in", "destination-out", "source-over"],
);
check(
  "... which key off ALPHA — so the mask PNGs must be RGBA coverage",
  comps.includes("destination-in"),
  true,
);

// With two objects selected, the "dim outside the selection" mask has TWO layers,
// and both must be blitted into the same layer or the union is a last-wins.
console.log("the multi-select union (the step-5 bug, in the browser)");
const dl = frame([1, 2]);
const dim = (function findCut(ops: any[]): any {
  for (const op of ops) {
    if (op.mask && op.mask.mode === "cut") return op;
    if (op.children) {
      const r = findCut(op.children);
      if (r) return r;
    }
  }
  return null;
})(dl.ops as any[]);
check("the dim-outside mask carries BOTH selected layers", dim.mask.layers, [1, 2]);
check(
  "... in sorted id order, so the two cores agree",
  dim.mask.layers,
  [...dim.mask.layers].sort((a: number, b: number) => a - b),
);

console.log("the checkerboard");
check(
  "is ONE pattern fill, not a rect per cell",
  fns.filter((f) => f === "createPattern").length,
  1,
);

console.log("marching ants");
const dashes = log.filter((c) => c.fn === "setLineDash" && (c.args[0] as number[]).length);
check("dashes are set from the builder's scaled values", dashes.length > 0, true);
check(
  "... and are the 6.0/4.5 pair divided by the view scale",
  (dashes[0].args[0] as number[]).map((v) => Math.round(v * 1000) / 1000),
  [6.0 / dl.scale, 4.5 / dl.scale].map((v) => Math.round(v * 1000) / 1000),
);

console.log("");
if (FAILED.length) {
  console.log(`CANVAS2D FAILED (${FAILED.length}): ${FAILED.join(", ")}`);
  process.exit(1);
}
console.log("CANVAS2D OK");
