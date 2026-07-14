/**
 * The conformance suite: does the TypeScript core agree with the Python one?
 *
 * It builds a display list for every fixture in spec/fixtures/ and compares it,
 * BYTE FOR BYTE, against the golden the Python engine froze. Same fixture in,
 * same JSON out — every eased curve, dash offset, pulse phase, morph lerp and
 * op ordering.
 *
 * That is what makes the web build a port rather than a rewrite: a disagreement
 * names the exact op that drifted instead of showing a blurry pixel diff.
 *
 *   bun run web/conform.ts
 */
import { AnimState } from "./core/anim";
import { parseColor } from "./core/color";
import { ObjectStore } from "./core/objects";
import { Pane } from "./core/pane";
import { build } from "./core/render/builder";
import { toJson, pyJson } from "./core/render/codec";

/** describe() is plain JSON-able data — no float markers needed. */
const pyJsonAny = (v: unknown) => pyJson(v as never);

const ROOT = new URL("..", import.meta.url).pathname;
const SCENE = `${ROOT}spec/assets/scene`;
const FIXTURES = `${ROOT}spec/fixtures`;
const GOLDENS = `${ROOT}spec/goldens/display_list`;

/** A frame-clock timestamp `ms` milliseconds in the past (now == 0). */
const msAgo = (ms: number | null | undefined) =>
  ms === null || ms === undefined ? null : -Math.trunc(ms * 1000);

interface Fixture {
  name: string;
  view: [number, number];
  seg_mode?: string;
  pane?: Record<string, number | boolean>;
  anim?: Record<string, number>;
  scanning?: boolean;
  selected?: number[];
  hover?: { gen?: number; spec?: number; depth?: number; drilled?: boolean };
  pop?: Record<string, number>;
  press?: {
    obj: number;
    pt: [number, number];
    press_ms_ago: number;
    release_ms_ago?: number;
  };
  morph?: { from: number; to: number; ms_ago: number };
  point_mask?: number;
  composite?: { ids: number[]; bg?: string };
}

/**
 * The mirror of spec/tools/rasterize.py:build_view — same fixture, same state.
 * No pixels are needed: the display list only ever names images by handle.
 */
function buildFixture(fx: Fixture, meta: any, objs: any[]) {
  const [vw, vh] = fx.view;

  const pane = new Pane();
  const anim = new AnimState();
  const objects = new ObjectStore();

  pane.setImageSize(meta.size[0], meta.size[1]);
  pane.setViewSize(vw, vh);

  if (fx.seg_mode) {
    objects.segMode = fx.seg_mode;
    objects.load(objs);
  }

  const p = fx.pane ?? {};
  pane.zoom = Number(p.zoom ?? 1.0);
  pane.ox = Number(p.ox ?? 0.0);
  pane.oy = Number(p.oy ?? 0.0);
  pane.rot = Number(p.rot ?? 0);
  pane.fh = Boolean(p.fh ?? false);
  pane.fv = Boolean(p.fv ?? false);

  const a = fx.anim ?? {};
  anim.pulse = Number(a.pulse ?? 0.5);
  anim.ant = Number(a.ant ?? 0.0);
  anim.scanPhase = Number(a.scan_phase ?? 0.0);
  anim.reveal = Number(a.reveal ?? 1.0);
  anim.revealT0 = null;
  anim.scanning = Boolean(fx.scanning ?? false);

  objects.setSelection(fx.selected ?? []);

  const h = fx.hover;
  if (h) {
    objects.hoverGen = h.gen ?? 0;
    objects.hoverSpec = h.spec ?? 0;
    objects.hoverDepth = h.depth ?? 0;
    anim.drilled = Boolean(h.drilled ?? false);
    objects.hoverId = anim.drilled ? objects.hoverSpec : objects.hoverGen;
  }

  for (const [oid, ms] of Object.entries(fx.pop ?? {})) {
    anim.pop.set(Number(oid), msAgo(ms)!);
  }

  const pr = fx.press;
  if (pr) {
    anim.pressObj = pr.obj;
    anim.pressPt = pr.pt;
    anim.pressT0 = msAgo(pr.press_ms_ago);
    anim.releaseT0 = msAgo(pr.release_ms_ago);
  }

  const mo = fx.morph;
  if (mo) {
    objects.morph = objects.buildMorph(mo.from, mo.to);
    anim.morphT0 = msAgo(mo.ms_ago);
  }

  const comp = fx.composite;
  if (comp) {
    // The clip layers are ["clip", i] handles — the shell owns those textures,
    // and the engine never names them by object id.
    const ids = comp.ids.filter((i) => objects.masks.has(i));
    objects.setComposite(
      ids.map((_, i) => ["clip", i]),
      comp.bg ? parseColor(comp.bg) : null,
    );
  }

  const pm = fx.point_mask;
  if (pm) {
    const obj = objs.find((o) => o.id === pm);
    objects.setPointMask("point", obj.contour);
    // set_point_mask begins a reveal, and the rasteriser clears reveal_t0
    // *before* this point — so revealT0 stays set (at 0) even though `reveal` is
    // then overridden to 1.0, and the frame is still reported as animating.
    anim.beginReveal(0);
    anim.reveal = Number(a.reveal ?? 1.0);
  }

  const sc = objects.scene("src", [meta.size[0], meta.size[1]]);
  return build(sc, pane, anim, 0); // the clock is pinned at 0, like the rasteriser
}

// ---------------------------------------------------------------- the run ---
const meta = await Bun.file(`${SCENE}/meta.json`).json();
const objs = meta.objects;

const names = [...new Bun.Glob("*.json").scanSync(FIXTURES)].sort();

let bad = 0;
let ok = 0;
for (const file of names) {
  const name = file.replace(/\.json$/, "");
  const fx: Fixture = await Bun.file(`${FIXTURES}/${file}`).json();
  fx.name = name;

  let got: string;
  try {
    got = toJson(buildFixture(fx, meta, objs));
  } catch (e) {
    console.log(`  THREW  ${name}: ${e}`);
    bad++;
    continue;
  }

  const goldenPath = `${GOLDENS}/${name}.json`;
  const want = (await Bun.file(goldenPath).text()).replace(/\n$/, "");

  if (want === got) {
    ok++;
    continue;
  }

  bad++;
  console.log(`  DIFF   ${name}`);
  await Bun.write(`${goldenPath}.ts.actual`, got + "\n");
  // point at the first differing line, not the whole tree
  const wl = want.split("\n");
  const gl = got.split("\n");
  for (let i = 0; i < Math.max(wl.length, gl.length); i++) {
    if (wl[i] !== gl[i]) {
      console.log(`    line ${i + 1}:`);
      console.log(`      py: ${wl[i] ?? "<eof>"}`);
      console.log(`      ts: ${gl[i] ?? "<eof>"}`);
      break;
    }
  }
}

if (bad) {
  console.log(`Display lists: ${ok} match, ${bad} differ`);
} else {
  console.log(`Display   OK — ${ok} lists byte-identical to the Python core`);
}

// ============================================================== UiSchema =====
// The sidebar contract. Both UIs build themselves from describe(), so the two
// cores must describe the same thing.
{
  const { newSettings } = await import("./core/interactables");
  const st = newSettings();
  const got = pyJsonAny(st.describe());
  const want = (
    await Bun.file(`${ROOT}spec/goldens/ui_schema.json`).text()
  ).replace(/\n$/, "");
  if (want === got) {
    console.log("UiSchema  OK — describe() is byte-identical");
  } else {
    console.log("UiSchema  DIFF");
    await Bun.write(`${ROOT}spec/goldens/ui_schema.json.ts.actual`, got + "\n");
    const wl = want.split("\n");
    const gl = got.split("\n");
    for (let i = 0; i < Math.max(wl.length, gl.length); i++) {
      if (wl[i] !== gl[i]) {
        console.log(`    line ${i + 1}:\n      py: ${wl[i]}\n      ts: ${gl[i]}`);
        break;
      }
    }
    bad++;
  }
}

// ================================================================== CV =======
// The kernels the web build has to run in the main thread: components, contour
// tracing, NMS, stability. spec/goldens/cv.json is what the Python side froze.
{
  const M = await import("./core/maskops");
  const cv = (await Bun.file(`${ROOT}spec/goldens/cv.json`).json()).cv;
  const W = 64, H = 64;

  const mk = (f: (x: number, y: number) => boolean): M.Mask => {
    const data = new Uint8Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) data[y * W + x] = f(x, y) ? 1 : 0;
    return { data, w: W, h: H };
  };
  const disc = (cx: number, cy: number, r: number) =>
    mk((x, y) => (x - cx) ** 2 + (y - cy) ** 2 <= r * r);
  const both = (a: M.Mask, b: M.Mask, op: (p: number, q: number) => boolean) =>
    mk((x, y) => op(a.data[y * W + x], b.data[y * W + x]));

  const big = disc(28, 30, 20);
  const small = disc(24, 26, 7);
  const ring = both(disc(30, 30, 16), disc(30, 30, 9), (p, q) => !!p && !q);
  const disjoint = both(disc(8, 8, 5), disc(56, 56, 5), (p, q) => !!p || !!q);
  const diagonal_touch = mk(
    (x, y) =>
      (x >= 4 && x < 12 && y >= 40 && y < 48) ||
      (x >= 12 && x < 20 && y >= 48 && y < 56),
  );
  const masks: Record<string, M.Mask> = {
    big, small, ring, disjoint, diagonal_touch,
  };

  let cvBad = 0;
  const cmp = (label: string, got: unknown, want: unknown) => {
    const g = JSON.stringify(got);
    const w = JSON.stringify(want);
    if (g !== w) {
      console.log(`  CV DIFF ${label}\n      py: ${w}\n      ts: ${g}`);
      cvBad++;
    }
  };

  for (const [name, m] of Object.entries(masks)) {
    const want = cv[name];
    cmp(`${name}.area`, M.maskArea(m), want.area);
    cmp(`${name}.bbox`, M.bbox(m), want.bbox);
    cmp(`${name}.components_8con`, M.label(m, 8)[1], want.components_8con);
    cmp(`${name}.components_4con`, M.label(m, 4)[1], want.components_4con);
    cmp(`${name}.contours`, M.contour(m), want.contours);
  }

  const sc = cv._scoring;
  const r4 = (x: number) => Math.round(x * 1e4) / 1e4;
  cmp("iou_big_small", r4(M.iou(big, small)), sc.iou_big_small);
  cmp("iou_self", r4(M.iou(big, big)), sc.iou_self);
  cmp("bbox_overlap", Number(M.bboxOverlap(M.bbox(big), M.bbox(small))), sc.bbox_overlap);

  const logit = (m: M.Mask, hi: number, lo: number) => {
    const f = new Float32Array(m.data.length);
    for (let i = 0; i < f.length; i++) f[i] = m.data[i] ? hi : lo;
    return f;
  };
  cmp("stability_sharp", r4(M.stability(logit(big, 6, -6))), sc.stability_sharp);
  cmp("stability_soft", r4(M.stability(logit(big, 0.4, -0.4))), sc.stability_soft);

  const kept = M.nms(
    [[0.9, big], [0.95, big], [0.8, small], [0.7, disjoint]],
    0.7,
  );
  cmp("nms.kept_scores", kept.map(([s]) => r4(s)), cv._nms.kept_scores);
  cmp("nms.kept_areas", kept.map(([, m]) => M.maskArea(m)), cv._nms.kept_areas);
  cmp("nms.n_kept", kept.length, cv._nms.n_kept);

  if (cvBad) {
    console.log(`CV        FAILED (${cvBad} kernels differ)`);
    bad += cvBad;
  } else {
    console.log("CV        OK — every kernel matches the Python golden");
  }
}

console.log("");
if (bad) {
  console.log("WEB CONFORMANCE FAILED");
  process.exit(1);
}
console.log("WEB CONFORMANCE OK");
