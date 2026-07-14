/**
 * DisplayList -> JSON — the mirror of engine/render/codec.py.
 *
 * This is the anti-drift spine, so it has to be byte-identical to what Python's
 * `json.dumps(encode(dl), indent=1, sort_keys=True)` produces. Two things make
 * that harder than calling JSON.stringify:
 *
 *  1. **Floats vs ints.** Python writes a float 320.0 as "320.0"; JSON.stringify
 *     writes "320". So the encoder marks every value that `_n()` produced as a
 *     float (the `F` wrapper) and the writer formats it Python's way. Object-id
 *     handles stay ints and must NOT gain a ".0".
 *  2. **Rounding.** Python's round() is half-to-even on the exact binary value.
 *     See pyround.ts — it disagrees with every naive JS approach on real data.
 *
 * Get either wrong and the goldens diff on noise instead of on meaning.
 */
import { n } from "../pyround";
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
  type Handle,
  type Op,
} from "./ops";

/** A number that Python would print as a float ("320.0", not "320"). */
class F {
  constructor(public v: number) {}
}

const f = (x: number) => new F(n(x));
const fs = (xs: readonly number[]) => xs.map(f);

type JVal =
  | F
  | number
  | string
  | boolean
  | null
  | JVal[]
  | { [k: string]: JVal };

// ---------------------------------------------------------------- writer ----
/** repr() of a Python float: always a "." (or an exponent). */
function floatRepr(v: number): string {
  if (Number.isInteger(v) && Math.abs(v) < 1e16) return `${v}.0`;
  return String(v);
}

/** Python's json.dumps defaults to ensure_ascii=True: "…" becomes "\u2026". */
function pyStr(s: string): string {
  let out = JSON.stringify(s);
  out = out.replace(/[\u007f-\uffff]/g, (c) =>
    "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"),
  );
  return out;
}

/** json.dumps(obj, indent=1, sort_keys=True), exactly. */
export function pyJson(value: JVal, indent = 1, level = 0): string {
  const pad = " ".repeat(indent * (level + 1));
  const closePad = " ".repeat(indent * level);

  if (value instanceof F) return floatRepr(value.v);
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value); // an int
  if (typeof value === "string") return pyStr(value);

  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    const items = value.map((v) => pad + pyJson(v, indent, level + 1));
    return "[\n" + items.join(",\n") + "\n" + closePad + "]";
  }

  const keys = Object.keys(value).sort();
  if (keys.length === 0) return "{}";
  const items = keys.map(
    (k) => pad + pyStr(k) + ": " + pyJson(value[k], indent, level + 1),
  );
  return "{\n" + items.join(",\n") + "\n" + closePad + "}";
}

// --------------------------------------------------------------- encoder ----
function handle(h: Handle): JVal {
  return Array.isArray(h) ? (h.slice() as JVal[]) : (h as JVal);
}

function encPath(p: Path): JVal {
  const out: { [k: string]: JVal } = {};
  if (p.polys.length) {
    out.polys = p.polys.map((poly) => poly.map((pt) => fs(pt)));
  }
  if (p.circles.length) {
    out.circles = p.circles.map(([c, r]) => [fs(c), f(r)]);
  }
  return out;
}

function encMask(m: MaskSpec): JVal {
  return {
    layers: m.layers.map(handle),
    rect: fs(m.rect),
    mode: m.mode,
  };
}

export function encodeOp(op: Op): JVal {
  if (op instanceof Push) {
    const d: { [k: string]: JVal } = { op: "push" };
    if (op.transform.length) {
      d.transform = op.transform.map(
        (s) => [s[0], ...fs(s.slice(1) as number[])] as JVal[],
      );
    }
    if (op.opacity !== 1.0) d.opacity = f(op.opacity);
    if (op.blurSigma) d.blur_sigma = f(op.blurSigma);
    if (op.mask !== null) d.mask = encMask(op.mask);
    d.children = op.children.map(encodeOp);
    return d;
  }
  if (op instanceof FillRect) {
    return { op: "fill_rect", rect: fs(op.rect), color: fs(op.color) };
  }
  if (op instanceof Checker) {
    return {
      op: "checker",
      rect: fs(op.rect),
      cell: f(op.cell),
      light: fs(op.light),
      dark: fs(op.dark),
    };
  }
  if (op instanceof DrawImage) {
    const d: { [k: string]: JVal } = {
      op: "draw_image",
      image: handle(op.image),
      rect: fs(op.rect),
      filter: op.filter,
    };
    if (op.tint) d.tint = fs(op.tint);
    return d;
  }
  if (op instanceof LinearGradient) {
    return {
      op: "linear_gradient",
      rect: fs(op.rect),
      p0: fs(op.p0),
      p1: fs(op.p1),
      stops: op.stops.map(([o, c]) => [f(o), fs(c)] as JVal[]),
    };
  }
  if (op instanceof StrokePath) {
    const d: { [k: string]: JVal } = {
      op: "stroke",
      path: encPath(op.path),
      color: fs(op.color),
      width: f(op.width),
    };
    if (op.dash.length) {
      d.dash = fs(op.dash);
      d.dash_offset = f(op.dashOffset);
    }
    return d;
  }
  throw new TypeError(`cannot encode ${op}`);
}

export function encode(dl: DisplayList): JVal {
  return {
    version: dl.version,
    scale: f(dl.scale),
    view_size: fs(dl.viewSize),
    animating: !!dl.animating,
    ops: dl.ops.map(encodeOp),
  };
}

export function toJson(dl: DisplayList, indent = 1): string {
  return pyJson(encode(dl), indent);
}
