/** Colour parsing — the mirror of engine/color.py. */

export type RGBA = [number, number, number, number];

/** "#rgb" / "#rrggbb" / "#rrggbbaa" -> (r, g, b, a) in 0..1. */
export function parseColor(spec: string | RGBA | number[] | null): RGBA | null {
  if (spec === null || spec === undefined) return null;
  if (Array.isArray(spec)) {
    const c = spec.map(Number);
    return (c.length === 4 ? c : [...c, 1.0]) as RGBA;
  }
  let s = String(spec).trim().replace(/^#/, "");
  if (s.length === 3 || s.length === 4) {
    s = s
      .split("")
      .map((ch) => ch + ch)
      .join("");
  }
  if (s.length !== 6 && s.length !== 8) {
    throw new Error(`cannot parse colour ${spec}`);
  }
  const v: number[] = [];
  for (let i = 0; i < s.length; i += 2) {
    v.push(parseInt(s.slice(i, i + 2), 16) / 255.0);
  }
  if (v.length === 3) v.push(1.0);
  return v as RGBA;
}
