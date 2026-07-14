/**
 * Python's round(), exactly.
 *
 * The display-list codec rounds every float to 4 decimals so the two cores agree
 * despite last-bit noise — which means the *rounding* has to agree first, and it
 * very nearly does not:
 *
 *   Python's round(x, n) is round-half-to-EVEN on the exact binary value.
 *   JS's toFixed(n)      is round-half-AWAY-from-zero on the exact binary value.
 *   Math.round(x*1e4)/1e4 is neither: x*1e4 is itself inexact.
 *
 * They disagree on exact ties, and exact ties are reachable: a tie needs
 * x*10^4 to land exactly on k+0.5, i.e. x = (2k+1)/20000. Since 20000 = 2^5·625,
 * that IS a dyadic rational whenever 625 divides (2k+1) — for example
 * 0.03125 (= 1/32), where x*1e4 is exactly 312.5.
 *
 *   round(0.03125, 4)  ->  Python 0.0312   (312 is even)
 *   (0.03125).toFixed(4) -> "0.0313"       (half away from zero)
 *
 * One digit, in a golden that is compared byte-for-byte. So this does the
 * rounding on the exact decimal expansion of the double, and breaks ties to even
 * — and tests/pyround_test.ts checks it against CPython's own answers.
 */

/** round-half-to-even at `nd` decimals, on the exact value of the double. */
export function pyRound(x: number, nd = 4): number {
  if (!Number.isFinite(x)) return x;
  const neg = x < 0;
  const a = Math.abs(x);

  // toFixed is correctly rounded to that many decimals of the EXACT value, so
  // 30 digits is far more than enough to see what lies past the cut.
  const s = a.toFixed(30);
  const dot = s.indexOf(".");
  const digits = s.slice(0, dot) + s.slice(dot + 1);
  const cut = dot + nd; // keep digits[0..cut)

  if (cut >= digits.length) return x;

  const keep = digits.slice(0, cut);
  const rest = digits.slice(cut);

  let up = false;
  if (rest[0] > "5") {
    up = true;
  } else if (rest[0] === "5") {
    if (/[1-9]/.test(rest.slice(1))) {
      up = true; // strictly above the tie
    } else {
      // an exact tie -> round half to EVEN, which is the whole point of this file
      up = Number(keep[keep.length - 1]) % 2 === 1;
    }
  }

  let n = BigInt(keep === "" ? "0" : keep);
  if (up) n += 1n;

  const out = Number(n) / Math.pow(10, nd);
  return neg ? -out : out;
}

/** The codec's `_n`: round, and normalise -0.0 to 0.0. */
export function n(x: number): number {
  const r = pyRound(x, 4);
  return r === 0 ? 0 : r; // `0 + -0` is +0
}

export function nums(xs: readonly number[]): number[] {
  return xs.map(n);
}
