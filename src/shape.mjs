// src/shape.mjs — pinned convergence predicates for per-episode return series.
//
// RE-IMPLEMENTATION (zero-dep, in-repo) of the CONCEPTS in
// SuperInstance/delta-shape src/shape.mjs (flatTail / zeroTail /
// firstCrossing / changePoints / shapeOf / shapeHash). Credit and lineage:
// delta-shape's thesis is that a series' SHAPE — the sign pattern of its
// deltas and where that pattern changes — is an identity independent of the
// values, so "flat for the 4th consecutive round" stops being an eyeball
// claim and becomes a checkable object. This module applies that doctrine to
// RL training: convergence is a SHAPE claim about the return series, and it
// is pinned numerically or it does not exist.
//
// Differences from upstream (all deliberate, documented):
//   1. flatTail gains a WINDOW + TOLERANCE: RL returns are floats, so exact
//      equality (upstream's integer doctrine) is replaced by "every value in
//      the trailing window lies within tolerance of the window mean".
//   2. certifyConvergence adds the BAR: shape alone can certify a uniformly
//      bad policy (perfectly flat at -100). Convergence = flat tail AND mean
//      above bar. Shape is necessary, not sufficient — the same correction
//      delta-shape applies to "rate not wall" style claims.
//   3. No floats in the SHAPE layer: slopeClass and changePoints compare
//      delta signs only, exactly like upstream.

import { createHash } from "node:crypto";

export const diffs = (xs) => xs.slice(1).map((v, i) => v - xs[i]);
export const slopeClass = (d) => (d > 0 ? "+" : d < 0 ? "-" : "0");
export const shapeOf = (xs) => diffs(xs).map(slopeClass).join("");

// indices i (into the delta string) where the class changes — the
// change-points of the series. A drift detector is a change-point alarm.
// (identical semantics to delta-shape)
export function changePoints(xs) {
  const s = shapeOf(xs);
  const out = [];
  for (let i = 1; i < s.length; i++) if (s[i] !== s[i - 1]) out.push(i);
  return out;
}

// first index where the series crosses threshold k (>= k); null if never.
// (identical semantics to delta-shape's firstCrossing; float compare because
// RL returns are floats — the integer cross-multiply trick only applies to
// integer series, which is upstream's home turf)
export function firstCrossing(xs, k) {
  for (let i = 0; i < xs.length; i++) if (xs[i] >= k) return i;
  return null;
}

// pinned-window flat tail. Examines the LAST min(window, len) values; ok iff
// every one of them lies within `tolerance` of the window mean. Returns the
// evidence, not just a boolean — a receipt states its numbers.
export function flatTail(xs, { window = 100, tolerance = 0.01 } = {}) {
  const n = Math.min(window, xs.length);
  if (n === 0) return { ok: false, n: 0, mean: null, maxDev: null, tolerance, window };
  const tail = xs.slice(xs.length - n);
  const mean = tail.reduce((a, b) => a + b, 0) / n;
  let maxDev = 0;
  for (const v of tail) maxDev = Math.max(maxDev, Math.abs(v - mean));
  return { ok: maxDev <= tolerance, n, mean, maxDev, tolerance, window };
}

// terminal zero run (kept for parity with upstream's zeroTail/extinct).
export function zeroTail(xs) {
  let n = 0;
  for (let i = xs.length - 1; i >= 0 && xs[i] === 0; i--) n++;
  return n;
}
export const extinct = (xs, k = 3) => zeroTail(xs) >= k;

// THE claim, translated: the shape hash. Translation-invariant by
// construction (deltas kill constants); sign-preserving under any strictly
// monotone transform. Same 16-hex truncation as upstream.
export const shapeHash = (xs) =>
  createHash("sha256").update(`${shapeOf(xs)}|${JSON.stringify(changePoints(xs))}`, "utf8").digest("hex").slice(0, 16);

// T-CONV pin: a return series is CONVERGED at series end iff
//   (a) the trailing window is full (len >= window),
//   (b) the trailing-window mean >= bar, AND
//   (c) flatTail(window, tolerance) is verified.
// The receipt states the numbers: mean, bar, shortfall (0 when passed),
// maxDev, changePoints of the whole series, shape hash. `refused` names the
// failed leg when not converged — NC2 consumes this.
export function certifyConvergence(xs, { window = 100, tolerance = 0.01, bar } = {}) {
  const flat = flatTail(xs, { window, tolerance });
  const n = xs.length;
  const full = n >= window;
  const mean = flat.mean;
  const passesBar = mean !== null && bar !== undefined && mean >= bar;
  const converged = full && flat.ok && passesBar;
  const refused = converged ? null : !full ? "window_not_full" : !passesBar ? "bar" : "not_flat";
  return {
    converged,
    refused,
    n,
    window_mean: mean,
    bar,
    shortfall: passesBar ? 0 : bar - mean,
    flat_tail: flat,
    change_points: changePoints(xs),
    n_change_points: changePoints(xs).length,
    shape_hash: shapeHash(xs),
  };
}

// scan a full return series for the FIRST episode index at which T-CONV
// pins. Returns { pinned, episode, receipt } — receipt is the certificate at
// the pinning episode (or at series end when never pinned).
export function firstConvergenceEpisode(xs, opts = {}) {
  const window = opts.window ?? 100;
  for (let e = window - 1; e < xs.length; e++) {
    const cert = certifyConvergence(xs.slice(0, e + 1), opts);
    if (cert.converged) return { pinned: true, episode: e, receipt: cert };
  }
  return { pinned: false, episode: null, receipt: certifyConvergence(xs, opts) };
}
