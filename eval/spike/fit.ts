// Candidate trainers for the spike, with the same holdout and temperature protocol as src/head.ts. Plain typed-array
// JavaScript, so anything here could run in the browser worker.
import { seeded } from "../data.ts";

export interface Cfg {
  solver: "lbfgs" | "adam";
  l2?: number | number[];   // a list: choose on the holdout by temperature-scaled log loss
  iters?: number;           // L-BFGS iterations or Adam epochs
  lr?: number;              // Adam
  smoothing?: number;       // label smoothing
  balanced?: boolean;       // weight examples by inverse class frequency
  prior?: "none" | "ridge"; // ridge: shrink toward the centroid and label-name start of src/head.ts, not toward zero
  centroidScale?: number;
  bestEpoch?: boolean;      // Adam: keep the epoch with the lowest holdout loss (jimothy's rule)
}
type Prob = { X: Float32Array | Float64Array; dim: number; C: number; y: Int32Array };

// Mean cross-entropy over rows + l2/2 |W - W0|^2, bias unregularised. Parameters: C*(dim+1), bias last in each row.
function lossGrad(p: Prob, rows: Int32Array, w: Float64Array, g: Float64Array, l2: number, W0: Float64Array | null, smoothing: number, weights: Float64Array | null) {
  const { X, dim, C, y } = p, D = dim + 1, z = new Float64Array(C);
  g.fill(0);
  let loss = 0, total = 0;
  const off = smoothing / C, on = 1 - smoothing;
  for (let r = 0; r < rows.length; r++) {
    const i = rows[r], x = i * dim, ew = weights ? weights[y[i]] : 1;
    total += ew;
    let max = -Infinity;
    for (let c = 0; c < C; c++) {
      const o = c * D;
      let v = w[o + dim];
      for (let j = 0; j < dim; j++) v += w[o + j] * X[x + j];
      z[c] = v;
      if (v > max) max = v;
    }
    let sum = 0;
    for (let c = 0; c < C; c++) { z[c] = Math.exp(z[c] - max); sum += z[c]; }
    for (let c = 0; c < C; c++) {
      const q = z[c] / sum, t = off + (c === y[i] ? on : 0);
      if (t > 0) loss -= ew * t * Math.log(q + 1e-300);
      const e = ew * (q - t), o = c * D;
      g[o + dim] += e;
      for (let j = 0; j < dim; j++) g[o + j] += e * X[x + j];
    }
  }
  for (let k = 0; k < g.length; k++) g[k] /= total;
  loss /= total;
  if (l2 > 0) for (let c = 0; c < C; c++) for (let j = 0; j < dim; j++) {
    const k = c * D + j, d = w[k] - (W0 ? W0[k] : 0);
    loss += (l2 / 2) * d * d; g[k] += l2 * d;
  }
  return loss;
}

function lbfgs(p: Prob, rows: Int32Array, w: Float64Array, l2: number, W0: Float64Array | null, cfg: Cfg, weights: Float64Array | null) {
  const n = w.length, m = 10, S: Float64Array[] = [], Y: Float64Array[] = [], rho: number[] = [];
  const dot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; };
  const g = new Float64Array(n), g2 = new Float64Array(n), d = new Float64Array(n), w2 = new Float64Array(n);
  const f0 = (a: Float64Array, b: Float64Array) => lossGrad(p, rows, a, b, l2, W0, cfg.smoothing ?? 0, weights);
  let f = f0(w, g), it = 0, evals = 1;
  const maxIter = cfg.iters ?? 300;
  for (; it < maxIter; it++) {
    d.set(g);
    const al: number[] = [];
    for (let i = S.length - 1; i >= 0; i--) { const a = rho[i] * dot(S[i], d); al[i] = a; for (let k = 0; k < n; k++) d[k] -= a * Y[i][k]; }
    const gamma = S.length ? dot(S[S.length - 1], Y[S.length - 1]) / dot(Y[S.length - 1], Y[S.length - 1]) : 1 / Math.sqrt(dot(g, g));
    for (let k = 0; k < n; k++) d[k] *= gamma;
    for (let i = 0; i < S.length; i++) { const b = rho[i] * dot(Y[i], d); for (let k = 0; k < n; k++) d[k] += S[i][k] * (al[i] - b); }
    const slope = -dot(g, d);
    let step = 1, f2 = Infinity;
    for (let ls = 0; ls < 25; ls++) {
      for (let k = 0; k < n; k++) w2[k] = w[k] - step * d[k];
      f2 = f0(w2, g2); evals++;
      if (f2 <= f + 1e-4 * step * slope) break;
      step /= 2;
    }
    if (!(f2 < f)) break;
    const s = new Float64Array(n), yy = new Float64Array(n);
    for (let k = 0; k < n; k++) { s[k] = w2[k] - w[k]; yy[k] = g2[k] - g[k]; }
    const sy = dot(s, yy);
    if (sy > 1e-12) { S.push(s); Y.push(yy); rho.push(1 / sy); if (S.length > m) { S.shift(); Y.shift(); rho.shift(); } }
    const done = (f - f2) / Math.max(1e-12, Math.abs(f)) < 1e-7;
    w.set(w2); g.set(g2); f = f2;
    if (done) { it++; break; }
  }
  return { iters: it, evals, loss: f };
}

function heldLoss(p: Prob, w: Float64Array, check: Int32Array) {
  const { X, dim, C, y } = p, D = dim + 1, z = new Float64Array(C);
  let loss = 0;
  for (const i of check) {
    let max = -Infinity;
    for (let c = 0; c < C; c++) { const o = c * D; let v = w[o + dim]; for (let j = 0; j < dim; j++) v += w[o + j] * X[i * dim + j]; z[c] = v; if (v > max) max = v; }
    let sum = 0;
    for (let c = 0; c < C; c++) sum += Math.exp(z[c] - max);
    loss -= z[y[i]] - max - Math.log(sum);
  }
  return loss / check.length;
}

// Full-batch Adam from the start point, as in jimothy's src/linear.ts (MIT, Andrew Prifer).
function adam(p: Prob, rows: Int32Array, w: Float64Array, l2: number, W0: Float64Array | null, cfg: Cfg, weights: Float64Array | null, check?: Int32Array) {
  const n = w.length, m = new Float64Array(n), v = new Float64Array(n), g = new Float64Array(n);
  const lr = cfg.lr ?? 0.05, epochs = cfg.iters ?? 100;
  let best = Infinity, bestEp = epochs, keep: Float64Array | null = null;
  for (let e = 1; e <= epochs; e++) {
    lossGrad(p, rows, w, g, l2, W0, cfg.smoothing ?? 0, weights);
    const c1 = 1 - 0.9 ** e, c2 = 1 - 0.999 ** e;
    for (let k = 0; k < n; k++) {
      m[k] = 0.9 * m[k] + 0.1 * g[k];
      v[k] = 0.999 * v[k] + 0.001 * g[k] * g[k];
      w[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
    }
    if (cfg.bestEpoch && check) {
      const l = heldLoss(p, w, check);
      if (l < best - 1e-7) { best = l; bestEp = e; keep = w.slice(); }
      if (e - bestEp >= 30) break;
    }
  }
  if (keep) w.set(keep);
  return { iters: bestEp, evals: epochs, loss: best };
}

export function fitCandidate(X: Float32Array, dim: number, y: Int32Array, C: number, cfg: Cfg, seed: number, priors: Float32Array | null, floor: number) {
  const N = y.length, D = dim + 1, P = C * D;
  const p: Prob = { X, dim, C, y };
  const rand = seeded(seed);
  const idx = Array.from({ length: N }, (_, i) => i).sort(() => rand() - 0.5);
  const nVal = Math.max(1, Math.floor(N * 0.1));
  const folds = nVal < 200 ? Array.from({ length: 5 }, (_, f) => idx.filter((_, i) => i % 5 === f)) : [idx.slice(0, nVal)];
  let weights: Float64Array | null = null;
  if (cfg.balanced) {
    const n = new Float64Array(C);
    for (let i = 0; i < N; i++) n[y[i]]++;
    weights = n.map((v) => (v ? N / (C * v) : 0));
  }
  const start = (rows: ArrayLike<number>) => {
    if (cfg.prior !== "ridge") return null;
    const w = new Float64Array(P), scale = cfg.centroidScale ?? 20;
    for (let r = 0; r < rows.length; r++) for (let j = 0; j < dim; j++) w[y[rows[r]] * D + j] += X[rows[r] * dim + j];
    if (priors) for (let c = 0; c < C; c++) for (let j = 0; j < dim; j++) w[c * D + j] += priors[c * dim + j];
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let j = 0; j < dim; j++) s += w[c * D + j] ** 2;
      s = Math.sqrt(s) || 1;
      for (let j = 0; j < dim; j++) w[c * D + j] *= scale / s;
    }
    return w;
  };
  const solve = (rows: Int32Array, l2: number, check?: Int32Array, fixedIters?: number) => {
    const W0 = start(rows), w = W0 ? W0.slice() : new Float64Array(P);
    const c = fixedIters ? { ...cfg, iters: fixedIters, bestEpoch: false } : cfg;
    const info = cfg.solver === "lbfgs" ? lbfgs(p, rows, w, l2, W0, c, weights) : adam(p, rows, w, l2, W0, c, weights, check);
    return { w, info };
  };
  const logitsOf = (w: Float64Array, i: number) => {
    const z = new Float64Array(C);
    for (let c = 0; c < C; c++) { const o = c * D; let v = w[o + dim]; for (let j = 0; j < dim; j++) v += w[o + j] * X[i * dim + j]; z[c] = v; }
    return z;
  };
  const logProb = (z: Float64Array, t: number, c: number) => {
    let max = -Infinity, sum = 0;
    for (const v of z) if (v > max) max = v;
    for (const v of z) sum += Math.exp((v - max) / t);
    return (z[c] - max) / t - Math.log(sum);
  };
  const fitT = (held: { z: Float64Array; y: number }[]) => {
    let temperature = 1, bestNll = Infinity;
    for (let k = 0; k <= 120; k++) {
      const t = 0.01 * 1000 ** (k / 120);
      let nll = 0;
      for (const h of held) nll -= logProb(h.z, t, h.y);
      if (nll < bestNll) { bestNll = nll; temperature = Number(t.toPrecision(3)); }
    }
    return { temperature, nll: bestNll / Math.max(1, held.length) };
  };
  const grid = Array.isArray(cfg.l2) ? cfg.l2 : [cfg.l2 ?? 1e-4];
  let best: { l2: number; nll: number; temperature: number; iters: number } | null = null;
  const trail: unknown[] = [];
  for (const l2 of grid) {
    const held: { z: Float64Array; y: number }[] = [], its: number[] = [];
    for (const check of folds) {
      const inCheck = new Set(check), rows = Int32Array.from(idx.filter((i) => !inCheck.has(i)));
      const { w, info } = solve(rows, l2, Int32Array.from(check));
      its.push(info.iters);
      const seen = new Set(Array.from(rows, (i) => y[i]));
      for (const i of check) if (seen.has(y[i])) held.push({ z: logitsOf(w, i), y: y[i] });
    }
    const { temperature, nll } = fitT(held);
    let right = 0;
    for (const h of held) if (h.z.indexOf(Math.max(...h.z)) === h.y) right++;
    trail.push({ l2, nll: +nll.toFixed(4), holdAcc: +(right / Math.max(1, held.length)).toFixed(4), temperature, iters: its });
    if (!best || nll < best.nll) best = { l2, nll, temperature: held.length < 50 ? Math.max(temperature, floor) : temperature, iters: its.sort((a, b) => a - b)[its.length >> 1] };
  }
  const { w, info } = solve(Int32Array.from(idx), best!.l2, undefined, cfg.solver === "adam" ? best!.iters : undefined);
  const weightsOut = new Float32Array(C * dim), bias = new Float32Array(C);
  for (let c = 0; c < C; c++) { bias[c] = w[c * D + dim]; for (let j = 0; j < dim; j++) weightsOut[c * dim + j] = w[c * D + j]; }
  return { weights: weightsOut, bias, temperature: best!.temperature, dim, info: { chosenL2: best!.l2, finalIters: info.iters, trail } };
}
