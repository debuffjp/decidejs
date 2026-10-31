// The trainable part: a softmax layer over sentence embeddings. Kept free of the encoder so experiments can fit
// heads on cached embeddings in seconds.
import type { Head, TrainOptions } from "./index.ts";

const BATCH = 64;
const CENTROID_SCALE = 20;

// Lets a tab paint and handle input between training batches. A message-channel task, unlike setTimeout, isn't
// throttled to once a second (or minute) when the tab is in the background.
function yieldToPage(): Promise<void> {
  return new Promise((resolve) => {
    const { port1, port2 } = new MessageChannel();
    port1.onmessage = () => { port1.close(); resolve(); };
    port2.postMessage(null);
  });
}

export function softmax(z: ArrayLike<number>, t: number): number[] {
  let max = -Infinity;
  for (let i = 0; i < z.length; i++) max = Math.max(max, z[i]);
  const e = Array.from(z, (v) => Math.exp((v - max) / t));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export function logits(head: Pick<Head, "labels" | "dim" | "weights" | "bias">, x: ArrayLike<number>): Float64Array {
  const { labels, dim, weights, bias } = head;
  const z = new Float64Array(labels.length);
  for (let c = 0; c < labels.length; c++) {
    let s = bias[c];
    const row = c * dim;
    for (let j = 0; j < dim; j++) s += weights[row + j] * x[j];
    z[c] = s;
  }
  return z;
}


/** Fits weights, biases and temperature to embeddings `X` with label indices `y`. */
export async function fitHead<L extends string>(X: ArrayLike<number>[], y: number[], labels: L[], opts: TrainOptions = {}, priors?: ArrayLike<number>[], priorWeight = 1) {
  const { epochs = 60, learningRate = 2, l2 = 1e-4, holdout = 0.1, seed = 7, onEpoch } = opts;
  const dim = X[0].length, C = labels.length;
  let s = seed;
  const rand = () => (s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
  const idx = X.map((_, i) => i).sort(() => rand() - 0.5);
  const nVal = Math.max(1, Math.floor(idx.length * holdout));
  const val = idx.slice(0, nVal);

  // One training run on `rows`. With `check`, keeps the epoch that scores best on it.
  async function run(rows: number[], nEpochs: number, check?: number[]) {
    const head = { labels, dim, weights: new Float32Array(C * dim), bias: new Float32Array(C) };
    // Start each label at the normalised mean of its examples (nearest centroid), scaled like zero-shot
    // similarities. With a handful of examples per label this is most of the accuracy; gradient descent refines it.
    for (const i of rows) for (let j = 0; j < dim; j++) head.weights[y[i] * dim + j] += X[i][j];
    if (priors) for (let c = 0; c < C; c++) for (let j = 0; j < dim; j++) head.weights[c * dim + j] += priorWeight * priors[c][j];
    for (let c = 0; c < C; c++) {
      const row = head.weights.subarray(c * dim, (c + 1) * dim);
      const norm = Math.hypot(...row) || 1;
      for (let j = 0; j < dim; j++) row[j] = (row[j] / norm) * CENTROID_SCALE;
    }
    const acc = () => check!.filter((i) => { const z = logits(head, X[i]); return z.indexOf(Math.max(...z)) === y[i]; }).length / check!.length;
    let best = { acc: check ? acc() : 0, weights: head.weights.slice(), bias: head.bias.slice(), epochs: 0 };
    const gW = new Float64Array(C * dim), gb = new Float64Array(C);
    const order = [...rows];
    for (let ep = 0; ep < nEpochs; ep++) {
      order.sort(() => rand() - 0.5);
      for (let b = 0; b < order.length; b += BATCH) {
        const ids = order.slice(b, b + BATCH);
        gW.fill(0); gb.fill(0);
        for (const i of ids) {
          const p = softmax(logits(head, X[i]), 1);
          p[y[i]] -= 1;
          for (let c = 0; c < C; c++) { gb[c] += p[c]; for (let j = 0; j < dim; j++) gW[c * dim + j] += p[c] * X[i][j]; }
        }
        await yieldToPage();
        for (let c = 0; c < C; c++) {
          head.bias[c] -= (learningRate * gb[c]) / ids.length;
          for (let j = 0; j < dim; j++) { const k = c * dim + j; head.weights[k] -= learningRate * (gW[k] / ids.length + l2 * head.weights[k]); }
        }
      }
      if (!check) continue;
      const a = acc();
      onEpoch?.(ep, a);
      if (a > best.acc) best = { acc: a, weights: head.weights.slice(), bias: head.bias.slice(), epochs: ep + 1 };
    }
    return check ? { ...head, weights: best.weights, bias: best.bias, epochs: best.epochs } : { ...head, epochs: nEpochs };
  }

  // Pick the epoch count and temperature on held-out examples, then retrain on every example for that many epochs:
  // with 5 examples per label, a holdout would otherwise cost each label one of its five. A holdout under 200
  // examples gave a temperature off by up to 0.40 calibration error, so small sets use 5 folds instead: every example
  // is held out once and the temperature is fitted on all of them.
  const folds = nVal < 200 ? Array.from({ length: 5 }, (_, f) => idx.filter((_, i) => i % 5 === f)) : [val];
  const heldOut: { z: Float64Array; y: number }[] = [], foldEpochs: number[] = [];
  for (const check of folds) {
    const probe = await run(idx.filter((i) => !check.includes(i)), epochs, check);
    foldEpochs.push(probe.epochs);
    for (const i of check) heldOut.push({ z: logits(probe, X[i]), y: y[i] });
  }
  let temperature = 1, bestNll = Infinity;
  for (let t = 0.2; t <= 5; t += 0.05) {
    const nll = heldOut.reduce((sum, h) => sum - Math.log(softmax(h.z, t)[h.y] + 1e-12), 0);
    if (nll < bestNll) { bestNll = nll; temperature = Number(t.toFixed(2)); }
  }
  const head = await run(idx, foldEpochs.sort((a, b) => a - b)[foldEpochs.length >> 1]);
  return { labels, dim, weights: head.weights, bias: head.bias, temperature };
}
