// decide(text, options): typed, calibrated decisions on device. Zero-shot compares the text with each option's
// embedding; a trained head (a softmax layer over the same embeddings, about 118 KB for 77 labels) is far more
// accurate. Both return the option literals as the choice type.
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

/** The result of `decide()`. */
export interface Decision<L extends string> {
  /** The most likely option or label. */
  choice: L;
  /**
   * Probability of `choice`, 0 to 1. Calibrated for trained heads (fitted on held-out examples). Zero-shot
   * confidence uses a fixed temperature and reads close to 1 even when wrong, so don't threshold on it.
   */
  confidence: number;
  /** Probability of every option; sums to 1. */
  probs: Record<L, number>;
}

/** A trained head: row-major weights (labels × dim), biases and the fitted temperature. */
export interface Head<L extends string = string> {
  model: string;
  labels: readonly L[];
  dim: number;
  weights: Float32Array;
  bias: Float32Array;
  temperature: number;
}

/** One labelled example for `train()`. */
export interface Example<L extends string> {
  text: string;
  label: L;
}

export interface TrainOptions {
  /** Passes over the examples. Default 60; the best holdout epoch is kept. */
  epochs?: number;
  /** Default 0.5. */
  learningRate?: number;
  /** Weight decay. Default 1e-4. */
  l2?: number;
  /** Share of examples held out for early stopping and temperature fitting. Default 0.1. */
  holdout?: number;
  /** Seed for the holdout split and shuffling, so runs repeat. Default 7. */
  seed?: number;
  /** Called after each epoch with accuracy on the holdout. */
  onEpoch?: (epoch: number, holdoutAccuracy: number) => void;
}

export interface DeciderOptions {
  /** A Hugging Face sentence-embedding model with ONNX weights. Default `Xenova/all-MiniLM-L6-v2` (23.7 MB at q8). */
  model?: string;
  /** Defaults to transformers.js' choice: wasm in browsers, cpu in Node. */
  device?: "wasm" | "webgpu" | "cpu";
  /** Weight precision. Default `q8`. */
  dtype?: "q8" | "fp16" | "fp32";
  /** Softmax temperature for zero-shot similarities. Default 0.05. */
  temperature?: number;
}

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

function softmax(z: ArrayLike<number>, t: number): number[] {
  let max = -Infinity;
  for (let i = 0; i < z.length; i++) max = Math.max(max, z[i]);
  const e = Array.from(z, (v) => Math.exp((v - max) / t));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

function toDecision<L extends string>(labels: readonly L[], p: number[]): Decision<L> {
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  const probs = Object.fromEntries(labels.map((l, i) => [l, p[i]])) as Record<L, number>;
  return { choice: labels[best], confidence: p[best], probs };
}

function logits(head: Pick<Head, "labels" | "dim" | "weights" | "bias">, x: ArrayLike<number>): Float64Array {
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

/**
 * Loads the encoder (downloaded once, then cached by the browser or transformers.js) and returns `decide()` and
 * `train()`. In a browser, prefer `createWorkerDecider()` from `decidejs/browser`: the encoder blocks the thread
 * it runs on.
 */
export async function createDecider({ model = "Xenova/all-MiniLM-L6-v2", device, dtype = "q8", temperature = 0.05 }: DeciderOptions = {}) {
  const embed: FeatureExtractionPipeline = await (pipeline as any)("feature-extraction", model, { device, dtype });
  const optionCache = new Map<string, number[][]>();

  async function vectors(texts: readonly string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += BATCH) {
      const t = await embed(texts.slice(i, i + BATCH) as string[], { pooling: "mean", normalize: true });
      out.push(...(t.tolist() as number[][]));
    }
    return out;
  }

  /**
   * Picks one of `options` (zero-shot) or one of a trained head's labels. The result's `choice` has the literal type
   * of the options or labels. A head trained on a different model throws.
   */
  async function decide<const L extends string>(text: string, using: readonly L[] | Head<L>): Promise<Decision<L>> {
    const [q] = await vectors([text]);
    if ("weights" in using) {
      if (using.model !== model) throw new Error(`head was trained on ${using.model}, decider uses ${model}`);
      return toDecision(using.labels, softmax(logits(using, q), using.temperature));
    }
    const key = using.join("\u0000");
    let opts = optionCache.get(key);
    if (!opts) optionCache.set(key, (opts = await vectors(using)));
    const sims = opts.map((o) => o.reduce((s, x, i) => s + x * q[i], 0));
    return toDecision(using, softmax(sims, temperature));
  }

  /**
   * Trains a head on labelled examples: mini-batch gradient descent on the embeddings, keeping the best holdout
   * epoch, then a temperature fitted on the holdout so `confidence` is calibrated. Every label needs at least a few
   * examples. Each label starts at the mean of its examples, so 5 per label already gave 75% on BANKING77 and
   * 85% on CLINC150 (see the README).
   */
  async function train<const L extends string>(examples: readonly Example<L>[], opts: TrainOptions = {}): Promise<Head<L>> {
    const { epochs = 60, learningRate = 0.5, l2 = 1e-4, holdout = 0.1, seed = 7, onEpoch } = opts;
    const labels = [...new Set(examples.map((e) => e.label))].sort() as L[];
    const y = examples.map((e) => labels.indexOf(e.label));
    const X = await vectors(examples.map((e) => e.text));
    const dim = X[0].length, C = labels.length;

    let s = seed;
    const rand = () => (s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
    const idx = X.map((_, i) => i).sort(() => rand() - 0.5);
    const nVal = Math.max(1, Math.floor(idx.length * holdout));
    const val = idx.slice(0, nVal), fit = idx.slice(nVal);

    const head = { labels, dim, weights: new Float32Array(C * dim), bias: new Float32Array(C) };
    // Start each label at the normalised mean of its examples (nearest centroid), scaled like zero-shot similarities.
    // With a handful of examples per label this is most of the accuracy; gradient descent refines it.
    for (const i of fit) for (let j = 0; j < dim; j++) head.weights[y[i] * dim + j] += X[i][j];
    for (let c = 0; c < C; c++) {
      const row = head.weights.subarray(c * dim, (c + 1) * dim);
      const norm = Math.hypot(...row) || 1;
      for (let j = 0; j < dim; j++) row[j] = (row[j] / norm) * CENTROID_SCALE;
    }
    const acc = () => val.filter((i) => { const z = logits(head, X[i]); return z.indexOf(Math.max(...z)) === y[i]; }).length / val.length;
    let best = { acc: acc(), weights: head.weights.slice(), bias: head.bias.slice() };
    const gW = new Float64Array(C * dim), gb = new Float64Array(C);

    for (let ep = 0; ep < epochs; ep++) {
      fit.sort(() => rand() - 0.5);
      for (let b = 0; b < fit.length; b += BATCH) {
        const ids = fit.slice(b, b + BATCH);
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
      const a = acc();
      onEpoch?.(ep, a);
      if (a > best.acc) best = { acc: a, weights: head.weights.slice(), bias: head.bias.slice() };
    }
    head.weights = best.weights; head.bias = best.bias;

    let temperature = 1, bestNll = Infinity;
    for (let t = 0.2; t <= 5; t += 0.05) {
      const nll = val.reduce((sum, i) => sum - Math.log(softmax(logits(head, X[i]), t)[y[i]] + 1e-12), 0);
      if (nll < bestNll) { bestNll = nll; temperature = Number(t.toFixed(2)); }
    }
    return { model, ...head, temperature };
  }

  return { model, decide, train };
}

/** Serialises a head to JSON with base64 weights, for a static file next to the app. */
export function saveHead(head: Head): string {
  const b64 = (a: Float32Array) => {
    const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(s);
  };
  return JSON.stringify({ ...head, weights: b64(head.weights), bias: b64(head.bias) });
}

export function loadHead<L extends string = string>(json: string): Head<L> {
  const raw = JSON.parse(json);
  const f32 = (s: string) => new Float32Array(Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer);
  return { ...raw, weights: f32(raw.weights), bias: f32(raw.bias) };
}
