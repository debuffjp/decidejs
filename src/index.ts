// decide(text, options): typed, calibrated decisions on device. Zero-shot compares the text with each option's
// embedding; a trained head (a softmax layer over the same embeddings, about 118 KB for 77 labels) is far more
// accurate. Both return the option literals as the choice type.
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import { fitHead, logits, softmax } from "./head.ts";

/** The result of `decide()`. */
export interface Decision<L extends string, C extends L | null = L> {
  /** The most likely option or label; `null` when below `minConfidence`. */
  choice: C;
  /**
   * Probability of `choice`, 0 to 1. Trained heads fit it on held-out examples. Zero-shot uses a fixed temperature
   * (0.05), which measured close to calibrated on two intent datasets (error 0.005-0.045 with 2 to 150 options);
   * check it on your own options before thresholding.
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

/**
 * What `decide()` chooses between: a list of labels, labels with descriptions (`{ refund: "wants their money back" }`,
 * where the description is what gets compared), or a trained head.
 */
export type Choices<L extends string> = readonly L[] | Readonly<Record<L, string>> | Head<L>;

const isHead = <L extends string>(c: Choices<L>): c is Head<L> => !Array.isArray(c) && (c as Head<L>).weights instanceof Float32Array;

const readable = (label: string) => label.replace(/[_-]+/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();

/** Options for one `decide()` call. */
export interface DecideOptions {
  /**
   * Below this confidence, `choice` is `null`: "none of these". Use it with trained heads, whose confidence is
   * calibrated. On CLINC150, 0.88 returned null for 81.7% of out-of-scope queries.
   */
  minConfidence?: number;
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

function toDecision<L extends string>(labels: readonly L[], p: number[]): Decision<L> {
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  const probs = Object.fromEntries(labels.map((l, i) => [l, p[i]])) as Record<L, number>;
  return { choice: labels[best], confidence: p[best], probs };
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
  async function decide<const L extends string>(text: string, using: Choices<L>, options?: { minConfidence?: undefined }): Promise<Decision<L>>;
  async function decide<const L extends string>(text: string, using: Choices<L>, options: DecideOptions): Promise<Decision<L, L | null>>;
  async function decide<const L extends string>(text: string, using: Choices<L>, { minConfidence }: DecideOptions = {}): Promise<Decision<L, L | null>> {
    const [q] = await vectors([text]);
    let d: Decision<L>;
    if (isHead(using)) {
      if (using.model !== model) throw new Error(`head was trained on ${using.model}, decider uses ${model}`);
      d = toDecision(using.labels, softmax(logits(using, q), using.temperature));
    } else {
      const labels = (Array.isArray(using) ? using : Object.keys(using)) as L[];
      // Bare labels are compared as readable text ("card_arrival" as "card arrival"): 2.6 points better on CLINC150's
      // dev split. Descriptions are compared as given.
      const texts = Array.isArray(using) ? labels.map(readable) : labels.map((l) => (using as Record<L, string>)[l]);
      const key = texts.join("\u0000");
      let opts = optionCache.get(key);
      if (!opts) optionCache.set(key, (opts = await vectors(texts)));
      const sims = opts.map((o) => o.reduce((s, x, i) => s + x * q[i], 0));
      d = toDecision(labels, softmax(sims, temperature));
    }
    return minConfidence !== undefined && d.confidence < minConfidence ? { ...d, choice: null } : d;
  }

  /**
   * Trains a head on labelled examples: mini-batch gradient descent on the embeddings, keeping the best holdout
   * epoch, then a temperature fitted on the holdout so `confidence` is calibrated. Every label needs at least a few
   * examples. Each label starts at the mean of its examples, so 5 per label already gave 77% on BANKING77 and
   * 86% on CLINC150 (see the README).
   */
  async function train<const L extends string>(examples: readonly Example<L>[], opts: TrainOptions = {}): Promise<Head<L>> {
    const labels = [...new Set(examples.map((e) => e.label))].sort() as L[];
    const y = examples.map((e) => labels.indexOf(e.label));
    const X = await vectors(examples.map((e) => e.text));

    return { model, ...(await fitHead(X, y, labels, opts)) };
  }

  return {
    model, decide, train,
    /** Embeds texts with the encoder: unit-length vectors, one per text. */
    embed: vectors,
  };
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
