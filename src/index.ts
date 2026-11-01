// decide(text, options): typed, calibrated decisions on device. Zero-shot compares the text with each option's
// embedding; a trained head (a softmax layer over the same embeddings, 156 KB for 77 labels) is far more
// accurate. Both return the option literals as the choice type.
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import { fitHead, logits, softmax } from "./head.ts";

/** The result of `decide()`. */
export interface Decision<L extends string, C extends L | null = L> {
  /** The most likely option or label; `null` when below `minConfidence`. */
  choice: C;
  /**
   * Probability of `choice`, 0 to 1. Trained heads fit it on held-out examples. Zero-shot uses a fixed temperature
   * per model (0.03 for the default bge-small, 0.05 for all-MiniLM-L6-v2), which measured close to calibrated on two
   * intent datasets (error 0.03-0.04 on their dev splits); check it on your own options before thresholding.
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

function isHead<L extends string>(c: Choices<L>): c is Head<L> {
  if (Array.isArray(c) || !("weights" in c)) return false;
  if (!((c as Head<L>).weights instanceof Float32Array)) throw new Error("decidejs: this head's weights aren't a Float32Array. Read head files with loadHead(), not JSON.parse.");
  return true;
}

const readable = (label: string) => label.replace(/[_-]+/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();

/** Options for one `decide()` call. */
export interface DecideOptions {
  /**
   * Below this confidence, `choice` is `null`: "none of these". Use it with trained heads, whose confidence is
   * calibrated. On CLINC150, 0.93 returned null for 86.3% of out-of-scope queries.
   */
  minConfidence?: number;
}

/** One labelled example for `train()`. */
export interface Example<L extends string> {
  text: string;
  label: L;
}

export interface TrainOptions<L extends string = string> {
  /** Use label names as a hint (default true). Turn off when labels are opaque ids like `c17`. */
  labelNames?: boolean;
  /**
   * Labels to include even without examples yet. They start from their readable name alone, as zero-shot does, and
   * can still be chosen. Needs `labelNames`.
   */
  labels?: readonly L[];
  /** Passes over the examples. Default 60; the best holdout epoch is kept. */
  epochs?: number;
  /** Default 2. */
  learningRate?: number;
  /** Weight decay. Default 1e-4. */
  l2?: number;
  /** Share of examples held out for early stopping and temperature fitting. Default 0.1. */
  holdout?: number;
  /** Seed for the holdout split and shuffling, so runs repeat. Default 7. */
  seed?: number;
  /**
   * Examples known to be right, used only to fit the temperature (how far to trust the head). Pass them when the
   * training examples come from a source that is sometimes wrong, such as an LLM through `withFallback()`: fitted on
   * those, the temperature learns to doubt the head. 2 per label is enough (eval/long-stream.ts).
   */
  calibrateOn?: readonly Example<L>[];
  /** Called after each epoch with accuracy on the holdout. */
  onEpoch?: (epoch: number, holdoutAccuracy: number) => void;
}

export interface DeciderOptions {
  /**
   * A Hugging Face sentence-embedding model with ONNX weights. Default `Xenova/bge-small-en-v1.5` (34 MB at q8).
   * Pre-release builds defaulted to `Xenova/all-MiniLM-L6-v2` (23 MB): smaller and 1.6x faster, 3 to 5 points less
   * accurate. Pass it to keep using heads trained on it.
   */
  model?: string;
  /** Defaults to transformers.js' choice: wasm in browsers, cpu in Node. */
  device?: "wasm" | "webgpu" | "cpu";
  /** Weight precision. Default `q8`. */
  dtype?: "q8" | "fp16" | "fp32";
  /** Called as model files download (first load only; later loads come from cache). */
  onProgress?: (progress: { file: string; loaded: number; total: number }) => void;
  /**
   * Softmax temperature for zero-shot similarities. Default 0.03 for bge-small (0.025 for described labels, which
   * score higher), 0.05 for other models. Setting it uses the same value for both.
   */
  temperature?: number;
}

const BATCH = 64;
const LABEL_NAME_WEIGHT = 1;
// Zero-shot temperatures that measured calibrated on the BANKING77 and CLINC150 dev splits (eval/gains.ts). Models
// differ in how spread out their similarities are: at 0.05, bge-small answered only 3% of texts at 0.9 confidence.
// Described labels get their own: on bge-small, 0.025 halved their calibration error on CLINC150 (0.066 to 0.020) and
// raised it on BANKING77 (0.032 to 0.042); names alone measured best at 0.03 (eval/descriptions.ts).
// Small heads (under 50 held-out answers) can't fit a trustworthy temperature, so it has a floor. On bge-small a floor
// of 1 left them underconfident (calibration error 0.09 to 0.13; only 43 to 69% of texts at 0.9, nearly all right);
// 0.5 gave 0.009 to 0.017. MiniLM's similarities are spread differently and keep 1 (eval/small-floor.ts).
const SMALL_HEAD_FLOOR: Record<string, number> = { "Xenova/bge-small-en-v1.5": 0.5 };
const TEMPERATURES: Record<string, [names: number, described: number]> = { "Xenova/bge-small-en-v1.5": [0.03, 0.025], "Xenova/all-MiniLM-L6-v2": [0.05, 0.05] };

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
export async function createDecider({ model = "Xenova/bge-small-en-v1.5", device, dtype = "q8", temperature, onProgress }: DeciderOptions = {}) {
  const embed: FeatureExtractionPipeline = await (pipeline as any)("feature-extraction", model, {
    device, dtype,
    progress_callback: onProgress && ((p: any) => p.status === "progress" && onProgress({ file: p.file, loaded: p.loaded, total: p.total })),
  });
  const optionCache = new Map<string, number[][]>();
  // The last training set's embeddings, so retraining after adding or fixing a few examples only encodes those.
  let trainCache = new Map<string, number[]>();

  async function vectors(texts: readonly string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += BATCH) {
      const t = await embed(texts.slice(i, i + BATCH) as string[], { pooling: "mean", normalize: true });
      out.push(...(t.tolist() as number[][]));
    }
    return out;
  }

  async function score<L extends string>(q: number[], using: Choices<L>, minConfidence?: number): Promise<Decision<L, L | null>> {
    let d: Decision<L>;
    if (isHead(using)) {
      if (using.model !== model) throw new Error(`decidejs: this head was trained on ${using.model}, but the decider uses ${model}. Pass { model: "${using.model}" } to createDecider.`);
      d = toDecision(using.labels, softmax(logits(using, q), using.temperature));
    } else {
      const labels = (Array.isArray(using) ? using : Object.keys(using)) as L[];
      if (labels.length < 2) throw new Error(`decidejs: decide() needs at least 2 options, got ${labels.length}`);
      // Bare labels are compared as readable text ("card_arrival" as "card arrival"): 2.6 points better on CLINC150's
      // dev split. A described label scores by the better of its name and its description: on the dev splits that
      // beat the description alone by 0.5 (BANKING77) and 3.1 points (CLINC150), and names alone by 7 and 10.
      const texts = Array.isArray(using) ? labels.map(readable) : labels.flatMap((l) => [readable(l), (using as Record<L, string>)[l]]);
      const key = texts.join("\u0000");
      let opts = optionCache.get(key);
      if (!opts) optionCache.set(key, (opts = await vectors(texts)));
      const all = opts.map((o) => o.reduce((s, x, i) => s + x * q[i], 0));
      const sims = Array.isArray(using) ? all : labels.map((_, c) => Math.max(all[2 * c], all[2 * c + 1]));
      const [names, described] = temperature !== undefined ? [temperature, temperature] : TEMPERATURES[model] ?? [0.05, 0.05];
      d = toDecision(labels, softmax(sims, Array.isArray(using) ? names : described));
    }
    return minConfidence !== undefined && d.confidence < minConfidence ? { ...d, choice: null } : d;
  }

  /**
   * Picks one of a list of labels or described labels (zero-shot), or one of a trained head's labels. The result's
   * `choice` has the literal type of the labels. A head trained on a different model throws.
   */
  async function decide<const L extends string>(text: string, using: Choices<L>, options?: { minConfidence?: undefined }): Promise<Decision<L>>;
  async function decide<const L extends string>(text: string, using: Choices<L>, options: DecideOptions): Promise<Decision<L, L | null>>;
  async function decide<const L extends string>(text: string, using: Choices<L>, { minConfidence }: DecideOptions = {}): Promise<Decision<L, L | null>> {
    const [q] = await vectors([text]);
    return score(q, using, minConfidence);
  }

  /** `decide()` for many texts, encoded in batches of 64: much faster than one call per text in Node. */
  async function decideMany<const L extends string>(texts: readonly string[], using: Choices<L>, options?: { minConfidence?: undefined }): Promise<Decision<L>[]>;
  async function decideMany<const L extends string>(texts: readonly string[], using: Choices<L>, options: DecideOptions): Promise<Decision<L, L | null>[]>;
  async function decideMany<const L extends string>(texts: readonly string[], using: Choices<L>, { minConfidence }: DecideOptions = {}): Promise<Decision<L, L | null>[]> {
    const qs = await vectors(texts);
    return Promise.all(qs.map((q) => score(q, using, minConfidence)));
  }

  /**
   * Trains a head on labelled examples: mini-batch gradient descent on the embeddings, keeping the best holdout
   * epoch, then a temperature fitted on the holdout so `confidence` is calibrated. Every label needs at least a few
   * examples. Each label starts at the mean of its examples and its name, so 5 per label already gave 79% on BANKING77 and
   * 87% on CLINC150 (see the README).
   */
  async function train<const L extends string>(examples: readonly Example<L>[], opts: TrainOptions<L> = {}): Promise<Head<L>> {
    const bad = examples.findIndex((e) => typeof e?.text !== "string" || typeof e?.label !== "string" || !e.label);
    if (bad >= 0) throw new Error(`decidejs: example ${bad} needs a string text and a non-empty string label, got ${JSON.stringify(examples[bad])}`);
    if (!examples.length) throw new Error("decidejs: train() needs at least 1 example");
    const labels = [...new Set([...(opts.labels ?? []), ...examples.map((e) => e.label)])].sort() as L[];
    if (opts.labels && opts.labelNames === false && labels.some((l) => !examples.some((e) => e.label === l))) {
      throw new Error("decidejs: labels without examples need labelNames, which is what they start from");
    }
    if (labels.length < 2) throw new Error(`decidejs: train() needs at least 2 labels, got ${labels.length}`);
    const y = examples.map((e) => labels.indexOf(e.label));
    const texts = [...new Set(examples.map((e) => e.text))];
    const fresh = texts.filter((t) => !trainCache.has(t));
    const encoded = await vectors(fresh);
    const cache = new Map(texts.map((t) => [t, trainCache.get(t)!]));
    fresh.forEach((t, i) => cache.set(t, encoded[i]));
    trainCache = cache;
    const X = examples.map((e) => cache.get(e.text)!);

    // Each label's readable name counts as 1 extra example in the starting point. With bge-small that beat no prior
    // by 1 to 14 points at 1 to 5 examples per label on the dev splits, and beat weight 3 (the MiniLM setting) by 0.9
    // to 2.5 points (eval/priors.ts).
    const priors = opts.labelNames === false ? undefined : await vectors(labels.map(readable));
    const cal = opts.calibrateOn;
    const unknown = cal?.find((e) => !labels.includes(e.label));
    if (unknown) throw new Error(`decidejs: calibrateOn has label ${JSON.stringify(unknown.label)}, which isn't one of the head's labels`);
    const calibration = cal?.length ? { X: await vectors(cal.map((e) => e.text)), y: cal.map((e) => labels.indexOf(e.label)) } : undefined;
    return { model, ...(await fitHead(X, y, labels, opts, priors, LABEL_NAME_WEIGHT, undefined, SMALL_HEAD_FLOOR[model] ?? 1, calibration)) };
  }

  return {
    model, decide, decideMany, train,
    /** Embeds texts with the encoder: unit-length vectors, one per text. */
    embed: vectors,
  };
}

export { saveHead, loadHead } from "./head-io.ts";
export { withFallback, NONE, type FallbackOptions, type RoutedDecision, type Trainable } from "./fallback.ts";
export { describeLabels, describePrompt, parseDescriptions, type DescribeOptions } from "./describe.ts";
