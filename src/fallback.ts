// Decide locally when confident; otherwise ask something slower (an LLM, a person, an API) and, optionally, keep its
// answers as training examples so the head needs it less over time. Can start from labels alone (zero-shot) and
// train its first head from the answers.
import type { Choices, Decision, Example, Head, TrainOptions } from "./index.ts";

/** What `withFallback()` needs from a decider: `createDecider()` and `createWorkerDecider()` both fit. */
export interface Trainable {
  decide<const L extends string>(text: string, using: Choices<L>): Promise<Decision<L>>;
  train<const L extends string>(examples: readonly Example<L>[], options?: TrainOptions<L>): Promise<Head<L>>;
}

export interface FallbackOptions<L extends string> {
  /** Below this confidence, call `ask`. Default 0.9. */
  minConfidence?: number;
  /**
   * Answers the texts the local decision isn't sure about: return one of `labels`, or `null` for none of them.
   * Answers that aren't one of the labels count as `null`.
   */
  ask: (text: string, labels: readonly L[], local: Decision<L>) => L | null | Promise<L | null>;
  /**
   * The examples a trained head was trained on. Pass them (or start from labels) to learn from `ask`: each answer
   * becomes an example, and `retrain()` trains on both.
   */
  examples?: readonly Example<L>[];
  /** Retrain on its own after this many new answers. Default: only when you call `retrain()`. */
  retrainEvery?: number;
  /**
   * Passed to `train()` when retraining. Starting from labels, `calibrateOn` defaults to the texts zero-shot answered
   * locally (up to 4 per label, once there are 50): they fit the head's temperature better than the LLM's answers,
   * which are sometimes wrong. Pass your own `calibrateOn` to replace them.
   */
  trainOptions?: Omit<TrainOptions<L>, "labels">;
}

/** A decision plus where it came from. `confidence` and `probs` are always the local decision's. */
export interface RoutedDecision<L extends string> extends Decision<L, L | null> {
  source: "local" | "fallback";
}

const isHead = (c: unknown): c is Head => typeof c === "object" && c !== null && "weights" in c;

/**
 * Routes each text to the local decision, or to `ask` when its confidence is below `minConfidence`.
 *
 * - With a trained head and its `examples`, answers from `ask` are kept as new examples.
 * - With labels (`["refund", "bug"]` or described labels), it starts zero-shot and always learns: `retrain()` (or
 *   `retrainEvery`) trains the first head from the answers. Labels with no answers yet start from their names.
 */
export function withFallback<const L extends string>(decider: Trainable, using: Choices<L>, options: FallbackOptions<L>) {
  const { minConfidence = 0.9, ask, retrainEvery, trainOptions } = options;
  const labels: readonly L[] = isHead(using) ? (using.labels as readonly L[]) : Array.isArray(using) ? using : (Object.keys(using) as L[]);
  const examples = options.examples ?? (isHead(using) ? undefined : []);
  const learned: Example<L>[] = [];
  let current: Choices<L> = using;
  let retraining: Promise<Head<L>> | undefined;
  let sinceRetrain = 0;
  // Zero-shot's confident answers: right about 93% of the time on BANKING77, and enough to calibrate on (eval/long-stream.ts).
  const confident: Example<L>[] = [], perLabel = new Map<L, number>();

  function retrain(): Promise<Head<L>> {
    if (!examples) return Promise.reject(new Error("decidejs: withFallback() needs `examples` to retrain a head"));
    if (!examples.length && !learned.length) return Promise.reject(new Error("decidejs: nothing to train on yet: no answers from `ask`"));
    const calibrateOn = trainOptions?.calibrateOn ?? (confident.length >= 50 ? confident : undefined);
    retraining ??= decider.train<L>([...examples, ...learned], { ...trainOptions, labels, calibrateOn }).then((h) => {
      current = h;
      return h;
    }).finally(() => (retraining = undefined));
    return retraining;
  }

  async function route(text: string): Promise<RoutedDecision<L>> {
    const local = await decider.decide(text, current);
    if (local.confidence >= minConfidence) {
      if (!isHead(current) && (perLabel.get(local.choice) ?? 0) < 4) {
        perLabel.set(local.choice, (perLabel.get(local.choice) ?? 0) + 1);
        confident.push({ text, label: local.choice });
      }
      return { ...local, source: "local" };
    }
    const answer = await ask(text, labels, local);
    const choice = answer !== null && labels.includes(answer) ? answer : null;
    if (examples && choice !== null) {
      learned.push({ text, label: choice });
      if (retrainEvery && ++sinceRetrain >= retrainEvery) {
        sinceRetrain = 0;
        // A failed background retrain keeps the current head; call retrain() yourself to see the error.
        retrain().catch(() => {});
      }
    }
    return { ...local, choice, source: "fallback" };
  }

  // defineProperties, not Object.assign: assign would copy the getters' values once.
  return Object.defineProperties(route as typeof route & {
    retrain: typeof retrain;
    /** The trained head in use, or `null` while still zero-shot. Replaced after each retrain. */
    readonly head: Head<L> | null;
    /** Answers from `ask` kept as examples since the start. */
    readonly learned: readonly Example<L>[];
  }, {
    retrain: { value: retrain },
    head: { get: () => (isHead(current) ? current : null) },
    learned: { get: () => learned },
  });
}
