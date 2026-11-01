// The shape of TypeSafe's SDK (`client.systemOne({ state, questions })`) on a local decider, so code written for Jev can
// run on the device and back again. Choice only: an encoder compares a text with options, so it has nothing to measure a
// lone Noul statement or a Score's ordered levels against.
import type { Choices, Decision, Head } from "./index.ts";

/** What this layer needs from a decider: `createDecider()` and `createWorkerDecider()` both fit. */
export interface Decides {
  decide<const L extends string>(text: string, using: Choices<L>): Promise<Decision<L>>;
}

export interface ChoiceQuestion<L extends string = string> {
  type: "choice";
  /** Kept for compatibility and ignored: decidekit compares the state with the options, it doesn't read a question. */
  instructions: string;
  /** Option names to descriptions (`null` for none), as in TypeSafe's API, or a trained head. */
  criteria: Readonly<Record<L, string | null>> | Head<L>;
}

export function choice<const L extends string>(instructions: string, criteria: Readonly<Record<L, string | null>> | Head<L>): ChoiceQuestion<L> {
  return { type: "choice", instructions, criteria };
}

const unsupported = (type: string) => (): never => {
  throw new Error(`decidekit: ${type} questions aren't supported locally, only choice. Ask a yes/no as a choice between two described options.`);
};
export const noul = unsupported("noul") as (instructions: string) => never;
export const score = unsupported("score") as (instructions: string, levels: unknown) => never;

export interface ChoiceAnswer<L extends string> {
  type: "choice";
  choice: L;
  /** TypeSafe's measure: 1 with all probability on one option, 0 when evenly spread, `(n * top - 1) / (n - 1)`. */
  confidence: number;
  probabilities: Record<L, number>;
}

/** A string, or an object whose string values are joined: braces and keys only add noise for an encoder. */
function stateText(state: string | Readonly<Record<string, unknown>>): string {
  if (typeof state === "string") return state;
  return Object.values(state).map((v) => {
    if (typeof v === "string") return v;
    if (typeof v === "number" || typeof v === "boolean") return String(v);
    throw new Error("decidekit: state values must be text, numbers or booleans");
  }).join("\n");
}

function options<L extends string>(criteria: ChoiceQuestion<L>["criteria"]): Choices<L> {
  if ("weights" in criteria) return criteria as Head<L>;
  const entries = Object.entries(criteria) as [L, string | null][];
  if (entries.every(([, d]) => d === null)) return entries.map(([l]) => l);
  // Same as a plain label: an option with no description is compared by its readable name.
  return Object.fromEntries(entries.map(([l, d]) => [l, d ?? l.replace(/[_-]+/g, " ")])) as Record<L, string>;
}

/**
 * Wraps a decider in TypeSafe's client shape. `confidence` follows TypeSafe's formula, so it reads lower than
 * `decide()`'s: 0.61 on the top of three options is 0.42 here. `Math.max(...probabilities)` is the calibrated chance of
 * being right. Thresholds carry over from neither model to the other.
 */
export function typesafeClient(decider: Decides) {
  return {
    async systemOne<const Q extends Record<string, ChoiceQuestion<string>>>({ state, questions }: { state: string | Readonly<Record<string, unknown>>; questions: Q }) {
      const text = stateText(state), answers: Record<string, ChoiceAnswer<string>> = {};
      for (const [id, q] of Object.entries(questions)) {
        const { choice, confidence, probabilities } = await decider.decide(text, options(q.criteria));
        const n = Object.keys(probabilities).length;
        answers[id] = { type: "choice", choice, confidence: Math.max(0, (n * confidence - 1) / (n - 1)), probabilities };
      }
      return { model: "decidekit", answers: answers as { [K in keyof Q]: Q[K] extends ChoiceQuestion<infer L> ? ChoiceAnswer<L> : never } };
    },
  };
}
