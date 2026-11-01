// decidejs as a decision model for switch (github.com/osolmaz/switch), a harness that runs a decision model through a
// task by asking it typed questions. The types below copy the parts of switch's `Decider` that this needs, so decidejs
// doesn't depend on switch: pass the result wherever switch takes a decider.
import { withFallback, type Trainable } from "./fallback.ts";
import type { Choices, Decision, Head } from "./index.ts";

type Json = string | number | boolean | null | readonly Json[] | { readonly [key: string]: Json };
type Question =
  | { readonly type: "bool"; readonly instructions: string; readonly criteria: { readonly true: string; readonly false: string } }
  | { readonly type: "choice"; readonly instructions: string; readonly criteria: Readonly<Record<string, string>> }
  | { readonly type: "score"; readonly instructions: string; readonly criteria: readonly string[] };
type Answer =
  | { readonly type: "bool"; readonly probability: number }
  | { readonly type: "choice"; readonly choice: string; readonly probabilities: Readonly<Record<string, number>>; readonly confidence: number }
  | { readonly type: "score"; readonly score: number; readonly confidence: number };

/** switch's decision request: the state the model reads and the questions, by id. */
export interface SwitchRequest {
  readonly decisionState: { readonly [key: string]: Json };
  readonly decisionImages?: readonly unknown[];
  readonly decisionQuestions: Readonly<Record<string, Question>>;
}
/** switch's decision result, with the fields switch reads. */
export interface SwitchResult {
  readonly decisionAnswers: Readonly<Record<string, Answer>>;
  readonly decisionUsage?: { input: number; output: number; cacheRead: number; cacheWrite: number; totalTokens: number; cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number } };
  readonly api: string;
  readonly provider: string;
  readonly model: string;
  readonly stopReason: "stop" | "error" | "aborted";
  readonly errorMessage?: string;
  readonly timestamp: number;
  readonly tooLong: boolean;
  readonly milliseconds: number;
}
/** switch's `Decider`. */
export interface SwitchDecider {
  readonly model: string;
  readonly window: number;
  readonly charsPerToken: number;
  decide(request: SwitchRequest): Promise<SwitchResult>;
}

type Local = Trainable & { model: string; decide<const L extends string>(text: string, using: Choices<L>): Promise<Decision<L>> };

export interface SwitchOptions {
  /** Trained heads for questions that repeat, by question id. Questions without one are decided zero-shot. */
  heads?: Readonly<Record<string, Head>>;
  /**
   * A stronger decider to ask when decidejs isn't sure, such as Jev through switch's pi-ai decider. Its answers to
   * choice questions train a head for that question, so a question that repeats asks it less and less (see `withFallback`).
   */
  teacher?: SwitchDecider;
  /** Below this confidence, choice questions go to `teacher`. Default 0.9. */
  minConfidence?: number;
  /**
   * What decidejs reads for a question. Its encoder reads about 256 tokens and drops the rest, so a long state or long
   * instructions can push out what matters; pick it here, such as the target of a search. Default: `textOf`.
   */
  text?: (state: SwitchRequest["decisionState"], question: Question, id: string) => string;
}

/** The text decidejs reads for a question: its instructions, then the state as plain lines rather than JSON. */
export function textOf(instructions: string, state: Json): string {
  const lines: string[] = [];
  const walk = (v: Json, key: string) => {
    if (v === null || typeof v !== "object") lines.push(key ? `${key}: ${v}` : String(v));
    else if (Array.isArray(v) && v.every((x) => x === null || typeof x !== "object")) lines.push(key ? `${key}: ${v.join(", ")}` : v.join(", "));
    else for (const [k, x] of Object.entries(v)) walk(x as Json, Array.isArray(v) ? key : k);
  };
  walk(state, "");
  return [instructions, ...lines].filter(Boolean).join("\n");
}

/**
 * Wraps a decidejs decider as a switch decider. Choice, bool and score questions become decidejs decisions over the
 * question's criteria; images are ignored. Every call is local, so usage costs nothing.
 */
export function switchDecider(local: Local, options: SwitchOptions = {}): SwitchDecider {
  const { heads = {}, teacher, minConfidence = 0.9, text: textFor = (state, q) => textOf(q.instructions, state) } = options;
  // One router per repeating choice question, keyed by its instructions and options.
  const routers = new Map<string, ReturnType<typeof withFallback<string>>>();
  // The state each pending text came from, so the teacher reads the same request.
  const states = new Map<string, SwitchRequest["decisionState"]>();

  function routerFor(id: string, q: Extract<Question, { type: "choice" }>) {
    const key = JSON.stringify([q.instructions, q.criteria]);
    let r = routers.get(key);
    if (!r) {
      r = withFallback(local, heads[id] ?? (q.criteria as Record<string, string>), {
        minConfidence,
        retrainEvery: 8,
        ask: async (text) => {
          const res = await teacher!.decide({ decisionState: states.get(text)!, decisionQuestions: { [id]: q } });
          const a = res.decisionAnswers[id];
          return a?.type === "choice" ? a.choice : null;
        },
      });
      routers.set(key, r);
    }
    return r;
  }

  async function answer(id: string, q: Question, state: SwitchRequest["decisionState"]): Promise<Answer> {
    const text = textFor(state, q, id);
    if (q.type === "choice") {
      const keys = Object.keys(q.criteria);
      // decidejs compares two or more options; a single one is the answer.
      if (keys.length === 1) return { type: "choice", choice: keys[0], probabilities: { [keys[0]]: 1 }, confidence: 1 };
      if (teacher) states.set(text, state);
      const d = teacher ? await routerFor(id, q)(text).finally(() => states.delete(text)) : await local.decide(text, heads[id] ?? (q.criteria as Record<string, string>));
      const choice = d.choice ?? (Object.entries(d.probabilities) as [string, number][]).sort((a, b) => b[1] - a[1])[0][0];
      return { type: "choice", choice, probabilities: d.probabilities, confidence: d.confidence };
    }
    if (q.type === "bool") {
      const d = await local.decide(text, heads[id] ?? { true: q.criteria.true, false: q.criteria.false });
      return { type: "bool", probability: (d.probabilities as Record<string, number>).true };
    }
    const levels = Object.fromEntries(q.criteria.map((c, i) => [String(i), c]));
    const d = await local.decide(text, heads[id] ?? levels);
    return { type: "score", score: Number(d.choice), confidence: d.confidence };
  }

  return {
    model: `decidejs/${local.model}`,
    // The encoder reads about 256 tokens and cuts the rest, so a long state still fits; tell switch the window is big
    // so it doesn't trim the choices to fit.
    window: 1_000_000,
    charsPerToken: 4,
    async decide(request) {
      const started = performance.now();
      const base = { api: "decidejs", provider: "decidejs", model: `decidejs/${local.model}`, timestamp: Date.now(), tooLong: false };
      try {
        const entries = await Promise.all(Object.entries(request.decisionQuestions).map(async ([id, q]) => [id, await answer(id, q, request.decisionState)] as const));
        const input = Math.round(JSON.stringify(request.decisionState).length / 4);
        const zero = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };
        return {
          ...base,
          decisionAnswers: Object.fromEntries(entries),
          decisionUsage: { input, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: input, cost: zero },
          stopReason: "stop",
          milliseconds: performance.now() - started,
        };
      } catch (e) {
        return { ...base, decisionAnswers: {}, stopReason: "error", errorMessage: e instanceof Error ? e.message : String(e), milliseconds: performance.now() - started };
      }
    },
  };
}
export type { Question as SwitchQuestion, Answer as SwitchAnswer };
