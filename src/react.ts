// React hooks over createWorkerDecider(): the encoder loads in a worker once per component tree that calls
// useDecider(), and useDecision() decides as the text changes, dropping answers for text that has since changed.
import { useEffect, useState } from "react";
import { createWorkerDecider } from "./browser.ts";
import type { Choices, DeciderOptions, Decision } from "./index.ts";

export type WorkerDecider = Awaited<ReturnType<typeof createWorkerDecider>>;

export interface DeciderState {
  /** The decider once the model has loaded; `null` before that. */
  decider: WorkerDecider | null;
  status: "loading" | "ready" | "error";
  /** Bytes of model files downloaded so far, on a first visit. */
  progress: { loaded: number; total: number } | null;
  error: Error | null;
}

/**
 * Starts a worker decider when the component mounts and terminates it on unmount. Call it once near the top of
 * the tree and pass `decider` down: each call starts its own worker and loads its own copy of the model. Options are
 * read on mount; change `model` or `dtype` by remounting (a `key`).
 */
export function useDecider(options: Omit<DeciderOptions, "onProgress"> = {}): DeciderState {
  const [state, setState] = useState<DeciderState>({ decider: null, status: "loading", progress: null, error: null });
  useEffect(() => {
    let alive = true, made: WorkerDecider | undefined;
    const files = new Map<string, [number, number]>();
    createWorkerDecider({
      ...options,
      onProgress: ({ file, loaded, total }) => {
        files.set(file, [loaded, total]);
        let l = 0, t = 0;
        for (const [a, b] of files.values()) { l += a; t += b; }
        if (alive) setState((s) => ({ ...s, progress: { loaded: l, total: t } }));
      },
    }).then((d) => {
      made = d;
      if (alive) setState((s) => ({ ...s, decider: d, status: "ready" }));
      else d.terminate();
    }, (e) => alive && setState((s) => ({ ...s, status: "error", error: e instanceof Error ? e : new Error(String(e)) })));
    return () => { alive = false; made?.terminate(); };
    // Options are read once; see the JSDoc.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return state;
}

export interface DecisionState<D> {
  /** The decision for the latest text, or `null` before the first one (and for empty text). */
  decision: D | null;
  /** True while a decision for newer text is on its way. */
  pending: boolean;
  error: Error | null;
}

/**
 * Decides `text` whenever it (or `using`) changes, after `debounceMs` of quiet (default 150). Answers for text that
 * has changed since are dropped. Keep `using` stable (a constant, state or useMemo), or every render decides again.
 */
export function useDecision<const L extends string>(decider: WorkerDecider | null, text: string, using: Choices<L> | null | undefined, options?: { minConfidence?: undefined; debounceMs?: number }): DecisionState<Decision<L>>;
export function useDecision<const L extends string>(decider: WorkerDecider | null, text: string, using: Choices<L> | null | undefined, options: { minConfidence: number; debounceMs?: number }): DecisionState<Decision<L, L | null>>;
export function useDecision<const L extends string>(decider: WorkerDecider | null, text: string, using: Choices<L> | null | undefined, { minConfidence, debounceMs = 150 }: { minConfidence?: number; debounceMs?: number } = {}): DecisionState<Decision<L, L | null>> {
  const [state, setState] = useState<DecisionState<Decision<L, L | null>>>({ decision: null, pending: false, error: null });
  useEffect(() => {
    if (!decider || !using || !text.trim()) {
      setState({ decision: null, pending: false, error: null });
      return;
    }
    let current = true;
    setState((s) => ({ ...s, pending: true }));
    const timer = setTimeout(() => {
      decider.decide(text, using, { minConfidence }).then(
        (decision) => current && setState({ decision, pending: false, error: null }),
        (e) => current && setState({ decision: null, pending: false, error: e instanceof Error ? e : new Error(String(e)) }),
      );
    }, debounceMs);
    return () => { current = false; clearTimeout(timer); };
  }, [decider, text, using, minConfidence, debounceMs]);
  return state;
}
