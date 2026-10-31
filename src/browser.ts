// The same decide() and train() as createDecider, running in a Web Worker so the page never freezes.
import type { DeciderOptions, Decision, Example, Head, TrainOptions } from "./index.ts";

/**
 * `createDecider()` in a Web Worker: same `decide()` and `train()`, plus `terminate()`. Each call costs about 1 ms
 * of message passing. Serve the page cross-origin isolated (COOP `same-origin`, COEP `credentialless`) so wasm
 * can use several threads. Pass your own `worker` if your bundler needs a different worker URL.
 */
export async function createWorkerDecider(options: DeciderOptions = {}, worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" })) {
  let next = 0;
  const pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void; progress?: (p: any) => void }>();
  worker.onmessage = ({ data }) => {
    const p = pending.get(data.id);
    if (!p) return;
    if (data.progress) return p.progress?.(data.progress);
    pending.delete(data.id);
    data.error ? p.reject(new Error(data.error)) : p.resolve(data.value);
  };
  const call = <T>(op: string, payload: object, progress?: (p: any) => void) =>
    new Promise<T>((resolve, reject) => {
      const id = next++;
      pending.set(id, { resolve, reject, progress });
      worker.postMessage({ id, op, ...payload });
    });

  await call("init", { options });
  return {
    decide: <const L extends string>(text: string, using: readonly L[] | Head<L>) => call<Decision<L>>("decide", { args: [text, using] }),
    train: <const L extends string>(examples: readonly Example<L>[], { onEpoch, ...opts }: TrainOptions = {}) =>
      call<Head<L>>("train", { args: [examples, opts] }, (p) => onEpoch?.(p.epoch, p.accuracy)),
    terminate: () => worker.terminate(),
  };
}
