// Runs a decider off the main thread. The encoder's wasm work never gives the page a turn (measured: a 171 s freeze
// while training in Chrome), so in a browser it belongs here.
import { createDecider, type DecideOptions, type DeciderOptions } from "./index.ts";

type Request =
  | { id: number; op: "init"; options: DeciderOptions }
  | { id: number; op: "decide"; args: [string, unknown, DecideOptions | undefined] }
  | { id: number; op: "decideMany"; args: [string[], unknown, DecideOptions | undefined] }
  | { id: number; op: "train"; args: [unknown[], Record<string, unknown>] };

let decider: Awaited<ReturnType<typeof createDecider>> | undefined;

self.onmessage = async ({ data }: MessageEvent<Request>) => {
  try {
    let value: unknown;
    if (data.op === "init") decider = await createDecider({ ...data.options, onProgress: (progress) => self.postMessage({ id: data.id, progress }) });
    else if (!decider) throw new Error("decider not initialised");
    else if (data.op === "decideMany") value = await decider.decideMany(data.args[0], data.args[1] as any, data.args[2] ?? {});
    else if (data.op === "decide") value = await decider.decide(data.args[0], data.args[1] as any, data.args[2] ?? {});
    else value = await decider.train(data.args[0] as any, {
      ...data.args[1],
      onEpoch: (epoch, accuracy) => self.postMessage({ id: data.id, progress: { epoch, accuracy } }),
    });
    self.postMessage({ id: data.id, value });
  } catch (e) {
    self.postMessage({ id: data.id, error: String(e) });
  }
};
