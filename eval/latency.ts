// Encoder latency on this machine's CPU (Node, q8): one text at a time and batched.
import { createDecider } from "../src/index.ts";
const texts = Array.from({ length: 60 }, (_, i) => `my card payment number ${i} still hasn't gone through after three days`);
for (const model of ["Xenova/all-MiniLM-L6-v2", "Xenova/bge-small-en-v1.5"]) {
  const d = await createDecider({ model });
  for (const t of texts.slice(0, 5)) await d.embed([t]);
  const ms: number[] = [];
  for (const t of texts) { const s = performance.now(); await d.embed([t]); ms.push(performance.now() - s); }
  ms.sort((a, b) => a - b);
  const s = performance.now(); await d.embed(texts); const batch = (performance.now() - s) / texts.length;
  console.log(model, "median single", ms[30].toFixed(1), "ms; p90", ms[54].toFixed(1), "ms; batched per text", batch.toFixed(1), "ms");
}
