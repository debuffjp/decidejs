// Embeds every text the accuracy spike needs once, through the public API, so heads can be compared on identical
// vectors. Writes data/spike/emb-<model>.bin (Float32 rows) and .json (the texts). Scores nothing.
//   [MODEL=id] node --experimental-strip-types eval/spike/embed.ts
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createDecider } from "../../src/index.ts";
import { loadData } from "../data.ts";

const { banking, clinc } = await loadData();
const hwu = async (s: string) => (await readFile(`data/hwu64/${s}.seq.in`, "utf8")).trim().split(/\r?\n/).map((t) => t.trim());
const hwuLabels = (await readFile("data/hwu64/train.label", "utf8")).trim().split(/\r?\n/).map((l) => l.trim());
const readable = (l: string) => l.replace(/[_-]+/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();
const labelNames = [...banking.train, ...clinc.train].map((r) => r.label).concat(hwuLabels).map(readable);
const texts = [...new Set([
  ...[banking.train, banking.test, clinc.train, clinc.val, clinc.test].flat().map((r) => r.text),
  ...(await hwu("train")), ...(await hwu("test")), ...labelNames,
])];
if (process.env.LOCAL_MODELS) {
  const { env } = await import("@huggingface/transformers");
  env.localModelPath = process.env.LOCAL_MODELS;
  env.allowRemoteModels = false;
}
const d = await createDecider({ model: process.env.MODEL });
const t0 = Date.now();
const rows = await d.embed(texts);
const buf = new Float32Array(texts.length * rows[0].length);
rows.forEach((r: ArrayLike<number>, i: number) => buf.set(r, i * rows[0].length));
await mkdir("data/spike", { recursive: true });
const name = d.model.replace(/\W/g, "_");
await writeFile(`data/spike/emb-${name}.bin`, new Uint8Array(buf.buffer));
await writeFile(`data/spike/emb-${name}.json`, JSON.stringify({ model: d.model, dim: rows[0].length, texts }));
console.log(JSON.stringify({ model: d.model, n: texts.length, dim: rows[0].length, seconds: (Date.now() - t0) / 1000, name }));
