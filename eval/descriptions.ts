// Zero-shot with one-sentence label descriptions (eval/descriptions-<dataset>.json, written once from the label names
// alone, as an LLM would at setup) against plain readable names. Dev split as in eval/gains.ts.
//   node --experimental-strip-types eval/descriptions.ts [model] [banking77|clinc150]
import { appendFile, readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { softmax } from "../src/head.ts";
import { loadData, seeded } from "./data.ts";

const model = process.argv[2] ?? "Xenova/all-MiniLM-L6-v2";
const dataset = process.argv[3] ?? "banking77";
const { banking, clinc } = await loadData();
const rand = seeded(11);
const shuffled = [...banking.train].sort(() => rand() - 0.5);
const dev = dataset === "clinc150" ? clinc.val : shuffled.slice(Math.floor(shuffled.length * 0.8));
const desc: Record<string, string> = JSON.parse(await readFile(`eval/descriptions-${dataset}.json`, "utf8"));
const cache: Record<string, number[]> = JSON.parse(await readFile(`data/emb-${model.replace(/\W/g, "_")}.json`, "utf8"));
const d = await createDecider({ model });
const labels = Object.keys(desc).sort();
const rows = dev.filter((r) => r.label in desc);
const readable = (l: string) => l.replace(/[_-]+/g, " ").replace(/\?$/, "").toLowerCase();
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const variants: Record<string, string[][]> = {
  names: labels.map((l) => [readable(l)]),
  descriptions: labels.map((l) => [desc[l]]),
  "name+description": labels.map((l) => [readable(l), desc[l]]),
};
for (const [idea, texts] of Object.entries(variants)) {
  // A label with several texts scores by its best match.
  const vecs = await Promise.all(texts.map((t) => d.embed(t)));
  const sims = rows.map((r) => vecs.map((vs) => Math.max(...vs.map((v) => dot(v, cache[r.text])))));
  const mean = labels.map((_, c) => sims.reduce((s, z) => s + z[c], 0) / sims.length);
  for (const [shift, temp] of [[0, 0.02], [0, 0.025], [0, 0.03], [0, 0.035]]) {
    const s = sims.map((z, i) => { const p = softmax(z.map((v, c) => v - shift * mean[c]), temp); return { right: labels[p.indexOf(Math.max(...p))] === rows[i].label, confidence: Math.max(...p) }; });
    // Expected calibration error, 15 bins, as in eval/suite.ts.
    const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
    for (const x of s) { const b = bins[Math.min(14, Math.floor(x.confidence * 15))]; b.n++; b.conf += x.confidence; b.right += x.right ? 1 : 0; }
    const ece = bins.reduce((sum, b) => sum + (b.n ? (b.n / s.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0);
    const line = { model, date: "2026-11-01", dataset, idea, temperature: temp, accuracy: +(s.filter((x) => x.right).length / rows.length).toFixed(4), ece: +ece.toFixed(4) };
    console.log(JSON.stringify(line));
    await appendFile("eval/results-gains.jsonl", JSON.stringify(line) + "\n");
  }
}
