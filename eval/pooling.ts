// Mean vs CLS pooling for bge-small (it was trained with CLS), zero-shot with name plus description and 5-shot heads,
// dev splits. Embeds everything itself, so it takes a few minutes.
//   node --experimental-strip-types eval/pooling.ts
import { appendFile, readFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";
import { fitHead, logits, softmax } from "../src/head.ts";
import { kShot, loadData, seeded } from "./data.ts";

const model = "Xenova/bge-small-en-v1.5";
const fe: any = await (pipeline as any)("feature-extraction", model, { dtype: "q8" });
const { banking, clinc } = await loadData();
const rand = seeded(11);
const shuffled = [...banking.train].sort(() => rand() - 0.5);
const sets = { banking77: { train: shuffled.slice(0, Math.floor(shuffled.length * 0.8)), dev: shuffled.slice(Math.floor(shuffled.length * 0.8)) }, clinc150: { train: clinc.train, dev: clinc.val } };
const readable = (l: string) => l.replace(/[_-]+/g, " ").replace(/\?$/, "").toLowerCase();
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
async function embed(texts: string[], pooling: string) {
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += 64) out.push(...(await fe(texts.slice(i, i + 64), { pooling, normalize: true })).tolist());
  return out;
}
for (const [dataset, { train, dev }] of Object.entries(sets)) {
  const desc: Record<string, string> = JSON.parse(await readFile(`eval/descriptions-${dataset}.json`, "utf8"));
  const labels = [...new Set(train.map((r) => r.label))].filter((l) => l in desc).sort();
  const rows = dev.filter((r) => labels.includes(r.label));
  for (const pooling of ["mean", "cls"]) {
    const X = await embed(rows.map((r) => r.text), pooling);
    const N = await embed(labels.map(readable), pooling), D = await embed(labels.map((l) => desc[l]), pooling);
    for (const t of [0.02, 0.03, 0.05]) {
      const s = X.map((q, i) => { const p = softmax(labels.map((_, c) => Math.max(dot(N[c], q), dot(D[c], q))), t); return { right: labels[p.indexOf(Math.max(...p))] === rows[i].label, confidence: Math.max(...p) }; });
      const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
      for (const x of s) { const b = bins[Math.min(14, Math.floor(x.confidence * 15))]; b.n++; b.conf += x.confidence; b.right += x.right ? 1 : 0; }
      const ece = bins.reduce((sum, b) => sum + (b.n ? (b.n / s.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0);
      const line = { model, date: "2026-11-01", dataset, pooling, idea: "zero-shot name+description", temperature: t, accuracy: +(s.filter((x) => x.right).length / s.length).toFixed(4), ece: +ece.toFixed(4) };
      console.log(JSON.stringify(line)); await appendFile("eval/results-pooling.jsonl", JSON.stringify(line) + "\n");
    }
    let right = 0, n = 0;
    for (const seed of [7, 8, 9]) {
      const ex = kShot(train.filter((r) => labels.includes(r.label)), 5, seed);
      const E = await embed(ex.map((r) => r.text), pooling);
      const head = await fitHead(E, ex.map((r) => labels.indexOf(r.label)), labels, { seed }, N, 1);
      X.forEach((q, i) => { const p = softmax(logits(head, q), head.temperature); right += labels[p.indexOf(Math.max(...p))] === rows[i].label ? 1 : 0; n++; });
    }
    const line = { model, date: "2026-11-01", dataset, pooling, idea: "5-shot head, names w1", accuracy: +(right / n).toFixed(4) };
    console.log(JSON.stringify(line)); await appendFile("eval/results-pooling.jsonl", JSON.stringify(line) + "\n");
  }
}
