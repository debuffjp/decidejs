// HWU64, a third intent set never used for tuning: do the settings picked on BANKING77 and CLINC150 (zero-shot
// temperature, prior weight, small-head floor) hold? Runs the public API end to end: zero-shot from label names, heads
// on 1, 5 and 10 examples per label (3 seeds) and on all 8,954 training examples, scored on the 1,076 test queries.
// Data: the DNNC/CPFT split (github.com/jianguoz/Few-Shot-Intent-Detection, Datasets/HWU64), saved under data/hwu64.
//   [MODEL=id] node --experimental-strip-types eval/hwu64.ts
import { appendFile, readFile } from "node:fs/promises";
import { createDecider, type Example } from "../src/index.ts";
import { kShot } from "./data.ts";

const split = async (s: string) => {
  const texts = (await readFile(`data/hwu64/${s}.seq.in`, "utf8")).trim().split(/\r?\n/);
  const labels = (await readFile(`data/hwu64/${s}.label`, "utf8")).trim().split(/\r?\n/);
  return texts.map((text, i) => ({ text: text.trim(), label: labels[i].trim() }));
};
const train = await split("train"), test = await split("test");
const labels = [...new Set(train.map((r) => r.label))].sort();
const d = await createDecider({ model: process.env.MODEL });

const ece = (s: { right: boolean; confidence: number }[]) => {
  const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
  for (const x of s) { const b = bins[Math.min(14, Math.floor(x.confidence * 15))]; b.n++; b.conf += x.confidence; b.right += x.right ? 1 : 0; }
  return bins.reduce((sum, b) => sum + (b.n ? (b.n / s.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0);
};
async function score(using: Parameters<typeof d.decideMany>[1]) {
  const s = (await d.decideMany(test.map((r) => r.text), using)).map((p, i) => ({ right: p.choice === test[i].label, confidence: p.confidence }));
  const local = s.filter((x) => x.confidence >= 0.9);
  return {
    accuracy: +(s.filter((x) => x.right).length / s.length).toFixed(4), ece: +ece(s).toFixed(4),
    localAt90: +(local.length / s.length).toFixed(4), rightWhenLocal: +(local.filter((x) => x.right).length / Math.max(1, local.length)).toFixed(4),
  };
}
const out = async (r: Record<string, unknown>) => { console.log(JSON.stringify(r)); await appendFile("eval/results-hwu64.jsonl", JSON.stringify({ date: "2026-11-01", model: d.model, ...r }) + "\n"); };

await out({ setup: "zero-shot names", ...(await score(labels)) });
for (const k of [1, 5, 10]) {
  for (const seed of [7, 8, 9]) {
    const head = await d.train(kShot(train, k, seed) as Example<string>[]);
    await out({ setup: `${k} per label`, seed, temperature: head.temperature, ...(await score(head)) });
  }
}
const head = await d.train(train);
await out({ setup: "all", examples: train.length, temperature: head.temperature, ...(await score(head)) });
