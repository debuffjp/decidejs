// Accuracy experiments before release, on dev splits only (CLINC150 val, a seeded 20% of BANKING77 train), using the
// embeddings eval/hill.ts cached. Writes one JSON line per (model, idea) to eval/results-gains.jsonl.
//   node --experimental-strip-types eval/gains.ts [model]
// Ideas: zero-shot with contextual calibration (subtract each label's mean similarity over unlabelled inputs), and a
// few-shot head mixed with a nearest-neighbour vote over the training examples.
import { appendFile, readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits, softmax } from "../src/head.ts";
import { kShot, loadData, seeded, type Row } from "./data.ts";

const model = process.argv[2] ?? "Xenova/all-MiniLM-L6-v2";
const { banking, clinc } = await loadData();
const rand = seeded(11);
const shuffled = [...banking.train].sort(() => rand() - 0.5);
const cut = Math.floor(shuffled.length * 0.8);
const sets = { banking77: { train: shuffled.slice(0, cut), dev: shuffled.slice(cut) }, clinc150: { train: clinc.train, dev: clinc.val } };
const cache: Record<string, number[]> = JSON.parse(await readFile(`data/emb-${model.replace(/\W/g, "_")}.json`, "utf8"));
const d = await createDecider({ model });
const readable = (l: string) => l.replace(/[_-]+/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const argmax = (p: ArrayLike<number>) => { let b = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i; return b; };
const ece = (s: { right: boolean; confidence: number }[]) => {
  const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
  for (const x of s) { const b = bins[Math.min(14, Math.floor(x.confidence * 15))]; b.n++; b.conf += x.confidence; b.right += x.right ? 1 : 0; }
  return bins.reduce((sum, b) => sum + (b.n ? (b.n / s.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0);
};
const summary = (s: { right: boolean; confidence: number }[]) => ({
  accuracy: +(s.filter((x) => x.right).length / s.length).toFixed(4),
  ece: +ece(s).toFixed(4),
  // What withFallback cares about: share answered locally at 0.9, and how often those answers are right.
  localAt90: +(s.filter((x) => x.confidence >= 0.9).length / s.length).toFixed(4),
  rightWhenLocal: +(s.filter((x) => x.confidence >= 0.9 && x.right).length / Math.max(1, s.filter((x) => x.confidence >= 0.9).length)).toFixed(4),
});
const out = async (r: Record<string, unknown>) => { console.log(JSON.stringify(r)); await appendFile("eval/results-gains.jsonl", JSON.stringify({ model, date: "2026-11-01", ...r }) + "\n"); };

for (const [dataset, { train, dev }] of Object.entries(sets)) {
  const labels = [...new Set(train.map((r) => r.label))].sort();
  const L = await d.embed(labels.map(readable));
  const devX = dev.map((r) => cache[r.text]);
  const sims = devX.map((q) => L.map((o) => dot(o, q)));

  // Zero-shot. Contextual calibration: a label that sits close to everything (a generic name) wins too often;
  // subtracting its mean similarity over the inputs removes that. The mean comes from the dev texts themselves,
  // without their labels, as an app could from the texts it has seen.
  const mean = labels.map((_, c) => sims.reduce((s, z) => s + z[c], 0) / sims.length);
  for (const [idea, shift] of [["zero-shot", 0], ["zero-shot+context", 1], ["zero-shot+half-context", 0.5]] as const) {
    for (const t of [0.05, 0.03, 0.02]) {
      const s = sims.map((z, i) => {
        const p = softmax(z.map((v, c) => v - shift * mean[c]), t);
        return { right: labels[argmax(p)] === dev[i].label, confidence: Math.max(...p) };
      });
      await out({ dataset, idea, temperature: t, ...summary(s) });
    }
  }

  // Few-shot: the trained head alone, then mixed with a k-nearest-neighbour vote over the training embeddings.
  for (const k of [5, 10]) {
    const runs: Record<string, { right: boolean; confidence: number }[]> = {};
    for (const seed of [7, 8, 9]) {
      const rows: Row[] = kShot(train, k, seed);
      const X = rows.map((r) => cache[r.text]), y = rows.map((r) => labels.indexOf(r.label));
      const head = await fitHead(X, y, labels, { seed }, L, 1);
      devX.forEach((q, i) => {
        const p = softmax(logits(head, q), head.temperature);
        const nn = X.map((x, j) => [dot(x, q), y[j]] as const).sort((a, b) => b[0] - a[0]);
        for (const nk of [3, 5, 10]) {
          const vote = new Float64Array(labels.length);
          for (const [s, c] of nn.slice(0, nk)) vote[c] += Math.exp((s - nn[0][0]) / 0.05);
          const vs = vote.reduce((a, b) => a + b, 0);
          for (const a of [0, 0.25, 0.5]) {
            if (a === 0 && nk !== 3) continue;
            const m = p.map((v, c) => (1 - a) * v + a * vote[c] / vs);
            (runs[a === 0 ? "head" : `head+knn${nk}@${a}`] ??= []).push({ right: labels[argmax(m)] === dev[i].label, confidence: Math.max(...m) });
          }
        }
      });
    }
    for (const [idea, s] of Object.entries(runs)) await out({ dataset, perLabel: k, idea, ...summary(s) });
  }
}
