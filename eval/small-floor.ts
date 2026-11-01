// The temperature floor for small heads (fewer than 50 held-out answers). Many small heads: 12-label subsets of
// BANKING77 and 10-label subsets of CLINC150, 1 to 4 examples per label, 6 seeds each, scored on dev messages for
// those labels. For each floor: calibration error, share answered at 0.9 or more, and how often those are right.
//   node --experimental-strip-types eval/small-floor.ts [model]
import { appendFile, readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits, softmax } from "../src/head.ts";
import { kShot, loadData, seeded } from "./data.ts";

const model = process.argv[2] ?? "Xenova/bge-small-en-v1.5";
const cache: Record<string, number[]> = JSON.parse(await readFile(`data/emb-${model.replace(/\W/g, "_")}.json`, "utf8"));
const { banking, clinc } = await loadData();
const rand = seeded(11);
const shuffled = [...banking.train].sort(() => rand() - 0.5);
const sets = { banking77: { train: shuffled.slice(0, Math.floor(shuffled.length * 0.8)), dev: shuffled.slice(Math.floor(shuffled.length * 0.8)), size: 12 }, clinc150: { train: clinc.train, dev: clinc.val, size: 10 } };
const d = await createDecider({ model });
const floors = [1, 0.7, 0.5, 0.3, 0];
for (const [dataset, { train, dev, size }] of Object.entries(sets)) {
  const allLabels = [...new Set(train.map((r) => r.label))].sort();
  for (const k of [1, 2, 4]) {
    const out: Record<number, { right: boolean; confidence: number }[]> = Object.fromEntries(floors.map((f) => [f, []]));
    for (let seed = 1; seed <= 6; seed++) {
      const pick = seeded(seed * 97);
      const labels = [...allLabels].sort(() => pick() - 0.5).slice(0, size).sort();
      const ex = kShot(train.filter((r) => labels.includes(r.label)), k, seed);
      const rows = dev.filter((r) => labels.includes(r.label));
      const names = await d.embed(labels.map((l) => l.replace(/[_-]+/g, " ")));
      for (const f of floors) {
        const head = await fitHead(ex.map((r) => cache[r.text]), ex.map((r) => labels.indexOf(r.label)), labels, { seed }, names, 1, 20, f);
        for (const r of rows) { const p = softmax(logits(head, cache[r.text]), head.temperature), c = Math.max(...p); out[f].push({ right: labels[p.indexOf(c)] === r.label, confidence: c }); }
      }
    }
    for (const f of floors) {
      const s = out[f];
      const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
      for (const x of s) { const b = bins[Math.min(14, Math.floor(x.confidence * 15))]; b.n++; b.conf += x.confidence; b.right += x.right ? 1 : 0; }
      const ece = bins.reduce((sum, b) => sum + (b.n ? (b.n / s.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0);
      const sure = s.filter((x) => x.confidence >= 0.9);
      const line = { model, date: "2026-11-01", dataset, perLabel: k, floor: f, accuracy: +(s.filter((x) => x.right).length / s.length).toFixed(3), ece: +ece.toFixed(3), at90: +(sure.length / s.length).toFixed(3), rightAt90: +(sure.filter((x) => x.right).length / Math.max(1, sure.length)).toFixed(3) };
      console.log(JSON.stringify(line));
      await appendFile("eval/results-small-floor.jsonl", JSON.stringify(line) + "\n");
    }
  }
}
