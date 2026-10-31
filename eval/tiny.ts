// Calibration with very little data: many small tasks sampled from CLINC150 (2 to 10 labels, k examples each),
// scored on the validation queries of those labels. Uses the hill cache. Usage: node --experimental-strip-types eval/tiny.ts
import { readFile } from "node:fs/promises";
import { fitHead, logits, softmax } from "../src/head.ts";
import { loadData, seeded, kShot } from "./data.ts";

const { clinc } = await loadData();
const cache: Record<string, number[]> = JSON.parse(await readFile("data/emb-Xenova_all_MiniLM_L6_v2.json", "utf8"));
const allLabels = [...new Set(clinc.train.map((r) => r.label))].sort();
const rand = seeded(3);

for (const k of [1, 2, 3, 5]) {
  const bins = Array.from({ length: 10 }, () => ({ n: 0, conf: 0, right: 0 }));
  let n = 0, right = 0, confSum = 0;
  for (let task = 0; task < 60; task++) {
    const nLabels = 2 + Math.floor(rand() * 9);
    const labels = [...allLabels].sort(() => rand() - 0.5).slice(0, nLabels).sort();
    const train = kShot(clinc.train.filter((r) => labels.includes(r.label)), k, task);
    const dev = clinc.val.filter((r) => labels.includes(r.label));
    const head = await fitHead(train.map((r) => cache[r.text]), train.map((r) => labels.indexOf(r.label)), labels, { seed: task });
    for (const r of dev) {
      const p = softmax(logits(head, cache[r.text]), head.temperature);
      const best = p.indexOf(Math.max(...p));
      const ok = labels[best] === r.label ? 1 : 0;
      const b = bins[Math.min(9, Math.floor(p[best] * 10))];
      b.n++; b.conf += p[best]; b.right += ok;
      n++; right += ok; confSum += p[best];
    }
  }
  const ece = bins.reduce((s, b) => s + (b.n ? Math.abs(b.conf - b.right) : 0), 0) / n;
  console.log(JSON.stringify({ k, accuracy: +(right / n).toFixed(3), meanConfidence: +(confSum / n).toFixed(3), ece: +ece.toFixed(3) }));
}
