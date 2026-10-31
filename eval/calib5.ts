// Calibration at 5 examples per label, BANKING77 test, 3 seeds (cached embeddings when available).
import { readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits, softmax } from "../src/head.ts";
import { kShot, loadData } from "./data.ts";
const { banking } = await loadData();
let cache: Record<string, number[]> = JSON.parse(await readFile("data/emb-Xenova_all_MiniLM_L6_v2.json", "utf8"));
const missing = banking.test.filter((r) => !cache[r.text]).map((r) => r.text);
if (missing.length) { const d = await createDecider(); const v = await d.embed(missing); missing.forEach((t, i) => (cache[t] = v[i])); }
const labels = [...new Set(banking.train.map((r) => r.label))].sort();
for (const seed of [7, 8, 9]) {
  const tr = kShot(banking.train, 5, seed);
  const head = await fitHead(tr.map((r) => cache[r.text]), tr.map((r) => labels.indexOf(r.label)), labels, { seed });
  const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
  let right = 0;
  for (const r of banking.test) {
    const p = softmax(logits(head, cache[r.text]), head.temperature), c = Math.max(...p), ok = labels[p.indexOf(c)] === r.label;
    const b = bins[Math.min(14, Math.floor(c * 15))]; b.n++; b.conf += c; b.right += ok ? 1 : 0; right += ok ? 1 : 0;
  }
  const ece = bins.reduce((s, b) => s + (b.n ? (b.n / banking.test.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0);
  console.log(JSON.stringify({ seed, accuracy: right / banking.test.length, ece, temperature: head.temperature }));
}
