// Head start-up on bge-small, dev splits, cached embeddings: how much weight the label-name prior gets, whether a
// one-sentence description is a better prior than the name, and the centroid scale.
//   node --experimental-strip-types eval/priors.ts [model]
import { appendFile, readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits, softmax } from "../src/head.ts";
import { kShot, loadData, seeded } from "./data.ts";

const model = process.argv[2] ?? "Xenova/bge-small-en-v1.5";
const { banking, clinc } = await loadData();
const rand = seeded(11);
const shuffled = [...banking.train].sort(() => rand() - 0.5);
const sets = { banking77: { train: shuffled.slice(0, Math.floor(shuffled.length * 0.8)), dev: shuffled.slice(Math.floor(shuffled.length * 0.8)) }, clinc150: { train: clinc.train, dev: clinc.val } };
const cache: Record<string, number[]> = JSON.parse(await readFile(`data/emb-${model.replace(/\W/g, "_")}.json`, "utf8"));
const d = await createDecider({ model });
const readable = (l: string) => l.replace(/[_-]+/g, " ").replace(/\?$/, "").toLowerCase();
for (const [dataset, { train, dev }] of Object.entries(sets)) {
  const desc: Record<string, string> = JSON.parse(await readFile(`eval/descriptions-${dataset}.json`, "utf8"));
  const labels = [...new Set(train.map((r) => r.label))].filter((l) => l in desc).sort();
  const rows = dev.filter((r) => labels.includes(r.label));
  const names = await d.embed(labels.map(readable)), descs = await d.embed(labels.map((l) => desc[l]));
  const both = names.map((n, c) => n.map((v, j) => v + descs[c][j]));
  const configs: [string, number[][] | undefined, number, number][] = [
    ["none", undefined, 0, 20], ["names w1", names, 1, 20], ["names w3", names, 3, 20], ["descriptions w3", descs, 3, 20],
    ["names+descriptions w3", both, 3, 20], ["names w3 scale 33", names, 3, 33], ["descriptions w3 scale 33", descs, 3, 33],
  ];
  for (const k of [1, 5, 10]) for (const [idea, priors, w, scale] of configs) {
    let right = 0, n = 0;
    for (const seed of [7, 8, 9]) {
      const ex = kShot(train.filter((r) => labels.includes(r.label)), k, seed);
      const head = await fitHead(ex.map((r) => cache[r.text]), ex.map((r) => labels.indexOf(r.label)), labels, { seed }, priors, w, scale);
      for (const r of rows) { const p = softmax(logits(head, cache[r.text]), head.temperature); right += labels[p.indexOf(Math.max(...p))] === r.label ? 1 : 0; n++; }
    }
    const line = { model, date: "2026-11-01", dataset, perLabel: k, idea, accuracy: +(right / n).toFixed(4) };
    console.log(JSON.stringify(line));
    await appendFile("eval/results-priors.jsonl", JSON.stringify(line) + "\n");
  }
}
