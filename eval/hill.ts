// Fast head experiments on cached embeddings. Scores only on dev splits (CLINC150 val, a seeded 20% of BANKING77
// train), never test. Usage: npm run eval:hill -- [model]  (first run embeds everything and caches it in data/)
import { readFile, writeFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits } from "../src/head.ts";
import { kShot, loadData, seeded, type Row } from "./data.ts";

const model = process.argv[2] ?? "Xenova/all-MiniLM-L6-v2";
const { banking, clinc } = await loadData();
const rand = seeded(11);
const shuffled = [...banking.train].sort(() => rand() - 0.5);
const cut = Math.floor(shuffled.length * 0.8);
const sets = { bTrain: shuffled.slice(0, cut), bDev: shuffled.slice(cut), cTrain: clinc.train, cDev: clinc.val };

const cacheFile = `data/emb-${model.replace(/\W/g, "_")}.json`;
let cache: Record<string, number[]>;
try { cache = JSON.parse(await readFile(cacheFile, "utf8")); } catch {
  const d = await createDecider({ model });
  const texts = [...new Set(Object.values(sets).flat().map((r) => r.text))];
  const vecs = await d.embed(texts);
  cache = Object.fromEntries(texts.map((t, i) => [t, vecs[i].map((v) => Math.round(v * 1e5) / 1e5)]));
  await writeFile(cacheFile, JSON.stringify(cache));
}

// PRIOR=w adds each label's readable name as w extra examples in the centroid start.
const priorWeight = Number(process.env.PRIOR ?? 0);
const labelVec = new Map<string, number[]>();
if (priorWeight) {
  const d = await createDecider({ model });
  const all = [...new Set([...sets.bTrain, ...sets.cTrain].map((r) => r.label))];
  const v = await d.embed(all.map((l) => l.replace(/[_-]+/g, " ").toLowerCase()));
  all.forEach((l, i) => labelVec.set(l, v[i]));
}
async function accuracy(train: Row[], dev: Row[], seed = 7) {
  const labels = [...new Set(train.map((r) => r.label))].sort();
  const priors = priorWeight ? labels.map((l) => labelVec.get(l)!) : undefined;
  const head = await fitHead(train.map((r) => cache[r.text]), train.map((r) => labels.indexOf(r.label)), labels, { seed }, priors, priorWeight);
  return dev.filter((r) => { const z = logits(head, cache[r.text]); return labels[z.indexOf(Math.max(...z))] === r.label; }).length / dev.length;
}
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
const few = (train: Row[], dev: Row[]) => Promise.all([7, 8, 9].map((s) => accuracy(kShot(train, 5, s), dev, s))).then(mean);

const t0 = performance.now();
const r = {
  b77Full: await accuracy(sets.bTrain, sets.bDev), b77Five: await few(sets.bTrain, sets.bDev),
  clincFull: await accuracy(sets.cTrain, sets.cDev), clincFive: await few(sets.cTrain, sets.cDev),
};
console.log(JSON.stringify({ score: mean(Object.values(r)), ...r, seconds: Math.round((performance.now() - t0) / 1000) }));
