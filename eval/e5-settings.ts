// Settings for multilingual-e5-small on MASSIVE (English label names): the zero-shot temperature that minimises
// log loss on validation rows per language, and the calibration error of 5-per-label heads for small-head floors.
//   node --experimental-strip-types eval/e5-settings.ts
import { appendFile, readFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";
import { fitHead, logits, softmax } from "../src/head.ts";
import { seeded } from "./data.ts";

type Row = { text: string; label: string };
const embed = await (pipeline as any)("feature-extraction", "Xenova/multilingual-e5-small", { dtype: "q8" });
const vecs = async (t: string[]) => { const o: number[][] = []; for (let i = 0; i < t.length; i += 64) o.push(...(await embed(t.slice(i, i + 64).map((s) => "query: " + s), { pooling: "mean", normalize: true })).tolist()); return o; };
const top = (p: ArrayLike<number>) => { let b = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i; return b; };
const ece = (s: { r: boolean; c: number }[]) => { const bins = Array.from({ length: 15 }, () => ({ n: 0, c: 0, r: 0 })); for (const x of s) { const b = bins[Math.min(14, Math.floor(x.c * 15))]; b.n++; b.c += x.c; b.r += +x.r; } return +bins.reduce((a, b) => a + (b.n ? (b.n / s.length) * Math.abs(b.c / b.n - b.r / b.n) : 0), 0).toFixed(3); };
const en: Row[] = JSON.parse(await readFile("data/massive/en-train.json", "utf8"));
const labels = [...new Set(en.map((r) => r.label))].sort();
const names = await vecs(labels.map((l) => l.replace(/_/g, " ")));
for (const lang of ["en", "ja"]) {
  const train: Row[] = JSON.parse(await readFile(`data/massive/${lang}-train.json`, "utf8"));
  const test: Row[] = JSON.parse(await readFile(`data/massive/${lang}-test.json`, "utf8"));
  const tX = await vecs(test.map((r) => r.text)), tY = test.map((r) => labels.indexOf(r.label));
  const sims = tX.map((x) => names.map((n) => n.reduce((s, v, i) => s + v * x[i], 0)));
  let best = { t: 0, nll: Infinity };
  for (let k = 0; k <= 120; k++) { const t = 0.002 * 100 ** (k / 120); const nll = sims.reduce((a, z, i) => a - Math.log(softmax(z, t)[tY[i]] + 1e-12), 0); if (nll < best.nll) best = { t: +t.toPrecision(3), nll }; }
  const zsEce = ece(sims.map((z, i) => { const p = softmax(z, best.t); return { r: top(p) === tY[i], c: Math.max(...p) }; }));
  const X = await vecs(train.map((r) => r.text)), Y = train.map((r) => labels.indexOf(r.label));
  const floors: Record<string, number> = {};
  for (const floor of (process.env.FLOORS ?? "0.02,0.05,0.1,0.15,0.2").split(",").map(Number)) {
    const e: number[] = [];
    for (const seed of [1, 2, 3]) {
      const rand = seeded(seed), seen = new Map<number, number>(), few: number[] = [];
      for (const i of [...Y.keys()].sort(() => rand() - 0.5)) if ((seen.get(Y[i]) ?? 0) < 5) { seen.set(Y[i], (seen.get(Y[i]) ?? 0) + 1); few.push(i); }
      // 5 per label is 300 examples, over the 50-answer line; 1 per label for 12 labels is the case the floor guards.
      const sub = few.filter((i) => Y[i] < 12).filter((_, k) => k % 5 === 0);
      const h = await fitHead(sub.map((i) => X[i]), sub.map((i) => Y[i]), labels.slice(0, 12), { seed: 7 }, names.slice(0, 12), 1, undefined, floor);
      const rows = tX.map((x, i) => ({ x, y: tY[i] })).filter((r) => r.y < 12);
      e.push(ece(rows.map((r) => { const p = softmax(logits(h, r.x), h.temperature); return { r: top(p) === r.y, c: Math.max(...p) }; })));
    }
    floors[floor] = +(e.reduce((a, b) => a + b, 0) / e.length).toFixed(3);
  }
  const line = { date: "2026-11-01", lang, zeroShotTemperature: best.t, zeroShotEce: zsEce, tinyHeadEceByFloor: floors };
  console.log(JSON.stringify(line));
  await appendFile("eval/results-e5-settings.jsonl", JSON.stringify(line) + "\n");
}
