// Heads trained on balanced few-shot examples assume every label is equally common. On imbalanced sets (toxicity: 7.6%
// toxic) that over-predicts the rare label. Re-weights each head's probabilities by label shares estimated from the
// training rows (p(c) times share(c), renormalised) and reports recall per label. Result (2026-11-01): accuracy on
// toxicity rose from 66.6% to 92.5%, but toxic recall fell to 0 to 38% (usually under 12%): the encoder barely
// separates the labels, so reweighting only turns the head into a majority guess. Not shipped as an option.
//   node --experimental-strip-types eval/prior-shift.ts toxic|tweet|emotion|agnews
import { appendFile, readFile } from "node:fs/promises";
import { createDecider, type Example } from "../src/index.ts";
import { kShot } from "./data.ts";

const name = process.argv[2];
const train: Example<string>[] = JSON.parse(await readFile(`data/mteb/${name}-train.json`, "utf8"));
const test: Example<string>[] = JSON.parse(await readFile(`data/mteb/${name}-test.json`, "utf8")).slice(0, 2000);
const labels = [...new Set([...train, ...test].map((r) => r.label))].sort();
const share = Object.fromEntries(labels.map((l) => [l, train.filter((r) => r.label === l).length / train.length]));
const d = await createDecider();
const recall = (s: { right: boolean }[], l: string) => +(s.filter((x, i) => x.right && test[i].label === l).length / Math.max(1, test.filter((r) => r.label === l).length)).toFixed(4);
for (const k of [8, 32]) for (const seed of [7, 8, 9]) {
  for (const shifted of [false, true]) {
    const head = await d.train(kShot(train, k, seed));
    const s = (await d.decideMany(test.map((r) => r.text), head)).map((p, i) => {
      const w = labels.map((l) => (p.probs as Record<string, number>)[l] * (shifted ? share[l] : 1));
      const sum = w.reduce((a, b) => a + b, 0), best = w.indexOf(Math.max(...w));
      return { right: labels[best] === test[i].label, confidence: w[best] / sum };
    });
    const local = s.filter((x) => x.confidence >= 0.9);
    const line = { date: "2026-11-01", set: name, k, seed, shifted, accuracy: +(s.filter((x) => x.right).length / s.length).toFixed(4), localAt90: +(local.length / s.length).toFixed(4), rightWhenLocal: +(local.filter((x) => x.right).length / Math.max(1, local.length)).toFixed(4), recall: Object.fromEntries(labels.map((l) => [l, recall(s, l)])) };
    console.log(JSON.stringify(line));
    await appendFile("eval/results-prior-shift.jsonl", JSON.stringify(line) + "\n");
  }
}
