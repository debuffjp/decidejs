// Zero-shot variants on dev splits (cached embeddings): how the option text is phrased.
import { readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { loadData, seeded } from "./data.ts";

const { banking, clinc } = await loadData();
const rand = seeded(11);
const bDev = [...banking.train].sort(() => rand() - 0.5).slice(Math.floor(banking.train.length * 0.8));
const cache: Record<string, number[]> = JSON.parse(await readFile("data/emb-Xenova_all_MiniLM_L6_v2.json", "utf8"));
const d = await createDecider();
const readable = (l: string) => l.replace(/[_-]+/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();
const variants: Record<string, (l: string) => string> = {
  raw: (l) => l,
  readable,
  template: (l) => `This is about ${readable(l)}.`,
};
for (const [name, dev] of [["banking77", bDev], ["clinc150", clinc.val]] as const) {
  const labels = [...new Set(dev.map((r) => r.label))].sort();
  const row: Record<string, number | string> = { dataset: name };
  for (const [v, fn] of Object.entries(variants)) {
    const L = await d.embed(labels.map(fn));
    const right = dev.filter((r) => {
      const q = cache[r.text];
      const sims = L.map((o) => o.reduce((s, x, i) => s + x * q[i], 0));
      return labels[sims.indexOf(Math.max(...sims))] === r.label;
    }).length;
    row[v] = Number((right / dev.length).toFixed(4));
  }
  console.log(JSON.stringify(row));
}
