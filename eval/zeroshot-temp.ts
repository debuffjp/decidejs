// Zero-shot calibration: ECE on dev splits for a range of fixed temperatures (readable labels, cached embeddings).
import { readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { softmax } from "../src/head.ts";
import { loadData, seeded } from "./data.ts";

const { banking, clinc } = await loadData();
const rand = seeded(11);
const bDev = [...banking.train].sort(() => rand() - 0.5).slice(Math.floor(banking.train.length * 0.8));
const cache: Record<string, number[]> = JSON.parse(await readFile("data/emb-Xenova_all_MiniLM_L6_v2.json", "utf8"));
const d = await createDecider();
const readable = (l: string) => l.replace(/[_-]+/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();
for (const [name, dev] of [["banking77", bDev], ["clinc150", clinc.val]] as const) {
  const labels = [...new Set(dev.map((r) => r.label))].sort();
  const L = await d.embed(labels.map(readable));
  const sims = dev.map((r) => L.map((o) => o.reduce((s, x, i) => s + x * cache[r.text][i], 0)));
  const out: string[] = [];
  for (const t of [0.01, 0.02, 0.03, 0.05, 0.08]) {
    const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
    dev.forEach((r, k) => {
      const p = softmax(sims[k], t), c = Math.max(...p), ok = labels[p.indexOf(c)] === r.label;
      const b = bins[Math.min(14, Math.floor(c * 15))]; b.n++; b.conf += c; b.right += ok ? 1 : 0;
    });
    out.push(`T${t}:${bins.reduce((s, b) => s + (b.n ? (b.n / dev.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0).toFixed(3)}`);
  }
  console.log(name, out.join(" "));
}

// Few options: the true label plus n-1 random others per query (CLINC150 dev).
{
  const dev = clinc.val, labels = [...new Set(dev.map((r) => r.label))].sort();
  const L = await d.embed(labels.map(readable));
  const r2 = seeded(5);
  for (const n of [2, 3, 5]) {
    const out: string[] = [];
    for (const t of [0.02, 0.05, 0.1, 0.2]) {
      const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
      let right = 0;
      for (const r of dev) {
        const opts = [labels.indexOf(r.label)];
        while (opts.length < n) { const o = Math.floor(r2() * labels.length); if (!opts.includes(o)) opts.push(o); }
        const p = softmax(opts.map((o) => L[o].reduce((s, x, i) => s + x * cache[r.text][i], 0)), t);
        const c = Math.max(...p), ok = p.indexOf(c) === 0;
        const b = bins[Math.min(14, Math.floor(c * 15))]; b.n++; b.conf += c; b.right += ok ? 1 : 0; right += ok ? 1 : 0;
      }
      out.push(`T${t}:${bins.reduce((s, b) => s + (b.n ? (b.n / dev.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0).toFixed(3)}`);
      if (t === 0.05) out.push(`acc:${(right / dev.length).toFixed(3)}`);
    }
    console.log(`options${n}`, out.join(" "));
  }
}
