// Decisions other than intents: four MTEB-style classification sets (emotion, toxicity, tweet sentiment, news topic),
// to check the "decide anything" claim and the settings tuned on intent sets. Per set: zero-shot from label names,
// then heads on 8 and 32 examples per label (3 seeds each), scored on 2,000 seeded test rows. Rows come from the
// Hugging Face datasets server and are cached under data/mteb. Not MTEB's own protocol (logistic regression on
// undersampled training data), so the numbers are not comparable with its leaderboard.
//   [MODEL=id] node --experimental-strip-types eval/mteb.ts emotion|toxic|tweet|agnews
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { createDecider, type Example } from "../src/index.ts";
import { kShot, seeded } from "./data.ts";

const SETS: Record<string, { dataset: string; names?: string[] }> = {
  emotion: { dataset: "mteb/emotion" },
  toxic: { dataset: "mteb/toxic_conversations_50k" },
  tweet: { dataset: "mteb/tweet_sentiment_extraction" },
  agnews: { dataset: "fancyzhx/ag_news", names: ["world", "sports", "business", "science_and_technology"] },
};
const name = process.argv[2];
const set = SETS[name];
if (!set) throw new Error(`usage: mteb.ts ${Object.keys(SETS).join("|")}`);

async function get(url: string): Promise<any> {
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) return r.json();
    if (attempt === 8) throw new Error(`${r.status} ${url}`);
    await new Promise((done) => setTimeout(done, 2000 * 2 ** attempt));
  }
}
// 20 pages of 100 rows at seeded offsets: some sets are sorted by label, so the first rows would be one label.
async function rows(split: string): Promise<Example<string>[]> {
  const file = `data/mteb/${name}-${split}.json`;
  try { return JSON.parse(await readFile(file, "utf8")); } catch {}
  const base = `https://datasets-server.huggingface.co/rows?dataset=${set.dataset}&config=default&split=${split}`;
  const total: number = (await get(`${base}&offset=0&length=1`)).num_rows_total;
  const rand = seeded(split === "train" ? 5 : 6), pages = Math.min(20, Math.ceil(total / 100));
  const offsets = [...new Set(Array.from({ length: pages }, () => Math.floor((rand() * total) / 100) * 100))];
  const out: Example<string>[] = [];
  for (const offset of offsets) {
    for (const { row } of (await get(`${base}&offset=${offset}&length=100`)).rows) {
      const label = (set.names ? set.names[row.label] : String(row.label_text)).replace(/\s+/g, "_");
      if (row.text?.trim()) out.push({ text: String(row.text).slice(0, 1000), label });
    }
  }
  await mkdir("data/mteb", { recursive: true });
  await writeFile(file, JSON.stringify(out));
  return out;
}

const train = await rows("train"), test = (await rows("test")).slice(0, 2000);
const labels = [...new Set([...train, ...test].map((r) => r.label))].sort();
const d = await createDecider({ model: process.env.MODEL });
const ece = (s: { right: boolean; confidence: number }[]) => {
  const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
  for (const x of s) { const b = bins[Math.min(14, Math.floor(x.confidence * 15))]; b.n++; b.conf += x.confidence; b.right += x.right ? 1 : 0; }
  return bins.reduce((sum, b) => sum + (b.n ? (b.n / s.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0);
};
async function score(using: Parameters<typeof d.decideMany>[1]) {
  const s = (await d.decideMany(test.map((r) => r.text), using)).map((p, i) => ({ right: p.choice === test[i].label, confidence: p.confidence }));
  const local = s.filter((x) => x.confidence >= 0.9);
  return {
    accuracy: +(s.filter((x) => x.right).length / s.length).toFixed(4), ece: +ece(s).toFixed(4),
    localAt90: +(local.length / s.length).toFixed(4), rightWhenLocal: +(local.filter((x) => x.right).length / Math.max(1, local.length)).toFixed(4),
  };
}
const majority = Math.max(...labels.map((l) => test.filter((r) => r.label === l).length)) / test.length;
const out = async (r: Record<string, unknown>) => { console.log(JSON.stringify(r)); await appendFile("eval/results-mteb.jsonl", JSON.stringify({ date: "2026-11-01", model: d.model, set: name, labels, majority: +majority.toFixed(4), ...r }) + "\n"); };

await out({ setup: "zero-shot names", ...(await score(labels)) });
for (const k of [8, 32]) {
  for (const seed of [7, 8, 9]) {
    const head = await d.train(kShot(train, k, seed));
    await out({ setup: `${k} per label`, seed, temperature: head.temperature, ...(await score(head)) });
  }
}
