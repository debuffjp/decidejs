// Does decidekit work outside English? MASSIVE (60 intents, the same utterances translated per language), English label
// names, for bge-small-en (the default) and multilingual-e5-small (needs "query: " before every text). Per language:
// zero-shot accuracy, a head trained on 10 examples per label from that language, one trained on all of that
// language's training rows, and one trained on English rows only (train once, use in every language). Test: 2,000
// rows per language. Calibration error for the full head.
//   MODEL=Xenova/multilingual-e5-small PREFIX="query: " node --experimental-strip-types eval/multilingual.ts
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";
import { fitHead, logits, softmax } from "../src/head.ts";
import { seeded } from "./data.ts";

type Row = { text: string; label: string };
const MODEL = process.env.MODEL ?? "Xenova/bge-small-en-v1.5", PREFIX = process.env.PREFIX ?? "";
const LANGS = (process.env.LANGS ?? "en,ja,de,es,zh-CN").split(",");
async function get(url: string): Promise<any> {
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) return r.json();
    if (attempt === 8) throw new Error(`${r.status} ${url}`);
    await new Promise((done) => setTimeout(done, 2000 * 2 ** attempt));
  }
}
async function rows(lang: string, split: string, max: number): Promise<Row[]> {
  const file = `data/massive/${lang}-${split}.json`;
  try { return JSON.parse(await readFile(file, "utf8")); } catch {}
  const base = `https://datasets-server.huggingface.co/rows?dataset=mteb/amazon_massive_intent&config=${lang}&split=${split}`;
  const total: number = Math.min(max, (await get(`${base}&offset=0&length=1`)).num_rows_total);
  const out: Row[] = [];
  for (let offset = 0; offset < total; offset += 100)
    for (const { row } of (await get(`${base}&offset=${offset}&length=100`)).rows) out.push({ text: row.text, label: row.label });
  await mkdir("data/massive", { recursive: true });
  await writeFile(file, JSON.stringify(out));
  return out;
}

const embed = await (pipeline as any)("feature-extraction", MODEL, { dtype: "q8" });
async function vecs(texts: string[]): Promise<number[][]> {
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += 64) out.push(...(await embed(texts.slice(i, i + 64).map((t) => PREFIX + t), { pooling: "mean", normalize: true })).tolist());
  return out;
}
const top = (p: ArrayLike<number>) => { let b = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i; return b; };
const ece = (s: { right: boolean; c: number }[]) => {
  const bins = Array.from({ length: 15 }, () => ({ n: 0, c: 0, r: 0 }));
  for (const x of s) { const b = bins[Math.min(14, Math.floor(x.c * 15))]; b.n++; b.c += x.c; b.r += +x.right; }
  return bins.reduce((sum, b) => sum + (b.n ? (b.n / s.length) * Math.abs(b.c / b.n - b.r / b.n) : 0), 0);
};

const enTrain = await rows("en", "train", 12000);
const labels = [...new Set(enTrain.map((r) => r.label))].sort();
const names = await vecs(labels.map((l) => l.replace(/_/g, " ")));
const enX = await vecs(enTrain.map((r) => r.text)), enY = enTrain.map((r) => labels.indexOf(r.label));
const enHead = await fitHead(enX, enY, labels, { seed: 7 }, names);
for (const lang of LANGS) {
  const train = (await rows(lang, "train", 12000)).filter((r) => labels.includes(r.label));
  const test = (await rows(lang, "test", 2000)).filter((r) => labels.includes(r.label));
  const tX = await vecs(test.map((r) => r.text)), tY = test.map((r) => labels.indexOf(r.label));
  const acc = (pred: (x: number[]) => number) => +(tX.filter((x, i) => pred(x) === tY[i]).length / tX.length).toFixed(3);
  const X = lang === "en" ? enX : await vecs(train.map((r) => r.text)), Y = train.map((r) => labels.indexOf(r.label));
  const rand = seeded(1), seen = new Map<number, number>(), few: number[] = [];
  for (const i of [...Y.keys()].sort(() => rand() - 0.5)) if ((seen.get(Y[i]) ?? 0) < 10) { seen.set(Y[i], (seen.get(Y[i]) ?? 0) + 1); few.push(i); }
  const fewHead = await fitHead(few.map((i) => X[i]), few.map((i) => Y[i]), labels, { seed: 7 }, names);
  const fullHead = lang === "en" ? enHead : await fitHead(X, Y, labels, { seed: 7 }, names);
  const headAcc = (h: typeof enHead) => acc((x) => top(logits(h, x)));
  const line = {
    date: "2026-11-01", model: MODEL, prefix: PREFIX || undefined, lang, test: tX.length,
    zeroShot: acc((x) => top(names.map((n) => n.reduce((s, v, i) => s + v * x[i], 0)))),
    tenPerLabel: headAcc(fewHead), full: headAcc(fullHead), englishHead: headAcc(enHead),
    temperature: fullHead.temperature,
    fullMeanConf: +(tX.reduce((a, x) => a + Math.max(...softmax(logits(fullHead, x), fullHead.temperature)), 0) / tX.length).toFixed(3),
    fullEce: +ece(tX.map((x, i) => { const p = softmax(logits(fullHead, x), fullHead.temperature); return { right: top(p) === tY[i], c: Math.max(...p) }; })).toFixed(3),
  };
  console.log(JSON.stringify(line));
  await appendFile("eval/results-multilingual.jsonl", JSON.stringify(line) + "\n");
}
