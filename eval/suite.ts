// Beyond one clean benchmark: accuracy with few examples per label (BANKING77, CLINC150), and whether confidence can
// flag out-of-scope requests (CLINC150's 1,000 out-of-scope test queries). Writes eval/results.json.
// Usage: npm run eval:suite -- [model] [out.json]
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createDecider, type Example, type Head } from "../src/index.ts";

type Row = Example<string>;
const cached = async (file: string, url: string) => {
  try { return await readFile(file, "utf8"); } catch {
    const text = await (await fetch(url)).text();
    await mkdir("data", { recursive: true });
    await writeFile(file, text);
    return text;
  }
};
const csv = (text: string): Row[] => [...text.matchAll(/("(?:[^"]|"")*"|[^,\n]*),([^\n]*?)\r?(?:\n|$)/g)].slice(1)
  .filter((m) => m[2]).map(([, t, label]) => ({ text: t.startsWith('"') ? t.slice(1, -1).replace(/""/g, '"') : t, label }));
const B77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/";
const banking = { train: csv(await cached("data/banking77-train.csv", B77 + "train.csv")), test: csv(await cached("data/banking77-test.csv", B77 + "test.csv")) };
const clincRaw = JSON.parse(await cached("data/clinc150.json", "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"));
const pairs = (a: [string, string][]): Row[] => a.map(([text, label]) => ({ text, label }));
const clinc = { train: pairs(clincRaw.train), val: pairs(clincRaw.val), test: pairs(clincRaw.test), oosVal: pairs(clincRaw.oos_val), oosTest: pairs(clincRaw.oos_test) };

function kShot(rows: Row[], k: number, seed: number): Row[] {
  let s = seed;
  const rand = () => (s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
  const by = new Map<string, Row[]>();
  for (const r of [...rows].sort(() => rand() - 0.5)) (by.get(r.label) ?? by.set(r.label, []).get(r.label)!).push(r);
  return [...by.values()].flatMap((g) => g.slice(0, k));
}

const d = await createDecider({ model: process.argv[2] });
const score = async (head: Head<string>, rows: Row[]) => {
  const out: { right: boolean; confidence: number }[] = [];
  for (const r of rows) { const x = await d.decide(r.text, head); out.push({ right: x.choice === r.label, confidence: x.confidence }); }
  return out;
};
const acc = (s: { right: boolean }[]) => s.filter((x) => x.right).length / s.length;
// Expected calibration error, 15 equal-width confidence bins.
const ece = (s: { right: boolean; confidence: number }[]) => {
  const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
  for (const x of s) { const b = bins[Math.min(14, Math.floor(x.confidence * 15))]; b.n++; b.conf += x.confidence; b.right += x.right ? 1 : 0; }
  return bins.reduce((sum, b) => sum + (b.n ? (b.n / s.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0);
};

const results: Record<string, unknown>[] = [];
const log = (r: Record<string, unknown>) => { results.push(r); console.error(JSON.stringify(r)); };

// Few-shot curve: 3 seeds per k (holdout needs at least 1 per label, so k starts at 5).
for (const [name, data] of [["banking77", banking], ["clinc150", clinc]] as const) {
  for (const k of [5, 10, 25, Infinity]) {
    const seeds = k === Infinity ? [7] : [7, 8, 9];
    const accs: number[] = [], eces: number[] = [];
    for (const seed of seeds) {
      const head = await d.train(k === Infinity ? data.train : kShot(data.train, k, seed), { seed, labelNames: process.env.LABEL_NAMES !== "0" });
      const s = await score(head, data.test);
      accs.push(acc(s)); eces.push(ece(s));
    }
    const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
    log({ dataset: name, perLabel: k === Infinity ? "all" : k, accuracy: accs, mean: mean(accs), ece: eces, meanEce: mean(eces) });
  }
}

// Out-of-scope: train on in-scope only, pick the confidence threshold on validation, apply it to test.
const head = await d.train(clinc.train);
const [valIn, valOut, testIn, testOut] = await Promise.all([score(head, clinc.val), score(head, clinc.oosVal), score(head, clinc.test), score(head, clinc.oosTest)]);
let best = { t: 0, f: -1 };
for (let t = 0; t <= 1; t += 0.01) {
  // Balanced: in-scope accuracy (rejecting counts as wrong) averaged with out-of-scope recall.
  const f = (valIn.filter((x) => x.right && x.confidence >= t).length / valIn.length + valOut.filter((x) => x.confidence < t).length / valOut.length) / 2;
  if (f > best.f) best = { t, f };
}
const t = best.t;
log({
  dataset: "clinc150-oos", threshold: Number(t.toFixed(2)),
  inScopeAccuracyNoThreshold: acc(testIn),
  inScopeAccuracyAtThreshold: testIn.filter((x) => x.right && x.confidence >= t).length / testIn.length,
  oosRecall: testOut.filter((x) => x.confidence < t).length / testOut.length,
  oosAcceptedAsInScope: testOut.filter((x) => x.confidence >= t).length / testOut.length,
});
await writeFile(process.argv[3] ?? "eval/results.json", JSON.stringify({ date: new Date().toISOString(), model: d.model, results }, null, 2) + "\n");
