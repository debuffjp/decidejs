// Trains a head through the library on BANKING77 and scores the test split, to check the library reproduces the
// spike's numbers (L6 87.7%). Usage: npm run eval -- [model]
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createDecider, loadHead, saveHead } from "../src/index.ts";

const CSV = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/";
async function rows(name: string): Promise<{ text: string; label: string }[]> {
  const file = `data/banking77-${name}.csv`;
  let text: string;
  try { text = await readFile(file, "utf8"); } catch {
    text = await (await fetch(CSV + name + ".csv")).text();
    await mkdir("data", { recursive: true });
    await writeFile(file, text);
  }
  // Quoted fields may contain commas, doubled quotes and newlines.
  const records = [...text.matchAll(/("(?:[^"]|"")*"|[^,\n]*),([^\n]*?)\r?(?:\n|$)/g)].slice(1);
  const out = records.filter((m) => m[2]).map(([, t, label]) => ({ text: t.startsWith('"') ? t.slice(1, -1).replace(/""/g, '"') : t, label }));
  return out;
}

const [train, test] = [await rows("train"), await rows("test")];
const decider = await createDecider({ model: process.argv[2] });
let t0 = performance.now();
const head = loadHead(saveHead(await decider.train(train)));
const trainSeconds = (performance.now() - t0) / 1000;
let right = 0;
t0 = performance.now();
for (const ex of test) if ((await decider.decide(ex.text, head)).choice === ex.label) right++;
console.log(JSON.stringify({
  model: decider.model, labels: head.labels.length, train: train.length, test: test.length,
  testAccuracy: right / test.length, temperature: head.temperature, trainSeconds,
  msPerDecision: (performance.now() - t0) / test.length, headJsonBytes: saveHead(head).length,
}, null, 2));
