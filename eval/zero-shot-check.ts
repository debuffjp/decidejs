// Does the replay's zero-shot (eval/real-stream.ts: precomputed embeddings, max of name and description) match
// decideMany() on the same 12 labels? Accuracy of each on the recorded stream's in-scope messages and on the test split.
//   node --experimental-strip-types eval/zero-shot-check.ts
import { readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { softmax } from "../src/head.ts";
import { loadData } from "./data.ts";

const recorded: { text: string; label: string | null }[] = JSON.parse(await readFile("eval/teacher-8b-stream.json", "utf8"));
const labels = [...new Set(recorded.filter((m) => m.label).map((m) => m.label!))].sort();
const all: Record<string, string> = JSON.parse(await readFile("eval/descriptions-banking77.json", "utf8"));
const described = Object.fromEntries(labels.map((l) => [l, all[l]]));
const { banking } = await loadData();
const d = await createDecider();
const names = await d.embed(labels.map((l) => l.replace(/_/g, " ")));
const descs = await d.embed(labels.map((l) => all[l]));
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
for (const [name, set] of [["stream", recorded.filter((m) => m.label)], ["test", banking.test.filter((r) => labels.includes(r.label))]] as const) {
  const texts = set.map((r) => r.text);
  const vecs = await d.embed(texts);
  const replay = vecs.filter((x, i) => { const p = softmax(labels.map((_, c) => Math.max(dot(names[c], x), dot(descs[c], x))), 0.025); return labels[p.indexOf(Math.max(...p))] === set[i].label; }).length;
  const many = (await d.decideMany(texts, described)).filter((r, i) => r.choice === set[i].label).length;
  console.log(JSON.stringify({ set: name, n: texts.length, replay: +(replay / texts.length).toFixed(4), decideMany: +(many / texts.length).toFixed(4) }));
}
