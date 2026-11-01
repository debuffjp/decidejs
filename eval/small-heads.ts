// How confident are small heads? Train on the first n gold-labelled lanes messages (seeded order) and report the
// fitted temperature, the share of other messages answered at 0.9 or more, and how often those are right, against
// zero-shot with descriptions. Explains why early retrains in eval/replay-teacher.ts sent more to the LLM.
import { readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits, softmax } from "../src/head.ts";
import { loadData, seeded } from "./data.ts";

const msgs: { text: string; label: string }[] = JSON.parse(await readFile("playground/lanes/messages.json", "utf8"));
const labels = [...new Set(msgs.map((m) => m.label))].sort();
const all: Record<string, string> = JSON.parse(await readFile("eval/descriptions-banking77.json", "utf8"));
const { banking } = await loadData();
const seen = new Set(msgs.map((m) => m.text));
const heldOut = banking.test.filter((r) => labels.includes(r.label) && !seen.has(r.text));
const d = await createDecider();
const X = await d.embed(msgs.map((m) => m.text)), H = await d.embed(heldOut.map((r) => r.text));
const names = await d.embed(labels.map((l) => l.replace(/_/g, " "))), descs = await d.embed(labels.map((l) => all[l]));
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
const report = (name: string, ps: number[][]) => {
  const conf = ps.map((p) => Math.max(...p)), sure = ps.map((p, i) => [Math.max(...p) >= 0.9, labels[p.indexOf(Math.max(...p))] === heldOut[i].label]);
  console.log(name.padEnd(22), "at 0.9+", (sure.filter(([s]) => s).length / ps.length).toFixed(3), "right when sure", (sure.filter(([s, r]) => s && r).length / Math.max(1, sure.filter(([s]) => s).length)).toFixed(3), "accuracy", (sure.filter(([, r]) => r).length / ps.length).toFixed(3), "mean conf", (conf.reduce((a, b) => a + b) / conf.length).toFixed(3));
};
report("zero-shot described", H.map((x) => softmax(labels.map((_, c) => Math.max(dot(names[c], x), dot(descs[c], x))), 0.025)));
const order = msgs.map((_, i) => i).sort(seeded(3) as any);
const rand = seeded(3); order.sort(() => rand() - 0.5);
for (const n of [16, 32, 48, 96, 160, 240]) {
  const ids = order.slice(0, n);
  const head = await fitHead(ids.map((i) => X[i]), ids.map((i) => labels.indexOf(msgs[i].label)), labels, { seed: 7 }, names, 1);
  report(`head n=${n} T=${head.temperature}`, H.map((x) => softmax(logits(head, x), head.temperature)));
}
