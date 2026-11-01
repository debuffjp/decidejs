// withFallback with a real teacher: the lanes demo's recorded Llama 3.1 8B answers (bare names, eval/teacher-8b-bare.json;
// with descriptions and sorted labels, playground/lanes/messages.json) against a perfect teacher. Streams the 211
// messages both recorded, in 5 shuffled orders, like the demo: zero-shot over descriptions until the first retrain,
// threshold 0.9, retrain every 8 answers. Scores the final head on the other BANKING77 test messages for the same 12
// labels. STOP=1 adds a stop rule: audit 1 in 10 confident local answers with the teacher, and once 20 audits show the
// teacher disagreeing with more than 15% of them, stop learning from it. PRIOR=descriptions|both starts each head from
// the descriptions (or the sum of name and description) instead of the name, with weight PRIOR_W (default 1).
// SWITCH_AT=n keeps zero-shot until n answers have been learned, then trains the first head.
//   node --experimental-strip-types eval/replay-teacher.ts
import { appendFile, readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits, softmax } from "../src/head.ts";
import { loadData, seeded } from "./data.ts";

type Msg = { text: string; label: string; llm?: string | null };
const described: Msg[] = JSON.parse(await readFile("playground/lanes/messages.json", "utf8"));
const bare: Msg[] = JSON.parse(await readFile("eval/teacher-8b-bare.json", "utf8"));
const labels = [...new Set(described.map((m) => m.label))].sort();
const all: Record<string, string> = JSON.parse(await readFile("eval/descriptions-banking77.json", "utf8"));
const both = described.map((m, i) => ({ text: m.text, label: m.label, described: m.llm ?? null, bare: bare[i].llm })).filter((m) => m.bare !== undefined);
const { banking } = await loadData();
const streamTexts = new Set(described.map((m) => m.text));
const heldOut = banking.test.filter((r) => labels.includes(r.label) && !streamTexts.has(r.text));

const d = await createDecider();
const vec = new Map<string, number[]>();
(await d.embed([...both.map((m) => m.text), ...heldOut.map((r) => r.text)])).forEach((v, i) => vec.set(i < both.length ? both[i].text : heldOut[i - both.length].text, v));
const names = await d.embed(labels.map((l) => l.replace(/_/g, " ")));
const descs = await d.embed(labels.map((l) => all[l]));
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
const stopRule = process.env.STOP === "1";
const switchAt = Number(process.env.SWITCH_AT ?? 0);
const priorKind = process.env.PRIOR ?? "names", priorW = Number(process.env.PRIOR_W ?? 1);
const priors = priorKind === "descriptions" ? descs : priorKind === "both" ? names.map((n, c) => n.map((v, j) => v + descs[c][j])) : names;

for (const teacher of ["perfect", "8b-bare", "8b-described"] as const) {
  const runs: Record<string, number>[] = [];
  for (const seed of [1, 2, 3, 4, 5]) {
    const rand = seeded(seed);
    const stream = [...both].sort(() => rand() - 0.5);
    let head: Awaited<ReturnType<typeof fitHead>> | null = null;
    const learned: { text: string; label: number }[] = [];
    let sinceRetrain = 0, sent = 0, right = 0, sentLate = 0, audits = 0, disagree = 0, stopped = false, wrongLearned = 0;
    for (const [i, m] of stream.entries()) {
      const x = vec.get(m.text)!;
      const p = head ? softmax(logits(head, x), head.temperature) : softmax(labels.map((_, c) => Math.max(dot(names[c], x), dot(descs[c], x))), 0.025);
      const best = p.indexOf(Math.max(...p));
      const answer = teacher === "perfect" ? m.label : teacher === "8b-bare" ? m.bare : m.described;
      if (p[best] >= 0.9) {
        if (labels[best] === m.label) right++;
        // Stop rule: occasionally pay for a check of a confident answer.
        if (stopRule && !stopped && i % 10 === 0) {
          audits++;
          if (answer !== labels[best]) disagree++;
          if (audits >= 20 && disagree / audits > 0.15) stopped = true;
        }
        continue;
      }
      sent++;
      if (i >= stream.length / 2) sentLate++;
      if (answer === m.label) right++;
      if (answer && !stopped) {
        learned.push({ text: m.text, label: labels.indexOf(answer) });
        if (answer !== m.label) wrongLearned++;
        if (++sinceRetrain >= 8 && learned.length >= switchAt) {
          head = await fitHead(learned.map((l) => vec.get(l.text)!), learned.map((l) => l.label), labels, { seed: 7 }, priors, priorW);
          sinceRetrain = 0;
        }
      }
    }
    const final = heldOut.filter((r) => {
      const x = vec.get(r.text)!;
      const p = head ? softmax(logits(head, x), head.temperature) : labels.map((_, c) => Math.max(dot(names[c], x), dot(descs[c], x)));
      return labels[p.indexOf(Math.max(...p))] === r.label;
    }).length / heldOut.length;
    runs.push({ sent: sent / stream.length, sentLate: sentLate / (stream.length / 2), right: right / stream.length, final, wrongLearned: wrongLearned / Math.max(1, learned.length), stopped: stopped ? 1 : 0 });
  }
  const mean = (k: string) => +(runs.reduce((s, r) => s + r[k], 0) / runs.length).toFixed(3);
  const line = { date: "2026-11-01", teacher, stopRule, prior: priorKind, priorWeight: priorW, switchAt, messages: both.length, heldOut: heldOut.length, sentToLlm: mean("sent"), sentSecondHalf: mean("sentLate"), answeredRight: mean("right"), finalHeadOnHeldOut: mean("final"), learnedWrong: mean("wrongLearned"), runsStopped: runs.filter((r) => r.stopped).length };
  console.log(JSON.stringify(line));
  await appendFile("eval/results-replay-teacher.jsonl", JSON.stringify(line) + "\n");
}
