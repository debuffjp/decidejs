// withFallback replayed with real Llama 3.1 8B answers (eval/teacher-8b-stream.json, collect-teacher-stream.mjs): the
// lanes demo's 12 BANKING77 labels, their 1,560 training messages plus 70 CLINC150 out-of-scope queries, in 5 shuffled
// orders. Starts from names and descriptions, retrains every 100 answers. Scores the final head on the 12 labels'
// BANKING77 test messages and on 200 held-out CLINC150 out-of-scope test queries. MODE: plain (calibrate on the LLM's
// answers, drop nulls), auto (calibrate on zero-shot's confident answers, as withFallback does), none (auto plus
// learning nulls as a label, withFallback's default), gated (none, but a null is learned only when the text's best
// similarity to any label name or description is under CUT, default 0.6: off-topic text sits far from every label,
// ambiguous in-scope text doesn't, eval/null-similarity.ts). CAP sets the zero-shot calibration set's per-label cap
// (default: 4, or enough to reach 50 with few labels).
//   MODE=none node --experimental-strip-types eval/real-stream.ts
import { appendFile, readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits, softmax } from "../src/head.ts";
import { loadData, seeded } from "./data.ts";

type Msg = { text: string; label: string | null; llm: string | null };
const mode = process.env.MODE ?? "none", EVERY = 100, CUT = Number(process.env.CUT ?? 0.6);
const recorded: Msg[] = JSON.parse(await readFile("eval/teacher-8b-stream.json", "utf8"));
const labels = [...new Set(recorded.filter((m) => m.label).map((m) => m.label!))].sort();
const all: Record<string, string> = JSON.parse(await readFile("eval/descriptions-banking77.json", "utf8"));
const { banking, clinc } = await loadData();
const test = banking.test.filter((r) => labels.includes(r.label));
const oosHeld = clinc.oosTest.slice(0, 200);
const d = await createDecider();
const texts = [...new Set([...recorded.map((m) => m.text), ...test.map((r) => r.text), ...oosHeld.map((r) => r.text)])];
const vec = new Map((await d.embed(texts)).map((v, i) => [texts[i], v]));
const names = await d.embed(labels.map((l) => l.replace(/_/g, " ")));
const descs = await d.embed(labels.map((l) => all[l]));
const [noneName] = await d.embed(["none of these"]);
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
const nearest = (x: number[]) => Math.max(...labels.map((_, c) => Math.max(dot(names[c], x), dot(descs[c], x))));
const CAP_N = Number(process.env.CAP ?? Math.max(4, Math.ceil(50 / labels.length)));
const zeroShot = (x: number[]) => softmax(labels.map((_, c) => Math.max(dot(names[c], x), dot(descs[c], x))), 0.025);
type Head = Awaited<ReturnType<typeof fitHead>>;
const NONE = labels.length;
const top = (p: number[]) => p.indexOf(Math.max(...p));
const predict = (h: Head | null, x: number[]) => (h ? softmax(logits(h, x), h.temperature) : zeroShot(x));
// A label index, -1 for a local "none", or undefined when it would be sent.
const local = (h: Head | null, x: number[]) => { const p = predict(h, x), b = top(p); return p[b] < 0.9 ? undefined : h && b === NONE ? -1 : b; };

const runs: Record<string, number>[] = [];
for (const seed of [1, 2, 3, 4, 5]) {
  const rand = seeded(seed);
  const stream = [...recorded].sort(() => rand() - 0.5);
  const learned: { x: number[]; y: number }[] = [], selfSet: { x: number[]; y: number }[] = [], perLabel = new Array(labels.length).fill(0);
  let head: Head | null = null, since = 0, sent = 0, sentLate = 0, right = 0, wrongLearned = 0;
  for (const [i, m] of stream.entries()) {
    const x = vec.get(m.text)!, truth = m.label ? labels.indexOf(m.label) : -1, z = zeroShot(x);
    if (!head && z[top(z)] >= 0.9 && perLabel[top(z)] < CAP_N) { perLabel[top(z)]++; selfSet.push({ x, y: top(z) }); }
    const l = local(head, x);
    if (l !== undefined) { if (l === truth) right++; continue; }
    sent++;
    if (i >= stream.length / 2) sentLate++;
    const answer = m.llm ? labels.indexOf(m.llm) : -1;
    if (answer === truth) right++;
    if (answer >= 0 || mode === "none" || (mode === "gated" && nearest(x) < CUT)) {
      learned.push({ x, y: answer >= 0 ? answer : NONE });
      if (answer !== truth) wrongLearned++;
      if (++since >= EVERY) {
        since = 0;
        const hasNone = learned.some((e) => e.y === NONE);
        const calibration = mode !== "plain" && selfSet.length >= 50 ? { X: selfSet.map((e) => e.x), y: selfSet.map((e) => e.y) } : undefined;
        head = await fitHead(learned.map((e) => e.x), learned.map((e) => e.y), hasNone ? [...labels, "none_of_these"] : labels, { seed: 7 }, hasNone ? [...names, noneName] : names, 1, undefined, 0.5, calibration);
      }
    }
  }
  const t = test.map((r) => local(head, vec.get(r.text)!)), o = oosHeld.map((r) => local(head, vec.get(r.text)!));
  runs.push({
    sent: sent / stream.length, sentSecondHalf: sentLate / (stream.length / 2), answeredRight: right / stream.length, learnedWrong: wrongLearned / Math.max(1, learned.length),
    testRightLocally: t.filter((v, i) => v === labels.indexOf(test[i].label)).length / test.length,
    testSent: t.filter((v) => v === undefined).length / test.length,
    testWrongLocally: t.filter((v, i) => v !== undefined && v !== labels.indexOf(test[i].label)).length / test.length,
    oosLabelledLocally: o.filter((v) => v !== undefined && v >= 0).length / o.length,
    oosNoneLocally: o.filter((v) => v === -1).length / o.length,
  });
}
const mean = Object.fromEntries(Object.keys(runs[0]).map((k) => [k, +(runs.reduce((s, r) => s + r[k], 0) / runs.length).toFixed(3)]));
const line = { date: "2026-11-01", mode, cut: mode === "gated" ? CUT : undefined, cap: CAP_N, messages: recorded.length, llmAgreement: +(recorded.filter((m) => m.llm === m.label).length / recorded.length).toFixed(3), llmNullOnInScope: +(recorded.filter((m) => m.label && m.llm === null).length / recorded.filter((m) => m.label).length).toFixed(3), llmNullOnOos: +(recorded.filter((m) => !m.label && m.llm === null).length / Math.max(1, recorded.filter((m) => !m.label).length)).toFixed(3), ...mean };
console.log(JSON.stringify(line));
await appendFile("eval/results-real-stream.jsonl", JSON.stringify(line) + "\n");
