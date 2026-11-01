// Off-topic traffic through withFallback: can the LLM's "none of these" (null) answers teach the head to stop giving
// off-topic texts a confident wrong label? CLINC150, cold start from names and descriptions, self-calibration (as
// withFallback does by default), mistakes weighted to hard texts (10% overall, as long-stream.ts HARD=1). The stream is
// the 15,000 in-scope training queries plus 700 out-of-scope ones (oos_train, oos_val and half of oos_test, 4.5%); the
// other 500 of oos_test are held out. For an off-topic text the stand-in LLM answers null, or zero-shot's best label
// when it errs. NONE picks what happens to null answers:
//   off   thrown away (withFallback today)
//   label learned as an extra label; a confident "none" is answered locally as null
//   ask   learned as an extra label, but a text whose best label is "none" is always sent
//   NONE=label node --experimental-strip-types eval/none-label.ts
import { appendFile, readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits, softmax } from "../src/head.ts";
import { seeded } from "./data.ts";

type Row = { text: string; label: string };
const mode = process.env.NONE ?? "off", ROUND = 1000, NOISE = 0.1;
const raw = JSON.parse(await readFile("data/clinc150.json", "utf8"));
const rows = (a: [string, string][]): Row[] => a.map(([text, label]) => ({ text, label }));
const train = rows(raw.train), test = rows(raw.test), oosTest = rows(raw.oos_test);
const labels = [...new Set(train.map((r) => r.label))].sort();
const descriptions: Record<string, string> = JSON.parse(await readFile("eval/descriptions-clinc150.json", "utf8"));
const d = await createDecider();
const cache: Record<string, number[]> = JSON.parse(await readFile(`data/emb-${d.model.replace(/\W/g, "_")}.json`, "utf8"));
const all = [...train, ...test, ...rows(raw.oos_train), ...rows(raw.oos_val), ...oosTest];
const missing = [...new Set(all.map((r) => r.text).filter((t) => !cache[t]))];
(await d.embed(missing)).forEach((v, i) => (cache[missing[i]] = v));
const names = await d.embed(labels.map((l) => l.replace(/_/g, " ")));
const descs = await d.embed(labels.map((l) => descriptions[l]));
const [noneName] = await d.embed(["none of these, off topic"]);
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
const zeroShot = (x: number[]) => softmax(labels.map((_, c) => Math.max(dot(names[c], x), dot(descs[c], x))), 0.025);
type Head = Awaited<ReturnType<typeof fitHead>>;
const NONE = labels.length; // index of the extra label when learning nulls
const headLabels = mode === "off" ? labels : [...labels, "__none__"];
const priors = mode === "off" ? names : [...names, noneName];
const predict = (head: Head | null, x: number[]) => (head ? softmax(logits(head, x), head.temperature) : zeroShot(x));
const top = (p: number[]) => p.indexOf(Math.max(...p));

const rand = seeded(3), teacher = seeded(11);
const heldOos = oosTest.slice(500);
const stream = [...train.map((r) => ({ ...r, oos: false })), ...[...rows(raw.oos_train), ...rows(raw.oos_val), ...oosTest.slice(0, 500)].map((r) => ({ ...r, oos: true }))].sort(() => rand() - 0.5);
const doubt = (x: number[]) => 1 - Math.max(...zeroShot(x));
const meanDoubt = stream.reduce((s, r) => s + doubt(cache[r.text]), 0) / stream.length;
// Answer: a label index, or -1 for none.
function decide(head: Head | null, x: number[]): { local: boolean; answer: number } {
  const p = predict(head, x), b = top(p);
  if (p[b] < 0.9 || (b === NONE && head && mode === "ask")) return { local: false, answer: -2 };
  return { local: true, answer: head && b === NONE ? -1 : b };
}
const learned: { x: number[]; y: number }[] = [];
const selfSet: { x: number[]; y: number }[] = [], perLabel = new Array(labels.length).fill(0);
let head: Head | null = null;
for (let r = 0; r * ROUND < stream.length; r++) {
  let sent = 0, right = 0, oosLabelled = 0, oosSeen = 0, inNone = 0;
  for (const q of stream.slice(r * ROUND, (r + 1) * ROUND)) {
    const x = cache[q.text], truth = q.oos ? -1 : labels.indexOf(q.label), z = zeroShot(x);
    if (z[top(z)] >= 0.9 && perLabel[top(z)] < 4) { perLabel[top(z)]++; selfSet.push({ x, y: top(z) }); }
    if (q.oos) oosSeen++;
    const dec = decide(head, x);
    let answer = dec.answer;
    if (!dec.local) {
      sent++;
      const errs = teacher() < Math.min(0.9, (NOISE * (1 - z[top(z)])) / meanDoubt);
      answer = !errs ? truth : z.map((v, c) => [v, c]).sort((a, b) => b[0] - a[0]).find(([, c]) => c !== truth)![1];
      if (answer >= 0) learned.push({ x, y: answer });
      else if (mode !== "off") learned.push({ x, y: NONE });
    } else {
      if (q.oos && answer >= 0) oosLabelled++;
      if (!q.oos && answer === -1) inNone++;
    }
    if (answer === truth) right++;
  }
  const n = Math.min(ROUND, stream.length - r * ROUND);
  const line = { date: "2026-11-01", mode, round: r + 1, sent: +(sent / n).toFixed(3), right: +(right / n).toFixed(3), oosSeen, oosGivenLabelLocally: oosLabelled, inScopeAnsweredNone: inNone };
  console.log(JSON.stringify(line));
  await appendFile("eval/results-none-label.jsonl", JSON.stringify(line) + "\n");
  const calibration = selfSet.length >= 50 ? { X: selfSet.map((e) => e.x), y: selfSet.map((e) => e.y) } : undefined;
  head = await fitHead(learned.map((l) => l.x), learned.map((l) => l.y), headLabels, { seed: 7 }, priors, 1, undefined, 0.5, calibration);
}
const outcome = (rs: Row[], oos: boolean) => {
  const ds = rs.map((q) => decide(head, cache[q.text]));
  const local = ds.filter((x) => x.local);
  return {
    sent: +(1 - local.length / rs.length).toFixed(3),
    ...(oos
      ? { labelledLocally: +(local.filter((x) => x.answer >= 0).length / rs.length).toFixed(3), noneLocally: +(local.filter((x) => x.answer === -1).length / rs.length).toFixed(3) }
      : { rightLocally: +(ds.filter((x, i) => x.local && x.answer === labels.indexOf(rs[i].label)).length / rs.length).toFixed(3), noneLocally: +(local.filter((x) => x.answer === -1).length / rs.length).toFixed(3) }),
  };
};
const line = { date: "2026-11-01", mode, final: true, inScopeTest: outcome(test, false), heldOutOos: outcome(heldOos, true) };
console.log(JSON.stringify(line));
await appendFile("eval/results-none-label.jsonl", JSON.stringify(line) + "\n");
