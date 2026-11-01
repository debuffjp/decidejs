// withFallback over a long stream, from labels alone: does learning from the LLM ever send less than zero-shot with
// descriptions? Streams the BANKING77 training set (10,003 messages, 77 labels) in rounds of 1,000, cold: zero-shot over
// names and descriptions until the first retrain, then a head trained the way train() does it (name prior weight 1,
// small-head floor 0.5). The stand-in LLM is wrong on NOISE of the messages it is sent, answering zero-shot's best label
// that isn't the true one (real Llama 3.1 8B mistakes were 97% in the head's top 3; on messages Decide found hard,
// 8B with descriptions was right 62.5% of the time). Reports per round the share sent, accuracy of what was answered and
// the same for zero-shot alone, plus the final head on the BANKING77 test set. CAL=k sets aside k correctly labelled
// examples per label (kept out of the stream) and fits each head's temperature on them, as train()'s calibrateOn does.
// SELF=k needs no labels: it keeps up to k stream messages per label that zero-shot answered with confidence 0.9 or
// more, labelled with zero-shot's answer, and calibrates on those.
//   NOISE=0.3 [CAL=2] node --experimental-strip-types eval/long-stream.ts
import { appendFile, readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { fitHead, logits, softmax } from "../src/head.ts";
import { kShot, loadData, seeded } from "./data.ts";

const noise = Number(process.env.NOISE ?? 0), cal = Number(process.env.CAL ?? 0), self = Number(process.env.SELF ?? 0), ROUND = 1000;
const { banking } = await loadData();
const labels = [...new Set(banking.train.map((r) => r.label))].sort();
const descriptions: Record<string, string> = JSON.parse(await readFile("eval/descriptions-banking77.json", "utf8"));
const d = await createDecider();
const cache: Record<string, number[]> = JSON.parse(await readFile(`data/emb-${d.model.replace(/\W/g, "_")}.json`, "utf8"));
const missing = [...banking.train, ...banking.test].map((r) => r.text).filter((t) => !cache[t]);
(await d.embed(missing)).forEach((v, i) => (cache[missing[i]] = v));
const names = await d.embed(labels.map((l) => l.replace(/_/g, " ")));
const descs = await d.embed(labels.map((l) => descriptions[l]));
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
const zeroShot = (x: number[]) => softmax(labels.map((_, c) => Math.max(dot(names[c], x), dot(descs[c], x))), 0.025);
type Head = Awaited<ReturnType<typeof fitHead>>;
const predict = (head: Head | null, x: number[]) => (head ? softmax(logits(head, x), head.temperature) : zeroShot(x));
const top = (p: number[]) => p.indexOf(Math.max(...p));

const rand = seeded(3), teacher = seeded(11);
const calSet = cal ? kShot(banking.train, cal, 9) : [];
const inCal = new Set(calSet.map((r) => r.text));
const stream = banking.train.filter((r) => !inCal.has(r.text)).sort(() => rand() - 0.5);
const learned: { x: number[]; y: number }[] = [];
let head: Head | null = null, wrong = 0;
const selfSet: { x: number[]; y: number; right: boolean }[] = [], perLabel = new Array(labels.length).fill(0);
for (let r = 0; r * ROUND < stream.length; r++) {
  let sent = 0, right = 0, zSent = 0, zRight = 0;
  for (const q of stream.slice(r * ROUND, (r + 1) * ROUND)) {
    const x = cache[q.text], truth = labels.indexOf(q.label);
    const z = zeroShot(x), p = predict(head, x);
    const ask = () => {
      if (teacher() >= noise) return truth;
      return z.map((v, c) => [v, c]).sort((a, b) => b[0] - a[0]).find(([, c]) => c !== truth)![1];
    };
    // Zero-shot alone, against a teacher with the same error rate.
    if (self && z[top(z)] >= 0.9 && perLabel[top(z)] < self) { perLabel[top(z)]++; selfSet.push({ x, y: top(z), right: top(z) === truth }); }
    if (z[top(z)] >= 0.9) { if (top(z) === truth) zRight++; } else { zSent++; if (ask() === truth) zRight++; }
    if (p[top(p)] >= 0.9) { if (top(p) === truth) right++; continue; }
    sent++;
    const y = ask();
    if (y === truth) right++; else wrong++;
    learned.push({ x, y });
  }
  const n = Math.min(ROUND, stream.length - r * ROUND);
  const line = { date: "2026-11-01", noise, cal, self, selfSet: selfSet.length, selfWrong: +(selfSet.filter((e) => !e.right).length / Math.max(1, selfSet.length)).toFixed(3), round: r + 1, learned: learned.length, learnedWrong: +(wrong / Math.max(1, learned.length)).toFixed(3), sent: +(sent / n).toFixed(3), right: +(right / n).toFixed(3), zeroShotSent: +(zSent / n).toFixed(3), zeroShotRight: +(zRight / n).toFixed(3) };
  console.log(JSON.stringify(line));
  await appendFile("eval/results-long-stream.jsonl", JSON.stringify(line) + "\n");
  const calibration = cal ? { X: calSet.map((r) => cache[r.text]), y: calSet.map((r) => labels.indexOf(r.label)) }
    : self && selfSet.length >= 50 ? { X: selfSet.map((e) => e.x), y: selfSet.map((e) => e.y) } : undefined;
  head = await fitHead(learned.map((l) => l.x), learned.map((l) => l.y), labels, { seed: 7 }, names, 1, undefined, 0.5, calibration);
}
const test = (h: Head | null) => +(banking.test.filter((q) => top(predict(h, cache[q.text])) === labels.indexOf(q.label)).length / banking.test.length).toFixed(3);
const line = { date: "2026-11-01", noise, cal, self, final: true, headOnTest: test(head), zeroShotOnTest: test(null) };
console.log(JSON.stringify(line));
await appendFile("eval/results-long-stream.jsonl", JSON.stringify(line) + "\n");
