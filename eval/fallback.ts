// withFallback() with learning, simulated on CLINC150: start from 5 examples per label, stream the rest of the training
// queries as traffic in rounds, let the true label stand in for the LLM below the threshold, retrain after each
// round. Reports the share sent to the fallback and the head's own accuracy on the validation split. Uses the hill
// embedding cache (npm run eval:hill once first). COLD=1 starts from labels alone: zero-shot until the first retrain,
// labels without answers starting from their names. Usage: [COLD=1] node --experimental-strip-types eval/fallback.ts [threshold]
import { readFile } from "node:fs/promises";
import { fitHead, logits, softmax } from "../src/head.ts";
import { createDecider } from "../src/index.ts";
import { kShot, loadData, seeded } from "./data.ts";

const threshold = Number(process.argv[2] ?? 0.9);
const { clinc } = await loadData();
const cache: Record<string, number[]> = JSON.parse(await readFile("data/emb-Xenova_all_MiniLM_L6_v2.json", "utf8"));
const labels = [...new Set(clinc.train.map((r) => r.label))].sort();
const cold = process.env.COLD === "1";
const base = cold ? [] : kShot(clinc.train, 5, 7);
const inBase = new Set(base.map((r) => r.text));
const rand = seeded(5);
const traffic = clinc.train.filter((r) => !inBase.has(r.text)).sort(() => rand() - 0.5);
// Label names, as train() and zero-shot decide() read them.
const names = await (await createDecider()).embed(labels.map((l) => l.replace(/[_-]+/g, " ").toLowerCase()));
const fit = (rows: typeof base) => fitHead(rows.map((r) => cache[r.text]), rows.map((r) => labels.indexOf(r.label)), labels, { seed: 7 }, cold ? names : undefined, 3); // the warm run in results-fallback-0.9.jsonl had no label names
type Fitted = Awaited<ReturnType<typeof fit>>;
const decide = (head: Fitted | null, text: string) => {
  const x = cache[text];
  const p = head ? softmax(logits(head, x), head.temperature) : softmax(names.map((n) => n.reduce((s, v, i) => s + v * x[i], 0)), 0.05);
  const best = p.indexOf(Math.max(...p));
  return { choice: labels[best], confidence: p[best] };
};

let head: Fitted | null = cold ? null : await fit(base);
const learned: typeof base = [];
const ROUND = 1000;
for (let r = 0; r * ROUND < traffic.length && r < 10; r++) {
  const batch = traffic.slice(r * ROUND, (r + 1) * ROUND);
  let sent = 0, right = 0, localRight = 0, local = 0;
  for (const q of batch) {
    const d = decide(head, q.text);
    if (d.confidence >= threshold) {
      local++;
      if (d.choice === q.label) { right++; localRight++; }
    } else {
      sent++;
      right++; // the stand-in LLM is always right
      learned.push(q);
    }
  }
  const dev = clinc.val.filter((q) => decide(head, q.text).choice === q.label).length / clinc.val.length;
  console.log(JSON.stringify({
    round: r + 1, examples: base.length + learned.length, sentToFallback: +(sent / batch.length).toFixed(3),
    answeredRight: +(right / batch.length).toFixed(3), localPrecision: +(localRight / Math.max(1, local)).toFixed(3),
    headDevAccuracy: +dev.toFixed(3),
  }));
  head = await fit([...base, ...learned]);
}
