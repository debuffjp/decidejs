// How good is the fallback teacher? Llama 3.1 8B and 3.3 70B on Workers AI, with bare label names and with label
// descriptions, on the 211 lanes demo messages (BANKING77 test, 12 labels), against decidejs on the same messages.
// Needs a Worker with the /ask prompt on TEACHER_URL (default http://localhost:8799; a throwaway one run with
// `wrangler dev --remote`). Appends to eval/results-teacher.jsonl.
//   node --experimental-strip-types eval/teacher-compare.ts
import { appendFile, readFile } from "node:fs/promises";

const url = process.env.TEACHER_URL ?? "http://localhost:8799";
const msgs: { text: string; label: string; llm: string | null }[] = JSON.parse(await readFile("playground/lanes/messages.json", "utf8"));
const rows = msgs.filter((m) => m.llm);
// ORDER=demo keeps the demo's label order instead of sorting: the prompt lists labels in this order. MODELS and
// DESCRIBED (0 or 1) narrow the run.
const demoOrder = [...new Set(msgs.map((m) => m.label))];
const labels = process.env.ORDER === "demo" ? JSON.parse((await readFile("playground/lanes/lanes.ts", "utf8")).match(/const LABELS = (\[[^\]]+\])/)![1]) as string[] : demoOrder.sort();
const all: Record<string, string> = JSON.parse(await readFile("eval/descriptions-banking77.json", "utf8"));
const descriptions = Object.fromEntries(labels.map((l) => [l, all[l]]));
const models = process.env.MODELS?.split(",") ?? ["@cf/meta/llama-3.1-8b-instruct-fast", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"];
for (const model of models) for (const described of process.env.DESCRIBED ? [process.env.DESCRIBED === "1"] : [false, true]) {
  let right = 0, none = 0;
  const answers: (string | null)[] = [];
  for (let i = 0; i < rows.length; i += 8) {
    const got = await Promise.all(rows.slice(i, i + 8).map(async (m) => {
      for (let tries = 0; tries < 3; tries++) {
        const r = await fetch(url, { method: "POST", body: JSON.stringify({ text: m.text, labels, model, descriptions: described ? descriptions : undefined }) });
        if (r.ok) return ((await r.json()) as { label: string | null }).label;
      }
      throw new Error("teacher failed 3 times");
    }));
    answers.push(...got);
  }
  answers.forEach((a, i) => { if (a === rows[i].label) right++; if (a === null) none++; });
  const line = { date: "2026-11-01", model, described, order: process.env.ORDER ?? "sorted", messages: rows.length, accuracy: +(right / rows.length).toFixed(3), none };
  console.log(JSON.stringify(line));
  await appendFile("eval/results-teacher.jsonl", JSON.stringify({ ...line, answers }) + "\n");
}
