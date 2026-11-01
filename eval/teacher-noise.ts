// What real LLM mistakes look like, to check eval/fallback.ts's NOISE model ("the head's best wrong guess"). Uses the
// Llama 3.1 8B answers recorded for the lanes demo (BANKING77 test messages, 12 labels). For each wrong answer: was it
// the head's top wrong guess, in the head's top 3, or far off? The head is zero-shot over label names and a 5-shot head.
//   node --experimental-strip-types eval/teacher-noise.ts
import { readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { kShot, loadData } from "./data.ts";

const msgs: { text: string; label: string; llm: string | null }[] = JSON.parse(await readFile("playground/lanes/messages.json", "utf8"));
const labels = [...new Set(msgs.map((m) => m.label))].sort();
const d = await createDecider();
const { banking } = await loadData();
const head = await d.train(kShot(banking.train.filter((r) => labels.includes(r.label)), 5, 7));
const recorded = msgs.filter((m) => m.llm);
for (const [name, using] of [["zero-shot", labels], ["5-shot head", head]] as const) {
  let wrong = 0, top1 = 0, top3 = 0, offList = 0, headRight = 0;
  for (const m of recorded) {
    const r = await d.decide(m.text, using as any);
    if (r.choice === m.label) headRight++;
    if (m.llm === m.label) continue;
    wrong++;
    if (!labels.includes(m.llm!)) { offList++; continue; }
    const ranked = Object.entries(r.probs as Record<string, number>).sort((a, b) => b[1] - a[1]).map(([l]) => l).filter((l) => l !== m.label);
    if (ranked[0] === m.llm) top1++;
    if (ranked.slice(0, 3).includes(m.llm!)) top3++;
  }
  console.log(JSON.stringify({ head: name, recorded: recorded.length, llmWrong: wrong, headAccuracy: +(headRight / recorded.length).toFixed(3), llmWrongIsHeadsTopWrong: +(top1 / wrong).toFixed(3), llmWrongInHeadsTop3Wrong: +(top3 / wrong).toFixed(3), llmOffList: offList }));
}
