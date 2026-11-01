// On the messages zero-shot decidekit would send to the LLM (confidence under 0.9, labels described), who is right
// more often: the teacher or decidekit' own best guess? If the teacher isn't clearly better there, learning from it
// can't help. Same 215 lanes messages as eval/replay-teacher.ts.
import { readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";

const described = JSON.parse(await readFile("playground/lanes/messages.json", "utf8"));
const bare = JSON.parse(await readFile("eval/teacher-8b-bare.json", "utf8"));
const all = JSON.parse(await readFile("eval/descriptions-banking77.json", "utf8"));
const labels = [...new Set(described.map((m: any) => m.label))].sort() as string[];
const rows = described.map((m: any, i: number) => ({ ...m, bare: bare[i].llm })).filter((m: any) => m.bare !== undefined);
const d = await createDecider();
const out = await d.decideMany(rows.map((r: any) => r.text), Object.fromEntries(labels.map((l) => [l, all[l]])));
const hard = rows.map((r: any, i: number) => ({ ...r, local: out[i] })).filter((r: any) => r.local.confidence < 0.9);
const pct = (f: (r: any) => boolean) => (100 * hard.filter(f).length / hard.length).toFixed(1) + "%";
console.log(JSON.stringify({ messages: rows.length, sentToLlm: hard.length, decidekitRight: pct((r) => r.local.choice === r.label), teacherBareRight: pct((r) => r.bare === r.label), teacherDescribedRight: pct((r) => r.llm === r.label) }));

// Baseline for eval/replay-teacher.ts: zero-shot with descriptions, never learning, on its held-out messages.
import { loadData } from "./data.ts";
const { banking } = await loadData();
const stream = new Set(described.map((m: any) => m.text));
const heldOut = banking.test.filter((r) => labels.includes(r.label) && !stream.has(r.text));
const z = await d.decideMany(heldOut.map((r) => r.text), Object.fromEntries(labels.map((l) => [l, all[l]])));
console.log(JSON.stringify({ heldOut: heldOut.length, zeroShotDescribedNoLearning: +(z.filter((r, i) => r.choice === heldOut[i].label).length / heldOut.length).toFixed(3) }));
