// decidekit on the same 211 lanes messages as eval/teacher-compare.ts: zero-shot with names, with descriptions, and a
// 5-shot head (BANKING77 train, 3 seeds).
import { readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";
import { kShot, loadData } from "./data.ts";

const msgs: { text: string; label: string; llm: string | null }[] = JSON.parse(await readFile("playground/lanes/messages.json", "utf8"));
const rows = msgs.filter((m) => m.llm);
const labels = [...new Set(msgs.map((m) => m.label))].sort();
const all: Record<string, string> = JSON.parse(await readFile("eval/descriptions-banking77.json", "utf8"));
const d = await createDecider();
const acc = async (using: any) => (await d.decideMany(rows.map((r) => r.text), using)).filter((r, i) => r.choice === rows[i].label).length / rows.length;
console.log("names", (await acc(labels)).toFixed(3));
console.log("descriptions", (await acc(Object.fromEntries(labels.map((l) => [l, all[l]])))).toFixed(3));
const { banking } = await loadData();
const heads = [];
for (const seed of [7, 8, 9]) heads.push(await acc(await d.train(kShot(banking.train.filter((r) => labels.includes(r.label)), 5, seed), { seed })));
console.log("5-shot head", (heads.reduce((a, b) => a + b) / 3).toFixed(3), heads.map((h) => h.toFixed(3)).join(" "));
