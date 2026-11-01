// Records Llama 3.1 8B's answer for every message in the lanes demo's 12-label BANKING77 stream (training split), plus
// 70 CLINC150 out-of-scope queries, with the demo's prompt (sorted labels, descriptions). One call each, so replays of
// any withFallback setting run offline. Needs a local Worker with an AI binding on TEACHER_URL (the decidejs eval
// teacher). Writes eval/teacher-8b-stream.json; resumes where it stopped and stops at the daily-limit error.
//   TEACHER_URL=http://127.0.0.1:8799 [LIMIT=n] node eval/collect-teacher-stream.mjs
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const OUT = "eval/teacher-8b-stream.json", MODEL = "@cf/meta/llama-3.1-8b-instruct-fast";
const LABELS = ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card", "card_not_working", "declined_card_payment", "card_payment_fee_charged", "exchange_rate", "top_up_failed", "pending_transfer", "cash_withdrawal_charge", "activate_my_card", "change_pin"];
const all = JSON.parse(readFileSync("eval/descriptions-banking77.json", "utf8"));
const descriptions = Object.fromEntries(LABELS.map((l) => [l, all[l]]));
const csv = readFileSync("data/banking77-train.csv", "utf8");
const rows = [...csv.matchAll(/("(?:[^"]|"")*"|[^,\n]*),([^\n]*?)\r?(?:\n|$)/g)].slice(1)
  .map(([, t, label]) => ({ text: t.startsWith('"') ? t.slice(1, -1).replace(/""/g, '"') : t, label }))
  .filter((r) => LABELS.includes(r.label));
const oos = JSON.parse(readFileSync("data/clinc150.json", "utf8")).oos_train.slice(0, 70).map(([text]) => ({ text, label: null }));
const done = existsSync(OUT) ? JSON.parse(readFileSync(OUT, "utf8")) : [];
const seen = new Set(done.map((r) => r.text));
const todo = [...rows, ...oos].filter((r) => !seen.has(r.text)).slice(0, Number(process.env.LIMIT ?? Infinity));
let tokens = 0;
for (let i = 0; i < todo.length; i += 8) {
  const batch = await Promise.all(todo.slice(i, i + 8).map(async (r) => {
    const res = await fetch(process.env.TEACHER_URL, { method: "POST", body: JSON.stringify({ text: r.text, labels: LABELS, descriptions, model: MODEL }) });
    const body = await res.text();
    if (!res.ok) throw new Error(`${res.status} ${body.slice(0, 200)}`);
    const j = JSON.parse(body);
    tokens += j.usage?.prompt_tokens ?? 0;
    return { ...r, llm: j.label };
  }));
  done.push(...batch);
  writeFileSync(OUT, JSON.stringify(done, null, 0) + "\n");
  if (i % 200 === 0) console.log(`${done.length} recorded, ${tokens} prompt tokens this run`);
}
console.log(`done: ${done.length} recorded, ${tokens} prompt tokens this run`);
