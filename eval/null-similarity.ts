// Where do Llama 3.1 8B's null answers fall? For each null in eval/teacher-8b-stream.json, the best cosine similarity
// to any label's name or description, split by in-scope and out-of-scope text, to see whether a similarity cut can tell
// "ambiguous" nulls from "off-topic" ones.
//   node --experimental-strip-types eval/null-similarity.ts
import { readFile } from "node:fs/promises";
import { createDecider } from "../src/index.ts";

type Msg = { text: string; label: string | null; llm: string | null };
const recorded: Msg[] = JSON.parse(await readFile("eval/teacher-8b-stream.json", "utf8"));
const labels = [...new Set(recorded.filter((m) => m.label).map((m) => m.label!))].sort();
const all: Record<string, string> = JSON.parse(await readFile("eval/descriptions-banking77.json", "utf8"));
const d = await createDecider();
const nulls = recorded.filter((m) => m.llm === null);
const X = await d.embed(nulls.map((m) => m.text));
const L = await d.embed([...labels.map((l) => l.replace(/_/g, " ")), ...labels.map((l) => all[l])]);
const best = X.map((x) => Math.max(...L.map((v) => v.reduce((s, a, i) => s + a * x[i], 0))));
for (const kind of ["in-scope", "out-of-scope"]) {
  const s = best.filter((_, i) => (kind === "in-scope") === (nulls[i].label !== null)).sort((a, b) => a - b);
  const q = (p: number) => s[Math.floor(p * (s.length - 1))].toFixed(3);
  console.log(kind, s.length, "min", q(0), "p25", q(0.25), "median", q(0.5), "p75", q(0.75), "max", q(1));
}
