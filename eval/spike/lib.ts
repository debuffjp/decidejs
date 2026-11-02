// Shared by the accuracy spike. Same dev splits as eval/gains.ts: a seeded 20% of BANKING77 train, CLINC150 val.
// Test splits and HWU64 are refused unless SPIKE_TEST names the reason, and each look is logged in test-looks.tsv.
import { appendFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { kShot, loadData, seeded, type Row } from "../data.ts";

export const readable = (l: string) => l.replace(/[_-]+/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();

export async function embeddings(model: string) {
  const name = model.replace(/\W/g, "_");
  const meta = JSON.parse(await readFile(`data/spike/emb-${name}.json`, "utf8"));
  const raw = await readFile(`data/spike/emb-${name}.bin`);
  const all = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
  const at = new Map<string, number>(meta.texts.map((t: string, i: number) => [t, i]));
  const dim: number = meta.dim;
  return { dim, vec: (t: string) => { const i = at.get(t); if (i === undefined) throw new Error(`no embedding: ${t}`); return all.subarray(i * dim, (i + 1) * dim); } };
}

const hwu = async (s: string): Promise<Row[]> => {
  const t = (await readFile(`data/hwu64/${s}.seq.in`, "utf8")).trim().split(/\r?\n/);
  const l = (await readFile(`data/hwu64/${s}.label`, "utf8")).trim().split(/\r?\n/);
  return t.map((text, i) => ({ text: text.trim(), label: l[i].trim() }));
};

export type Split = "dev" | "test";
export async function dataset(name: string, split: Split, id: string): Promise<{ train: Row[]; eval: Row[] }> {
  if (split === "test" || name === "hwu64") {
    if (!process.env.SPIKE_TEST) throw new Error("test or HWU64 look refused: set SPIKE_TEST=<reason>");
    appendFileSync("eval/spike/test-looks.tsv", [new Date().toISOString(), id, name, split, process.env.SPIKE_TEST].join("\t") + "\n");
  }
  if (name === "hwu64") return { train: await hwu("train"), eval: await hwu("test") };
  const { banking, clinc } = await loadData();
  if (name === "clinc150") return split === "dev" ? { train: clinc.train, eval: clinc.val } : { train: clinc.train, eval: clinc.test };
  if (split === "test") return { train: banking.train, eval: banking.test };
  const rand = seeded(11);
  const s = [...banking.train].sort(() => rand() - 0.5);
  const cut = Math.floor(s.length * 0.8);
  return { train: s.slice(0, cut), eval: s.slice(cut) };
}
export { kShot };

export function ece(s: { right: boolean; confidence: number }[]) {
  const bins = Array.from({ length: 15 }, () => ({ n: 0, conf: 0, right: 0 }));
  for (const x of s) { const b = bins[Math.min(14, Math.floor(x.confidence * 15))]; b.n++; b.conf += x.confidence; b.right += x.right ? 1 : 0; }
  return bins.reduce((sum, b) => sum + (b.n ? (b.n / s.length) * Math.abs(b.conf / b.n - b.right / b.n) : 0), 0);
}
