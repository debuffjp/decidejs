// Dataset loading shared by the eval scripts: BANKING77 (PolyAI CSVs) and CLINC150 (oos-eval data_full.json).
import { mkdir, readFile, writeFile } from "node:fs/promises";
import type { Example } from "../src/index.ts";

export type Row = Example<string>;

async function cached(file: string, url: string) {
  try { return await readFile(file, "utf8"); } catch {
    const text = await (await fetch(url)).text();
    await mkdir("data", { recursive: true });
    await writeFile(file, text);
    return text;
  }
}

// Quoted fields may contain commas and doubled quotes.
const csv = (text: string): Row[] => [...text.matchAll(/("(?:[^"]|"")*"|[^,\n]*),([^\n]*?)\r?(?:\n|$)/g)].slice(1)
  .filter((m) => m[2]).map(([, t, label]) => ({ text: t.startsWith('"') ? t.slice(1, -1).replace(/""/g, '"') : t, label }));
const pairs = (a: [string, string][]): Row[] => a.map(([text, label]) => ({ text, label }));

export async function loadData() {
  const B77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/";
  const clinc = JSON.parse(await cached("data/clinc150.json", "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"));
  return {
    banking: { train: csv(await cached("data/banking77-train.csv", B77 + "train.csv")), test: csv(await cached("data/banking77-test.csv", B77 + "test.csv")) },
    clinc: { train: pairs(clinc.train), val: pairs(clinc.val), test: pairs(clinc.test), oosVal: pairs(clinc.oos_val), oosTest: pairs(clinc.oos_test) },
  };
}

export function seeded(seed: number) {
  let s = seed;
  return () => (s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
}

/** k examples per label, shuffled by seed. */
export function kShot(rows: Row[], k: number, seed: number): Row[] {
  const rand = seeded(seed);
  const by = new Map<string, Row[]>();
  for (const r of [...rows].sort(() => rand() - 0.5)) (by.get(r.label) ?? by.set(r.label, []).get(r.label)!).push(r);
  return [...by.values()].flatMap((g) => g.slice(0, k));
}
