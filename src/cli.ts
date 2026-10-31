#!/usr/bin/env node
// decidejs train <examples.csv|.jsonl> [-o head.json] [--model id] [--no-label-names]
// decidejs decide <head.json> "<text>" [--min-confidence 0.9]
import { readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { createDecider, loadHead, saveHead, type Example } from "./index.ts";

const USAGE = `usage:
  decidejs train <examples.csv|.jsonl> [-o head.json] [--model <hf-id>] [--no-label-names]
  decidejs decide <head.json> "<text>" [--min-confidence <0-1>]

CSV needs a header with "text" and "label" columns. JSONL needs {"text", "label"} per line.`;

function parseCsv(source: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], field = "", quoted = false;
  for (let i = 0; i < source.length; i++) {
    const ch = source[i];
    if (quoted) {
      if (ch === '"' && source[i + 1] === '"') { field += '"'; i++; } else if (ch === '"') quoted = false; else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (ch !== "\r") field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.some((f) => f !== ""));
}

export async function readExamples(file: string): Promise<Example<string>[]> {
  const source = await readFile(file, "utf8");
  if (file.endsWith(".jsonl")) return source.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  const [header, ...rows] = parseCsv(source);
  const t = header.indexOf("text"), l = header.indexOf("label");
  if (t < 0 || l < 0) throw new Error(`${file}: the header needs "text" and "label" columns, found ${header.join(", ")}`);
  return rows.map((r) => ({ text: r[t], label: r[l] }));
}

async function main(argv: string[]) {
  const { values, positionals } = parseArgs({
    args: argv, allowPositionals: true,
    options: {
      out: { type: "string", short: "o" }, model: { type: "string" }, "no-label-names": { type: "boolean" },
      "min-confidence": { type: "string" }, help: { type: "boolean", short: "h" },
    },
  });
  const [command, file, text] = positionals;
  if (values.help || !command) return console.log(USAGE);

  if (command === "train" && file) {
    const examples = await readExamples(file);
    const counts = new Map<string, number>();
    for (const e of examples) counts.set(e.label, (counts.get(e.label) ?? 0) + 1);
    const fewest = Math.min(...counts.values());
    console.error(`${examples.length} examples, ${counts.size} labels, fewest per label: ${fewest}`);
    if (fewest < 3) console.error("warning: labels with fewer than 3 examples will be unreliable");
    const d = await createDecider({ model: values.model });
    const t0 = performance.now();
    const head = await d.train(examples, { labelNames: !values["no-label-names"] });
    const out = values.out ?? "head.json";
    const json = saveHead(head);
    await writeFile(out, json);
    console.error(`wrote ${out} (${Math.round(json.length / 1024)} KB) in ${((performance.now() - t0) / 1000).toFixed(1)} s, temperature ${head.temperature}`);
    return;
  }
  if (command === "decide" && file && text !== undefined) {
    const head = loadHead(await readFile(file, "utf8"));
    const d = await createDecider({ model: head.model });
    const min = values["min-confidence"];
    const r = min === undefined ? await d.decide(text, head) : await d.decide(text, head, { minConfidence: Number(min) });
    console.log(JSON.stringify({ choice: r.choice, confidence: Number(r.confidence.toFixed(4)) }));
    return;
  }
  console.error(USAGE);
  process.exitCode = 2;
}

await main(process.argv.slice(2)).catch((e) => { console.error(String(e?.message ?? e)); process.exitCode = 1; });
