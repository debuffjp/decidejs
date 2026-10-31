// Trains the ready-made heads in heads/ on the public datasets' training data (CLINC150: train plus val).
// npm run heads
import { writeFile } from "node:fs/promises";
import { createDecider, saveHead } from "../src/index.ts";
import { loadData } from "../eval/data.ts";

const data = await loadData();
const d = await createDecider();
for (const [name, rows] of [["banking77", data.banking.train], ["clinc150", [...data.clinc.train, ...data.clinc.val]]] as const) {
  const t0 = performance.now();
  const json = saveHead(await d.train(rows));
  await writeFile(`heads/${name}.head.json`, json);
  console.log(name, rows.length, "examples", Math.round(json.length / 1024), "KB", ((performance.now() - t0) / 1000).toFixed(0), "s");
}
