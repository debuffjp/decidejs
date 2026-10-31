import { readFile } from "node:fs/promises";
import { createDecider, loadHead } from "../src/index.ts";
import { loadData } from "./data.ts";

const data = await loadData();
const d = await createDecider();
for (const [name, rows] of [["banking77", data.banking.test], ["clinc150", data.clinc.test]] as const) {
  const head = loadHead(await readFile(`heads/${name}.head.json`, "utf8"));
  const t0 = performance.now();
  const rs = await d.decideMany(rows.map((r) => r.text), head);
  const acc = rs.filter((r, i) => r.choice === rows[i].label).length / rows.length;
  console.log(name, (acc * 100).toFixed(1) + "%", "T", head.temperature, ((performance.now() - t0) / rows.length).toFixed(2), "ms/text batched");
}
const oos = await d.decideMany(data.clinc.oosTest.map((r) => r.text), loadHead(await readFile("heads/clinc150.head.json", "utf8")), { minConfidence: 0.93 });
console.log("oos null", (oos.filter((r) => r.choice === null).length / oos.length * 100).toFixed(1) + "%");
