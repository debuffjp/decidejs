// Times train() on BANKING77, then again after adding 10 examples: the second run reuses the cached embeddings.
import { createDecider } from "../src/index.ts";
import { loadData } from "./data.ts";

const { banking } = await loadData();
const d = await createDecider();
const first = banking.train.slice(0, -10);
let t0 = performance.now();
await d.train(first);
const a = (performance.now() - t0) / 1000;
t0 = performance.now();
await d.train(banking.train);
console.log(JSON.stringify({ examples: banking.train.length, firstSeconds: +a.toFixed(1), retrainSeconds: +((performance.now() - t0) / 1000).toFixed(1) }));
