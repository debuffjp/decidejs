// Head math without the encoder: synthetic embeddings, so these run in milliseconds and offline.
import { test } from "node:test";
import assert from "node:assert/strict";
import { fitHead, logits, softmax } from "../src/head.ts";
import { loadHead, saveHead } from "../src/index.ts";

// Three labels, each a noisy cluster around its own axis in 8 dimensions.
function clusters(perLabel: number, seed = 1) {
  let s = seed;
  const rand = () => (s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31 - 0.5;
  const X: number[][] = [], y: number[] = [];
  for (let c = 0; c < 3; c++) for (let i = 0; i < perLabel; i++) {
    const v = Array.from({ length: 8 }, (_, j) => (j === c ? 1 : 0) + rand() * 0.6);
    const n = Math.hypot(...v);
    X.push(v.map((x) => x / n)); y.push(c);
  }
  return { X, y };
}
const argmax = (z: ArrayLike<number>) => Array.from(z).indexOf(Math.max(...Array.from(z)));

test("softmax sums to 1 and survives large logits", () => {
  const p = softmax([1000, 1001, 999], 1);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.equal(argmax(p), 1);
});

test("fitHead separates clusters", async () => {
  const { X, y } = clusters(30);
  const head = await fitHead(X, y, ["a", "b", "c"], { epochs: 20 });
  const test = clusters(20, 99);
  const right = test.X.filter((x, i) => argmax(logits(head, x)) === test.y[i]).length;
  assert.ok(right / test.X.length > 0.95, `accuracy ${right / test.X.length}`);
  assert.ok(head.temperature > 0);
});

test("fitHead with 5 examples per label still works (centroid start)", async () => {
  const { X, y } = clusters(5);
  const head = await fitHead(X, y, ["a", "b", "c"]);
  const test = clusters(20, 42);
  const right = test.X.filter((x, i) => argmax(logits(head, x)) === test.y[i]).length;
  assert.ok(right / test.X.length > 0.9, `accuracy ${right / test.X.length}`);
});

test("fitHead is repeatable for a seed", async () => {
  const { X, y } = clusters(10);
  const a = await fitHead(X, y, ["a", "b", "c"], { seed: 3, epochs: 5 });
  const b = await fitHead(X, y, ["a", "b", "c"], { seed: 3, epochs: 5 });
  assert.deepEqual(a.weights, b.weights);
});

test("saveHead and loadHead round-trip exactly", async () => {
  const { X, y } = clusters(10);
  const head = { model: "m", ...(await fitHead(X, y, ["a", "b", "c"], { epochs: 3 })) };
  const back = loadHead(saveHead(head));
  assert.deepEqual(back.weights, head.weights);
  assert.deepEqual(back.bias, head.bias);
  assert.deepEqual(back.labels, head.labels);
  assert.equal(back.temperature, head.temperature);
});
