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

test("tiny training sets never get a temperature below 1", async () => {
  const { X, y } = clusters(2);
  const head = await fitHead(X, y, ["a", "b", "c"]);
  assert.ok(head.temperature >= 1, `temperature ${head.temperature}`);
});

test("fitHead fits the temperature on calibration examples when given, so noisy labels don't make it doubt itself", async () => {
  const { X, y } = clusters(60);
  // A third of the training labels are wrong, as from a noisy teacher.
  const noisy = y.map((c, i) => (i % 3 === 0 ? (c + 1) % 3 : c));
  const clean = clusters(20, 42);
  const plain = await fitHead(X, noisy, ["a", "b", "c"], { epochs: 20 });
  const calibrated = await fitHead(X, noisy, ["a", "b", "c"], { epochs: 20 }, undefined, 1, undefined, 1, clean);
  assert.ok(calibrated.temperature < plain.temperature, `${calibrated.temperature} < ${plain.temperature}`);
});

test("adam from 8 examples per label, sgd below", async () => {
  const many = { ...clusters(8), labels: ["a", "b", "c"] }, few = { ...clusters(7), labels: ["a", "b", "c"] };
  const same = (a: { weights: Float32Array }, b: { weights: Float32Array }) => assert.deepEqual([...a.weights], [...b.weights]);
  same(await fitHead(many.X, many.y, many.labels, { epochs: 5 }), await fitHead(many.X, many.y, many.labels, { epochs: 5, solver: "adam" }));
  same(await fitHead(few.X, few.y, few.labels, { epochs: 5 }), await fitHead(few.X, few.y, few.labels, { epochs: 5, solver: "sgd" }));
  assert.notDeepEqual(
    [...(await fitHead(many.X, many.y, many.labels, { epochs: 5, solver: "adam" })).weights],
    [...(await fitHead(many.X, many.y, many.labels, { epochs: 5, solver: "sgd" })).weights],
  );
});

test("a set learningRate or l2 keeps sgd", async () => {
  const { X, y } = clusters(8), labels = ["a", "b", "c"];
  const sgd = await fitHead(X, y, labels, { epochs: 5, solver: "sgd", learningRate: 1 });
  assert.deepEqual([...(await fitHead(X, y, labels, { epochs: 5, learningRate: 1 })).weights], [...sgd.weights]);
});

test("adam separates clusters and reports epochs", async () => {
  const { X, y } = clusters(20), labels = ["a", "b", "c"];
  const seen: number[] = [];
  const head = await fitHead(X, y, labels, { onEpoch: (e) => seen.push(e) });
  assert.ok(seen.length > 0);
  const right = X.filter((x, i) => { const z = logits({ labels, dim: 8, weights: head.weights, bias: head.bias }, x); return z.indexOf(Math.max(...z)) === y[i]; });
  assert.equal(right.length, X.length);
});

test("adam yields inside an epoch", async () => {
  const { X, y } = clusters(100), labels = ["a", "b", "c"];
  let ticks = 0, on = true;
  const tick = () => { ticks++; if (on) setImmediate(tick); };
  tick();
  await fitHead(X, y, labels, { epochs: 1, solver: "adam", holdout: 0.5 });
  on = false;
  assert.ok(ticks >= 6, String(ticks));
});
