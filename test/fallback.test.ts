import { test } from "node:test";
import assert from "node:assert/strict";
import { NONE, withFallback, type Trainable } from "../src/fallback.ts";
import type { Choices, Decision, Example, Head, TrainOptions } from "../src/index.ts";

// A decider whose confidence is whatever the text says ("0.95 ..."), so routing is easy to test.
function fake() {
  const trained: Example<string>[][] = [], trainLabels: (readonly string[] | undefined)[] = [], trainOpts: TrainOptions<string>[] = [];
  const decider: Trainable = {
    async decide<L extends string>(text: string, using: Choices<L>): Promise<Decision<L>> {
      const confidence = Number(text.split(" ")[0]);
      const labels = (Array.isArray(using) ? using : "weights" in using ? (using as Head<L>).labels : Object.keys(using)) as L[];
      const head = { labels };
      const probs = Object.fromEntries(head.labels.map((l, i) => [l, i === 0 ? confidence : (1 - confidence) / (head.labels.length - 1)])) as Record<L, number>;
      const named = text.split(" ")[1] as L;
      return { choice: labels.includes(named) ? named : head.labels[0], confidence, probs };
    },
    async train<L extends string>(examples: readonly Example<L>[], options?: TrainOptions<L>) {
      trained.push([...examples]);
      trainLabels.push(options?.labels);
      trainOpts.push(options ?? {});
      return head(["a", "b"] as L[], trained.length);
    },
  };
  return { decider, trained, trainLabels, trainOpts };
}
const head = <L extends string>(labels: L[], version = 0): Head<L> =>
  ({ model: "m", labels, dim: 1, weights: new Float32Array(labels.length), bias: new Float32Array(labels.length), temperature: version });

test("confident texts stay local; the rest go to ask", async () => {
  const { decider } = fake();
  const asked: string[] = [];
  const route = withFallback(decider, head(["a", "b"]), { ask: (text) => (asked.push(text), "b") });
  const local = await route("0.95 hello");
  assert.equal(local.source, "local");
  assert.equal(local.choice, "a");
  const fell = await route("0.5 hmm");
  assert.equal(fell.source, "fallback");
  assert.equal(fell.choice, "b");
  assert.equal(fell.confidence, 0.5);
  assert.deepEqual(asked, ["0.5 hmm"]);
});

test("an answer that isn't a label is null", async () => {
  const { decider } = fake();
  const route = withFallback(decider, head(["a", "b"]), { ask: () => "zebra" as "a" });
  assert.equal((await route("0.1 x")).choice, null);
});

test("learns from answers and retrains on examples plus answers", async () => {
  const { decider, trained } = fake();
  const examples = [{ text: "x", label: "a" as const }, { text: "y", label: "b" as const }];
  const route = withFallback(decider, head(["a", "b"]), { ask: () => "b", examples, retrainEvery: 2 });
  await route("0.1 one");
  assert.equal(trained.length, 0);
  await route("0.2 two");
  await route.retrain();
  assert.deepEqual(trained[0].map((e) => e.text), ["x", "y", "0.1 one", "0.2 two"]);
  assert.equal(route.learned.length, 2);
  assert.notEqual(route.head!.temperature, 0);
});

test("retrain without examples throws", async () => {
  const route = withFallback(fake().decider, head(["a", "b"]), { ask: () => null });
  await assert.rejects(route.retrain(), /needs `examples`/);
});

test("starts from labels alone and trains its first head from the answers", async () => {
  const { decider, trained, trainLabels } = fake();
  const route = withFallback(decider, ["a", "b", "c"], { ask: () => "b" });
  assert.equal(route.head, null);
  await assert.rejects(route.retrain(), /nothing to train on/);
  assert.equal((await route("0.95 sure")).source, "local");
  assert.equal((await route("0.3 unsure")).choice, "b");
  await route.retrain();
  assert.deepEqual(trained[0], [{ text: "0.3 unsure", label: "b" }]);
  assert.deepEqual(trainLabels[0], ["a", "b", "c"]);
  assert.ok(route.head);
});

test("starting from labels, calibrates on zero-shot's confident answers once there are 50, up to 4 per label", async () => {
  const { decider, trainOpts } = fake();
  const route = withFallback(decider, ["a", "b"], { ask: () => "b" });
  await route("0.3 unsure");
  for (let i = 0; i < 3; i++) await route(`0.95 sure ${i}`);
  await route.retrain();
  assert.equal(trainOpts[0].calibrateOn, undefined); // 3 is under 50
  const many = withFallback(decider, Array.from({ length: 20 }, (_, i) => `l${i}`), { ask: () => "l1" });
  for (let i = 0; i < 100; i++) await many(`0.95 l${i % 20} ${i}`);
  await many("0.3 unsure");
  await many.retrain();
  assert.equal(trainOpts[1].calibrateOn?.length, 80); // 4 per label
  const own = [{ text: "x", label: "a" as const }];
  const mine = withFallback(decider, ["a", "b"], { ask: () => "b", trainOptions: { calibrateOn: own } });
  await mine("0.3 unsure");
  await mine.retrain();
  assert.equal(trainOpts[2].calibrateOn, own);
});

test("learns null answers as NONE and answers null locally when sure it's none", async () => {
  const { decider, trained, trainLabels } = fake();
  const route = withFallback(decider, ["a", "b"], { ask: () => null });
  assert.equal((await route("0.3 off-topic")).choice, null);
  assert.deepEqual(route.learned, [{ text: "0.3 off-topic", label: NONE }]);
  await route.retrain();
  assert.deepEqual(trainLabels[0], ["a", "b", NONE]);
  assert.equal(trained[0][0].label, NONE);
  const sure = await withFallback(decider, head(["a", NONE]), { ask: () => "a" })(`0.95 ${NONE}`);
  assert.equal(sure.source, "local");
  assert.equal(sure.choice, null);
});

test("learnNone: false keeps null answers out", async () => {
  const route = withFallback(fake().decider, ["a", "b"], { ask: () => null, learnNone: false });
  await route("0.3 off-topic");
  assert.equal(route.learned.length, 0);
});
