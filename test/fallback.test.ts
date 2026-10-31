import { test } from "node:test";
import assert from "node:assert/strict";
import { withFallback, type Trainable } from "../src/fallback.ts";
import type { Decision, Example, Head } from "../src/index.ts";

// A decider whose confidence is whatever the text says ("0.95 ..."), so routing is easy to test.
function fake() {
  const trained: Example<string>[][] = [];
  const decider: Trainable = {
    async decide<L extends string>(text: string, head: Head<L>): Promise<Decision<L>> {
      const confidence = Number(text.split(" ")[0]);
      const probs = Object.fromEntries(head.labels.map((l, i) => [l, i === 0 ? confidence : (1 - confidence) / (head.labels.length - 1)])) as Record<L, number>;
      return { choice: head.labels[0], confidence, probs };
    },
    async train<L extends string>(examples: readonly Example<L>[]) {
      trained.push([...examples]);
      return head(["a", "b"] as L[], trained.length);
    },
  };
  return { decider, trained };
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
  assert.notEqual(route.head.temperature, 0);
});

test("retrain without examples throws", async () => {
  const route = withFallback(fake().decider, head(["a", "b"]), { ask: () => null });
  await assert.rejects(route.retrain(), /needs `examples`/);
});
