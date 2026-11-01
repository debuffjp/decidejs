import assert from "node:assert/strict";
import { test } from "node:test";
import { choice, noul, score, typesafeClient } from "../src/typesafe.ts";

const seen: unknown[] = [];
const decider = {
  async decide(text: string, using: any) {
    seen.push({ text, using });
    const labels: string[] = Array.isArray(using) ? using : Object.keys(using);
    const probabilities = Object.fromEntries(labels.map((l, i) => [l, i === 0 ? 0.61 : (1 - 0.61) / (labels.length - 1)]));
    return { choice: labels[0], confidence: 0.61, probabilities } as any;
  },
};

test("returns TypeSafe's answer shape and confidence formula", async () => {
  const { answers } = await typesafeClient(decider).systemOne({
    state: "my shoes are the wrong size",
    questions: { department: choice("Which team?", { returns: null, shipping: null, billing: null }) },
  });
  const a = answers.department;
  const typed: "returns" | "shipping" | "billing" = a.choice;
  assert.equal(typed, "returns");
  assert.equal(a.type, "choice");
  assert.ok(Math.abs(a.confidence - 0.415) < 1e-9);
  assert.equal(a.probabilities.returns, 0.61);
});

test("answers each question independently; null criteria become names or fill in beside descriptions", async () => {
  seen.length = 0;
  const { answers } = await typesafeClient(decider).systemOne({
    state: { subject: "Refund", document: "I was charged twice", attempts: 2 },
    questions: { a: choice("?", { x: null, y: null }), b: choice("?", { late_fee: null, z: "Something else" }) },
  });
  assert.deepEqual(Object.keys(answers), ["a", "b"]);
  assert.deepEqual(seen[0], { text: "Refund\nI was charged twice\n2", using: ["x", "y"] });
  assert.deepEqual((seen[1] as any).using, { late_fee: "late fee", z: "Something else" });
});

test("rejects what it can't answer", async () => {
  assert.throws(() => noul("Is this a refund?"), /only choice/);
  assert.throws(() => score("How bad?", ["a", "b"]), /only choice/);
  await assert.rejects(typesafeClient(decider).systemOne({ state: { photo: {} }, questions: { a: choice("?", { x: null, y: null }) } }), /state values/);
});
