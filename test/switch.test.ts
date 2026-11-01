import { test } from "node:test";
import assert from "node:assert/strict";
import { switchDecider, textOf, type SwitchDecider } from "../src/switch.ts";
import type { Choices, Decision, Example, Head } from "../src/index.ts";

// A decider that picks the option whose text appears in the read text, with the confidence the text starts with.
function fake() {
  const read: string[] = [], trained: Example<string>[][] = [];
  const local = {
    model: "m",
    async decide<L extends string>(text: string, using: Choices<L>): Promise<Decision<L>> {
      read.push(text);
      const entries = (Array.isArray(using) ? using.map((l) => [l, l]) : "weights" in using ? (using as Head<L>).labels.map((l) => [l, l]) : Object.entries(using)) as [L, string][];
      const confidence = Number(text.split(" ")[0]) || 0.99;
      const hit = entries.find(([, t]) => text.includes(t)) ?? entries[0];
      const probabilities = Object.fromEntries(entries.map(([l]) => [l, l === hit[0] ? confidence : (1 - confidence) / (entries.length - 1)])) as Record<L, number>;
      return { choice: hit[0], confidence, probabilities };
    },
    async train<L extends string>(examples: readonly Example<L>[]) {
      trained.push([...examples]);
      return { model: "m", labels: ["a", "b"] as L[], dim: 1, weights: new Float32Array(2), bias: new Float32Array(2), temperature: 1 };
    },
  };
  return { local, read, trained };
}

const ask = (decider: SwitchDecider, state: Record<string, string>, questions: Parameters<SwitchDecider["decide"]>[0]["decisionQuestions"]) =>
  decider.decide({ decisionState: state, decisionQuestions: questions });

test("answers choice, bool and score questions in switch's shapes", async () => {
  const { local } = fake();
  const r = await ask(switchDecider(local), { text: "the moon and the sun" }, {
    c: { type: "choice", instructions: "which", criteria: { a: "planet", b: "moon" } },
    y: { type: "bool", instructions: "is it", criteria: { true: "sun", false: "nothing" } },
    s: { type: "score", instructions: "how", criteria: ["nothing", "sun", "both"] },
  });
  assert.equal(r.stopReason, "stop");
  assert.deepEqual(r.decisionAnswers.c && { ...r.decisionAnswers.c, probabilities: undefined }, { type: "choice", choice: "b", confidence: 0.99, probabilities: undefined });
  assert.equal(r.decisionAnswers.y.type === "bool" && r.decisionAnswers.y.probability, 0.99);
  assert.deepEqual(r.decisionAnswers.s, { type: "score", score: 1, confidence: 0.99 });
  assert.equal(r.decisionUsage?.cost.total, 0);
  assert.equal(r.model, "decidejs/m");
});

test("a single option is the answer, and errors come back as a failed result", async () => {
  const { local } = fake();
  const d = switchDecider(local);
  const one = await ask(d, {}, { c: { type: "choice", instructions: "", criteria: { only: "x" } } });
  assert.deepEqual(one.decisionAnswers.c, { type: "choice", choice: "only", probabilities: { only: 1 }, confidence: 1 });
  const broken = switchDecider({ ...local, decide: async () => { throw new Error("boom"); } });
  const r = await ask(broken, {}, { c: { type: "choice", instructions: "", criteria: { a: "a", b: "b" } } });
  assert.equal(r.stopReason, "error");
  assert.equal(r.errorMessage, "boom");
});

test("text picks what decidejs reads; by default the instructions then the state as lines", async () => {
  assert.equal(textOf("Pick one.", { target: "Moon", path: ["A", "B"], at: { title: "NASA" } }), "Pick one.\ntarget: Moon\npath: A, B\ntitle: NASA");
  const { local, read } = fake();
  await ask(switchDecider(local, { text: (s) => String(s.target) }), { target: "moon", other: "x" }, { c: { type: "choice", instructions: "long", criteria: { a: "planet", b: "moon" } } });
  assert.deepEqual(read, ["moon"]);
});

test("unsure choices go to the teacher, which sees the same request and teaches decidejs", async () => {
  const { local, trained } = fake();
  const seen: unknown[] = [];
  const teacher: SwitchDecider = {
    model: "t", window: 1000, charsPerToken: 3,
    async decide(request) {
      seen.push(request.decisionState);
      return { decisionAnswers: { c: { type: "choice", choice: "a", probabilities: { a: 1, b: 0 }, confidence: 1 } }, api: "t", provider: "t", model: "t", stopReason: "stop", timestamp: 0, tooLong: false, milliseconds: 1 };
    },
  };
  const d = switchDecider(local, { teacher, minConfidence: 0.9, text: (s) => String(s.text) });
  const q = { c: { type: "choice" as const, instructions: "", criteria: { a: "planet", b: "moon" } } };
  const sure = await ask(d, { text: "0.95 moon" }, q);
  assert.equal(sure.decisionAnswers.c.type === "choice" && sure.decisionAnswers.c.choice, "b");
  for (let i = 0; i < 8; i++) {
    const r = await ask(d, { text: `0.5 moon ${i}` }, q);
    assert.equal(r.decisionAnswers.c.type === "choice" && r.decisionAnswers.c.choice, "a");
  }
  assert.deepEqual(seen[0], { text: "0.5 moon 0" });
  assert.equal(seen.length, 8);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(trained.length, 1);
  assert.equal(trained[0].length, 8);
});
