import { test } from "node:test";
import assert from "node:assert/strict";
import { createDecider } from "../src/index.ts";

const d = await createDecider();

test("train() rejects bad input with a clear message", async () => {
  await assert.rejects(d.train([]), /at least 2 labels/);
  await assert.rejects(d.train([{ text: "a", label: "x" }, { text: "b", label: "x" }]), /at least 2 labels, got 1/);
  await assert.rejects(d.train([{ text: "a", label: "x" }, { text: 3 as any, label: "y" }]), /example 1 needs a string text/);
  await assert.rejects(d.train([{ text: "a", label: "" }, { text: "b", label: "y" }]), /example 0/);
});

test("decideMany() matches decide()", async () => {
  const texts = ["my card never arrived", "I want my money back"];
  const many = await d.decideMany(texts, ["card_arrival", "refund"]);
  for (const [i, t] of texts.entries()) {
    const one = await d.decide(t, ["card_arrival", "refund"]);
    assert.equal(many[i].choice, one.choice);
    assert.ok(Math.abs(many[i].confidence - one.confidence) < 1e-4);
  }
  assert.deepEqual(many.map((r) => r.choice), ["card_arrival", "refund"]);
});

test("a head read with JSON.parse throws instead of acting as labels", async () => {
  const head = await d.train([{ text: "a", label: "x" }, { text: "b", label: "y" }]);
  const { saveHead } = await import("../src/index.ts");
  await assert.rejects(d.decide("a", JSON.parse(saveHead(head))), /loadHead/);
});
