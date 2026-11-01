import { test } from "node:test";
import assert from "node:assert/strict";
import { describeLabels, describePrompt, parseDescriptions } from "../src/describe.ts";

test("describePrompt lists every label", () => {
  const p = describePrompt(["card_arrival", "refund"], { context: "message to a bank" });
  assert.match(p, /- card_arrival\n- refund$/);
  assert.match(p, /message to a bank/);
});

test("parseDescriptions reads JSON wrapped in prose and drops bad entries", () => {
  const reply = 'Sure!\n```json\n{"card_arrival": "My card hasn\'t come yet.", "refund": 3, "other": "x"}\n```';
  assert.deepEqual(parseDescriptions(["card_arrival", "refund"], reply), { card_arrival: "My card hasn't come yet." });
  assert.deepEqual(parseDescriptions(["a"], "no json here"), {});
});

test("describeLabels falls back to the readable name", async () => {
  const d = await describeLabels(["card_arrival", "lost_card"], () => '{"card_arrival": "Where is my card?"}');
  assert.deepEqual(d, { card_arrival: "Where is my card?", lost_card: "lost card" });
});
