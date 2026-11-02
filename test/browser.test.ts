import { test } from "node:test";
import assert from "node:assert/strict";
import { createWorkerDecider } from "../src/browser.ts";

/** A worker that answers init, then lets the test fire its error events. */
function fakeWorker() {
  const worker = {
    onmessage: null as ((e: any) => void) | null,
    onerror: null as ((e: any) => void) | null,
    onmessageerror: null as ((e: any) => void) | null,
    postMessage({ id, op }: { id: number; op: string }) {
      if (op === "init") worker.onmessage?.({ data: { id } });
    },
    terminate() {},
  };
  return worker;
}

const settle = (p: Promise<unknown>) =>
  Promise.race([p, new Promise((_, reject) => setTimeout(() => reject(new Error("still pending")), 100))]);

test("a worker that fails to load rejects createWorkerDecider", async () => {
  const worker = fakeWorker();
  worker.postMessage = () => {};
  const started = createWorkerDecider({}, worker as any);
  worker.onerror?.({ message: "" });
  await assert.rejects(settle(started), /worker failed/);
});

test("a worker crash rejects pending and later calls", async () => {
  const worker = fakeWorker();
  const d = await createWorkerDecider({}, worker as any);
  const pending = d.decide("a", ["x", "y"]);
  worker.onerror?.({ message: "out of memory" });
  await assert.rejects(settle(pending), /worker failed: out of memory/);
  await assert.rejects(settle(d.decide("a", ["x", "y"])), /worker failed: out of memory/);
});

test("a message the page can't read rejects pending calls", async () => {
  const worker = fakeWorker();
  const d = await createWorkerDecider({}, worker as any);
  const pending = d.decide("a", ["x", "y"]);
  worker.onmessageerror?.({});
  await assert.rejects(settle(pending), /worker failed/);
});

test("calls after terminate() reject", async () => {
  const d = await createWorkerDecider({}, fakeWorker() as any);
  d.terminate();
  await assert.rejects(settle(d.decide("a", ["x", "y"])), /terminated/);
});
