// Compile-time checks of decide()'s result types; \`npm run typecheck\` fails if they regress.
import { createDecider } from "../src/index.ts";

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
const expect = <T extends true>() => {};

declare const d: Awaited<ReturnType<typeof createDecider>>;
const a = await d.decide("x", ["yes", "no"]);
expect<Equal<typeof a.choice, "yes" | "no">>();
const b = await d.decide("x", ["yes", "no"], { minConfidence: 0.9 });
expect<Equal<typeof b.choice, "yes" | "no" | null>>();

import { createWorkerDecider } from "../src/browser.ts";
declare const w: Awaited<ReturnType<typeof createWorkerDecider>>;
const c = await w.decide("x", ["yes", "no"], { minConfidence: 0.9 });
expect<Equal<typeof c.choice, "yes" | "no" | null>>();

const e = await d.decide("x", { refund: "wants their money back", bug: "something is broken" });
expect<Equal<typeof e.choice, "refund" | "bug">>();

const m = await d.decideMany(["x"], ["yes", "no"]);
expect<Equal<(typeof m)[number]["choice"], "yes" | "no">>();
const n = await w.decideMany(["x"], ["yes", "no"], { minConfidence: 0.5 });
expect<Equal<(typeof n)[number]["choice"], "yes" | "no" | null>>();

import { useDecision, type WorkerDecider } from "../src/react.ts";
declare const wd: WorkerDecider | null;
const hd = useDecision(wd, "x", ["yes", "no"]);
expect<Equal<NonNullable<typeof hd.decision>["choice"], "yes" | "no">>();
const hn = useDecision(wd, "x", ["yes", "no"], { minConfidence: 0.9 });
expect<Equal<NonNullable<typeof hn.decision>["choice"], "yes" | "no" | null>>();

import { withFallback } from "../src/index.ts";
declare const hh: import("../src/index.ts").Head<"a" | "b">;
const route = withFallback(d, hh, { ask: async () => "a" as const });
const rr = await route("x");
expect<Equal<typeof rr.choice, "a" | "b" | null>>();
