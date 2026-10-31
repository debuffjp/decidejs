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
