// Installs the packed package into an empty project and uses it the way a user would: every export resolves and
// typechecks, decide() and train() run on the real model, and the CLI trains and decides. Catches a wrong `exports`
// map or a file missing from `files`, which the unit tests can't see.
//   node scripts/smoke.mjs
import { execSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const run = (cmd, cwd) => execSync(cmd, { cwd, stdio: "inherit", shell: true });
const root = resolve(import.meta.dirname, "..");
const dir = mkdtempSync(join(tmpdir(), "decidejs-smoke-"));
try {
  run(`npm pack --pack-destination "${dir}"`, root);
  const tgz = readdirSync(dir).find((f) => f.endsWith(".tgz"));
  writeFileSync(join(dir, "package.json"), JSON.stringify({ name: "smoke", private: true, type: "module" }));
  run(`npm i --no-audit --no-fund ./${tgz} typescript@5 @types/react@18`, dir);
  writeFileSync(join(dir, "use.mts"), `
import { createDecider, saveHead, loadHead, NONE } from "decidejs";
import type { createWorkerDecider } from "decidejs/browser";
import type { useDecider } from "decidejs/react";
import type * as sw from "decidejs/switch";
import { choice as ask, typesafeClient } from "decidejs/typesafe";
const d = await createDecider();
const r = await d.decide("my card still hasn't arrived", ["card_arrival", "lost_card", "refund"] as const);
const choice: "card_arrival" | "lost_card" | "refund" = r.choice;
if (choice !== "card_arrival") throw new Error("decide: " + choice);
const head = await d.train([{ text: "where is my card", label: "arrival" }, { text: "I want my money back", label: "refund" }, { text: "has my card shipped", label: "arrival" }, { text: "refund me please", label: "refund" }]);
const again = loadHead(saveHead(head));
if ((await d.decide("give me a refund", again)).choice !== "refund") throw new Error("train");
const { answers } = await typesafeClient(d).systemOne({ state: { document: "my card still has not arrived" }, questions: { topic: ask("What is this about?", { card_arrival: null, refund: null }) } });
const topic: "card_arrival" | "refund" = answers.topic.choice;
if (topic !== "card_arrival") throw new Error("systemOne: " + topic);
console.log("smoke: library ok", NONE);
`);
  writeFileSync(join(dir, "tsconfig.json"), JSON.stringify({ compilerOptions: { module: "nodenext", target: "es2022", strict: true, noEmit: true, skipLibCheck: false, types: [] }, files: ["use.mts"] }));
  run("npx tsc -p .", dir);
  run("node use.mts", dir);
  writeFileSync(join(dir, "ex.csv"), "text,label\nwhere is my card,arrival\nhas my card shipped,arrival\nI want my money back,refund\nrefund me please,refund\n");
  run("npx decidejs train ex.csv -o head.json --types", dir);
  run('npx decidejs decide head.json "give me a refund"', dir);
  console.log("smoke: ok");
} finally {
  rmSync(dir, { recursive: true, force: true });
}
