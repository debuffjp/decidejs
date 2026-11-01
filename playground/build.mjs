// Builds the playground into playground/dist: npm run playground (then serve it cross-origin isolated).
import { build } from "esbuild";
import { cp, mkdir } from "node:fs/promises";
const out = new URL("./dist/", import.meta.url);
await mkdir(new URL("heads/", out), { recursive: true });
await build({
  entryPoints: { app: "playground/app.ts", worker: "src/worker.ts", "lanes/lanes": "playground/lanes/lanes.ts" },
  bundle: true, format: "esm", platform: "browser", outdir: "playground/dist", minify: true,
  alias: { "onnxruntime-web/webgpu": "onnxruntime-web/wasm" },
});
await cp(new URL("./index.html", import.meta.url), new URL("index.html", out));
await cp(new URL("./lanes/index.html", import.meta.url), new URL("lanes/index.html", out));
for (const h of ["banking77", "clinc150"]) await cp(new URL(`../heads/${h}.head.json`, import.meta.url), new URL(`heads/${h}.head.json`, out));
console.log("built playground/dist");
