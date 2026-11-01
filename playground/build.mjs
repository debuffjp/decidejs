// Builds debuff.jp/decidejs into playground/dist: the front page, the playground and the two lanes demo.
// npm run playground, then serve playground/dist cross-origin isolated.
import { build } from "esbuild";
import { cp, mkdir } from "node:fs/promises";
const out = new URL("./dist/", import.meta.url);
await mkdir(new URL("heads/", out), { recursive: true });
await build({
  entryPoints: { "playground/app": "playground/playground/app.ts", worker: "src/worker.ts", "lanes/lanes": "playground/lanes/lanes.ts" },
  bundle: true, format: "esm", platform: "browser", outdir: "playground/dist", minify: true,
  alias: { "onnxruntime-web/webgpu": "onnxruntime-web/wasm" },
});
for (const page of ["index.html", "ui.css", "playground/index.html", "lanes/index.html"]) await cp(new URL(`./${page}`, import.meta.url), new URL(page, out));
for (const h of ["banking77", "clinc150"]) await cp(new URL(`../heads/${h}.head.json`, import.meta.url), new URL(`heads/${h}.head.json`, out));
console.log("built playground/dist");
