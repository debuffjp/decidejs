# Runtime, DX and integrations: ranked ideas

Baseline (from the repo): first visit downloads the 34 MB bge-small model plus 12.9 MB wasm (3.3 MB gzipped); a warm-cache start still took 3.8 s (docs/guide-browser.md). Threads need COOP/COEP. Bundlers need an `onnxruntime-web/webgpu` alias (docs/bundlers.md). `createWorkerDecider` only wraps decide, decideMany and train.

1. **Ship a quantized default (q8 or q4) and say its size up front.** Impact: high on first load. A quantized all-MiniLM-L6-v2 is about 23 MB ([SitePoint](https://www.sitepoint.com/optimizing-transformers-js-production/)), and bge-small int8 would be a fraction of 34 MB. Cost: low (a `dtype` option plus a BANKING77 re-run to check the accuracy and calibration hit). Evidence: Xenova conversions publish int8, uint8, q4 and q4f16 files ([HF commit](https://huggingface.co/Xenova/bge-base-en/commit/5d55cede3294019419efcb9c4dc81f08f94de2bb)).

2. **Ready-state API: `d.cached()`, `preload()` and storage quota.** Impact: high on DX. Progress events don't fire on cache hits (guide-browser.md:39), so apps can't tell "first visit" from "warm". Expose a check via the Cache API and `navigator.storage.estimate()` and call `navigator.storage.persist()` ([HF forum](https://discuss.huggingface.co/t/transformers-js-retrieving-the-size-of-models-in-mb-gb-before-running/174303)). Cost: low.

3. **Kill the bundler alias.** Impact: high on adoption, since alias config is the top setup failure in transformers.js apps. Import `onnxruntime-web/wasm` directly from a `decidekit/wasm` entry, so no user config is needed. Cost: medium (depends on how transformers.js 4 lets you inject the backend).

4. **Node/Bun/Deno/edge story.** Impact: high for server and agent users. Node already works (CLI). Test Bun and Deno in CI. For Cloudflare Workers, run the head-only path (dot product plus softmax) on embeddings from Workers AI's bge-small-en, so training in the browser and deciding at the edge share one head file. Cost: medium. Evidence: needs measuring; the head is plain JSON (156 KB).

5. **Agent-framework adapters: Vercel AI SDK, LangChain router, MCP tool.** Impact: high on discoverability. `withFallback` already routes to an LLM when unsure. Ship it as an AI SDK middleware ("decide locally, else call the model") and a `decide` MCP tool backed by a head file. Cost: low-medium. Evidence: no measurement yet; it matches the "learns from an LLM" pitch.

6. **Typed schemas and multi-question decisions.** Impact: medium. `decide(text, { intent: [...], urgency: ["low","high"] })` encodes once and runs several heads, so you get N decisions for one embedding cost. Accept Standard Schema enums (zod, valibot) as choices. Cost: low, since the encoding is shared.

7. **WebGPU path, measured.** Impact: medium on batch and training (training took 228 s in a worker). Benchmark `device: "webgpu"` for `train`/`decideMany`. Small batches are likely slower than wasm because of dispatch overhead. Cost: low to measure.

8. **Fuller worker and React hooks.** Impact: medium. The worker lacks progress for cache hits, cancellation (AbortSignal) and transferable batches. Add a SharedWorker option so tabs share one model instance. Cost: low.

9. **Devtools: `decidekit inspect head.json`.** Prints labels, a reliability diagram and confusable label pairs, and `--watch` logs low-confidence calls in dev. Impact: medium on trust. Cost: low.

10. **Shared cross-site model cache.** Impact: high in theory, but low today. Browsers partition HTTP and Cache API storage by top-level site, so a shared CDN copy is not reused across sites. Only revisit if a cross-origin storage proposal ships. Cost: none now.

Adoption blockers seen in comparable libraries: large first downloads with no clear "it's cached" state (transformers.js apps), bundler and wasm path config, and no server parity. compromise and wink stay popular because they are tiny and synchronous. ml5 trades accuracy for zero setup. decidekit should advertise a "small" tier (MiniLM q8, about 23 MB) and a no-config import.

Note: one web search backed this. The Cloudflare, WebGPU and cross-site cache claims need verifying before acting on them.
