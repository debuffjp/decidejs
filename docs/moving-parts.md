# Moving parts

Every piece of decidejs on one page: what it does, where it lives, the numbers that shaped it and where to read more.
Read [how-it-works.md](how-it-works.md) for the maths and the guides for how to use each piece.

```
text ──► encoder (bge-small, 384 numbers) ──┬──► zero-shot: cosine to each label's name or description ──► softmax
                                            └──► trained head: one softmax layer, calibrated temperature
                                                         │
                       withFallback: below minConfidence ask an LLM ──► answer becomes a training example ──► retrain
```

## 1. The encoder

**What:** turns a text into 384 numbers so that texts with similar meaning land close together. The default is
`Xenova/bge-small-en-v1.5` at 8-bit weights (34 MB), run by transformers.js on onnxruntime's wasm build. Mean pooling,
normalised.

**Where:** `createDecider()` in `src/index.ts`.

**Numbers:** 3.2 ms per text in Node, 1.1 ms batched; 7.7 ms median and 13.8 ms p95 per decision in Chrome with
threads. Chosen over all-MiniLM-L6-v2 (23 MB, 1.6x faster) for 3 to 5 points of accuracy; gte-small was rejected
because its zero-shot confidence couldn't be calibrated.

**Read more:** [guide-training.md](guide-training.md) (model comparison), [how-it-works.md](how-it-works.md).

## 2. Zero-shot scoring

**What:** compares the text with each option and turns the similarities into probabilities. A bare label is read as
text ("card_arrival" as "card arrival"). A described label (`{ card_arrival: "My card still hasn't arrived." }`) scores
by the better of its name and its description. The softmax temperature is fixed per model: 0.03 for bge-small names,
0.025 for described labels, 0.05 for other models.

**Where:** `score()` in `src/index.ts`. `describeLabels()` in `src/describe.ts` asks an LLM to write the descriptions.

**Numbers:** on the dev splits, names alone 66% (BANKING77) and 73% (CLINC150); with descriptions 72% and 83%. Unlike
an LLM prompt, the order of the options never changes the answer (Llama 3.1 8B moved 9 points with label order).

**Read more:** [how-it-works.md](how-it-works.md#describe-your-labels), [guide-confidence.md](guide-confidence.md).

## 3. Trained heads

**What:** one softmax layer over the frozen embeddings: a weight vector and a bias per label. Training starts each
label at the average of its examples plus its name (weight 1), then runs gradient descent, picks the epoch count on
held-out examples and retrains on everything.

**Where:** `train()` in `src/index.ts`, `fitHead()` in `src/head.ts`.

**Numbers:** BANKING77 82.3% with 5 examples per label, 90.6% with all 10,003 (the shipped head); CLINC150 89.6% and
95.0%. A head is 156 KB for 77 labels. Training all of BANKING77 takes 228 s in a browser worker.

**Read more:** [guide-training.md](guide-training.md).

## 4. Calibration

**What:** makes `confidence` mean what it says. Heads fit a temperature on held-out answers (5 folds when there are
fewer than 200, so every example is held out once). Zero-shot uses the fixed per-model temperatures above.

**Where:** the end of `fitHead()` in `src/head.ts`; `TEMPERATURES` in `src/index.ts`.

**Numbers:** calibration error 0.01 to 0.03 for heads with 10 or more examples per label, 0.07 at 5 per label on
BANKING77 (the loosest spot), 0.02 to 0.04 for zero-shot.

**Read more:** [guide-confidence.md](guide-confidence.md).

## 5. Learning from an LLM

**What:** `withFallback()` answers locally when `confidence >= minConfidence` and otherwise calls your `ask`. Each
answer becomes a training example; with `retrainEvery` it retrains on its own.

**Where:** `src/fallback.ts`.

**Numbers:** on CLINC150 from label names alone, the share sent to the LLM fell from 68% to 16% over ten retrains
with a perfect teacher. A real teacher matters: a teacher wrong 20% of the time left it at 52%. On all 77 BANKING77 labels, a teacher
wrong 15% of the time kept sends at 64% until `calibrateOn` (2 correct examples per label) brought them to 24%. Give the LLM the
same label descriptions (Llama 3.1 8B: 84% to 90%) and list labels in a stable, grouped order.

**Read more:** [guide-fallback.md](guide-fallback.md).

## 6. The browser worker

**What:** the encoder's wasm never gives the page a turn, so in a browser it runs in a Web Worker.
`createWorkerDecider()` has the same `decide()`, `decideMany()` and `train()`, plus `terminate()`.

**Where:** `src/browser.ts` (page side), `src/worker.ts` (worker side).

**Numbers:** training BANKING77 on the main thread froze the page for 171 s; in a worker the longest stall was 18 ms.
Each call costs about 1 ms of message passing.

**Read more:** [guide-browser.md](guide-browser.md).

## 7. Cross-origin isolation

**What:** with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless`, the page
is cross-origin isolated and onnxruntime's wasm can use several threads.

**Numbers:** with all-MiniLM-L6-v2, p95 decision time went from 14.3 ms to 7.8 ms.

**Read more:** [guide-browser.md](guide-browser.md).

## 8. Bundlers and download size

**What:** the worker starts with `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`, which
Vite, webpack 5 and esbuild-based tools understand. An alias from `onnxruntime-web/webgpu` to `onnxruntime-web/wasm`
drops the WebGPU build decidejs doesn't use by default.

**Numbers:** the alias halves the wasm (23.6 MB to 12.9 MB, 3.3 MB gzipped). First visit: the 34 MB model plus the
wasm. decidejs adds 3.6 KB to the page bundle itself.

**Read more:** [bundlers.md](bundlers.md).

## 9. Caching

**What:** transformers.js caches the model files after the first download (the browser's Cache Storage in a page,
a local folder in Node). Zero-shot option embeddings are cached per set of options, and `train()` keeps the last
training set's embeddings, so retraining after a few new examples only encodes those.

**Where:** `optionCache` and `trainCache` in `src/index.ts`.

## 10. Head files

**What:** `saveHead()` writes JSON with base64 weights; `loadHead()` reads it. It lives apart from the encoder so a page
can load heads without pulling the model code in. A head records the model it was trained on, and `decide()` refuses
a head from a different model.

**Where:** `src/head-io.ts`.

## 11. React, the CLI and switch

- `decidejs/react`: `useDecider()` and `useDecision()`, which decides as the user types and drops stale answers.
  [guide-react.md](guide-react.md).
- `decidejs` CLI: `train` a head from a CSV or JSONL file and `decide` from the shell. [api.md](api.md#cli).
- `decidejs/switch`: a decider for the switch agent harness, answering choice, yes/no and score questions locally,
  optionally learning from a teacher. [guide-switch.md](guide-switch.md).
