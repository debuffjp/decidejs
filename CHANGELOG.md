# Changelog

## 0.2.1 (2026-11-02)

### Browser

- `createWorkerDecider()` rejects when the worker fails to load, crashes or sends an unreadable message. Before, the call never settled. Calls after a failure or after `terminate()` reject too.
- The wasm download is 12.9 MB (3.3 MB gzipped) unless you pass `device: "webgpu"`. Earlier docs said an alias got this size, but in Chrome with an esbuild build, 0.2.0 fetched the 23.6 MB file with or without it. The alias is now optional and only trims about 14.5 KB gzipped of worker JS. Paths you set in `wasmPaths` are left alone.

## 0.2.0 (2026-11-02)

### Training

- From 8 examples per label (mean across labels), `train()` fits the head with full-batch Adam, starting from the same centroid and label-name point. Test accuracy with all data: BANKING77 90.6% to 93.1%, CLINC150 94.6% to 96.1%, HWU64 90.15% to 91.17% (one seed, on cached embeddings; through train() in eval/results-h14.json, BANKING77 93.1% and CLINC150 96.3%). At 5 per label nothing changes. At 10 per label: BANKING77 84.4% to 87.5%, CLINC150 92.2% to 92.5%, HWU64 85.50% to 85.38%. Adam's settings come from jimothy's `src/linear.ts` (MIT, Andrew Prifer).
- New option `solver`. Setting `learningRate` or `l2` without it keeps SGD, so tuned calls behave as before. The default `epochs` is 100 with Adam.
- Saved heads keep working unchanged. Retrained heads can score differently, so pick `minConfidence` thresholds again: the CLINC150 out-of-scope threshold moved from 0.9 to 0.98.
- Training takes longer: in Node, BANKING77 went from 84 s to 123 s on cached embeddings. In Chrome, BANKING77 took 208 s with a longest page stall of 19 ms.

## 0.1.1 (2026-11-02)

### Models

- The default model id is `debuffjp/decide-small` (Decide Small). Before 0.1.1 it was `Xenova/bge-small-en-v1.5`. The
  weights are identical, so decisions don't change. For other languages the id is `debuffjp/decide-multilingual`
  (Decide Multilingual), identical to `Xenova/multilingual-e5-small`.
- The old ids and heads recording them keep working. They resolve to the new ids, so `d.model` returns the new id. A
  head trained on 0.1.1 records the new id, so a 0.1.0 decider needs that id passed as `model` to use it.
- Downloads of both models are pinned to a revision.
- New option `modelHost` serves model files from `<modelHost>/<model>/` instead of Hugging Face. It applies to the
  whole page or process.

### Docs

- Label names count as 1 example in every doc, and the advice to turn them off with bge-small is gone: it applied to the
  old weight of 3.
- Calibration, out-of-scope and zero-shot numbers are now bge-small's or labelled with the model they were measured on.
- `decide()` documents that a described label scores by the better of its name and description.
- The API reference covers `decidekit/typesafe`.

## 0.1.0 (2026-11-02)

The first release. Numbers are on public test splits unless noted; the scripts that produced them are in `eval/`.

### Deciding

- `createDecider()` picks one of your labels for a text, in Node or the browser, with `confidence` and `probabilities`. Labels
  are the type: `choice` has the literal type of the labels you pass.
- Zero-shot from label names, or from described labels (`{ card_arrival: "My card hasn't arrived." }`), which score by
  the better of name and description. Zero-shot decisions report `similarity`, the best cosine similarity to any
  option, which is low for off-topic text.
- `minConfidence` returns `null` below a threshold. `decideMany()` for batches.
- Default encoder: `Xenova/bge-small-en-v1.5` at 8-bit (34 MB). Any transformers.js feature-extraction model can be
  passed as `model`.
- Supported encoders with measured settings: `Xenova/bge-small-en-v1.5` (default) and `Xenova/multilingual-e5-small`.
  The temperature fit now searches 0.01 to 10, so encoders with close similarities calibrate.
- `describeLabels()` asks an LLM once to write label descriptions (7 to 10 points better zero-shot on the dev splits).

### Training

- `train()` fits one softmax layer on frozen embeddings: 90.7% on BANKING77 and 94.4% on CLINC150 with all examples,
  82.3% and 89.6% with 5 per label. HWU64, never used for tuning: 90.2%.
- Calibrated: the temperature is fitted on held-out examples (5 folds for small sets), with a per-model floor for
  small heads. Calibration error 0.01 to 0.03 for heads with 10 or more examples per label.
- `labels` adds labels without examples yet (they start from their names). `calibrateOn` fits the temperature on
  examples known to be right, for training data from a source that is sometimes wrong.
- `saveHead()` and `loadHead()`. Ready-made heads for BANKING77 and CLINC150.

### Learning from an LLM

- `withFallback()` answers locally when confident and calls your `ask` otherwise; each answer becomes a training
  example and `retrainEvery` retrains in the background. It can start from labels alone.
- Built for an LLM that is sometimes wrong: starting from labels, it fits confidence on zero-shot's own confident
  answers rather than the LLM's, and learns `null` answers as the label `NONE` (`learnNone`) when the text is far
  from every label (`noneBelow`), so it can answer `null` itself for off-topic text.
- `route.saved()` returns the learned answers and the calibration set as plain JSON; pass them back as `restore` after
  a reload. With real Llama 3.1 8B answers on
  12 labels, the share sent fell from 30% (zero-shot) to 19%, and off-topic text given a wrong label from 6.7% to 2.4%.

### Browser and tools

- `decidekit/typesafe`: `typesafeClient()` gives a decider the shape of TypeSafe's SDK (`systemOne({ state, questions })`,
  `choice()`, `probabilities`, their `confidence` formula). Choice questions only.
- `decidekit/browser`: `createWorkerDecider()` runs the encoder in a Web Worker (training BANKING77 froze the page for
  171 s on the main thread; the longest stall in a worker was 18 ms).
- `decidekit/react`: `useDecider()` and `useDecision()`.
- `decidekit/switch`: a decider for the switch agent harness.
- A CLI: `decidekit train` and `decidekit decide`.

### Known limits

- The default encoder reads English. `Xenova/multilingual-e5-small` is supported for other languages (76% on Japanese
  and Chinese MASSIVE intents, calibrated), but `withFallback`'s off-topic gate doesn't fire with it.
- Weak at tone: 61% on tweet sentiment and 49% on emotion with 32 examples per label. See the FAQ.
