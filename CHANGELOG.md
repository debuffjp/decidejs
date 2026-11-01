# Changelog

## 0.1.0 (unreleased)

The first release. Numbers are on public test splits unless noted; the scripts that produced them are in `eval/`.

### Deciding

- `createDecider()` picks one of your labels for a text, in Node or the browser, with `confidence` and `probs`. Labels
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

- `decidejs/typesafe`: `typesafeClient()` gives a decider the shape of TypeSafe's SDK (`systemOne({ state, questions })`,
  `choice()`, `probabilities`, their `confidence` formula). Choice questions only.
- `decidejs/browser`: `createWorkerDecider()` runs the encoder in a Web Worker (training BANKING77 froze the page for
  171 s on the main thread; the longest stall in a worker was 18 ms).
- `decidejs/react`: `useDecider()` and `useDecision()`.
- `decidejs/switch`: a decider for the switch agent harness.
- A CLI: `decidejs train` and `decidejs decide`.

### Known limits

- The default encoder reads English. `Xenova/multilingual-e5-small` is supported for other languages (76% on Japanese
  and Chinese MASSIVE intents, calibrated), but `withFallback`'s off-topic gate doesn't fire with it.
- Weak at tone: 61% on tweet sentiment and 49% on emotion with 32 examples per label. See the FAQ.
