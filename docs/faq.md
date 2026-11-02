# FAQ

## Which languages does it support?

English with the default model. For other languages, use Decide Multilingual:

```ts
const d = await createDecider({ model: "debuffjp/decide-multilingual" });
```

Decide Multilingual is an unmodified copy of `Xenova/multilingual-e5-small`, from intfloat's multilingual-e5-small, which
is why the table below calls it e5. Before 0.1.1 this example used the Xenova id, which still works. decidekit knows its settings (the "query: " prefix it expects, its temperatures), so confidence stays calibrated. It
downloads about 118 MB against 34 MB. On MASSIVE (60 intents, labels named in English, 2,000 test messages per
language, eval/multilingual.ts):

| | Default model, Japanese | e5, Japanese | e5, Chinese | e5, German | e5, English | Default model, English |
|---|---|---|---|---|---|---|
| Zero-shot | 6.4% | 43.5% | 38.5% | 40.0% | 51.4% | 61.9% |
| 10 examples per label | 30.8% | 68.4% | 69.5% | 60.4% | 70.2% | 76.1% |
| All training data | 55.0% | 76.5% | 76.5% | 71.6% | 78.6% | 85.6% |
| Trained on English examples only | 9.0% | 70.1% | 65.3% | 60.8% | | |

The last row means one head trained on English examples works in other languages, at some cost. Calibration error of
the trained e5 heads is about 0.03. For English text, keep the default model: it is 7 points more accurate.

Two settings are tuned for the default model only: the off-topic gate in `withFallback` (`noneBelow`) never fires with
e5, whose similarities sit around 0.8 even for unrelated text, so it won't learn "none of these"; and the label
description temperature hasn't been measured for e5. Any other Hugging Face model with ONNX weights can be passed as
`model` too, unmeasured.

## Can it read long texts?

Short texts work best: about a paragraph. The encoder reads a limited number of tokens and ignores the rest, so a long
document is judged on its opening. To classify a long document, split it into paragraphs and decide each one, or decide
on the part that carries the meaning (a subject line, a title, the first paragraph).

## Does my users' text leave the device?

No. In the browser, the model and the head run in the user's tab and training runs there too. decidekit makes no network
requests of its own beyond downloading the model files, from Hugging Face or your own `modelHost` (and whatever head file
you fetch yourself).

## Does it work offline?

After the first visit, yes: the model files are cached by the browser, and the rest is your own static files. The first
load needs the network. If the browser clears its cache, the model downloads again. In Node, transformers.js keeps the
files in its cache directory after the first download.

## What does it cost to run?

Nothing per call. The costs are the first download (34 MB of model plus 12.9 MB of wasm, 3.3 MB gzipped, with the
wasm alias in [bundlers.md](bundlers.md)) and the user's CPU: about 8 ms median per decision in Chrome, 1.1 ms per text
in Node with `decideMany()`. The pre-release default, all-MiniLM-L6-v2, needed about 36 MB and 3-5 ms.

## Should I use this or an LLM?

Use decidekit when the answer is one of a fixed set of labels and you can collect examples. Use an LLM when the answer is
open-ended, needs reasoning over facts the text doesn't contain, or when you have no examples and need more than the
66-73% zero-shot reaches with label names (72-83% with one-sentence descriptions). LLM structured output gives a typed choice too, but needs an API, costs per call and its
confidence isn't calibrated.

They combine: decide locally, and send only answers below `minConfidence` (where `choice` is `null`) to a larger model.

## How does it compare with jimothy?

[jimothy](https://www.npmjs.com/package/jimothy) (MIT) also builds typed local classifiers with calibrated confidence. It
reports 92.4% on BANKING77 on its own split, with all-MiniLM-L6-v2. decidekit 0.2 reaches 93.1% on the dataset's test split with Decide Small (90.7% before 0.2). The splits and encoders differ, so that isn't a direct comparison. decidekit 0.2's Adam settings come from jimothy's source. jimothy trains in Node and runs inference in a worker. Pick it if you train ahead of time in Node. Pick decidekit if training has to happen in the page, from the
user's own data.

## And the transformers.js zero-shot pipeline?

It is NLI-based, its result is untyped and its scores are not calibrated. decidekit's zero-shot compares embeddings, types
the choice as your labels and can move to a trained head with the same `decide()` call.

## Does it use WebGPU?

Not by default. `createDecider({ device: "webgpu" })` passes the device through to transformers.js, but WebGPU in the
worker is not done yet and has not been measured. The default is wasm in browsers and cpu in Node. If you do use WebGPU,
leave out the wasm alias from [bundlers.md](bundlers.md).

## Does it work on phones?

Not tested yet. It needs the same downloads as on a desktop (34 MB of model plus the wasm on the first visit) and decisions will be
slower on a phone's CPU. Measure on the devices you care about before relying on it.

## Node or browser?

Both run the same code. In Node, use `createDecider()` from `decidekit`; `decideMany()` is the fast path for batches. In
a browser, use `createWorkerDecider()` from `decidekit/browser`, which runs the same decider in a Web Worker so the page
doesn't freeze. A common split: train in Node or with the CLI, ship the head file, decide in the browser. You can also
train in the browser when the examples come from the user.

## What licence do heads have?

A head you train on your own data is yours. decidekit itself is MIT. The ready-made heads in
[heads/](../heads/README.md) are derived from public datasets and carry their licences: BANKING77 is CC BY 4.0
(PolyAI) and CLINC150 is CC BY 3.0 (Clinc). Credit the dataset if you ship one of them.

## How accurate will it be on my data?

It depends on how distinct your labels are and how many examples you have. On two public intent benchmarks, a trained
head reached 83.7-90.2% with 5 examples per label and 93.1-96.3% with all the training data in 0.2 (82.3-89.6% and 90.7-94.4% before 0.2; 78.7-87.3% and
90.6-94.0% with the pre-release default, all-MiniLM-L6-v2). Small tasks with 2 to 10
labels reached 93.4% with one example per label. Hold back some labelled examples and measure: the recipe is in
[guide-confidence.md](guide-confidence.md#pick-a-threshold-on-your-own-data).

## What kinds of decisions is it good at?

Deciding what a text is about: intents, topics, routing, categories. It is weak at deciding how a text feels. Measured
with bge-small (2,000 test rows each, [eval/mteb.ts](../eval/mteb.ts), [eval/hwu64.ts](../eval/hwu64.ts)):

| Task | Labels | Zero-shot | Trained | Confident answers right |
|---|---|---|---|---|
| HWU64 intents (never used for tuning) | 64 | 59.8% | 85.8% (10 per label), 90.2% (all) | 98% |
| AG News topic | 4 | 67.5% | 84.4% (32 per label) | 95% |
| Tweet sentiment | 3 | 53.3% | 61.3% (32 per label) | 89% |
| Emotion | 6 | 50.6% | 49.4% (32 per label) | 66% |
| Toxicity (7.6% toxic) | 2 | 65.7% | 66.6% (32 per label) | 56% |

Sentiment, emotion and toxicity are limits of the encoder, which was trained on meaning, not tone: more examples
barely help. Toxicity is the worst case, because rare labels make it worse: always answering "not toxic" would be
right 92% of the time. Telling the head how rare toxic text is ([eval/prior-shift.ts](../eval/prior-shift.ts)) only
made it answer "not toxic" to nearly everything. A 4 times larger encoder, bge-base (`model: "Xenova/bge-base-en-v1.5"`),
helped little: 54.6% on emotion, 64.2% on toxicity (lower), 62.0% on tweet sentiment at 32 per label, and 90.6% on HWU64.
Use an LLM or a dedicated classifier for these for now.

## Why not fine-tune the encoder?

Training only the head keeps training cheap enough for a browser tab (147-217 s for 10,003 examples, 0.1 s for 9) and
lets one cached encoder serve every head. We tried fine-tuning on SNIPS and PAWS on 2026-11-02: it lost 2.5 points on BANKING77 and 1.2 on CLINC150 on the dev splits. That rules out this data mix only. Before 0.2 this answer called fine-tuning the likely route past about 91% on BANKING77; better head training reached 93.1% instead.

## How big is a head?

156 KB for 77 labels and 303 KB for 150, as JSON with base64 weights. It grows with the number of labels, not the number
of examples.

## Can I change the labels without retraining?

No. A head knows only the labels it was trained on. To add, remove or rename a label, change the examples and call
`train()` again; on the same decider, only new texts are encoded.

## npm warns that install scripts were blocked

npm 11 skips install scripts for packages your project hasn't allowed and says so for three of decidekit's
dependencies: onnxruntime-node, sharp and protobufjs. decidekit works anyway: onnxruntime-node ships its binaries in
the package, and decidekit doesn't use sharp (transformers.js loads it for images). The packed-package test in CI
installs this way. You don't need to approve them.

## Why does `decide()` throw "this head was trained on ..."?

A head only works with the encoder it was trained on. Create the decider with the head's model:
`createDecider({ model: head.model })`.

## Is `confidence` meaningful with zero-shot?

Roughly. The fixed temperature came out close to calibrated on two intent datasets, but that may not hold for your
labels. Check it on labelled text of your own before thresholding, or train a head, which fits its own temperature. See
[guide-confidence.md](guide-confidence.md#zero-shot-confidence).

## Why does it say "File doesn't exist" when it loads the model?

In Node on Windows, transformers.js caches the model inside `node_modules`. In a deeply nested project the path can pass the Windows limit of 260 characters, and loading fails with "Load model ... failed. File doesn't exist". Move the project to a shorter path or enable long paths in Windows.
