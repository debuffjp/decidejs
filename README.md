# decidejs

Typed, calibrated text decisions in the browser or Node. Your labels are the type. No server, no API key.

```ts
import { createDecider } from "decidejs";

const d = await createDecider();
const r = await d.decide("my card never arrived", ["card_arrival", "refund", "other"]);
r.choice; // "card_arrival" | "refund" | "other"
```

Give it 5 examples per label and it reaches 77% on BANKING77 (77 intents); give it the full training set and it
reaches 90.6%, with each answer taking 3-6 ms in Chrome. Training runs in the user's tab, so their data never
leaves the device.

## Zero-shot

Pass labels, or labels with descriptions. Labels like `card_arrival` are read as "card arrival".

```ts
await d.decide(text, ["question", "bug_report", "praise"]);
await d.decide(text, { refund: "wants their money back", bug: "something is broken" });
```

Zero-shot is for trying things out. Its `confidence` isn't calibrated, so don't threshold on it.

## Train a head

A head is a small layer trained on your examples: 160 KB for 77 labels. Train it once and ship it as a static
file, or train it in the page.

```ts
import { createDecider, saveHead, loadHead } from "decidejs";

const d = await createDecider();
const head = await d.train(examples); // { text, label }[]
await writeFile("intent.head.json", saveHead(head));

// later, in the app
const r = await d.decide(text, loadHead(json)); // choice is typed as your labels
```

Each label starts at the average of its examples, so a handful per label already works. Training then fits a
temperature on held-out examples, so `confidence` is calibrated: of the answers given at 0.9, about 90% are right.

## "None of these"

```ts
const r = await d.decide(text, head, { minConfidence: 0.88 });
r.choice; // "card_arrival" | "refund" | ... | null
```

Below the threshold, `choice` is `null`. Use it to send unclear messages to a person or a bigger model.

## In the browser

Use the worker version. The encoder blocks whatever thread it runs on (we measured a 171 s freeze while training
on the main thread; in a worker the page's longest stall was 18 ms).

```ts
import { createWorkerDecider } from "decidejs/browser";

const d = await createWorkerDecider();
const r = await d.decide(text, head);
```

Tested from npm in a Vite 7 app. Serve the page cross-origin isolated if you can:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

That lets wasm use several threads: in our runs it halved training time and cut p95 decision time from 14.3 ms to 7.8 ms.

Download on first visit: the model (23.7 MB) and onnxruntime's wasm (23.6 MB, 5.8 MB gzipped), both cached after.

## Numbers

All-MiniLM-L6-v2 at q8, Ryzen 7 260 laptop, Node 24 and headless Chrome 156, 2026-10-31. Accuracy and calibration
error (ECE, lower is better) on each dataset's test split, mean of 3 seeds (one run for "all"):
[eval/results.json](eval/results.json).

| Examples per label | 5 | 10 | 25 | all |
|---|---|---|---|---|
| BANKING77 (77 labels) | 77.4% | 82.6% | 87.7% | 90.6% |
| CLINC150 (150 labels) | 86.0% | 89.0% | 92.8% | 94.0% |
| ECE, BANKING77 | 0.086 | 0.046 | 0.030 | 0.020 |
| ECE, CLINC150 | 0.033 | 0.012 | 0.012 | 0.012 |

Calibration is loosest at 5 examples per label on BANKING77 (0.07-0.12 across seeds). Add examples before relying on
a threshold there.

**In Chrome** (BANKING77, full training set, worker, cross-origin isolated, [bench/results.json](bench/results.json)):
90.6% accuracy, 3.1 ms median and 5.7 ms p95 per decision, 147 s to train on 10,003 examples.

**Text that fits no label:** CLINC150 has 1,000 test queries that match none of its 150 intents. Trained on
in-scope examples only, with `minConfidence` 0.88 picked on the validation split, 81.7% of them come back `null`,
and 86.6% of in-scope queries are answered and right (94.0% are right with no threshold).

Reproduce: `npm run eval:suite` (Node), `npm run bench` then `npm run bench:run` (Chrome).

## What it's for

Routing support messages, triaging feedback, search and command intent, moderation hints while someone types,
labelling a dataset in the browser, and a cheap first pass before an LLM. Details, code and poor fits:
[docs/use-cases.md](docs/use-cases.md). Runnable examples are in [examples/](examples/).

## Compared with

| | Runs in the browser | Trains in the browser | Typed result | Calibrated confidence |
|---|---|---|---|---|
| decidejs | yes (worker) | yes | yes | trained heads |
| [jimothy](https://www.npmjs.com/package/jimothy) | inference (worker) | no, Node | yes | yes |
| transformers.js zero-shot pipeline | yes | no | no | no |
| MediaPipe text classifier | yes | no, offline | no | no |
| LLM structured output (AI SDK `Output.choice`) | no, API | n/a | yes | no |

jimothy reports 92.4% on BANKING77, above our 90.6%. Pick it if you train in Node and want the higher number. Pick
decidejs if training has to happen in the page, from the user's own data.

## Community

Questions and show-and-tell in [Discussions](https://github.com/debuffjp/decidejs/discussions), chat in the
[debuff Discord](https://discord.gg/jshbDHSpCY). Results on your own data, good or bad, help most: there's an
issue template for them. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Status

0.1. English only with the default model; short texts (about a paragraph). Not done yet: WebGPU in the worker, a
fallback to a hosted model below the threshold, and fine-tuning the encoder (the likely route past about 91%).

Made by the [debuff](https://debuff.jp) team. MIT.
