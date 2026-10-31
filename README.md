# decidejs

Typed, calibrated decisions in the browser or Node. Your options are the type. No server, no API key.

```ts
import { createDecider } from "decidejs";

const { decide } = await createDecider();
const r = await decide("my card never arrived", ["card_arrival", "refund", "other"]);
r.choice; // "card_arrival" | "refund" | "other"
r.confidence; // 0..1; only calibrated once you train a head
```

It embeds the text with a small sentence encoder (all-MiniLM-L6-v2, 23.7 MB, downloaded once) and picks the closest
option. That's zero-shot: quick to try, rough on hard tasks. For real accuracy, train a head.

## Train a head

A head is a small layer trained on your examples: 160 KB for 77 labels. Train it once, ship it as a static file.

```ts
import { createDecider, saveHead, loadHead } from "decidejs";

const d = await createDecider();
const head = await d.train(examples); // { text, label }[]
await writeFile("intent.head.json", saveHead(head));

// later, in the app
const r = await d.decide(text, loadHead(json)); // choice is typed as your labels
```

Training also fits a temperature on held-out examples, so `confidence` means what it says: of the decisions made at
0.9, about 90% are right. Use it to decide when to ask a person or a bigger model.

## In the browser

Use the worker version. The encoder's wasm work blocks the page otherwise (we measured a 171 s freeze while
training).

```ts
import { createWorkerDecider } from "decidejs/browser";

const d = await createWorkerDecider();
const r = await d.decide(text, head);
```

Serve the page cross-origin isolated if you can:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

That lets wasm use several threads. It halves training time and brings decisions under 10 ms.

## Numbers

BANKING77 (77 banking intents, 10,003 training and 3,080 test examples). Ryzen 7 260 laptop, one run each,
2026-10-31.

| Setup | Accuracy | Decision p50 / p95 | Train |
|---|---|---|---|
| zero-shot, Chrome | 52.5% | 7.7 / 12.2 ms | none |
| trained head, Node 24 | 87.7% | 1.6 ms mean | 89 s |
| trained head, Chrome worker | 87.9% | 7.6 / 14.3 ms | 228 s |
| trained head, Chrome worker, isolated | 87.9% | 4.4 / 7.8 ms | 116 s |

The browser rows were measured before training started each label at the mean of its examples, which lifted the
Node figure to 89.4% (below). Reproduce with `npm run eval` (Node) and `npm run bench` then `npm run bench:run`
(headless Chrome).

### With few examples

Accuracy by examples per label, mean of 3 seeds (one run for "all"), Node, all-MiniLM-L6-v2 q8
([eval/results-centroid.json](eval/results-centroid.json)).

| Examples per label | 5 | 10 | 25 | all |
|---|---|---|---|---|
| BANKING77 (77 labels) | 75.0% | 80.6% | 85.0% | 89.4% |
| CLINC150 (150 labels) | 84.8% | 88.5% | 91.0% | 93.9% |

Without the mean-of-examples start, 5 per label gave 54.4% and 64.8% ([eval/results-baseline.json](eval/results-baseline.json)).

### Text that fits no label

CLINC150 includes 1,000 test queries that match none of its 150 intents. Trained on in-scope examples only, with
a confidence threshold of 0.93 picked on the validation split: **87.8%** of out-of-scope queries fall below it.
The cost is in-scope answers: 93.9% are right with no threshold, and 80.1% are both right and above it. Pick the
threshold for your own trade-off.

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

jimothy reports 92.4% on BANKING77, above our 87.9%. Pick it if you train in Node and want the higher number.
Pick decidejs if training has to happen in the page, from the user's own data.

## Community

Questions and show-and-tell in [Discussions](https://github.com/debuffjp/decidejs/discussions), chat in the
[debuff Discord](https://discord.gg/jshbDHSpCY). Results on your own data, good or bad, help most: there's an
issue template for them. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Status

0.1, a prototype. Zero-shot confidence is not calibrated: it uses a fixed temperature and reads near 1.0 even on
hard cases, so only trust `confidence` from a trained head. Not done yet: WebGPU in the worker, a fallback to a hosted model when confidence is low, and a
smaller runtime than transformers.js (1.2 MB of JavaScript).

Made by the [debuff](https://debuff.jp) team. MIT.
