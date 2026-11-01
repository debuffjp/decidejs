# decidejs

[![npm](https://img.shields.io/npm/v/decidejs?color=cffb2d&labelColor=11140d)](https://www.npmjs.com/package/decidejs) [![ci](https://github.com/debuffjp/decidejs/actions/workflows/ci.yml/badge.svg)](https://github.com/debuffjp/decidejs/actions/workflows/ci.yml) [![MIT](https://img.shields.io/badge/licence-MIT-cffb2d?labelColor=11140d)](LICENSE) [![Discord](https://img.shields.io/badge/discord-debuff-a78bfa?labelColor=11140d)](https://discord.gg/jshbDHSpCY)

**[Try it in your browser](https://debuff.jp/decidejs/playground/)** · [Watch it learn from an LLM](https://debuff.jp/decidejs/lanes/) · [Launch post](https://debuff.jp/blog/introducing-decidejs) · [Research: when the LLM is wrong](https://debuff.jp/research/learning-from-a-noisy-llm)

A decision model for the browser and Node: it picks one of your labels for a piece of text, with confidence you can trust, and learns from an LLM when it isn't sure. Your labels are the type. No server, no API key.

A System One model in the browser: like TypeSafe's Jev, but on the user's device, so each decision is free, private and a few milliseconds. Jev leaves unsure cases to your code; decidejs can ask an LLM about them and learn from the answers.

```ts
import { createDecider } from "decidejs";

const d = await createDecider();
const r = await d.decide("my card never arrived", ["card_arrival", "refund", "other"]);
r.choice; // "card_arrival" | "refund" | "other"
```

Give it 5 examples per label and it reaches 82% on BANKING77 (77 intents); give it the full training set and it
reaches 90.7%, with each answer taking about 8 ms (median) in Chrome. Training runs in the user's tab, so their data never
leaves the device.

## Zero-shot

Pass labels, or labels with descriptions. Labels like `card_arrival` are read as "card arrival".

```ts
await d.decide(text, ["question", "bug_report", "praise"]);
await d.decide(text, { refund: "wants their money back", bug: "something is broken" });
```

Zero-shot is for trying things out: 61% on BANKING77 and 70% on CLINC150 (dev splits), against 90-94% trained.
Its confidence came out close to calibrated on those datasets (error 0.005-0.045 with 2 to 150 options), but check
it on your own labels before relying on a threshold.

## Train a head

A head is a small layer trained on your examples: 156 KB for 77 labels. Train it once and ship it as a static
file, or train it in the page.

```ts
import { createDecider, saveHead, loadHead } from "decidejs";

const d = await createDecider();
const head = await d.train(examples); // { text, label }[]
await writeFile("intent.head.json", saveHead(head));

// later, in the app
const r = await d.decide(text, loadHead(json)); // choice is typed as your labels
const rs = await d.decideMany(texts, head); // many texts, encoded in batches
```

Each label starts at the average of its examples plus its own name, so a handful per label already works. If your
labels are opaque ids (`c17`), pass `{ labelNames: false }`. Training then fits a temperature on held-out examples,
so `confidence` is calibrated: of the answers given at 0.9, about 90% are right.

Or train from the command line and ship the file:

```sh
npx decidejs train examples.csv -o intent.head.json   # CSV with text,label columns, or JSONL
npx decidejs decide intent.head.json "my card never arrived"
```

Add `--types` to also write `intent.head.labels.ts`, so `loadHead<Label>(json)` types the choice as your labels.

## "None of these"

```ts
const r = await d.decide(text, head, { minConfidence: 0.9 });
r.choice; // "card_arrival" | "refund" | ... | null
```

Below the threshold, `choice` is `null`. Use it to send unclear messages to a person or a bigger model. It needs
enough examples to mean something: with 3 examples per label, a head still gave "hello" 0.89. The numbers below come
from CLINC150's 100 examples per label. With fewer, check the threshold on some off-topic text of your own, or add an
`other` label trained on it.

## Fall back to an LLM

```ts
const route = withFallback(d, head, { minConfidence: 0.9, ask: (text, labels) => askMyLlm(text, labels), examples });
const r = await route(text); // r.source: "local" | "fallback"
```

Below the threshold, `ask` answers and its answer becomes a training example. Simulated on CLINC150 from 5 examples
per label with an LLM that is always right, the share sent to the LLM fell from 41% to 14-17% over 10,000 queries
while the head went from 86% to 94%. With real Llama 3.1 8B answers, starting from 12 described labels alone, it fell
from 30% to 19%.

Real LLMs are sometimes wrong, and `withFallback()` is built for that: it sets its confidence on its own sure answers
rather than the LLM's, and learns the LLM's "none of these" (`null`) as a label, `NONE`, so it can say `null` itself
for off-topic text. Details and numbers: [docs/guide-fallback.md](docs/guide-fallback.md).

## Coming from Jev

`decidejs/typesafe` has the shape of TypeSafe's SDK, for Choice questions:

```ts
import { choice, typesafeClient } from "decidejs/typesafe";

const client = typesafeClient(await createDecider());
const { answers } = await client.systemOne({
  state: "I was charged twice",
  questions: { category: choice("What is this about?", { billing: null, technical: null, other: null }) },
});
```

The [guide](docs/guide-typesafe.md) covers what differs, including what `confidence` means.

## switch

`decidejs/switch` makes decidejs a decision model for [switch](https://github.com/osolmaz/switch), the harness that
runs a decision model through a task. Zero-shot, it won two of the three WikiRace races from switch's example on a
laptop CPU. See the [switch guide](docs/guide-switch.md).

## React

`useDecider()` and `useDecision(decider, text, labels)` from `decidejs/react`: [docs/guide-react.md](docs/guide-react.md).

## In the browser

Use the worker version. The encoder blocks whatever thread it runs on (we measured a 171 s freeze while training
on the main thread; in a worker the page's longest stall was 18 ms).

```ts
import { createWorkerDecider, loadHead } from "decidejs/browser";

const head = loadHead(await (await fetch("/intent.head.json")).text());
const d = await createWorkerDecider({
  onProgress: ({ file, loaded, total }) => showProgress(file, loaded / total), // first visit only
});
const r = await d.decide(text, head);
```

Import `loadHead` from `decidejs/browser` too, not from `decidejs`: that keeps the encoder out of your page bundle.

Tested from npm in Vite 7, webpack 5 and Next.js 16 apps; see [docs/bundlers.md](docs/bundlers.md) for the config, including a one-line alias
that halves the wasm download. Serve the page cross-origin isolated if you can:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

That lets wasm use several threads: in our runs with all-MiniLM-L6-v2 it cut p95 decision time from 14.3 ms to 7.8 ms. With the default bge-small, isolated, p95 is 13.8 ms.

Download on first visit: the model (34 MB) and onnxruntime's wasm (23.6 MB, 5.8 MB gzipped), both cached after.

## Numbers

bge-small-en-v1.5 (the default) at q8, Ryzen 7 260 laptop, Node 24 and headless Chrome 156, 2026-11-01. Accuracy
and calibration error (ECE, lower is better) on each dataset's test split, mean of 3 seeds (one run for "all"):
[eval/results-bge-small.json](eval/results-bge-small.json).

| Examples per label | 5 | 10 | 25 | all |
|---|---|---|---|---|
| BANKING77 (77 labels) | 82.3% | 83.8% | 88.7% | 90.7% |
| CLINC150 (150 labels) | 89.6% | 92.0% | 93.4% | 94.4% |
| ECE, BANKING77 | 0.071 | 0.029 | 0.011 | 0.017 |
| ECE, CLINC150 | 0.028 | 0.010 | 0.015 | 0.008 |

Pre-release builds defaulted to all-MiniLM-L6-v2 (23.7 MB, 2026-10-31, [eval/results.json](eval/results.json)):

| Examples per label | 5 | 10 | 25 | all |
|---|---|---|---|---|
| BANKING77 (77 labels) | 78.7% | 83.0% | 87.4% | 90.6% |
| CLINC150 (150 labels) | 87.3% | 89.8% | 92.8% | 94.0% |
| ECE, BANKING77 | 0.065 | 0.048 | 0.025 | 0.019 |
| ECE, CLINC150 | 0.029 | 0.013 | 0.012 | 0.012 |

Calibration is loosest at 5 examples per label on BANKING77. On small tasks (2 to 10 labels sampled from CLINC150, scored on its validation
queries, [eval/tiny.ts](eval/tiny.ts), measured with all-MiniLM-L6-v2), 1 example per label was right 93% of the time with calibration error 0.020, and
3 per label 98% with 0.016. Add examples before relying on
a threshold there.

**In Chrome** (BANKING77, full training set, worker, cross-origin isolated, 16 threads): 7.7 ms median and 13.8 ms
p95 per decision, 228 s to train on 10,003 examples with no page stall longer than 18 ms. all-MiniLM-L6-v2 in the
same run: 4.2 ms median, 7.7 ms p95. In Node: 3.2 ms median for a single text, 1.1 ms per text batched (MiniLM 2.0
and 0.7). Earlier MiniLM run: [bench/results.json](bench/results.json).

**Text that fits no label:** CLINC150 has 1,000 test queries that match none of its 150 intents. Trained on
in-scope examples only, with `minConfidence` 0.93 picked on the validation split, the shipped bge-small head returns
`null` for 89.9% of them. With all-MiniLM-L6-v2 that was 86.3%, and 82.9% of in-scope queries were answered and right
(94.0% right with no threshold).

**The smaller model.** `createDecider({ model: "Xenova/all-MiniLM-L6-v2" })` (23.7 MB, the pre-release default) is
about 1.8x faster per decision in Chrome and 3 to 5 points less accurate with few examples, about the same with all
the data. Pass it to keep using heads trained on it. Before this default changed, the README suggested
`{ labelNames: false }` with bge-small: label names slightly hurt it on the dev splits.

Reproduce: `npm run eval:suite` (Node), `npm run bench` then `npm run bench:run` (Chrome).

## Playground

Try it at [debuff.jp/decidejs](https://debuff.jp/decidejs/). To run it locally: `npm run playground`, then serve `playground/dist` cross-origin isolated (`node bench/serve.mjs` style headers).
Try zero-shot labels, the ready-made heads, a head trained in the page and "none of these".
[Two lanes](https://debuff.jp/decidejs/lanes/) shows `withFallback` live: decidejs starts from twelve label names,
asks a small LLM when it's unsure, learns from each answer, and replays the first 20 messages at the end.

## Ready-made heads

[heads/](heads/) has a BANKING77 support-intent head (90.6%) and a CLINC150 assistant-intent head (95.0%), so you
can try a trained head before collecting examples.

## What it's for

Routing support messages, triaging feedback, search and command intent, moderation hints while someone types,
labelling a dataset in the browser, a cheap first pass before an LLM that learns from it, games, extensions
and offline apps, and text that can't leave the device. Details, code and poor fits:
[docs/use-cases.md](docs/use-cases.md). Runnable examples are in [examples/](examples/).

## Compared with

| | Runs in the browser | Trains in the browser | Typed result | Calibrated confidence |
|---|---|---|---|---|
| decidejs | yes (worker) | yes | yes | trained heads |
| [jimothy](https://www.npmjs.com/package/jimothy) | inference (worker) | no, Node | yes | yes |
| [Jev](https://typesafe.ai/) (TypeSafe) | no, API | no | yes | yes |
| transformers.js zero-shot pipeline | yes | no | no | no |
| MediaPipe text classifier | yes | no, offline | no | no |
| LLM structured output (AI SDK `Output.choice`) | no, API | n/a | yes | no |

jimothy reports 92.4% on BANKING77, above our 90.7%. Pick it if you train in Node and want the higher number. Pick
decidejs if training has to happen in the page, from the user's own data.

## Docs

[docs/README.md](docs/README.md) indexes the guides (training, confidence and thresholds, the browser), the
API reference, how it works and the FAQ.

## Community

Questions and show-and-tell in [Discussions](https://github.com/debuffjp/decidejs/discussions), chat in the
[debuff Discord](https://discord.gg/jshbDHSpCY). Results on your own data, good or bad, help most: there's an
issue template for them. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Status

0.1. English with the default model, other languages with `multilingual-e5-small` ([FAQ](docs/faq.md#which-languages-does-it-support)); short texts (about a paragraph). Not done yet: WebGPU in the worker, a
fallback to a hosted model below the threshold, and fine-tuning the encoder (the likely route past about 91%).

Made by the [debuff](https://debuff.jp) team. MIT.
