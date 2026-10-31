# decidejs

Typed, calibrated text decisions in the browser or Node. Your labels are the type. No server, no API key.

```ts
import { createDecider } from "decidejs";

const d = await createDecider();
const r = await d.decide("my card never arrived", ["card_arrival", "refund", "other"]);
r.choice; // "card_arrival" | "refund" | "other"
```

Give it 5 examples per label and it reaches 79% on BANKING77 (77 intents); give it the full training set and it
reaches 90.6%, with each answer taking 3-5 ms (median) in Chrome. Training runs in the user's tab, so their data never
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
per label, the share sent to the LLM fell from 41% to 14-17% over 10,000 queries while the head went from 86% to 94%. It can also start from labels alone, with no examples:
[docs/guide-fallback.md](docs/guide-fallback.md).

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

That lets wasm use several threads: in our runs it cut p95 decision time from 14.3 ms to 7.8 ms.

Download on first visit: the model (23.7 MB) and onnxruntime's wasm (23.6 MB, 5.8 MB gzipped), both cached after.

## Numbers

All-MiniLM-L6-v2 at q8, Ryzen 7 260 laptop, Node 24 and headless Chrome 156, 2026-10-31. Accuracy and calibration
error (ECE, lower is better) on each dataset's test split, mean of 3 seeds (one run for "all"):
[eval/results.json](eval/results.json).

| Examples per label | 5 | 10 | 25 | all |
|---|---|---|---|---|
| BANKING77 (77 labels) | 78.7% | 83.0% | 87.4% | 90.6% |
| CLINC150 (150 labels) | 87.3% | 89.8% | 92.8% | 94.0% |
| ECE, BANKING77 | 0.065 | 0.048 | 0.025 | 0.019 |
| ECE, CLINC150 | 0.029 | 0.013 | 0.012 | 0.012 |

Calibration is loosest at 5 examples per label on BANKING77. On small tasks (2 to 10 labels sampled from CLINC150, scored on its validation
queries, [eval/tiny.ts](eval/tiny.ts)), 1 example per label was right 93% of the time with calibration error 0.020, and
3 per label 98% with 0.016. Add examples before relying on
a threshold there.

**In Chrome** (BANKING77, full training set, worker, cross-origin isolated, [bench/results.json](bench/results.json)):
90.6% accuracy, 3.1 ms median and 5.7 ms p95 per decision, 147 s to train on 10,003 examples.

**Text that fits no label:** CLINC150 has 1,000 test queries that match none of its 150 intents. Trained on
in-scope examples only, with `minConfidence` 0.93 picked on the validation split, 86.3% of them come back `null`,
and 82.9% of in-scope queries are answered and right (94.0% are right with no threshold).

**With few examples, try bge-small.** `createDecider({ model: "Xenova/bge-small-en-v1.5" })` (34 MB, MIT) does
better at 5 per label (82.3% on BANKING77, 89.6% on CLINC150) and about the same with all the data (90.7%, 94.4%).
It takes about twice as long per decision in Chrome. Pass `{ labelNames: false }` to `train()` with it: label names
helped MiniLM but slightly hurt bge-small on the dev splits. [eval/results-bge-small.json](eval/results-bge-small.json).

Reproduce: `npm run eval:suite` (Node), `npm run bench` then `npm run bench:run` (Chrome).

## Playground

Try it at [debuff.jp/decidejs](https://debuff.jp/decidejs/). To run it locally: `npm run playground`, then serve `playground/dist` cross-origin isolated (`node bench/serve.mjs` style headers).
Try zero-shot labels, the ready-made heads, a head trained in the page and "none of these".

## Ready-made heads

[heads/](heads/) has a BANKING77 support-intent head (90.6%) and a CLINC150 assistant-intent head (94.2%), so you
can try a trained head before collecting examples.

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

## Docs

[docs/README.md](docs/README.md) indexes the guides (training, confidence and thresholds, the browser), the
API reference, how it works and the FAQ.

## Community

Questions and show-and-tell in [Discussions](https://github.com/debuffjp/decidejs/discussions), chat in the
[debuff Discord](https://discord.gg/jshbDHSpCY). Results on your own data, good or bad, help most: there's an
issue template for them. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Status

0.1. English only with the default model; short texts (about a paragraph). Not done yet: WebGPU in the worker, a
fallback to a hosted model below the threshold, and fine-tuning the encoder (the likely route past about 91%).

Made by the [debuff](https://debuff.jp) team. MIT.
