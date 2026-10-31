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

Reproduce with `npm run eval` (Node) and `npm run bench` then `npm run bench:run` (headless Chrome).

## Status

0.1, a prototype. Zero-shot confidence is not calibrated: it uses a fixed temperature and reads near 1.0 even on
hard cases, so only trust `confidence` from a trained head. Not done yet: WebGPU in the worker, a fallback to a hosted model when confidence is low, and a
smaller runtime than transformers.js (1.2 MB of JavaScript).

Made by the [debuff](https://debuff.jp) team. MIT.
