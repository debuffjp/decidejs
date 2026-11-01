# Training a head

A head is a small layer trained on your examples. It sits on top of the encoder and turns an embedding into a choice
between your labels. Trained heads are far more accurate than zero-shot (90-94% against 61-70% on the two intent
benchmarks below) and their confidence is calibrated.

## Collect examples

An example is a piece of text and the label it should get:

```ts
const examples = [
  { text: "my card still hasn't come", label: "card_arrival" },
  { text: "can I get a refund for this", label: "refund" },
  // ...
];
```

Take them from the place the model will be used: past support tickets, real search queries, reviews. Text written
by the people who will type into your app beats text you make up, because it has their spelling, length and habits.

- Write texts the way users write them. If messages are short and lowercase, the examples should be too.
- Vary each label. Five rewordings of one sentence teach less than five different ways people ask for the thing.
- Keep labels distinct. If you can't tell two labels apart from the text alone, the head can't either: merge them.
- Duplicate texts are fine; they are encoded once.
- Texts are read up to about a paragraph. Longer text is cut by the encoder (see the [FAQ](faq.md#can-it-read-long-texts)).

## How many per label

Accuracy on each dataset's test split, mean of 3 seeds, default model (bge-small-en-v1.5 at q8):

| Examples per label | 5 | 10 | 25 | all |
|---|---|---|---|---|
| BANKING77 (77 labels) | 82.3% | 83.8% | 88.7% | 90.7% |
| CLINC150 (150 labels) | 89.6% | 92.0% | 93.4% | 94.4% |
| Calibration error (ECE), BANKING77 | 0.071 | 0.029 | 0.011 | 0.017 |
| Calibration error (ECE), CLINC150 | 0.028 | 0.010 | 0.015 | 0.008 |

Source: `eval/results-bge-small.json`. With all-MiniLM-L6-v2, the pre-release default (`eval/results.json`):

| Examples per label | 5 | 10 | 25 | all |
|---|---|---|---|---|
| BANKING77 (77 labels) | 78.7% | 83.0% | 87.4% | 90.6% |
| CLINC150 (150 labels) | 87.3% | 89.8% | 92.8% | 94.0% |
| Calibration error (ECE), BANKING77 | 0.065 | 0.048 | 0.025 | 0.019 |
| Calibration error (ECE), CLINC150 | 0.029 | 0.013 | 0.012 | 0.012 |
 BANKING77's labels are close together (many card and transfer intents), so it needs more
examples than CLINC150, whose 150 intents spread over 10 domains.

Small tasks need fewer. On 60 tasks per size with 2 to 10 labels sampled from CLINC150, scored on its validation
queries (`eval/tiny.ts`):

| Examples per label | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| Accuracy | 93.4% | 96.9% | 97.8% | 98.4% |
| ECE | 0.020 | 0.028 | 0.016 | 0.008 |

A rough plan:

1. Start with 5 per label and check it on texts you held back.
2. Look at the mistakes. Add examples to the labels that get confused, not evenly to all.
3. Go to 25 per label where accuracy matters, and more if your labels are close together like BANKING77's.

The CLI warns when a label has fewer than 3 examples. Thresholds on `confidence` need more data than accuracy does: see
[guide-confidence.md](guide-confidence.md).

## Label names

By default each label's name is used as a hint: `card_arrival` is read as "card arrival" and counted as 1 extra
example at the starting point of training. With bge-small at 5 examples per label it raised dev accuracy from 80.8% to
81.8% on BANKING77 and from 90.4% to 91.4% on CLINC150; with 1 example per label, from 58% to 73% and 69% to 84%.
Pre-release builds (all-MiniLM-L6-v2) counted the name as 3 examples: there it raised test accuracy at 5 per label from
77.4% to 78.7% on BANKING77 and from 86.0% to 87.3% on CLINC150.

Turn it off with `{ labelNames: false }` (CLI: `--no-label-names`) when:

- labels are opaque ids (`c17`, `INTENT_042`) or codes whose words mean nothing;
- a label's name says something different from what its examples mean (`other`, `misc`, an internal team name);
- you use bge-small, where label names slightly hurt on the dev splits.

Labels are compared as words: `_` and `-` become spaces and camelCase is split. `cardArrival`, `card-arrival` and
`card_arrival` all read as "card arrival".

## Train

```ts
import { createDecider, saveHead } from "decidekit";
import { writeFile } from "node:fs/promises";

const d = await createDecider();
const head = await d.train(examples, {
  onEpoch: (epoch, acc) => console.log(epoch, acc),
});
await writeFile("support.head.json", saveHead(head));
```

The head's labels are the distinct labels in `examples`, sorted. Training repeats exactly for the same examples and
`seed`. Other options are in the [API reference](api.md#trainoptions); the defaults are the ones the tables above were
measured with.

In a browser, train with `createWorkerDecider()` from `decidekit/browser`: the same `train()`, run in a worker. On the main
thread, training BANKING77 froze the page for 171 s; in a worker the longest stall was 18 ms. Training 10,003 examples
in a tab took 147-217 s; 9 examples with 3 labels took 0.1 s.

## Retrain

Add or fix examples and call `train()` again on the same decider. It keeps the embeddings of the previous training set,
so only the new texts are encoded. The rest of training starts from scratch, so the new head does not depend on the old
one.

```ts
examples.push({ text: "card not here yet", label: "card_arrival" });
const updated = await d.train(examples);
```

The cache holds the last training set only, so alternating between two unrelated sets encodes each set every time.

A loop that works well for labelling in the browser: label a few rows per class, train, run `decideMany()` over the
unlabelled rows, review the lowest-confidence ones first, add them, retrain.

## Ship a head

`saveHead()` writes JSON with base64 weights: 156 KB for 77 labels, 303 KB for 150. Serve it as a static file and load
it with `loadHead()`:

```ts
import { createWorkerDecider, loadHead } from "decidekit/browser";

const head = loadHead(await (await fetch("/support.head.json")).text());
const d = await createWorkerDecider();
const r = await d.decide(text, head);
```

A head only works with the encoder it was trained on. It records the model id, and `decide()` throws if the decider uses
a different one. When you change model, retrain.

## The CLI

Train from a CSV or JSONL file without writing code:

```sh
npx decidekit train tickets.csv -o support.head.json --types
npx decidekit decide support.head.json "my card never arrived" --min-confidence 0.9
```

The CSV needs a header with `text` and `label` columns; JSONL needs one `{"text", "label"}` object per line. Full flags
are in the [API reference](api.md#cli).

### --types

`--types` writes `support.head.labels.ts` next to the head:

```ts
// Written by decidekit train --types. Labels of the head next to this file.
export const labels = ["card_arrival","refund"] as const;
export type Label = (typeof labels)[number];
```

Pass the type to `loadHead` so `choice` is typed as your labels:

```ts
import type { Label } from "./support.head.labels.ts";

const head = loadHead<Label>(json);
const r = await d.decide(text, head);
r.choice; // "card_arrival" | "refund"
```

Commit the labels file with the head and regenerate both together: `loadHead` trusts the type you give it and does not
check it against the file.

## Choosing a model

Two encoders have been measured. Both are English.

| | all-MiniLM-L6-v2 | bge-small-en-v1.5 (default) |
|---|---|---|
| Model id | `Xenova/all-MiniLM-L6-v2` | `Xenova/bge-small-en-v1.5` |
| Download | 23.7 MB at q8 | 34 MB at q8 |
| 5 per label, BANKING77 | 78.7% | 82.3% |
| 5 per label, CLINC150 | 87.3% | 89.6% |
| All data, BANKING77 | 90.6% | 90.7% |
| All data, CLINC150 | 94.0% | 94.4% |
| Time per decision, Chrome median | 4.2 ms | 7.7 ms |
| Label names | help | slightly hurt; pass `labelNames: false` |

The bge-small numbers are without label names (`eval/results-bge-small.json`). With only a handful of examples per label,
bge-small is the better start, which is why it became the default. Pre-release builds defaulted to all-MiniLM-L6-v2.
With plenty of data the two are close, and MiniLM is smaller and faster. Heads trained on MiniLM need
`createDecider({ model: "Xenova/all-MiniLM-L6-v2" })`.

```ts
const d = await createDecider({ model: "Xenova/bge-small-en-v1.5" });
const head = await d.train(examples, { labelNames: false });
```

Any Hugging Face sentence-embedding model with ONNX weights can be passed as `model`, but others have not been measured.
