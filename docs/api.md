# API reference

decidejs 0.1. Two entry points and a CLI:

- `decidejs`: `createDecider`, `saveHead`, `loadHead` and the types. Runs in Node and the browser.
- `decidejs/browser`: `createWorkerDecider`, `saveHead`, `loadHead` and the same types. Browser only (it starts a Web Worker).
- `npx decidejs`: `train` and `decide` from the command line (Node 22.18 or later).

## createDecider(options?)

```ts
function createDecider(options?: DeciderOptions): Promise<{
  model: string;
  decide: ...;      // see decide()
  decideMany: ...;  // see decideMany()
  train: ...;       // see train()
  embed: (texts: readonly string[]) => Promise<number[][]>;
}>;
```

Loads the sentence encoder through transformers.js and resolves once it is ready. The first call downloads the model
files; later loads come from the cache (the browser's cache, or transformers.js' cache directory in Node).

The encoder runs on the thread that calls it. In a browser, use [`createWorkerDecider()`](#createworkerdecideroptions-worker)
instead, or the page freezes while it works.

```ts
import { createDecider } from "decidejs";

const d = await createDecider();
const r = await d.decide("my card never arrived", ["card_arrival", "refund", "other"]);
```

### DeciderOptions

| Option | Type | Default | What it does |
|---|---|---|---|
| `model` | `string` | `"Xenova/bge-small-en-v1.5"` | Hugging Face id of a sentence-embedding model with ONNX weights. Before this release the default was `"Xenova/all-MiniLM-L6-v2"`: pass it to keep using heads trained on it. |
| `device` | `"wasm" \| "webgpu" \| "cpu"` | transformers.js' choice (wasm in browsers, cpu in Node) | Where the encoder runs. |
| `dtype` | `"q8" \| "fp16" \| "fp32"` | `"q8"` | Weight precision of the encoder. |
| `onProgress` | `(p: { file: string; loaded: number; total: number }) => void` | none | Called as model files download. Only fires on a download, so not on cached loads. |
| `temperature` | `number` | `0.03` for bge-small, `0.05` for other models (`0.05` for all models before this release) | Softmax temperature for zero-shot similarities. Has no effect on trained heads, which carry their own. |

The returned `model` is the model id in use. Trained heads record it, and `decide()` refuses a head trained on a
different model.

### decide(text, using, options?)

```ts
decide<const L extends string>(text: string, using: Choices<L>): Promise<Decision<L>>;
decide<const L extends string>(text: string, using: Choices<L>, options: DecideOptions): Promise<Decision<L, L | null>>;
```

Picks one option for `text`. `using` is one of:

- **An array of labels** (zero-shot). Each label is compared with the text as readable words: `_` and `-` become
  spaces, camelCase is split and the result is lowercased, so `card_arrival` and `cardArrival` are both read as
  "card arrival".
- **A record of label to description** (zero-shot). The descriptions are compared with the text, as written; the
  keys are what `choice` returns.
- **A trained [`Head`](#head)**. The head's weights score the text and its fitted temperature turns the scores into
  probabilities.

`const L` means a literal array keeps its literal types without `as const`: `choice` is typed
`"card_arrival" | "refund" | "other"`, not `string`.

With `options.minConfidence` set, `choice` is `null` when `confidence` is below it, and the return type includes
`null`. `confidence` and `probs` are filled in either way.

```ts
const r = await d.decide(text, { refund: "wants their money back", bug: "something is broken" });
r.choice; // "refund" | "bug"

const s = await d.decide(text, head, { minConfidence: 0.9 });
if (s.choice === null) sendToAPerson(text);
```

Zero-shot option embeddings are cached per decider, keyed on the exact list of option texts, so calling `decide()`
again with the same options only encodes the new text.

Throws:

- `decidejs: decide() needs at least 2 options, got N` when an array or record has fewer than 2 entries.
- `decidejs: this head was trained on <head model>, but the decider uses <model>. Pass { model: "<head model>" } to createDecider.`
  when a head's `model` does not match the decider's.

### decideMany(texts, using, options?)

```ts
decideMany<const L extends string>(texts: readonly string[], using: Choices<L>): Promise<Decision<L>[]>;
decideMany<const L extends string>(texts: readonly string[], using: Choices<L>, options: DecideOptions): Promise<Decision<L, L | null>[]>;
```

`decide()` for many texts. Results come back in the order of `texts`. Texts are encoded in batches of 64, which is much
faster than one `decide()` per text in Node: 0.8-1.5 ms per text with a trained head. Throws the same errors as
`decide()`.

```ts
const rs = await d.decideMany(reviews, head);
const bugs = reviews.filter((_, i) => rs[i].choice === "bug");
```

### train(examples, options?)

```ts
train<const L extends string>(examples: readonly Example<L>[], options?: TrainOptions): Promise<Head<L>>;
```

Trains a head on labelled examples and resolves with it. The head's labels are the distinct labels in `examples`,
sorted. [how-it-works.md](how-it-works.md) explains the steps; [guide-training.md](guide-training.md) covers how many
examples you need.

Retraining on the same decider reuses the embeddings of the previous training set: only texts that were not in the last
call are encoded. The cache holds the last set only.

```ts
const head = await d.train([
  { text: "where is my card", label: "card_arrival" },
  { text: "I want my money back", label: "refund" },
  // ...
]);
```

#### TrainOptions

| Option | Type | Default | What it does |
|---|---|---|---|
| `labelNames` | `boolean` | `true` | Use each label's readable name as a hint at the starting point (it counts as 1 extra example; 3 before the switch to bge-small). Turn off for opaque ids like `c17`. |
| `epochs` | `number` | `60` | Most passes over the examples. The best epoch on held-out examples is kept. |
| `learningRate` | `number` | `2` | Gradient descent step size. |
| `l2` | `number` | `1e-4` | Weight decay. |
| `holdout` | `number` | `0.1` | Share of examples held out to pick the epoch count and fit the temperature. When that share is under 200 examples, 5 folds are used instead (every example is held out once). |
| `seed` | `number` | `7` | Seed for the split and the shuffling, so a run repeats exactly. |
| `onEpoch` | `(epoch: number, holdoutAccuracy: number) => void` | none | Called after each epoch of the held-out runs with accuracy on the held-out examples. `epoch` counts from 0. With 5 folds it runs once per fold, so `epoch` starts again at 0 for each fold. It is not called during the final retrain on all examples. |

Throws:

- `decidejs: example N needs a string text and a non-empty string label, got {...}` for the first bad example.
- `decidejs: train() needs at least 2 labels, got N`.

Training yields to the event loop between batches of 64, so a page stays responsive, but the encoding step before it
does not. In a browser, train in a worker.

### embed(texts)

```ts
embed(texts: readonly string[]): Promise<number[][]>;
```

The encoder on its own: one vector per text, mean pooled and normalised to unit length (384 numbers with the default
model). Useful for your own similarity search or for checking which examples sit close together.

```ts
const [a, b] = await d.embed(["refund please", "I want my money back"]);
const similarity = a.reduce((s, x, i) => s + x * b[i], 0); // cosine, since both have length 1
```

## createWorkerDecider(options?, worker?)

```ts
import { createWorkerDecider } from "decidejs/browser";

function createWorkerDecider(options?: DeciderOptions, worker?: Worker): Promise<{
  decide: ...;      // same overloads as createDecider's
  decideMany: ...;
  train: <const L extends string>(examples: readonly Example<L>[], options?: TrainOptions) => Promise<Head<L>>;
  terminate: () => void;
}>;
```

`createDecider()` running in a Web Worker, so the page keeps painting and handling input while the encoder works. Takes
the same [`DeciderOptions`](#decideroptions); `onProgress` and `onEpoch` are relayed from the worker. Resolves once the
worker has loaded the model.

Differences from `createDecider()`:

- No `embed` and no `model` on the result.
- `terminate()` stops the worker and rejects any calls still pending ("decidejs: the worker was terminated").
- Errors thrown in the worker come back as `Error`s with the original message.
- Each call costs about 1 ms of message passing.

`worker` defaults to `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`. Pass your own when your
bundler needs a different worker URL. See [guide-browser.md](guide-browser.md) and [bundlers.md](bundlers.md).

```ts
const d = await createWorkerDecider({
  onProgress: ({ loaded, total }) => (bar.value = loaded / total),
});
const r = await d.decide(input.value, ["question", "bug_report", "praise"]);
```

## describeLabels(labels, complete, options?)

Asks an LLM once for a one-sentence description of each label, written as a typical message under it, and returns a
`Record<L, string>` to pass to `decide()`. `complete(prompt)` is your LLM call; it should return the reply text.
Labels the reply misses get their readable name. `options.context` says what the texts are ("message to a bank's
support chat"). On the dev splits, descriptions raised zero-shot accuracy by 6 to 10 points.

`describePrompt(labels, options?)` returns the prompt and `parseDescriptions(labels, reply)` reads a reply, for when
you'd rather run the LLM call yourself (a batch job, a different process).

## saveHead(head)

```ts
function saveHead(head: Head): string;
```

Serialises a head to a JSON string with the weights and biases as base64. Write it to a file and serve it next to your
app.

```ts
import { writeFile } from "node:fs/promises";
await writeFile("intent.head.json", saveHead(head));
```

## loadHead(json)

```ts
function loadHead<L extends string = string>(json: string): Head<L>;
```

Parses a string written by `saveHead()`. The labels are typed `string` unless you pass the label type, for example
the one `decidejs train --types` writes:

```ts
import { loadHead } from "decidejs/browser";
import type { Label } from "./intent.head.labels.ts";

const head = loadHead<Label>(await (await fetch("/intent.head.json")).text());
```

`loadHead` does not check the labels against `L`, and throws whatever `JSON.parse` or `atob` throws on malformed input.
Pass heads through `loadHead()`, not plain `JSON.parse()`: a parsed object without `Float32Array` weights is not
recognised as a head.

Both `saveHead` and `loadHead` are exported from `decidejs` and from `decidejs/browser`. Import them from
`decidejs/browser` in a page: that entry does not pull the encoder into your bundle (3.6 KB against 482 KB).

## Types

### Decision

```ts
interface Decision<L extends string, C extends L | null = L> {
  choice: C;                 // the most likely option; null when below minConfidence
  confidence: number;        // probability of the top option, 0 to 1
  probs: Record<L, number>;  // probability of every option, sums to 1
}
```

`confidence` is the top option's probability even when `choice` is `null`.

### Head

```ts
interface Head<L extends string = string> {
  model: string;             // encoder the head was trained on
  labels: readonly L[];      // sorted
  dim: number;               // embedding size (384 for the default model)
  weights: Float32Array;     // labels.length x dim, row-major
  bias: Float32Array;        // one per label
  temperature: number;       // fitted on held-out examples
}
```

### Choices

```ts
type Choices<L extends string> = readonly L[] | Readonly<Record<L, string>> | Head<L>;
```

### DecideOptions

```ts
interface DecideOptions {
  minConfidence?: number;    // below this, choice is null
}
```

### Example

```ts
interface Example<L extends string> {
  text: string;
  label: L;
}
```

### TrainOptions, DeciderOptions

See the tables under [train()](#trainoptions) and [createDecider()](#decideroptions).

## CLI

```sh
npx decidejs train <examples.csv|.jsonl> [-o head.json] [--model <hf-id>] [--no-label-names] [--types]
npx decidejs decide <head.json> "<text>" [--min-confidence <0-1>]
npx decidejs --help
```

### train

Reads examples, trains a head and writes it with `saveHead()`.

- Input: a `.jsonl` file with one `{"text": ..., "label": ...}` object per line, or any other file read as CSV. The CSV
  needs a header row with `text` and `label` columns (other columns are ignored); fields can be quoted with `"`, and
  `""` inside quotes is a literal quote.
- `-o, --out <file>`: where to write the head. Default `head.json`.
- `--model <id>`: encoder to train on. Default `Xenova/bge-small-en-v1.5` (`Xenova/all-MiniLM-L6-v2` before this release).
- `--no-label-names`: same as `labelNames: false`.
- `--types`: also write `<out without .json>.labels.ts`:

  ```ts
  // Written by decidejs train --types. Labels of the head next to this file.
  export const labels = ["card_arrival","refund"] as const;
  export type Label = (typeof labels)[number];
  ```

It prints the example count, label count and fewest examples per label to stderr, warns when a label has fewer than 3,
and ends with the file size, training time and fitted temperature. Other `TrainOptions` keep their defaults.

```sh
npx decidejs train tickets.csv -o support.head.json --types
```

### decide

Loads a head, creates a decider with the head's model and prints one line of JSON to stdout:

```sh
npx decidejs decide support.head.json "my card never arrived" --min-confidence 0.9
```

The output has the shape `{"choice":"card_arrival","confidence":<number>}`.

`confidence` is rounded to 4 decimals; `choice` is `null` below `--min-confidence`.

Exit codes: 0 on success, 1 on an error (the message goes to stderr), 2 on a usage error.

## withFallback() and decidejs/react

`withFallback(decider, head, options)` (from `decidejs` and `decidejs/browser`) is documented in
[guide-fallback.md](guide-fallback.md). `useDecider()` and `useDecision()` (from `decidejs/react`) are documented in
[guide-react.md](guide-react.md).

## decidejs/switch

### switchDecider(decider, options?)

Wraps a decider from `createDecider()` as a [switch](https://github.com/osolmaz/switch) `Decider`. Answers `choice`, `bool` and `score` questions locally; images are ignored.

- `text(state, question, id)`: what decidejs reads for a question. Default: `textOf(question.instructions, state)`.
- `heads`: trained heads by question id.
- `teacher`: a switch decider to ask when a choice is below `minConfidence` (default 0.9); its answers train a head per repeating question every 8 answers.

### textOf(instructions, state)

The instructions, then the state as `key: value` lines. See the [switch guide](guide-switch.md).
