# Falling back to an LLM

Answer what the head is sure about on the device and send the rest to something slower: an LLM, an API, a person.
`withFallback()` does the routing and, if you give it your training examples, keeps the fallback's answers as new
examples, so the head learns from exactly the texts it was unsure about.

```ts
import { createDecider, withFallback } from "decidejs";

const d = await createDecider();
const head = await d.train(examples);

const route = withFallback(d, head, {
  minConfidence: 0.9,
  ask: async (text, labels) => askMyLlm(text, labels), // one of labels, or null
  examples,           // learn from the answers
  retrainEvery: 200,  // retrain in the background after 200 new answers
});

const r = await route("is there a fee for sending money abroad?");
r.choice; // one of your labels, or null
r.source; // "local" or "fallback"
```

It works the same with `createWorkerDecider()` from `decidejs/browser`.

## What it saved on CLINC150

Simulated with [eval/fallback.ts](../eval/fallback.ts): a head trained on 5 examples per label (150 labels), then
10 rounds of 1,000 real queries from CLINC150's training split as traffic. Below 0.9 the true label stands in for
the LLM, the answer becomes an example, and the head retrains after each round. Head accuracy is on the validation
split ([eval/results-fallback-0.9.jsonl](../eval/results-fallback-0.9.jsonl)).

| Round | Examples | Sent to the fallback | Local answers right | Head accuracy |
|---|---|---|---|---|
| 1 | 750 | 40.8% | 98.3% | 86.2% |
| 2 | 1,158 | 31.6% | 98.8% | 87.9% |
| 3 | 1,474 | 24.6% | 98.3% | 90.5% |
| 5 | 1,952 | 20.6% | 99.0% | 92.1% |
| 10 | 2,828 | 17.1% | 99.5% | 94.4% |

(Examples is the training set the head in that round was trained on.)

- The share sent to the fallback fell from 41% to 14-17% within 10,000 queries.
- Answers kept locally were right 98-99.5% of the time, so a 0.9 threshold held up as the head changed.
- The head reached 94.4% with about 3,000 examples. A head trained on all 15,000 scored 94.6% on the same split in
  the hill harness (with label names): learning only from the texts it was unsure about needed a fifth of the labels.

What this doesn't show: a real LLM is sometimes wrong, and its mistakes become training examples; here the stand-in
was always right. Check a sample of learned examples (`route.learned`) before you trust them, or have a person
confirm them. The traffic was in-scope CLINC150 queries in random order; real traffic drifts.

## Options

| Option | Default | |
|---|---|---|
| `minConfidence` | 0.9 | Below this, `ask` is called. |
| `ask(text, labels, local)` | required | Returns one of `labels` or `null`; may be async. `local` is the head's own decision. Any other answer counts as `null`. |
| `examples` | none | The head's training examples. With them, every non-null answer is kept as an example. |
| `retrainEvery` | none | Retrain in the background after this many new answers. Needs `examples`. |
| `trainOptions` | `{}` | Passed to `train()` on retrain. |

The returned function also has:

- `route.retrain()`: train a new head on `examples` plus the learned answers now; resolves to the new head and
  switches to it. Calls while a retrain runs share it.
- `route.head`: the head in use.
- `route.learned`: the answers kept so far. Save them with your examples to keep what was learned across sessions.

A background retrain that fails keeps the current head; call `retrain()` yourself to see the error.

## Writing `ask`

Keep the labels in the prompt and ask for exactly one of them. With the AI SDK (5.x), structured output does this:

```ts
import { generateObject } from "ai";

const ask = async (text: string, labels: readonly string[]) => {
  const { object } = await generateObject({
    model, output: "enum", enum: [...labels, "none"],
    prompt: `Classify this support message: ${text}`,
  });
  return object === "none" ? null : object;
};
```

Any other client works the same way: return a label string or `null`. `withFallback()` checks the answer is one of
the head's labels.

Costs: each fallback is one LLM call. Over the 10,000 simulated queries, 2,249 went to the fallback, against 10,000
calls with an LLM alone.
