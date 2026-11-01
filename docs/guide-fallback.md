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
confirm them (and see [When the LLM is wrong](#when-the-llm-is-wrong)). The traffic was in-scope CLINC150 queries in random order; real traffic drifts.

## Starting from labels alone

Pass labels (or described labels) in place of a head. It starts zero-shot, always learns, and `retrain()` (or
`retrainEvery`) trains the first head from the answers. Labels nobody has answered yet start from their names.

```ts
const route = withFallback(d, ["refund", "bug_report", "question"], { ask, retrainEvery: 100 });
route.head; // null until the first retrain
```

The same simulation with no examples at all ([eval/results-fallback-cold-0.9.jsonl](../eval/results-fallback-cold-0.9.jsonl),
`COLD=1`):

| Round | Examples | Sent to the fallback | Local answers right | Head accuracy |
|---|---|---|---|---|
| 1 (zero-shot) | 0 | 67.1% | 97.6% | 69.6% |
| 2 | 671 | 33.3% | 95.1% | 81.4% |
| 3 | 1,004 | 23.3% | 97.7% | 87.2% |
| 5 | 1,475 | 17.2% | 99.3% | 92.1% |
| 10 | 2,383 | 14.5% | 98.8% | 93.3% |

By round 5 it matched the run that started from 5 examples per label (92.1%), and 2,528 of the 10,000 queries went
to the fallback, against 2,249 for that run. The first head, trained on answers alone, was the weakest point:
95.1% of its local answers were right in round 2.

## When the LLM is wrong

A real LLM gets some answers wrong, and more of the hard texts the head sends it. Those mistakes become training
examples. The head still gets more accurate, but it also fits its temperature (how far it trusts itself) on those
answers, so it learns to doubt itself: its confidence rarely reaches `minConfidence` and it keeps asking.

Started from labels, `withFallback()` handles this itself. While it is still zero-shot, it keeps up to 4 texts per
label (more with fewer than 13 labels, so it can reach 50; before that fix, 12 labels never did) that zero-shot answered locally (about 93% right on BANKING77) and, once it has 50, fits each new head's
temperature on those instead of on the LLM's answers. Nothing to set up.

With your own examples, or to do better than that, pass `calibrateOn`: a few
examples you know are right, 2 per label is enough. It replaces the automatic set:

```ts
const checked = [{ text: "where is my new card", label: "card_arrival" }, /* 2 per label */];
const route = withFallback(d, labels, { ask, retrainEvery: 1000, trainOptions: { calibrateOn: checked } });
```

Simulated with [eval/long-stream.ts](../eval/long-stream.ts): all 77 BANKING77 labels, starting from labels with
descriptions, 10,000 training messages as traffic, retraining every 1,000. The stand-in LLM is wrong on a share of
what it is sent, picking the closest wrong label (real Llama 3.1 8B mistakes were 97% in the head's top 3). Numbers
are for the last 1,000 messages ([eval/results-long-stream.jsonl](../eval/results-long-stream.jsonl)).

| LLM wrong | `calibrateOn` | Sent to the LLM | Answered right | Head on the test set |
|---|---|---|---|---|
| 0% | no | 23% | 98% | 92.0% |
| 0% | automatic | 31% | 99% | 91.9% |
| 15% | no | 64% | 90% | 90.2% |
| 15% | 2 per label | 24% | 95% | 90.1% |
| 15% | automatic | 31% | 93.5% | 90.4% |
| 30% | no | 86% | 71% | 85.6% |
| 30% | 2 per label | 34% | 86% | 86.2% |
| 30% | automatic | 36% | 85% | 86.1% |

Zero-shot alone, never learning, sent 59% and answered 97%, 87% and 80% right at 0%, 15% and 30% wrong; it scores
74% on the test set. At 15% wrong, `calibrateOn` brought the share sent down to what a perfect LLM gave. At 30% it beat not learning at
all, which plain learning didn't. "automatic" is the default described above (`SELF=4` in the script, last 4 rounds
averaged); with a perfect LLM it sends a few points more, with a real one far less.

Real LLM mistakes cluster on hard texts, which are the ones `withFallback()` sends. With `HARD=1` the chance of a
wrong answer grows as zero-shot's confidence falls, keeping 10% wrong over all texts (13 to 16% of what was sent).
Last 4 rounds, sent and answered right: plain 47% and 92.5%, automatic 33% and 93%, 2 checked per label 23% and
93.5%; zero-shot alone 58% and 87%. The order holds.

The same on CLINC150 (`DATASET=clinc150`: 15,000 queries, 150 labels, `HARD=1`, 10% wrong; last 4 rounds): plain
43% sent and 93.4% right, automatic 21% and 95.4%, 2 checked per label 20% and 95.5%; zero-shot alone 44% and 89%.
Here the automatic default matched hand-checked examples. Final heads on the test set: 93.3%, 91.8% and 93.6% (fewer
answers sent means fewer learned).

The cost is on text that fits none of the labels. A head that trusts itself more is also surer about those: of
CLINC150's 1,000 out-of-scope test queries, the final heads answered 1.4% (plain), 9.7% (automatic) and 17.7% (2
checked per label) locally with 0.9 or more, each one a wrong label; zero-shot answered 4.7%. The automatic default is
about where a head trained on all of CLINC150 sits (10.1% at its best threshold, see
[guide-confidence.md](guide-confidence.md#none-of-these)).

## Off-topic text

When `ask` returns `null` ("none of these"), `withFallback()` keeps the text as an example of a label of its own,
`NONE` (`"none_of_these"`, exported). The next head learns what off-topic text looks like and answers `null` itself
when it is sure, without calling `ask`. Turn it off with `learnNone: false`.

Simulated with [eval/none-label.ts](../eval/none-label.ts): CLINC150 from labels with descriptions, 15,000 in-scope
queries plus 700 out-of-scope ones (4.5%), the LLM wrong on 10% (more on hard texts), scored on 500 held-out
out-of-scope queries and the in-scope test set:

| | `learnNone: false` (before) | default |
|---|---|---|
| Off-topic given a confident wrong label | 7.6% | 3.8% |
| Off-topic answered `null` locally | 0% | 56% |
| Off-topic sent to the LLM | 92% | 40% |
| In-scope sent to the LLM | 25.9% | 24.7% |
| In-scope wrongly answered `null` | 0% | 0.6% |

LLMs also answer `null` for texts that do belong to a label but are ambiguous: Llama 3.1 8B did for 6.6% of
in-scope messages on the lanes demo's 12 BANKING77 labels. Those sit close to the labels (median similarity 0.77,
against 0.46 for off-topic text), so a `null` is learned as `NONE` only when the text's zero-shot `similarity` is below
`noneBelow` (default 0.6, set for bge-small; `0` learns every `null`).

With real Llama 3.1 8B answers ([eval/real-stream.ts](../eval/real-stream.ts): the 12 labels' 1,662 training messages
plus 70 off-topic queries, 5 orders, retraining every 100; scored on the labels' test messages and 200 held-out
off-topic queries):

| | Sent, 2nd half | In-scope right locally | In-scope wrong locally | Off-topic given a label | Off-topic `null` locally |
|---|---|---|---|---|---|
| No calibration fix, nulls dropped | 24.6% | 78.3% | 1.5% | 6.7% | 0% |
| Calibration fix, nulls dropped | 23.0% | 85.0% | 2.6% | 20.3% | 0% |
| Every null learned (`noneBelow: 0`) | 20.7% | 81.3% | 3.6% | 0.8% | 91.9% |
| Default (`noneBelow: 0.6`) | 19.1% | 83.3% | 2.8% | 2.4% | 76.6% |

The calibration fix alone made off-topic text worse; with `NONE` learned it is the best of both.

Before this default, `null` answers were dropped. `route.head.labels` and `route.learned` now include `NONE` once
`ask` has returned `null`; `probs` has a `none_of_these` entry. Give the LLM the same label descriptions too: they took Llama 3.1 8B from 84% to 90%
on the lanes demo's messages.

On CLINC150's 150 labels (eval/none-label.ts, one seed, a stand-in LLM that says `null` only for off-topic text),
the 0.6 gate keeps the safety gain: 4.4% of held-out off-topic texts got a confident wrong label, against 3.8% with
no gate and 7.6% without `NONE`. It costs local "none" answers: 24% of off-topic texts against 56% ungated, because
with many labels off-topic text sits closer to some label. Raising `noneBelow` to 0.65 or 0.7 brings that back to
51 to 56% but sent 4 to 5 points more in-scope text. The stand-in never says `null` for in-scope text, which real
LLMs do, so this measures only the gate's cost. With many labels and an LLM you trust to say `null` only for
off-topic text, try `noneBelow: 0.7`.

## Options

| Option | Default | |
|---|---|---|
| `minConfidence` | 0.9 | Below this, `ask` is called. |
| `ask(text, labels, local)` | required | Returns one of `labels` or `null`; may be async. `local` is the head's own decision. Any other answer counts as `null`. |
| `examples` | none | The head's training examples. With them, every non-null answer is kept as an example. |
| `retrainEvery` | none | Retrain in the background after this many new answers. Needs `examples` or labels. |
| `trainOptions` | `{}` | Passed to `train()` on retrain. Starting from labels, `calibrateOn` defaults to texts zero-shot answered confidently. |
| `learnNone` | `true` | Learn `null` answers as the label `NONE`, so the head can answer `null` itself. |
| `noneBelow` | `0.6` | Learn a `null` as `NONE` only when the text's zero-shot similarity to every label is below this. `0` learns every `null`. |
| `restore` | none | `{ learned, confident }` from an earlier `route.saved()`. Doesn't retrain by itself. |

The returned function also has:

- `route.retrain()`: train a new head on `examples` plus the learned answers now; resolves to the new head and
  switches to it. Calls while a retrain runs share it.
- `route.head`: the head in use.
- `route.learned`: the answers kept so far.
- `route.saved()`: the answers plus the calibration set, as plain JSON. Store it, pass it back as `restore`, then
  call `retrain()` (or pass a saved head as the labels):

  ```ts
  const route = withFallback(decider, labels, { ask, restore: JSON.parse(localStorage.decideState ?? "{}") });
  if (route.learned.length) await route.retrain();
  // later
  localStorage.decideState = JSON.stringify(route.saved());
  ```

  Before this, only `route.learned` could be saved, so a reload lost the calibration set and the next head fitted its
  confidence on the LLM's answers.

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
