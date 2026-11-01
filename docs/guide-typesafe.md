# Coming from Jev

[Jev](https://typesafe.ai/) is TypeSafe's hosted [System One model](https://docs.typesafe.ai/concepts/system-one). `decidejs/typesafe` gives a decider the same call shape as [their JavaScript SDK](https://docs.typesafe.ai/sdk/javascript), so Choice questions move between the two with a changed import and client:

```ts
import { createDecider } from "decidejs";
import { choice, typesafeClient } from "decidejs/typesafe";

const client = typesafeClient(await createDecider());
const { answers } = await client.systemOne({
  state: "My running shoes arrived in the wrong size.",
  questions: {
    department: choice("Which team should handle this?", {
      returns: "Exchanges, wrong or damaged items",
      shipping: "Delivery status, delays, lost packages",
      billing: "Charges, invoices, payment problems",
    }),
  },
});
answers.department.choice; // "returns" | "shipping" | "billing"
```

`criteria` can also be a trained head from `train()`. In the browser, pass `createWorkerDecider()` instead.

## What differs

| | Jev | `decidejs/typesafe` |
|---|---|---|
| Choice | yes | yes |
| Noul, Score | yes | throw an error |
| `instructions` | read by the model | ignored |
| `state` | text, objects and arrays | text, or an object of text, numbers and booleans, joined by lines |
| Images, `assets` | experimental | no |
| Response | `model`, `answers`, `usage` | `model`, `answers` |
| Runs | TypeSafe's API | the user's device |

**`instructions` are ignored.** An encoder compares the state with the options. It doesn't read a question, so everything has to be in the option names and descriptions. A question like "Which team should handle this?" with bare option names will do worse here than on Jev: describe the options.

**Noul and Score throw.** A Noul has one statement and an encoder has nothing to weigh it against, so any probability would be unmeasured. Ask a yes/no as a choice between two described options, and check it on your own data.

**Accuracy is not Jev's.** We haven't run the two side by side on the same tasks. This layer matches the shape, and answers won't always match.

## `confidence`

`decide()` and Jev use the same field names (`choice`, `probabilities`, `confidence`), but `confidence` means something different, and this layer follows TypeSafe. [Their `confidence`](https://docs.typesafe.ai/confidence) measures how concentrated the probabilities are, from 1 with everything on one option to 0 when evenly spread: `(n * top - 1) / (n - 1)` for `n` options. `decide()` returns the top probability itself.

| Top probability, 3 options | `decide()` `confidence` | `systemOne()` `confidence` |
|---|---|---|
| 0.61 | 0.61 | 0.415 |
| 0.90 | 0.90 | 0.85 |

The calibration measured in the [confidence guide](guide-confidence.md) is for the top probability: read it with `Math.max(...Object.values(answer.probabilities))`. We keep the top probability as the default because it can be read directly: an answer at 0.9 is right about 90% of the time, at any number of labels. TypeSafe's measure has a fair zero, an even split, which top probability lacks: 0.5 is weak with two labels and strong with fifty.

A threshold tuned on Jev won't carry over to decidejs, or the other way: pick it again on your own data.

## Jev as the fallback

Jev can be the slower model behind `withFallback()`. This uses TypeSafe's own SDK and your API key. We haven't run this example against the live API:

```ts
import { withFallback, createDecider } from "decidejs";
import { choice, TypeSafeClient } from "@typesafe-ai/sdk";

const jev = new TypeSafeClient();
const criteria = { returns: null, shipping: null, billing: null };
const route = withFallback(await createDecider(), ["returns", "shipping", "billing"], {
  ask: async (text) => (await jev.systemOne({ state: text, questions: { team: choice("Which team should handle this?", criteria) } })).answers.team.choice,
  retrainEvery: 100,
});
```

Jev never answers `null` for a Choice unless you give it an option for that, so `withFallback` won't learn "none of these" this way. Map a `none_of_the_above` option to `null` in `ask` if your traffic has off-topic text.
