# How it works

decidekit has two parts: an encoder that turns text into a vector, and a head that turns the vector into a choice
between your labels. The encoder is a pretrained model you download once. The head is small and trained on your
examples, in seconds to minutes, wherever your code runs.

## The encoder

The default is Decide Small (`debuffjp/decide-small`), an unmodified copy of the sentence-embedding model
bge-small-en-v1.5, at 8-bit weights (34 MB), run through transformers.js. The files download from Hugging Face on first
use and are then cached.
Pre-release builds used all-MiniLM-L6-v2 (23.7 MB), which also returns 384 numbers.
It reads a text and returns 384 numbers. decidekit averages the model's output over the text's tokens (mean pooling)
and scales the result to length 1, so texts with similar meaning end up close together and the dot product of two
vectors is their cosine similarity.

The encoder is never changed by training. That keeps training cheap enough for a browser tab, and it means one
download serves every head trained on that model. A head records the model id it was trained on, and `decide()` throws
if you pair it with a different encoder.

## Zero-shot

With a list of labels and no examples, decidekit encodes each label as text and compares it with the text you pass. Bare
labels are turned into words first (`card_arrival` becomes "card arrival"), which scored 2.6 points higher on CLINC150's
dev split than the raw label. With a record of descriptions, both the readable name and the description are encoded,
and each label scores by the better of the two. Before 0.1.0, the description replaced the name.

The cosine similarities go through a softmax at a fixed temperature per model: 0.03 for bge-small (0.025 when the
labels are described, since descriptions score higher), 0.05 for
all-MiniLM-L6-v2 and any other model. Models spread their similarities differently: at 0.05, bge-small answered only
3% of texts at 0.9 confidence. At these temperatures zero-shot came out close to calibrated on the BANKING77 and
CLINC150 dev splits (ECE 0.03-0.04 for bge-small; 0.005-0.045 for MiniLM at 0.05 in earlier runs). Zero-shot accuracy
from label names alone was 66.2% on BANKING77 and 72.8% on CLINC150 (dev splits; MiniLM 61.4% and 69.3%): enough to
try an idea, not enough to ship most of them.

### Describe your labels

A one-sentence description per label helps more than any other zero-shot change we measured. Pass a record instead of
an array:

```ts
const r = await d.decide(text, {
  card_arrival: "My card still hasn't arrived.",
  refund: "I want my money back for something I bought.",
  lost_card: "I lost my card or it was stolen.",
});
```

Write each description the way a user would say it. On the dev splits, descriptions plus names reached 72.4% on
BANKING77 and 83.1% on CLINC150 with bge-small (68.2% and 79.4% with MiniLM), against 66.2% and 72.8% for names alone.
An LLM can write them once at setup from your label names; check a few by hand. `describeLabels()` does that with any
LLM you pass it:

```ts
import { describeLabels } from "decidekit";

const described = await describeLabels(["card_arrival", "refund", "lost_card"], (prompt) => myLlm(prompt), {
  context: "message to a bank's support chat",
});
// Save `described` and reuse it; decide(text, described) from then on.
```

Each label is scored by the better of its readable name and its description, so a weak description costs little.

## The head

A head is one softmax layer: a weight vector and a bias per label. A label's score is the dot product of its weights
with the text's embedding plus its bias; softmax over the scores, divided by the head's temperature, gives the
probabilities. 77 labels take 156 KB as a saved file, 150 labels 303 KB.

That is the whole classifier, which is why it trains in a tab and decides in a few milliseconds: 7.7 ms median per
decision in Chrome with a worker (4.2 ms with all-MiniLM-L6-v2).

## Training, step by step

### 1. Embed every example

Each distinct text is encoded once, in batches of 64. On a retrain, texts from the previous training set reuse their
cached embeddings, so only new texts go through the encoder.

### 2. Start at the centroids

Before any gradient descent, each label's weights are set to the normalised mean of its examples' embeddings, scaled by
20. Scored that way, a text goes to the label whose average example it is closest to: a nearest-centroid classifier.
The scale of 20 matches all-MiniLM-L6-v2's 0.05 zero-shot temperature (1 / 0.05), so the starting head behaves like
zero-shot over your examples. With bge-small (zero-shot temperature 0.03) the starting head is a little flatter;
gradient descent and the fitted temperature correct for it.

With a handful of examples per label, this starting point is most of the accuracy. Gradient descent then refines it.

### 3. Add the label names

Unless `labelNames` is `false`, each label's readable name is encoded and added to its mean as if it were 1 more
example. A label with 5 examples starts at a point between its examples and its name. With bge-small at 5 examples per
label it took dev accuracy from 80.8% to 81.8% on BANKING77 and from 90.4% to 91.4% on CLINC150 (`eval/results-priors.jsonl`).
Before 0.1.0 the name counted as 3 examples: with all-MiniLM-L6-v2 at 5 per label that took test accuracy from 77.4% to
78.7% on BANKING77 and from 86.0% to 87.3% on CLINC150.

### 4. Gradient descent

From 8 examples per label (mean across labels), 0.2 uses full-batch Adam: learning rate 0.05, no weight decay, up to 100 epochs, keeping the epoch with the lowest loss on the held-out examples and stopping after 30 epochs without a better one. Below 8, and in every version before 0.2, it uses the method below.

Mini-batch gradient descent on softmax cross-entropy: batches of 64, learning rate 2, weight decay 1e-4, up to 60
epochs. After each epoch, accuracy on the held-out examples is checked, and the best epoch is kept. Its count is
recorded, and the best epoch can be 0: if the centroid start is never beaten on held-out examples, gradient descent is
not used at all.

### 5. Fit the temperature

The held-out examples' scores are used to fit the temperature, as described in
[guide-confidence.md](guide-confidence.md#how-the-temperature-is-fitted). In short: a grid search from 0.2 to 5 for the
lowest negative log-likelihood, with 5 folds when the holdout is under 200 examples, held-out examples skipped when
their label has nothing left to train on, and a floor under 50 held-out answers (0.5 for bge-small, 1 for other models; 1 for all in pre-release builds). The fold change took ECE at 1
example per label, with all-MiniLM-L6-v2, from 0.459 to 0.020.

### 6. Retrain on all of it

The held-out runs only pick the epoch count and the temperature. The head you get is trained again on every example,
for the median of the best epoch counts across the folds. With 5 examples per label, holding one out for good would
cost each label a fifth of its data.

## Yielding to the page

Training yields between batches with a `MessageChannel` message. A `setTimeout` would also work in a foreground tab,
but browsers throttle timers in background tabs to once a second or slower; a message-channel task is not throttled that
way. This keeps the training loop from blocking a thread it shares with anything else.

It does not make the main thread safe for the encoder: one encoder call is one long wasm task with no yield points.
That is why the browser entry runs everything in a worker. On the main thread, training BANKING77 froze the page for
171 s; in the worker, the page's longest stall was 18 ms.

## What decidekit does not do

- It does not fine-tune the encoder. Only the head is trained. A fine-tune on SNIPS and PAWS lost to the plain encoder on the dev splits (2026-11-02). Before 0.2 this page called fine-tuning the likely route past about 91% on BANKING77.
- It does not run the encoder on WebGPU in the worker yet.
- It does not fall back to a larger hosted model below the threshold. You can build that on `choice === null`.
