# Confidence and thresholds

Every `Decision` has a `confidence`: the probability of `choice`, from 0 to 1. For a trained head it is calibrated, so
you can act on it: send low-confidence answers to a person, ask the user to confirm or answer "none of these".

## What calibrated means

Take all the answers a head gave with confidence around 0.9. If about 90% of them are right, the head is calibrated
at 0.9. Calibrated everywhere means a confidence of `p` is right about `p` of the time, at every `p`.

The error is measured as ECE (expected calibration error): group answers by confidence, compare each group's average
confidence with its accuracy, and average the gaps weighted by group size. 0 is perfect. On the test splits:

| Examples per label | 5 | 10 | 25 | all |
|---|---|---|---|---|
| ECE, BANKING77 | 0.065 | 0.048 | 0.025 | 0.019 |
| ECE, CLINC150 | 0.029 | 0.013 | 0.012 | 0.012 |

Calibration is loosest with few examples on labels that sit close together (BANKING77 at 5 per label). On small tasks
of 2 to 10 labels, ECE was 0.020 at 1 example per label and 0.008 at 5.

Calibrated does not mean accurate. A head that is right 80% of the time and says 0.8 is calibrated; it is still wrong
one time in five. What calibration buys is that the number tells you which answers to trust.

## How the temperature is fitted

A head scores each label, and softmax turns the scores into probabilities. Dividing the scores by a temperature before
softmax makes the probabilities sharper (below 1) or flatter (above 1) without changing which label wins. `train()`
picks the temperature so the probabilities match how often the head is right on examples it did not train on:

1. Hold out examples: 10% by default (`holdout`). When that is under 200 examples, use 5 folds instead, so every
   example is held out once.
2. Train on the rest and score the held-out examples. Skip held-out examples whose label has no training example left
   in that fold: that label could never be chosen, and counting it as a miss pulls the temperature too high.
3. Try temperatures from 0.2 to 5 in steps of 0.05 and keep the one with the lowest negative log-likelihood on the
   held-out answers.
4. With under 50 held-out answers, never go below 1. A perfect score on a few examples would otherwise push every
   answer to confidence 1.0.
5. Retrain on every example and keep the temperature.

The head stores it as `head.temperature`. The 5-fold step matters with little data: at 1 example per label it took ECE
from 0.459 to 0.020.

## Pick a threshold on your own data

`minConfidence` turns low-confidence answers into `null`:

```ts
const r = await d.decide(text, head, { minConfidence: 0.9 });
if (r.choice === null) askAPerson(text);
```

The right threshold depends on your labels, your data and what a wrong answer costs. Measure it: keep some labelled
examples out of training, then count, at each threshold, how many answers are held back and how many of the rest are
right.

```ts
import { createDecider, type Example } from "decidejs";

const d = await createDecider();

// Hold out a fifth of the examples. Shuffle first if they're sorted by label.
const cut = Math.floor(examples.length * 0.8);
const train = examples.slice(0, cut), test = examples.slice(cut);

const head = await d.train(train);
const rs = await d.decideMany(test.map((e) => e.text), head);

for (const t of [0, 0.5, 0.7, 0.8, 0.9, 0.95]) {
  const answered = rs.flatMap((r, i) => (r.confidence >= t ? [r.choice === test[i].label] : []));
  const right = answered.filter(Boolean).length;
  console.log(
    `threshold ${t}: null ${(100 * (1 - answered.length / rs.length)).toFixed(1)}%,`,
    `right when answered ${(100 * right / Math.max(1, answered.length)).toFixed(1)}%`,
  );
}
```

Pick the lowest threshold whose "right when answered" is good enough for the job, and check that the null rate is one
you can live with. Then train the head you ship on all the examples, and keep the threshold.

The held-out set needs to be big enough: with 20 test examples, one answer moves the result by 5 points. If you also
expect text that fits none of your labels, put some of it in the test set (labelled with a name that is not one of the
head's labels, so it always counts as wrong when answered) and check how much of it comes back null.

## Zero-shot confidence

Zero-shot `decide()` (a label array or a record of descriptions) has no examples to fit a temperature on. It uses a
fixed one, 0.05 by default (`createDecider({ temperature })`). On BANKING77 and CLINC150 that came out close to
calibrated, ECE 0.005-0.045 with 2 to 150 options. That was measured on two intent datasets only. Your labels or
descriptions may be scored sharper or flatter, so check zero-shot confidence on some labelled text of your own before
putting a threshold on it, or train a head.

## "None of these"

Two ways to handle text that fits none of your labels.

**A threshold.** Train on your real labels only and set `minConfidence`. Off-topic text tends to score low on every
label, so it comes back `null`. On CLINC150, trained on in-scope examples only, a threshold of 0.93 picked on the
validation split returned `null` for 86.3% of the 1,000 out-of-scope test queries. The cost: 82.9% of in-scope queries
were answered and right, against 94.0% right with no threshold, because some correct answers also fall below 0.93.

This needs enough examples per label to mean something. With 3 examples per label, a head gave "hello" a confidence of
0.89. CLINC150 has 100 examples per label.

**An `other` label.** Collect examples of the off-topic text you see (greetings, spam, questions about something else)
and train them as a label of their own. This works when off-topic text has a recognisable shape, and it lets a small
head say "none of these" without a high threshold. It only catches the kinds of off-topic text you gave it examples of.
Train it with `labelNames: false`, or name it something descriptive: the word "other" says little as a hint.

The two combine: an `other` label for the off-topic text you know about, a threshold for the rest.
