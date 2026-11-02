# Accuracy spike for decidekit (2026-11-02)

The spike notes as written before 0.2 shipped. "Unknown" below is out of date: 0.2 added the page yield and was measured in Chrome (see the CHANGELOG). `run.ts` now asks for `solver: "sgd"` so its baseline still means the old head, and it has not been rerun since that change. The fine-tuning scripts and `embed.ts` caches were not kept: only their results are here, as the `F-` rows of `results.jsonl`. No datasets or fine-tuned weights are included.

**Result: head H14 passes on BANKING77 and CLINC150, with two HWU64 notes to judge. The fine-tuned encoder was reverted.**
H14 trains the softmax head with full-batch Adam from the existing centroid and label-name start, with no weight decay,
and keeps today's SGD below 8 examples per label. The technique comes from jimothy's `src/linear.ts` (MIT, Andrew Prifer).

Raw lines are in this folder. `results.jsonl` holds every run,
`test-looks.tsv` holds every test look. `src/` is unchanged. Heads were scored on cached q8 embeddings from the public
`embed()`, not through `train()` and `decide()`.

## Step 1: jimothy (measured)

jimothy commit f2ad9b4, its own split (8,023 train, 1,969 dev, 3,079 test), README settings.

| Measure | Value |
| --- | --- |
| Test accuracy, `report.json` | 92.66% (README 92.37%: reproduces, slightly higher) |
| Test accuracy, its `predict` command on the same rows | 92.53%, cause of the gap not investigated |
| ECE, 15 bins, from `predict` | 0.014 |
| Backend | Xenova/all-MiniLM-L6-v2 q8 |
| Bundle | 24 MB (encoder 23 MB, model.json 568 KB) |
| Training, tuning and evaluation | 12 min 10 s on a loaded machine (`elapsedMs` 729,744) |

What produces the number: full-batch Adam from zero, lr 0.05, 100 epochs. L2 = 0 won, and the best epoch was the cap.
Its encoder is the older MiniLM, so the gain is in the optimiser (H5 below supports this on bge-small).

## Step 2: decision log, dev splits, bge-small q8

Dev is a seeded 20% of BANKING77 train and CLINC150 val. All data, seed 7, one run each.

| id | Change | BANKING77 | CLINC150 | Verdict |
| --- | --- | --- | --- | --- |
| H0 | Baseline | 88.46 | 95.43 | |
| H1 | SGD, L2 1e-3 | 86.41 | 94.47 | revert |
| H1 | SGD, L2 1e-5 | 90.51 | 96.27 | superseded |
| H1 | SGD, L2 1e-6 | 90.55 | 96.27 | superseded |
| H1 | SGD, L2 0 | 90.55 | 96.30 | superseded |
| H5 | Adam from zero, 100 epochs | 92.35 | 96.67 | superseded |
| H6 | H5, 200 epochs | 92.90 | 96.63 | no clear gain, about 2x time |
| H12 | H5, 400 epochs | 92.95 | 96.57 | no clear gain, about 4x time |
| H7 | H5, lr 0.01, 300 epochs | 91.55 | 96.47 | revert |
| H8 | H5, label smoothing 0.05 | 92.55 | 96.63 | no clear gain |
| H9 | H5, class balanced | 92.20 | 96.67 | revert |
| H13 | H12, L2 1e-5 | 92.75 | 96.77 | no clear gain |
| H13 | H12, L2 1e-4 | 89.51 | 95.77 | revert |
| H10 | L-BFGS, L2 1e-3 | 76.61 | 92.80 | revert |
| H10 | L-BFGS, L2 1e-4 | 90.11 | 95.67 | revert |
| H10 | L-BFGS, L2 1e-5 | 92.75 | 96.67 | near H11, 165 s and 420 s |
| H11 | H5, centroid and label-name start | 92.85 | 96.90 | near miss, see below |
| H14 | H11, today's SGD under 8 per label | same | same | **keep** |

Few-shot on dev, mean of 3 seeds:

| Per label | Dataset | H0 | H5 | H11 |
| --- | --- | --- | --- | --- |
| 5 | BANKING77 | 81.64 | 81.86 | 82.63 |
| 5 | CLINC150 | 91.36 | 90.12 | 91.07 |
| 10 | BANKING77 | 83.91 | 86.01 | 86.31 |
| 10 | CLINC150 | 92.96 | 92.88 | 93.51 |

H11 fell 0.29 on CLINC150 at 5 per label, over the 0.2 limit, so H14 keeps SGD there. Its dev results at 5 per label
matched H0 exactly. The cutoff of 8 was never tested on its own: only 5 and 10 were measured.

The gaps between H6, H8, H11, H12 and H13 are small, at most 12 dev answers, too few to separate on one seed.

## Test and HWU64: one look per candidate

Each configuration was scored once. H14 was fixed before its first look and nothing changed afterwards.

| Dataset | Per label | Baseline | H14 | Change | Baseline ECE | H14 ECE |
| --- | --- | --- | --- | --- | --- | --- |
| BANKING77 | all | 90.62 | 93.12 | +2.50 | 0.013 | 0.013 |
| BANKING77 | 10 | 84.43 | 87.52 | +3.09 | 0.019 | 0.015 |
| BANKING77 | 5 | 83.53 | 83.53 | 0 | 0.049 | 0.049 |
| CLINC150 | all | 94.62 | 96.13 | +1.51 | 0.009 | 0.006 |
| CLINC150 | 10 | 92.16 | 92.50 | +0.34 | 0.010 | 0.014 |
| CLINC150 | 5 | 90.37 | 90.37 | 0 | 0.020 | 0.020 |
| HWU64 | all | 90.15 | 91.17 | +1.02 | 0.018 | 0.019 |
| HWU64 | 10 | 85.50 | 85.38 | -0.12 | 0.039 | 0.026 |
| HWU64 | 5 | 82.59 | 82.59 | 0 | 0.043 | 0.043 |

Few-shot rows are the mean of seeds 7, 8 and 9. All-data rows are seed 7.

Paired test on all data (answers only one head got right; McNemar chi-squared):

| Dataset | Baseline only | H14 only | Chi-squared |
| --- | --- | --- | --- |
| BANKING77 | 41 | 118 | 36.3 |
| CLINC150 | 39 | 107 | 30.7 |
| HWU64 | 18 | 29 | 2.1, not significant at this size |

## Pass mark

| Condition | Result | Status |
| --- | --- | --- |
| Mean gain at all data at least 0.5 | +2.00 | pass |
| No loss above 0.2 on either | +2.50, +1.51 | pass |
| HWU64 doesn't fall | all data +1.02, 5 per label 0, 10 per label -0.12 | **pass at all data; falls at 10** |
| 5 and 10 per label within 0.2 | 0, +3.09, 0, +0.34 | pass |
| ECE at most 0.03 at 10+ per label | BANKING77 and CLINC150 pass; HWU64 10 per label mean 0.026, worst seed 0.034 (baseline 0.039, 0.048) | **one HWU64 seed over** |
| Zero-shot unchanged | code path not touched; not rerun | pass by construction |
| Runs in browser worker | typed-array JavaScript, no new dependency; not run in a worker | **not measured** |
| Training time, Node, loaded machine | BANKING77 123 s (84 s), CLINC150 332 s (167 s) | stated |
| Head size | unchanged shape: 118,580 bytes for 77 labels, 231,000 for 150 (raw Float32) | stated |

The pass mark doesn't say whether "HWU64 doesn't fall" covers 10 per label, so that call is left to the maintainer. The
-0.12 is 4 answers across 3 seeds of 1,076.

## Step 3: contrastive adapter (not run)

Skipped. H14 already gives four times the required mean gain, and the adapter would add about 590 KB per head
(NOTES-accuracy.md estimate). The one place it could still help is 5 per label, which H14 leaves unchanged. Its gain
there is unknown.

## Step 4: fine-tune (trained once, reverted)

Environment: Python 3.12 and torch cu128, CUDA available on the RTX 5060.

| Dataset | Licence evidence | Decision |
| --- | --- | --- |
| SNIPS 2017 custom-intent-engines | sonos/nlu-benchmark LICENSE, CC0 1.0 | used: 13,576 texts, 7 intents, 3 dropped as benchmark matches |
| PAWS-Wiki labelled final | google-research-datasets/paws LICENSE: "may be freely used for any purpose" | used: positive pairs. Sentences come from Wikipedia, so get a licence opinion before redistributing |
| SLURP / MASSIVE en | alexa/massive NOTICE.md, CC BY 4.0 | **rejected**: 5,133 of 13,514 texts are verbatim in HWU64, 56 of 60 labels shared |
| NLU-Evaluation-Data | CC BY 4.0 | rejected: source of HWU64 |
| Others (TOPv2, SGD, Taskmaster, Quora, ATIS) | not verified | not used |

The PAWS Google bucket returned 403, so the data came from the Hugging Face parquet copy.

Training: BAAI/bge-small-en-v1.5, MultipleNegativesRanking, 2 epochs, lr 2e-5, 41,119 pairs. Batch 64 failed with CUDA
out of memory, reporting 5.59 GiB free (cause unknown; the machine was busy). Batch 32 succeeded on the first retry in
271 s, peak 1.46 GB. Held-out SNIPS accuracy@1 went from 0.949 to 0.985.

Dev results with H14 heads, exported to ONNX and quantised to q8 (33.8 MB). The control is unmodified bge through the
same export.

| Dataset | Per label | Control | Fine-tuned |
| --- | --- | --- | --- |
| BANKING77 | all | 92.45 | 89.91 |
| BANKING77 | 10 | 85.92 | 81.48 |
| CLINC150 | all | 96.63 | 95.40 |
| CLINC150 | 10 | 93.22 | 90.20 |

Reverted on dev, so no test or HWU64 look was spent on it. Measured for this one data mix and these settings only.
Inferred, not tested: a 7-intent set with Wikipedia paraphrases pulls the encoder away from fine distinctions.

## Recommendation

Ship H14 in `src/head.ts`, after a worker run. Credit jimothy. Keep decide-small as it is.

## Unknown

- Worker behaviour: H14 as written has no `yieldToPage` between epochs, and it needs one. Browser training time is unmeasured.
- `train()` end to end: the suite and `eval/hwu64.ts` were not rerun with the change, and neither was out-of-scope rejection.
- `onEpoch`, `calibrateOn` and the teacher replay paths with H14.
- Behaviour between 5 and 10 per label, and with unequal label counts around the cutoff.
- Seed variance at all data: one seed only.
- Baseline test numbers differ a little from the quoted ones (90.62 vs 90.7, 94.62 vs 94.4). Probably code changes since
  31 October; not checked.

