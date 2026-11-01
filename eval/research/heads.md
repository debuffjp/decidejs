# Heads, training and calibration: ideas ranked

Scope: `src/head.ts` (softmax over frozen embeddings, centroid init x20, label-name priors, 5-fold temperature, median-epoch retrain). Baseline to beat: bge-small 5-shot 82.3% BANKING77 / 89.6% CLINC150, all-shot 90.7 / 94.4, training 228 s in a worker. Test every idea on the dev splits in `eval/gains.ts` first, test only for winners (per NOTES-accuracy.md).

1. **Closed-form or L-BFGS logistic regression instead of 60 SGD epochs.** Gain: training 228 s to a few seconds; accuracy equal or +0.5 (convex optimum, no epoch selection noise). Cost: small (L-BFGS on a 384xC problem is ~200 lines, no deps). Keep centroid+prior init as a warm start and as an L2 pull target (regularise toward the prior weights, not zero). Test: `eval/suite.ts` accuracy and timing; `eval/retrain.ts` for wall time. Evidence: linear probes on frozen features are usually fit with L-BFGS (CLIP, https://arxiv.org/abs/2103.00020); LP++ shows a well-initialised linear probe beats adapters few-shot (https://arxiv.org/abs/2404.02117).

2. **Conformal abstention with a coverage guarantee.** Gain: replace "confidence >= 0.93" with a threshold that guarantees e.g. 95% of answered items are covered, computed from the 5-fold held-out scores already collected. Cost: tiny (sort held-out scores, take a quantile). Test: `eval/fallback.ts`, compare LLM-call rate at equal error. Evidence: https://arxiv.org/abs/2107.07511; APS/RAPS sets https://arxiv.org/abs/2009.14193.

3. **Mahalanobis or energy OOD score for out-of-scope.** Gain: better CLINC150 oos rejection (now 89.9% at 0.93) at no accuracy cost. Cost: small: shared covariance (shrinkage) from training embeddings, or energy = logsumexp of logits (zero cost). Test: CLINC150 oos split in `eval/heads-check.ts`, AUROC and recall at fixed in-scope accuracy. Evidence: https://arxiv.org/abs/1807.03888, https://arxiv.org/abs/2010.03759.

4. **Shrunk-covariance NCM / LDA head for 1 to 5 shots.** Gain: +1 to +3 at 1 to 5 shots, where SGD overfits; closed form, milliseconds. Cost: small. Pick between LDA and logistic by fold score. Test: `eval/tiny.ts` and 5-shot suite. Evidence: prototypical nets https://arxiv.org/abs/1703.05175; Tip-Adapter's cache+prior blend https://arxiv.org/abs/2111.03930.

5. **Vector scaling (per-class bias plus temperature) or Dirichlet calibration.** Gain: ECE down where classes are imbalanced; matrix scaling overfits under ~50 per class, so try vector only. Cost: tiny, reuses fold logits. Test: `eval/calib5.ts` ECE/NLL. Evidence: https://arxiv.org/abs/1706.04599, https://arxiv.org/abs/1910.12656.

6. **Active learning: send the LLM the texts with smallest margin or lowest conformal p-value, deduplicated by embedding clusters.** Gain: same accuracy with fewer LLM labels. Cost: medium (needs a loop in `eval/fallback.ts` that simulates a budget). Test: accuracy vs labels curve. Evidence: survey https://arxiv.org/abs/2009.00236; cold-start clustering https://arxiv.org/abs/2010.09535.

7. **SetFit-style contrastive tuning.** Gain: +2 to +5 at 8 shots in the paper, but needs fine-tuning the encoder, which breaks "frozen, shared embeddings" and is slow in a browser. Cost: high. Test only as a ceiling in Node. Evidence: https://arxiv.org/abs/2209.11055.

8. **WebGPU matmul for training.** Gain: maybe 5 to 20x on the SGD loop, moot if idea 1 lands. Cost: medium plus fallback path. Do last.

Order of work: 1, then 3 and 2 together (same held-out data), then 4, 5. Log choices in decisions.
