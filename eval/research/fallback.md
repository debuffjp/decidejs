# Fallback and learning loop: ranked ideas

Context: `withFallback()` learns every answer below 0.9 with full trust. eval/fallback.ts uses gold labels as the teacher, but Llama 3.1 8B agrees with BANKING77 gold only 80.5% of the time, so about 1 in 5 learned examples could be wrong. Ideas ordered by expected gain per unit of effort.

1. **Noisy-teacher simulation in eval/fallback.ts.** Gain: makes every other result honest. Cost: about 20 lines. Test: `NOISE=0.195` swaps the gold label for the head's top-2 or a random label (LLM errors cluster on near-miss intents, so prefer confusable labels). Report head accuracy per round against the clean run. Evidence: BANKING77 is built from fine-grained, confusable intents ([Casanueva et al. 2020](https://arxiv.org/abs/2003.04807)).

2. **Agreement gating (learn only when teacher and head top-k agree, or the teacher is self-consistent).** Gain: most label noise is likely filtered, with fewer examples learned. Cost: low; an `accept(text, answer, local)` hook, or default to keep only answers in the local top-5. Test: the noisy run from idea 1, measuring learned-label precision and head accuracy. Evidence: confident learning prunes labels that disagree with model probabilities ([Northcutt et al.](https://arxiv.org/abs/1911.00068)); self-consistency voting ([Wang et al.](https://arxiv.org/abs/2203.11171)).

3. **Teacher-confidence weights.** Gain: soft trust without dropping data. Cost: medium, since `Example` needs `weight` and `fitHead` needs a weighted loss. `ask` may return `{label, confidence}` from logprobs; weight = p_teacher, or the teacher's distribution as a soft target. Test: the idea 1 harness with a simulated calibrated teacher. Evidence: learning from soft LLM labels ([Wang et al. 2021](https://arxiv.org/abs/2108.13487)); noise-tolerant losses ([Zhang and Sabuncu, GCE](https://arxiv.org/abs/1805.07836)).

4. **Cascade cost model in place of a fixed 0.9.** Gain: about 30-50% fewer calls at equal accuracy is plausible. Cost: low. Pick the threshold from `cost_per_call` and `cost_of_error` with a held-out calibration set, and re-pick after each retrain (the head sharpens, so the right threshold moves). Test: sweep thresholds per round in eval/fallback.ts and plot calls against error. Evidence: [FrugalGPT](https://arxiv.org/abs/2305.05176), [RouteLLM](https://arxiv.org/abs/2406.18665), [cascade deferral analysis](https://arxiv.org/abs/2404.10136).

5. **Learn from a margin or entropy band; skip near-duplicates.** Gain: fewer redundant examples per retrain. Cost: low. Uncertainty sampling plus a cosine-dedup check against learned texts, and BADGE-style diversity when batching. Test: compare the examples count needed to reach 94% in the table in docs/guide-fallback.md. Evidence: [Settles survey](https://burrsettles.com/pub/settles.activelearning.pdf), [BADGE](https://arxiv.org/abs/1906.03671).

6. **Incremental head updates.** Gain: retrain latency drops from a full fit to milliseconds, so learning applies immediately. Cost: medium. Warm-start `fitHead` from current weights for a few epochs, or update class prototypes online, and keep the full retrain every N. Test: eval/retrain.ts timing plus accuracy parity with full retrain.

7. **Batch LLM calls.** Gain: one prompt classifies 10-20 texts and cuts per-call overhead and token cost (label list sent once). Cost: medium (async queue, partial failure). Test: offline only; measure agreement against gold for batched against single prompts on a BANKING77 sample. Evidence: FrugalGPT names query concatenation as a lever.

8. **Ask for a description or contrastive reason, not only a label.** Gain: one LLM answer improves the label for all future texts. Cost: medium. When the head confuses A and B, ask "how does A differ from B" and append to the label descriptions (eval/descriptions.ts already exists). Test: hill harness with descriptions updated from top confusion pairs. Evidence: rationales as extra supervision ([Distilling step-by-step](https://arxiv.org/abs/2305.02301)).

9. **Drift detection.** Gain: catches new intents and shifted traffic. Cost: low. Track the rolling fallback rate and null-answer rate with ADWIN ([Bifet and Gavalda 2007](https://doi.org/10.1137/1.9781611972771.42)); emit an event on a rise. Test: inject CLINC150 out-of-scope queries mid-stream in eval/fallback.ts.

10. **Audit sampling of local answers.** Gain: estimates real local error and catches confident mistakes the loop never sees. Cost: low. Send about 1% of confident answers to `ask` as well. Test: compare the estimate to measured "Local answers right" in the fallback run.

Do first: 1, 2, 4. Without 1 none of the gains above is measured against a realistic teacher.
