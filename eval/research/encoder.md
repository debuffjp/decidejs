# Encoder and zero-shot scoring: ideas, ranked

Baseline: bge-small q8, mean pooling, cosine against the better of name and description, T=0.03. Dev: BANKING77 72.4%, CLINC150 83.1%. Test every idea on cached dev embeddings with `eval/zeroshot.ts` / `eval/gains.ts`, log it to `results-gains.jsonl`, re-check ECE with `eval/zeroshot-temp.ts`, and only check winners on test.

1. **Several descriptions per label, scored by max or log-sum-exp.** Expected gain: +2 to 4 points. The step from one description to name+description already gave +0.6 to 1.8. Cost: N times the label encodes, done once at setup, and no runtime cost. Test: extend `descriptions.ts` to 3 to 5 phrasings per label, then compare max, mean-of-vectors (a centroid) and LSE. Evidence: the repo's own name+description gain; prototype averaging in CLIP prompt ensembling (https://arxiv.org/abs/2103.00020).

2. **HyDE-style synthetic examples feeding the head.** Have an LLM write about 5 example user messages per label and train the existing few-shot head on them. The 5-shot results are 82.3 / 89.6 against 72.4 / 83.1 zero-shot, so this is the biggest possible gain (+5 to 8) if synthetic examples get close to real ones. Cost: one LLM call at setup and a seconds-long training run. Test: `suite.ts` with synthetic examples in place of train, scored on dev. Use a different LLM to reduce leakage. Evidence: HyDE (https://arxiv.org/abs/2212.10496).

3. **Query-side instruction prefix.** bge v1.5 recommends "Represent this sentence for searching relevant passages: " for short queries (https://huggingface.co/BAAI/bge-small-en-v1.5). Try it on the text side only, then on both sides. Expected gain: 0 to 2 points; it may hurt, because these are symmetric short texts. Cost: none. Re-tune T afterwards. Test: a flag in `zeroshot.ts`.

4. **Newer small encoders.** Candidates: granite-embedding-small-english-r2 (47M parameters, ModernBERT, 8k context; https://huggingface.co/ibm-granite/granite-embedding-small-english-r2), snowflake-arctic-embed-xs (22M; https://huggingface.co/Snowflake/snowflake-arctic-embed-xs), and EmbeddingGemma-300m (multilingual, Matryoshka, ships ONNX for transformers.js; https://huggingface.co/blog/embeddinggemma). Expected gain: 1 to 4 points from EmbeddingGemma, at about 5 times the download (more than 150 MB at q8 or q4). Cost: an ONNX export where none exists, plus a temperature sweep per model. Test: run the bge-small comparison again (`suite.ts`, `zeroshot-temp.ts`) and record download size and latency in `latency.ts`.

5. **Multilingual tier.** The options are EmbeddingGemma or multilingual-e5-small (118M; https://huggingface.co/intfloat/multilingual-e5-small, needs the "query: " prefix). This adds a capability rather than accuracy. Test: MASSIVE or the multilingual Amazon intents with English descriptions.

6. **Negative descriptions / contrastive margin.** Score each label as sim(pos) - λ·sim(neg), where neg is "not this: …" text written for confusable pairs. Expected gain: about 1 point, concentrated on BANKING77's near-duplicate labels. Cost: authoring the negatives. Test: sweep λ on dev and look at the confusion pairs. Evidence is weak; treat this as speculative.

7. **Hierarchical labels.** Pick the domain first (CLINC150 has 10), then the intent within it, or add a domain-description similarity to the score. Expected gain: 0 to 2 points on CLINC; the domain text also helps out-of-scope rejection. Test: CLINC's domain map in `zeroshot.ts`.

8. **Static-embedding fast tier (model2vec potion).** Under 10 MB, about 0.1 ms per text, roughly 90% of MiniLM's MTEB score (https://github.com/MinishLab/model2vec, https://huggingface.co/blog/static-embeddings). Use it as an instant first pass or a low-end-device fallback; it loses accuracy. Test: `latency.ts` and `suite.ts` with a potion tokenizer-lookup path.

9. **Pooling.** bge was trained with CLS pooling (see the model card), but decidejs uses mean pooling. Try `pooling: "cls"`. Expected gain: 0 to 1.5 points. Cost: one line. This is the cheapest test on the list, so run it first.

10. **dtype and device.** fp16 against q8: expect about 0.0 to 0.5 points from fp16, at twice the download. int4 (q4) weights halve the size, but small BERT models lose more under it, so measure the drop. WebGPU via transformers.js v3 (https://huggingface.co/blog/transformersjs-v3) helps batch encoding of labels and examples while training, and probably not single-text latency at about 8 ms. Test: `latency.ts` per dtype/device, with accuracy from `suite.ts`.

Suggested order: 9 and 3 (free), then 1, then 2, then 4.
