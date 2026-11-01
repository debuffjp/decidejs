# Accuracy push before release (2026-11-01)

Goal: make decidekit as accurate as we can before it goes public, without cheating our own evals. Every idea was
scored on dev splits (CLINC150 val, a seeded 20% of BANKING77 train) using cached embeddings, and only the winners
were checked on test. Raw lines are in [results-gains.jsonl](results-gains.jsonl) and
[results-wikirace.jsonl](results-wikirace.jsonl).

## Kept

### 1. Describe your labels (zero-shot, +7 to +10 points)

One sentence per label, written from the label name alone, in the user's voice ("My card still hasn't arrived in
the post."), as an LLM could write once at setup. Files: [descriptions-banking77.json](descriptions-banking77.json),
[descriptions-clinc150.json](descriptions-clinc150.json). Script: [descriptions.ts](descriptions.ts).

Accuracy on dev, temperature 0.05:

| Encoder | Dataset | Names | Descriptions | Better of name and description |
| --- | --- | --- | --- | --- |
| MiniLM-L6 | BANKING77 | 61.4% | 67.8% | 68.2% |
| MiniLM-L6 | CLINC150 | 69.3% | 76.3% | 79.4% |
| bge-small | BANKING77 | 66.2% | 71.8% | 72.4% |
| bge-small | CLINC150 | 72.8% | 81.3% | 83.1% |
| gte-small | BANKING77 | 62.2% | 67.7% | 69.0% |
| gte-small | CLINC150 | 69.9% | 79.3% | 81.0% |

Calibration error (MiniLM, 15 bins) stayed 0.03 to 0.05. Shipped: a `{label: description}` record already worked;
`score()` in src/index.ts now takes the better of the readable name and the description for each label.

Caveat: Claude wrote the descriptions and has seen these public datasets, so some leakage is possible. The
descriptions are generic ("How do I change my PIN?"), not copied from the data, and the gain on CLINC150 is larger
than on BANKING77, the opposite of what memorising one dataset would give. Re-check with descriptions from a
different LLM before quoting it as a headline.

### 2. bge-small-en-v1.5 as the default encoder

Few-shot test accuracy, mean of 3 seeds ([results-bge-small.json](results-bge-small.json) vs the MiniLM suite):

| Per label | BANKING77 MiniLM | BANKING77 bge | CLINC150 MiniLM | CLINC150 bge |
| --- | --- | --- | --- | --- |
| 5 | 78.7% | 82.3% | 87.3% | 89.6% |
| all | 90.6% | 90.7% | 94.0% | 94.4% |

Shipped heads on test: BANKING77 90.6% (unchanged), CLINC150 94.2% to 95.0%; out-of-scope rejected at 0.93: 89.9%.
Fair WikiRace (20 races): 17/20 median 5 hops to 19/20 median 4.

Zero-shot needed a new temperature. At 0.05, bge's similarities sit so close together that only 3% of answers
reached 0.9 confidence (ECE 0.23). At 0.03: ECE 0.028 (BANKING77) and 0.041 (CLINC150), 28% and 35% answered at
0.9, 98% and 96% of those right. `createDecider` now picks the temperature per model.

Cost: 34 MB download instead of 23.7 MB. Node 3.2 ms per text (MiniLM 2.0); Chrome, isolated, 16 threads: p50 7.7
ms, p95 13.8 ms (MiniLM 4.2 and 7.7). Still well under a keystroke. Training BANKING77 in a worker: 228 s (176 s).

### 3. WikiRace: score links against the target's title as well as its summary

Same idea as name plus description. MiniLM: 17/20 either way, median hops 5 to 3. bge: 19/20, median 4 to 3. Kept in
the eval as STRATEGY=both; switch users get the same effect by putting the target's title in the state.

## Rejected

- **Contextual calibration** (subtract each label's mean similarity over unlabelled inputs, Zhao et al. 2021):
  -0.8 to +0.2 points, ECE worse on BANKING77. Embedding cosine doesn't have the label-prior bias it fixes in LLMs.
- **kNN vote mixed into the trained head**: +1.2 to +2.2 points on BANKING77, -0.2 to -2.0 on CLINC150, ECE up to
  2x worse and fewer answers at 0.9. Not worth the complexity.
- **gte-small**: few-shot between MiniLM and bge, and zero-shot similarities so compressed that even temperature 0.02
  left ECE at 0.14 to 0.22.
- **Cross-encoder rerank (WikiRace)**: 12/20, worse than 17/20.
- **Lookahead (WikiRace)**: 19/20 median 3, but it reads pages the player hasn't visited. Not allowed by the game, so
  it's a ceiling, not a result.

## Not tried yet

- A contrastive adapter (a 384x384 layer trained on pairs, SetFit-like) before the head: +1 to 3 points few-shot
  expected, 590 KB per head.
- Descriptions as priors for trained heads (today the readable name is the prior, weight 3).
- A `describe()` helper that asks the fallback LLM to write descriptions once.
- Larger opt-in encoders (EmbeddingGemma, nomic with matryoshka) and a static-embedding tiny tier (Model2Vec).
- WikiRace: the sentence around each link (the game shows it) as the link's text.
