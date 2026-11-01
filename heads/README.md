# Ready-made heads

Heads trained on public datasets, so you can see decidekit work before collecting your own examples. Both use the
default model (`Xenova/bge-small-en-v1.5`, q8) and load with `loadHead()`. Heads shipped before this release were
trained on `Xenova/all-MiniLM-L6-v2` (BANKING77 90.6%, CLINC150 94.2%).

| Head | Labels | Trained on | Test accuracy | Size |
|---|---|---|---|---|
| [banking77.head.json](banking77.head.json) | 77 card and banking support intents | BANKING77 train (10,003) | 90.6% (temperature 0.45) | 156 KB |
| [clinc150.head.json](clinc150.head.json) | 150 assistant intents in 10 domains | CLINC150 train and val (18,000) | 95.0% (temperature 0.35) | 303 KB |

```ts
import { createDecider, loadHead } from "decidekit";

const head = loadHead(await (await fetch("/heads/banking77.head.json")).text());
const d = await createDecider();
const r = await d.decide("my card hasn't arrived yet", head, { minConfidence: 0.9 });
```

The CLINC150 head has no out-of-scope label. Use `minConfidence` for that: at 0.93, it returned `null` for 91.1% of
CLINC150's out-of-scope test queries. That threshold was picked on the validation split, which this head also
trained on, so expect a little less on your own text.

The heads aren't in the npm package, to keep installs small. Copy the file you need next to your app.

Rebuild: `npm run heads`. Check: `node --experimental-strip-types eval/heads-check.ts`.

## Licences

The heads are derived from these datasets and carry their licences:

- BANKING77: Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders", 2020. PolyAI,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- CLINC150: Larson et al., "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction", 2019.
  Clinc, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
