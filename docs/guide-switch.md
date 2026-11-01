# decidejs in switch

[switch](https://github.com/osolmaz/switch) is a harness for decision models: code works out the options at each
step of a task, and the model picks one. `decidejs/switch` turns a decidejs decider into a switch decider, so a switch
domain can run on a 23 MB model on your machine, free, with no API key.

```ts
import { createDecider } from "decidejs";
import { switchDecider } from "decidejs/switch";
import { run } from "@osolmaz/switch";

const decider = switchDecider(await createDecider());
const result = await run(domain, machine, decider);
```

## What it answers

| switch question | decidejs decision |
|---|---|
| `choice` | One of the criteria, compared by their text |
| `bool` | `true` or `false`, compared by what each means |
| `score` | One of the levels, compared by their descriptions |

Images are ignored. Every answer is local, so usage reports a cost of 0.

## Choose what it reads

The encoder reads about 256 tokens and drops the rest. switch domains often give long instructions and a large state,
which can push out the part that matters. Pass `text` to pick it:

```ts
// WikiRace: compare the links with what the target is about.
switchDecider(local, { text: (state) => String(state.about_the_target) });
```

Without `text`, decidejs reads the question's instructions, then the state as `key: value` lines.

## Trained heads and a teacher

- `heads`: a trained head per question id, for questions that repeat (a rubric criterion, a fixed menu). Questions
  without a head are decided zero-shot.
- `teacher`: another switch decider, such as Jev through switch's pi-ai decider. When decidejs is less sure than
  `minConfidence` (default 0.9), the teacher answers the same request, and its answers train a head for that
  question every 8 answers. A question that repeats asks the teacher less and less. This is
  [withFallback](guide-fallback.md) per question.

## WikiRace

[eval/wikirace.ts](../eval/wikirace.ts) runs the three races from switch's WikiRace example with decidejs zero-shot,
reading only the target's first sentences. Each step offers every link of the current article (Jev takes at most 255);
a link to the target itself is taken directly. Results on 2026-11-01, one run each (Wikipedia changes, so runs differ):

| Race | decidejs | Jev, from switch's README |
|---|---|---|
| Rubber duck to Moon | 2 hops (NASA) | 3 hops |
| Baseball to Sun | 7 hops | gave up after 4 |
| DNA to Manipuri pony | lost after 12 hops (reached "History of the horse in the Indian subcontinent") | 6 hops |

Through switch's own WikiRace domain and `run()` loop, which shows at most 200 links a step and opens sections
for the rest, the same decider (reading the target's summary) reached Moon in 12 hops and Sun in 18, and hit the
40-step limit on DNA to Manipuri pony after reaching Manipur. Its paths wander: it revisits articles and follows
lexical look-alikes. That is the work in the next section.

About 4 s of decisions for all three, on a laptop CPU. Zero-shot decidejs picks the link closest in meaning to the
target, so it walks into near-misses such as "Pony (disambiguation)"; reading the summary instead of the title helped
(with the title it lost two races). It has no plan, which is where a teacher or a trained head comes in.

## Where it falls short

Zero-shot decidejs ranks options by closeness in meaning to one text. It has no plan and no memory of where it has
been, so in a search it takes near-misses and loops. Trained heads fix this for questions that repeat; for searches,
the next steps are a reranker that reads the target and the option together, looking one step ahead (decidejs is
local, so peeking at each candidate's links is cheap) and a teacher. Reports and ideas are welcome in
[Discussions](https://github.com/debuffjp/decidejs/discussions).
