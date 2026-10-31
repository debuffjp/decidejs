# React

`decidejs/react` has two hooks over the worker decider. React is an optional peer dependency (18 or later): nothing
extra is installed for apps that don't use it. Tested with React 19 in Next.js 16 (`next dev` and `next build`).

```tsx
"use client";
import { useState } from "react";
import { useDecider, useDecision } from "decidejs/react";

const LABELS = ["question", "bug_report", "praise"] as const;

export function Triage() {
  const { decider, status, progress } = useDecider();
  const [text, setText] = useState("");
  const { decision, pending } = useDecision(decider, text, LABELS, { minConfidence: 0.8 });

  if (status === "loading") return <p>Loading model {progress ? `${Math.round((100 * progress.loaded) / progress.total)}%` : ""}</p>;
  return (
    <>
      <textarea value={text} onChange={(e) => setText(e.target.value)} />
      <p>{pending ? "..." : decision?.choice ?? "none of these"}</p>
    </>
  );
}
```

## useDecider(options?)

Starts a worker decider on mount and terminates it on unmount. Takes the same options as `createWorkerDecider()`
except `onProgress`, which the hook handles.

Returns `{ decider, status, progress, error }`:

- `decider`: the worker decider once ready, `null` before. Use it directly for `train()` or `decideMany()`.
- `status`: `"loading"`, `"ready"` or `"error"`.
- `progress`: `{ loaded, total }` bytes across the model files while they download (first visit only), else `null`.
- `error`: the load error when `status` is `"error"`.

Each call starts its own worker and loads its own copy of the model, so call it once near the top of the tree and
pass `decider` down (props or context). Options are read on mount: to switch model, remount with a `key`.

## useDecision(decider, text, using, options?)

Decides `text` with `using` (labels, described labels or a head) whenever either changes, after `debounceMs` of
quiet (default 150 ms). Answers for text that has changed since are dropped, so fast typing never shows a stale
label.

Returns `{ decision, pending, error }`. `decision` is `null` before the first answer, while `decider` or `using` is
`null`, and for empty text. With `minConfidence`, `decision.choice` is typed `L | null`.

Keep `using` stable: a constant outside the component, state, or `useMemo`. A new array on every render decides
again on every render.

## Next.js

Use the hooks in a client component (`"use client"`). Next.js still renders client components on the server once; the
hooks only start the worker in an effect, which never runs there. Add the cross-origin isolation headers and the wasm
alias from [bundlers.md](bundlers.md#nextjs).
