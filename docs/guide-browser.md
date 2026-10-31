# In the browser

## Use the worker

The encoder runs wasm that never gives the thread a turn. On the main thread, training BANKING77 froze the page for
171 s. In a worker, the page's longest stall while training was 18 ms. So in a browser, use `createWorkerDecider()`:

```ts
import { createWorkerDecider, loadHead } from "decidejs/browser";

const d = await createWorkerDecider();
const r = await d.decide(input.value, ["question", "bug_report", "praise"]);
```

It has the same `decide()`, `decideMany()` and `train()` as `createDecider()`, plus `terminate()` to stop the worker.
Each call costs about 1 ms of message passing. The full list of differences is in the
[API reference](api.md#createworkerdecideroptions-worker).

Create one decider and reuse it. Each `createWorkerDecider()` starts a new worker and loads the model again (from the
cache after the first time, which still took 3.8 s in our runs).

## Show download progress

The first visit downloads the model. `onProgress` reports each file as it comes in:

```ts
const bar = document.querySelector("progress")!;
const label = document.querySelector("#status")!;

const d = await createWorkerDecider({
  onProgress: ({ file, loaded, total }) => {
    bar.value = total ? loaded / total : 0;
    label.textContent = `downloading ${file}`;
  },
});
label.textContent = "ready";
```

It fires per file, so the bar fills once for each file. It does not fire when the files come from the cache, so show
a plain "loading" message until the promise resolves.

To show progress while training, pass `onEpoch` to `train()`: it reports the epoch and held-out accuracy as training
runs (see the [API reference](api.md#trainoptions) for when it fires).

## Cross-origin isolation

Serve the page with these two headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

They make the page cross-origin isolated, which lets onnxruntime's wasm use several threads. In headless Chrome with
BANKING77 (`bench/results.json`):

| | Median per decision | p95 per decision |
|---|---|---|
| Cross-origin isolated | 3-5 ms | 6-8 ms |
| Not isolated | | 14.3 ms |

`credentialless` lets the page keep loading cross-origin images and scripts without credentials and without each
server opting in. Check that third-party embeds you rely on (sign-in pop-ups, payment frames) still work with
`same-origin` before turning it on. Without the headers, decidejs still works, single-threaded.

Header config for Vite, webpack and Next.js is in [bundlers.md](bundlers.md).

## First-load size

| File | Size |
|---|---|
| Model, all-MiniLM-L6-v2 at q8 | 23.7 MB |
| onnxruntime wasm | 23.6 MB (5.8 MB gzipped) |
| onnxruntime wasm with the alias below | 12.9 MB (3.3 MB gzipped) |
| A head, 77 labels | 156 KB |
| A head, 150 labels | 303 KB |

With the alias, the first visit downloads about 36 MB. Everything is cached after that. bge-small is 34 MB in place
of the 23.7 MB model.

The alias swaps onnxruntime's WebGPU build for its plain wasm build, which decidejs uses by default:

```js
// vite.config.js
export default {
  resolve: { alias: { "onnxruntime-web/webgpu": "onnxruntime-web/wasm" } },
};
```

Leave it out if you pass `device: "webgpu"`. The webpack and Next.js versions are in [bundlers.md](bundlers.md).

## Caching

The model files are cached by the browser after the first download, so later visits load them locally. The cache
belongs to the page's origin and the browser may clear it, so be ready for a second download. Head files are ordinary static files: give them a content hash in the file name or a cache header as you
would any other asset.

## Bundlers

decidejs starts its worker with `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`, the pattern
Vite, webpack 5 and esbuild-based tools recognise. Tested from the packed package in Vite 7.1, webpack 5.112 and
Next.js 16.4 (Turbopack). Config for each: [bundlers.md](bundlers.md).

If your setup needs a different worker URL, create the `Worker` yourself, pointing at decidejs' `dist/worker.js`
however your bundler serves it, and pass it as the second argument: `createWorkerDecider(options, worker)`.

## Keep the encoder out of the page bundle

Import from `decidejs/browser`, not `decidejs`, in page code. `decidejs` pulls in transformers.js and the encoder; the
browser entry only holds the worker client and the head functions. A page that only loads a head came to 3.6 KB from
`decidejs/browser` against 482 KB from `decidejs`.

```ts
// good: the encoder stays in the worker's chunk
import { createWorkerDecider, loadHead } from "decidejs/browser";

// pulls the encoder into the page bundle
import { loadHead } from "decidejs";
```

Types work the same from either entry.

## Server rendering

`decidejs/browser` starts a Web Worker, so import it only in the browser. In Next.js, use a dynamic `import()` inside
`useEffect` in a client component; [bundlers.md](bundlers.md#nextjs) has the code.
