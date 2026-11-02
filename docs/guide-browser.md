# In the browser

## Use the worker

The encoder runs wasm that never gives the thread a turn. On the main thread, training BANKING77 froze the page for
171 s. In a worker, the page's longest stall while training was 18 ms. So in a browser, use `createWorkerDecider()`:

```ts
import { createWorkerDecider, loadHead } from "decidekit/browser";

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
BANKING77 (16 threads, 2026-11-01):

| | Median per decision | p95 per decision |
|---|---|---|
| Cross-origin isolated, bge-small (default) | 7.7 ms | 13.8 ms |
| Cross-origin isolated, all-MiniLM-L6-v2 | 4.2 ms | 7.7 ms |
| Not isolated, all-MiniLM-L6-v2 (2026-10-31 run) | | 14.3 ms |

Not isolated has only been measured with all-MiniLM-L6-v2, in a run where isolation took p95 from 14.3 ms to 7.8 ms. Training BANKING77 in a worker took 228 s with bge-small,
and the page never stalled for more than 18 ms.

`credentialless` lets the page keep loading cross-origin images and scripts without credentials and without each
server opting in. Check that third-party embeds you rely on (sign-in pop-ups, payment frames) still work with
`same-origin` before turning it on. Without the headers, decidekit still works, single-threaded.

Header config for Vite, webpack and Next.js is in [bundlers.md](bundlers.md).

## First-load size

| File | Size |
|---|---|
| Model, Decide Small at q8 (default) | 34 MB |
| Model, all-MiniLM-L6-v2 at q8 (pre-release default) | 23.7 MB |
| onnxruntime wasm | 23.6 MB (5.8 MB gzipped) |
| onnxruntime wasm with the alias below | 12.9 MB (3.3 MB gzipped) |
| A head, 77 labels | 156 KB |
| A head, 150 labels | 303 KB |

With the alias, the first visit downloads the 34 MB model plus 12.9 MB of wasm (3.3 MB gzipped). Everything is cached
after that. With all-MiniLM-L6-v2 the first visit was about 36 MB.

The alias swaps onnxruntime's WebGPU build for its plain wasm build, which decidekit uses by default:

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

## Host the model files yourself

By default the files come from Hugging Face on first use and are then cached. To serve them from your own host, set
`modelHost` (since 0.1.1):

```ts
const d = await createWorkerDecider({ modelHost: "https://cdn.example.com/models" });
```

Files are read from `<modelHost>/<model>/`, with the tokenizer and config files alongside the `onnx` folder:

```
https://cdn.example.com/models/debuffjp/decide-small/config.json
https://cdn.example.com/models/debuffjp/decide-small/tokenizer.json
https://cdn.example.com/models/debuffjp/decide-small/tokenizer_config.json
https://cdn.example.com/models/debuffjp/decide-small/onnx/model_quantized.onnx
```

Copy them from [debuffjp/decide-small](https://huggingface.co/debuffjp/decide-small) or
[debuffjp/decide-multilingual](https://huggingface.co/debuffjp/decide-multilingual). The revision pin doesn't apply here:
your host serves whichever files you copied.

`modelHost` applies to the whole page or process: it sets transformers.js' host, so later deciders in the same thread
use it too, even without passing it. A worker decider sets it inside its worker. If the host is on another origin, it
needs to allow CORS requests from your page.

## Bundlers

decidekit starts its worker with `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`, the pattern
Vite, webpack 5 and esbuild-based tools recognise. Tested from the packed package in Vite 7.1, webpack 5.112 and
Next.js 16.4 (Turbopack). Config for each: [bundlers.md](bundlers.md).

If your setup needs a different worker URL, create the `Worker` yourself, pointing at decidekit's `dist/worker.js`
however your bundler serves it, and pass it as the second argument: `createWorkerDecider(options, worker)`.

## Keep the encoder out of the page bundle

Import from `decidekit/browser`, not `decidekit`, in page code. `decidekit` pulls in transformers.js and the encoder; the
browser entry only holds the worker client and the head functions. A page that only loads a head came to 3.6 KB from
`decidekit/browser` against 482 KB from `decidekit`.

```ts
// good: the encoder stays in the worker's chunk
import { createWorkerDecider, loadHead } from "decidekit/browser";

// pulls the encoder into the page bundle
import { loadHead } from "decidekit";
```

Types work the same from either entry.

## Server rendering

`decidekit/browser` starts a Web Worker, so import it only in the browser. In Next.js, use a dynamic `import()` inside
`useEffect` in a client component; [bundlers.md](bundlers.md#nextjs) has the code.
