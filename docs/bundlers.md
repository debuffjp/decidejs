# Bundlers

`decidekit/browser` starts its worker with `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`,
the pattern Vite, webpack 5 and esbuild-based tools recognise.

## Vite

Tested with Vite 7.1: built, served cross-origin isolated, run in headless Chrome 156 (zero-shot and training).

```js
// vite.config.js
export default {
  worker: { format: "es" },
  optimizeDeps: { exclude: ["decidekit"] },
  resolve: {
    // Halves the wasm download (23.6 MB to 12.9 MB; 5.8 MB to 3.3 MB gzipped). decidekit runs on wasm by default,
    // so it doesn't need the WebGPU build. Leave this out if you pass device: "webgpu".
    alias: { "onnxruntime-web/webgpu": "onnxruntime-web/wasm" },
  },
};
```

Add the cross-origin isolation headers to the dev server and your host:

```js
server: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "credentialless" } },
```

## webpack 5

Tested with webpack 5.112 from the packed package, served cross-origin isolated, run in headless Chrome (zero-shot
and training). It works with no config: webpack splits the worker into its own chunk. The alias brings the wasm
from 22.5 MiB to 12.3 MiB:

```js
// webpack.config.js
export default {
  experiments: { topLevelAwait: true }, // only if your own code awaits at the top level
  resolve: { alias: { "onnxruntime-web/webgpu$": "onnxruntime-web/wasm" } },
};
```

## Next.js

Tested with Next.js 16.4 (Turbopack, App Router), `next build` then `next start`, in headless Chrome. Import
`decidekit/browser` inside a client component, and only in the browser: a dynamic `import()` inside `useEffect`
keeps it out of server rendering.

```js
// next.config.mjs
export default {
  turbopack: { resolveAlias: { "onnxruntime-web/webgpu": "onnxruntime-web/wasm" } },
  async headers() {
    return [{ source: "/:path*", headers: [
      { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
      { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
    ] }];
  },
};
```

```js
"use client";
useEffect(() => {
  import("decidekit/browser").then(async ({ createWorkerDecider }) => {
    const d = await createWorkerDecider();
    // ...
  });
}, []);
```

Next.js with webpack (`next build --webpack`) is not tested; the webpack alias above should apply.

## Without a bundler

`npm run bench` builds `bench/browser.js` and `bench/worker.js` with esbuild (same alias), which you can copy and
serve as static files.
