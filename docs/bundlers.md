# Bundlers

`decidejs/browser` starts its worker with `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`,
the pattern Vite, webpack 5 and esbuild-based tools recognise.

## Vite

Tested with Vite 7.1: built, served cross-origin isolated, run in headless Chrome 156 (zero-shot and training).

```js
// vite.config.js
export default {
  worker: { format: "es" },
  optimizeDeps: { exclude: ["decidejs"] },
  resolve: {
    // Halves the wasm download (23.6 MB to 12.9 MB; 5.8 MB to 3.3 MB gzipped). decidejs runs on wasm by default,
    // so it doesn't need the WebGPU build. Leave this out if you pass device: "webgpu".
    alias: { "onnxruntime-web/webgpu": "onnxruntime-web/wasm" },
  },
};
```

Add the cross-origin isolation headers to the dev server and your host:

```js
server: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "credentialless" } },
```

## webpack 5 and Next.js

Not tested yet. The same alias goes in `resolve.alias`. Reports welcome in
[Discussions](https://github.com/debuffjp/decidejs/discussions).

## Without a bundler

`npm run bench` builds `bench/browser.js` and `bench/worker.js` with esbuild (same alias), which you can copy and
serve as static files.
