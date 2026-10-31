# Contributing

Issues, results and pull requests are all welcome. Chat happens in the [debuff Discord](https://discord.gg/jshbDHSpCY).

## Setup

```sh
npm ci
npm run typecheck
npm run build
```

`npm run eval` trains on BANKING77 in Node and prints accuracy. `npm run eval:suite` runs the few-shot and
out-of-scope tests. `npm run bench`, then `npm run bench:run` in a second terminal, measures the browser
build in headless Chrome. Use headless Chrome for timings, not a background tab: background tabs throttle timers.

## Numbers

Every accuracy or speed claim in the README comes from a run you can repeat. Say the dataset, model, machine,
browser and number of runs. A pull request that changes the numbers includes the new `eval/results.json`.

## Most wanted

- Results on your own data: dataset, labels, examples per label, accuracy. Open a discussion, even when the numbers
  are bad.
- Multilingual encoders: which one, how big, how accurate.
- Bundler recipes for the worker (Vite, webpack, Next.js).
- A smaller runtime than transformers.js.

## Style

Small pull requests. TypeScript strict. No new runtime dependencies without saying why in the pull request.

Prose in docs, comments and commit messages follows the rules in `scripts/lint-prose.mjs` (no em dashes, no
filler words or phrases). `npm run lint:prose` checks them, and CI runs it.
