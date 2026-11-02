# decidekit docs

Start with the [main README](../README.md) for a five-line example and the headline numbers.

## Guides

- [Training a head](guide-training.md): collecting examples, how many per label, label names, retraining, shipping heads, the CLI and choosing a model.
- [Confidence and thresholds](guide-confidence.md): what calibrated means, how the temperature is fitted, picking `minConfidence` on your own data and "none of these".
- [In the browser](guide-browser.md): the worker, download progress, cross-origin isolation, first-load size and keeping the encoder out of the page bundle.
- [Falling back to an LLM](guide-fallback.md): `withFallback()`, learning from the fallback, measured savings.
- [React](guide-react.md): `useDecider()` and `useDecision()`.
- [Other languages](faq.md#which-languages-does-it-support): Decide Multilingual, measured on Japanese, Chinese, German and Spanish.
- [Coming from Jev](guide-typesafe.md): TypeSafe's client shape on a local decider, what differs and using Jev as the fallback.
- [Bundlers](bundlers.md): Vite, webpack and Next.js config.
- [switch](guide-switch.md): decidekit as a decision model in the switch harness, with trained heads and a teacher, and WikiRace results.
- [Use cases](use-cases.md): good fits, poor fits and example code.

## Reference

- [API reference](api.md): every export of `decidekit` and `decidekit/browser`, and the CLI.
- [How it works](how-it-works.md): the encoder, the head and each training step, with the numbers behind them.
- [Moving parts](moving-parts.md): every piece on one page (encoder, scoring, heads, calibration, fallback, worker, isolation, bundlers, caching, head files), where it lives and the numbers behind it.
- [FAQ](faq.md).
- [Ready-made heads](../heads/README.md): BANKING77 and CLINC150 heads to try before collecting examples.

## Write-ups

- [Introducing decidekit](https://debuff.jp/blog/introducing-decidekit): what it is and the headline numbers.
- [When the LLM is wrong](https://debuff.jp/research/learning-from-a-noisy-llm): how `withFallback` learns from an LLM that makes mistakes.
