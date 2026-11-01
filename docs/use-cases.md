# Use cases

decidejs answers one question about a piece of text: which of these labels fits? It runs where the text already
is (the user's tab, your Node process), so each answer is free, private and a few milliseconds. Use it when a
fixed set of labels covers the question and you can collect a few dozen to a few hundred examples per label.

## Good fits

**Route support messages.** "Billing", "bug", "account", "other". Train on your past tickets, route in the form
before it's sent, and show the right help article while the user is still typing.

```ts
const head = loadHead(await (await fetch("/support.head.json")).text());
const { choice, confidence } = await d.decide(message, head);
if (confidence > 0.8) showArticleFor(choice);
```

**Triage feedback and reviews.** "Praise", "bug report", "feature request", "complaint". Sort thousands of app
reviews or survey answers in Node in a few seconds, with no API bill.

**Command palettes and search intent.** Map "make it darker" to `theme.dark` and "where's my invoice" to
`billing.invoices`. Zero-shot works for a first try; a head trained on real queries is much better.

**Moderation hints before posting.** Flag "this reads as a personal attack" while someone types. The text never
leaves their device. Use it as a nudge, not as the only moderation. Measured on a public toxicity set with the default
encoder, a head on 32 examples per label was right 66.6% of the time and caught 68 to 86% of toxic comments while
flagging about a third of clean ones (see the [FAQ](faq.md#what-kinds-of-decisions-is-it-good-at)): fine for a gentle
"are you sure?", not for blocking.

**Label a dataset in the browser.** Load a CSV in a tab, label 20 rows per class, train, review the low-confidence
rows first. Nothing uploads.

**Cheap first pass before an LLM.** Decide the easy 80% locally and send only low-confidence cases to a large model.
`confidence` from a trained head is calibrated, so a threshold means what it says.

**Let an LLM teach it.** Start from label names alone and wrap your LLM call in `withFallback`: unsure messages go to
the LLM, its answers become training examples, and the share it sees keeps falling. On CLINC150 (150 intents) the LLM
saw 67% of messages at first and 14.5% after ten retrains, with the local head at 93.3%. Watch it happen in the
[two-lanes demo](https://debuff.jp/decidejs/lanes/); the [fallback guide](guide-fallback.md) has the code.

**Games.** Read typed player chat or commands ("attack the goblin", "open the door") with no server, offline, at a
few milliseconds a message. Label names alone are enough to prototype a command set.

**Browser extensions and offline apps.** Sort tabs, notes, emails or field reports where a server call is slow,
unavailable or unwelcome. The model downloads once and is cached.

**Sensitive text.** Health, finance, HR and internal tools, where policy says the text can't go to a third-party API.
Training and deciding both run on the user's device.

**Agent and app plumbing.** "Is this message a question, a command or chit-chat?" "Does this log line look like an
error?" Small decisions an app makes thousands of times.

## Poor fits

- **Open-ended questions.** If the answer isn't one of a fixed set of labels, use an LLM.
- **Long documents.** The encoder reads 128 to 256 tokens (about a paragraph) and ignores the rest.
- **Reasoning over facts.** "Is this refund allowed under our policy?" needs the policy, which a sentence encoder
  can't read alongside the text.
- **Languages other than English** with the default model. Pass `model: "Xenova/multilingual-e5-small"` instead:
  76% on Japanese and Chinese intents against 55% and 40% ([FAQ](faq.md#which-languages-does-it-support)).
- **Tone: sentiment, emotion, toxicity.** The default encoder reads what a text is about better than how it feels:
  61% on tweet sentiment, 49% on emotion with 32 examples per label ([FAQ](faq.md#what-kinds-of-decisions-is-it-good-at)).
- **Labels you can't give examples for**, when accuracy matters. Zero-shot got 61-70% on our intent benchmarks against
  90-94% for a trained head.

## Share yours

Built something with decidejs? Post it in [Show and tell](https://github.com/debuffjp/decidejs/discussions) or the
[debuff Discord](https://discord.gg/jshbDHSpCY). Heads trained on public data are welcome there too.
