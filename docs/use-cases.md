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
leaves their device. Use it as a nudge, not as the only moderation.

**Label a dataset in the browser.** Load a CSV in a tab, label 20 rows per class, train, review the low-confidence
rows first. Nothing uploads.

**Cheap first pass before an LLM.** Decide the easy 80% locally and send only low-confidence cases to a large model.
`confidence` from a trained head is calibrated, so a threshold means what it says.

**Agent and app plumbing.** "Is this message a question, a command or chit-chat?" "Does this log line look like an
error?" Small decisions an app makes thousands of times.

## Poor fits

- **Open-ended questions.** If the answer isn't one of a fixed set of labels, use an LLM.
- **Long documents.** The encoder reads 128 to 256 tokens (about a paragraph) and ignores the rest.
- **Reasoning over facts.** "Is this refund allowed under our policy?" needs the policy, which a sentence encoder
  can't read alongside the text.
- **Languages other than English** with the default model. A multilingual encoder can be passed as `model`, but we
  haven't measured one yet.
- **Labels you can't give examples for**, when accuracy matters. Zero-shot is a starting point, not a result, and its
  `confidence` isn't calibrated.

## Share yours

Built something with decidejs? Post it in [Show and tell](https://github.com/debuffjp/decidejs/discussions) or the
[debuff Discord](https://discord.gg/jshbDHSpCY). Heads trained on public data are welcome there too.
