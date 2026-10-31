// Sorts feedback into four labels with a head trained on a handful of examples.
// Run: node --experimental-strip-types examples/node-triage.ts
import { createDecider } from "../src/index.ts";

const examples = [
  ["Love the new dark mode, great work", "praise"],
  ["This is my favourite app, thank you", "praise"],
  ["Five stars, does exactly what I need", "praise"],
  ["The app crashes when I open settings", "bug"],
  ["Export button does nothing on Firefox", "bug"],
  ["Login fails with an error since the update", "bug"],
  ["Please add a way to export to CSV", "feature"],
  ["Would be great to have keyboard shortcuts", "feature"],
  ["Can you support Google Calendar sync?", "feature"],
  ["Way too expensive for what it does", "complaint"],
  ["Support never answered my email", "complaint"],
  ["The ads are really annoying", "complaint"],
].map(([text, label]) => ({ text, label: label as "praise" | "bug" | "feature" | "complaint" }));

const d = await createDecider();
const head = await d.train(examples, { holdout: 0.25 });
for (const text of ["it freezes when I upload a photo", "dark theme please", "honestly brilliant"]) {
  const { choice, confidence } = await d.decide(text, head);
  console.log(choice.padEnd(10), confidence.toFixed(2), text);
}
