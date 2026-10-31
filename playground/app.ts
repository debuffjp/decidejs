import { createWorkerDecider, loadHead, type Choices, type Head } from "../src/browser.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const status = (s: string) => ($("status").textContent = s);
const examples: Record<string, string> = {
  zero: "the app crashes when I open settings",
  banking77: "my card still hasn't arrived",
  clinc150: "what's the weather like in Tokyo",
  train: "my parcel is late",
};
const heads: Record<string, Head> = {};
let trained: Head | undefined;

const files = new Map<string, [number, number]>();
const d = await createWorkerDecider({
  onProgress: ({ file, loaded, total }) => {
    files.set(file, [loaded, total]);
    let a = 0, b = 0;
    for (const [x, y] of files.values()) { a += x; b += y; }
    status(`Downloading the model (first visit only): ${(a / 1e6).toFixed(1)} of ${(b / 1e6).toFixed(1)} MB`);
  },
});
status(`Ready. ${crossOriginIsolated ? "Multi-threaded." : "Single-threaded (page not cross-origin isolated)."}`);
$<HTMLButtonElement>("go").disabled = false;

function labelsFromBox(): Choices<string> {
  const lines = $<HTMLTextAreaElement>("labels").value.split("\n").map((l) => l.trim()).filter(Boolean);
  if (!lines.some((l) => l.includes(":"))) return lines;
  // Labels without a description are compared by their readable name, as decide() does for a plain list.
  return Object.fromEntries(lines.map((l) => {
    const [label, ...rest] = l.split(":");
    return [label.trim(), rest.join(":").trim() || label.trim().replace(/[_-]+/g, " ")];
  }));
}

async function choices(): Promise<Choices<string> | undefined> {
  const mode = $<HTMLSelectElement>("mode").value;
  if (mode === "zero") return labelsFromBox();
  if (mode === "train") return trained;
  heads[mode] ??= loadHead(await (await fetch(`heads/${mode}.head.json`)).text());
  return heads[mode];
}

$("mode").addEventListener("change", () => {
  const mode = $<HTMLSelectElement>("mode").value;
  $("zero-box").hidden = mode !== "zero";
  $("train-box").hidden = mode !== "train";
  $<HTMLInputElement>("text").value = examples[mode];
  $("result").textContent = "";
});

$("train").addEventListener("click", async () => {
  const rows = $<HTMLTextAreaElement>("examples").value.split("\n").map((l) => {
    const i = l.indexOf(",");
    return { label: l.slice(0, i).trim(), text: l.slice(i + 1).trim() };
  }).filter((e) => e.label && e.text);
  const button = $<HTMLButtonElement>("train");
  button.disabled = true;
  try {
    const t0 = performance.now();
    trained = await d.train(rows, { onEpoch: (e) => ($("train-status").textContent = `epoch ${e}`) });
    $("train-status").textContent = `Trained on ${rows.length} examples, ${trained.labels.length} labels, in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
  } catch (e) {
    $("train-status").textContent = e instanceof Error ? e.message : String(e);
  }
  button.disabled = false;
});

$("go").addEventListener("click", decide);
$("text").addEventListener("keydown", (e) => e.key === "Enter" && decide());

async function decide() {
  const using = await choices();
  if (!using) return status("Train a head first.");
  const min = $<HTMLInputElement>("min").checked ? Number($<HTMLInputElement>("min-value").value) : undefined;
  try {
    const t0 = performance.now();
    const r = await d.decide($<HTMLInputElement>("text").value, using, { minConfidence: min });
    status(`${(performance.now() - t0).toFixed(1)} ms`);
    const result = $("result");
    result.textContent = "";
    const choice = result.appendChild(document.createElement("div"));
    choice.className = "choice";
    choice.textContent = r.choice === null ? `None of these (nothing reached ${min})` : `${r.choice}  ${(r.confidence * 100).toFixed(0)}%`;
    const top = Object.entries(r.probs as Record<string, number>).sort((a, b) => b[1] - a[1]).slice(0, 5);
    for (const [label, p] of top) {
      const row = result.appendChild(document.createElement("div"));
      row.className = "p";
      row.innerHTML = `<span></span><div class="track"><div class="fill"></div></div><span></span>`;
      row.children[0].textContent = label;
      row.querySelector<HTMLElement>(".fill")!.style.width = `${p * 100}%`;
      row.children[2].textContent = `${(p * 100).toFixed(1)}%`;
    }
  } catch (e) {
    status(e instanceof Error ? e.message : String(e));
  }
}
