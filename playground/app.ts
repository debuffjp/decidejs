import { createWorkerDecider, loadHead, type Choices, type Head } from "../src/browser.ts";

type Mode = "zero" | "banking77" | "clinc150" | "train";
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const status = (s: string) => ($("status").textContent = s);
const tries: Record<Mode, string[]> = {
  zero: ["the app crashes when I open settings", "love the new dark mode", "can you add CSV export?", "how do I change my avatar"],
  banking77: ["my card still hasn't arrived", "why was I charged a fee at the ATM", "the exchange rate looks wrong", "I think someone stole my card"],
  clinc150: ["what's the weather like in Tokyo", "set a timer for ten minutes", "translate thank you into Spanish", "what's my credit score"],
  train: ["my parcel is late", "I forgot my password", "send the money back to my card"],
};
const about: Partial<Record<Mode, string>> = {
  banking77: "A ready-made head trained on BANKING77: 77 banking support intents, 90.6% accurate on its test set.",
  clinc150: "A ready-made head trained on CLINC150: 150 assistant intents, 94.2% accurate on its test set.",
};
const heads: Partial<Record<Mode, Head>> = {};
let mode: Mode = "zero", trained: Head | undefined, ready = false;

const files = new Map<string, [number, number]>();
const d = await createWorkerDecider({
  onProgress: ({ file, loaded, total }) => {
    files.set(file, [loaded, total]);
    let a = 0, b = 0;
    for (const [x, y] of files.values()) { a += x; b += y; }
    status(`Downloading the model, first visit only: ${(a / 1e6).toFixed(1)} of ${(b / 1e6).toFixed(1)} MB`);
  },
});
ready = true;
status(crossOriginIsolated ? "Ready · multi-threaded" : "Ready · single-threaded");

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
  if (mode === "zero") return labelsFromBox();
  if (mode === "train") return trained;
  heads[mode] ??= loadHead(await (await fetch(`heads/${mode}.head.json`)).text());
  return heads[mode];
}

function chips() {
  const box = $("chips");
  box.textContent = "";
  for (const t of tries[mode]) {
    const b = box.appendChild(document.createElement("button"));
    b.textContent = t;
    b.addEventListener("click", () => {
      $<HTMLInputElement>("text").value = t;
      void decide();
    });
  }
}

for (const tab of document.querySelectorAll<HTMLButtonElement>("#tabs button")) {
  tab.addEventListener("click", () => {
    mode = tab.dataset.mode as Mode;
    for (const t of document.querySelectorAll<HTMLButtonElement>("#tabs button")) t.setAttribute("aria-selected", String(t === tab));
    $("zero-box").hidden = mode !== "zero";
    $("train-box").hidden = mode !== "train";
    $("head-box").hidden = !about[mode];
    $("head-about").textContent = about[mode] ?? "";
    $<HTMLInputElement>("text").value = tries[mode][0];
    chips();
    void decide();
  });
}
chips();

$("train").addEventListener("click", async () => {
  const rows = $<HTMLTextAreaElement>("examples").value.split("\n").map((l) => {
    const i = l.indexOf(",");
    return { label: l.slice(0, i).trim(), text: l.slice(i + 1).trim() };
  }).filter((e) => e.label && e.text);
  const button = $<HTMLButtonElement>("train");
  button.disabled = true;
  try {
    const t0 = performance.now();
    trained = await d.train(rows, { onEpoch: (e) => ($("train-status").textContent = `Training, epoch ${e}`) });
    $("train-status").textContent = `Trained on ${rows.length} examples in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
    void decide();
  } catch (e) {
    $("train-status").textContent = e instanceof Error ? e.message : String(e);
  }
  button.disabled = false;
});

let timer: ReturnType<typeof setTimeout> | undefined;
const soon = () => {
  clearTimeout(timer);
  timer = setTimeout(decide, 150);
};
$("text").addEventListener("input", soon);
$("labels").addEventListener("input", soon);
$("min").addEventListener("change", decide);
$("min-value").addEventListener("input", () => {
  $("min-label").textContent = Number($<HTMLInputElement>("min-value").value).toFixed(2);
  if ($<HTMLInputElement>("min").checked) soon();
});

let latest = 0;
async function decide() {
  if (!ready) return;
  const text = $<HTMLInputElement>("text").value.trim();
  const result = $("result");
  const using = await choices();
  if (!using) {
    result.innerHTML = `<p class="hint">Press Train to make a head from your examples, then try it here.</p>`;
    return;
  }
  if (!text) return void (result.textContent = "");
  const min = $<HTMLInputElement>("min").checked ? Number($<HTMLInputElement>("min-value").value) : undefined;
  const ticket = ++latest;
  try {
    const t0 = performance.now();
    const r = await d.decide(text, using, { minConfidence: min });
    if (ticket !== latest) return; // a newer keystroke already asked
    status(`${(performance.now() - t0).toFixed(0)} ms · ${crossOriginIsolated ? "multi-threaded" : "single-threaded"}`);
    result.textContent = "";
    const top = Object.entries(r.probs as Record<string, number>).sort((a, b) => b[1] - a[1]).slice(0, 5);
    const choice = result.appendChild(document.createElement("div"));
    choice.className = r.choice === null ? "choice none" : "choice";
    choice.innerHTML = "<b></b><span class=pct></span>";
    choice.children[0].textContent = r.choice === null ? "None of these" : r.choice;
    choice.children[1].textContent = r.choice === null ? `best was ${(r.confidence * 100).toFixed(0)}%, below ${min}` : `${(r.confidence * 100).toFixed(0)}% sure`;
    top.forEach(([label, p], i) => {
      const row = result.appendChild(document.createElement("div"));
      row.className = i ? "p" : "p top";
      row.innerHTML = `<span></span><div class="track"><div class="fill"></div></div><span></span>`;
      row.children[0].textContent = label;
      row.querySelector<HTMLElement>(".fill")!.style.width = `${p * 100}%`;
      row.children[2].textContent = `${(p * 100).toFixed(1)}%`;
    });
  } catch (e) {
    status(e instanceof Error ? e.message : String(e));
  }
}
void decide();
