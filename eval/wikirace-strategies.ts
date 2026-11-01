// WikiRace strategies for decidejs: 20 races between Wikipedia articles, every link offered at each step, the
// target's first sentences as the query. The first three are switch's WikiRace cases (github.com/osolmaz/switch), where
// Jev took 6 hops, 3 hops and gave up. Pages are cached in data/wiki/, so runs repeat and only the first reads Wikipedia.
//
//   STRATEGY=base|rerank|look node --experimental-strip-types eval/wikirace-strategies.ts
//
// base: the link closest in meaning to the target's summary (what zero-shot decide() picks), never a page already
// visited. rerank: base's top 20, reordered by a cross-encoder that reads the summary and the link together.
// look: base's top 5, each scored by its own similarity or its best link's, a little discounted (one step ahead).
import { createHash } from "node:crypto";
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { AutoModelForSequenceClassification, AutoTokenizer } from "@huggingface/transformers";
import { createDecider } from "../src/index.ts";

const RACES: [string, string][] = [
  ["DNA", "Manipuri pony"], ["Rubber duck", "Moon"], ["Baseball", "Sun"],
  ["Pizza", "Albert Einstein"], ["Volcano", "Jazz"], ["Coffee", "Roman Empire"], ["Bicycle", "Photosynthesis"],
  ["Chess", "Antarctica"], ["Mount Everest", "William Shakespeare"], ["Penguin", "Electricity"],
  ["Banana", "World War II"], ["Tokyo", "Dinosaur"], ["Guitar", "Black hole"], ["Association football", "Mathematics"],
  ["Tea", "Apollo 11"], ["Western honey bee", "Computer"], ["Olympic Games", "Ancient Egypt"], ["Chocolate", "Volcano"],
  ["Leonardo da Vinci", "Internet"], ["Samurai", "Coral reef"],
];
const STRATEGY = process.env.STRATEGY ?? "base";
const MAX_HOPS = 15, RERANK = 20, LOOK = 5, DISCOUNT = 0.9;
const HEADERS = { "user-agent": "decidejs-eval/0.1 (https://github.com/debuffjp/decidejs)" };
const CACHE = new URL("../data/wiki/", import.meta.url);
await mkdir(CACHE, { recursive: true });

async function cached<T>(key: string, get: () => Promise<T>): Promise<T> {
  const file = new URL(createHash("sha1").update(key).digest("hex") + ".json", CACHE);
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch {
    const v = await get();
    await writeFile(file, JSON.stringify(v));
    return v;
  }
}

const links = (title: string) => cached(`links:${title}`, async () => {
  const out: string[] = [];
  let cont: Record<string, string> = {}, resolved = title;
  for (;;) {
    const q = new URLSearchParams({ action: "query", format: "json", formatversion: "2", prop: "links", plnamespace: "0", pllimit: "max", redirects: "1", titles: title, ...cont });
    const j = await (await fetch(`https://en.wikipedia.org/w/api.php?${q}`, { headers: HEADERS })).json();
    const page = j.query.pages[0];
    resolved = page.title;
    out.push(...(page.links ?? []).map((l: { title: string }) => l.title));
    if (!j.continue) break;
    cont = j.continue;
  }
  return { title: resolved, links: [...new Set(out)] };
});

const summary = (title: string) => cached(`summary:${title}`, async () => {
  const j = await (await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}?redirect=true`, { headers: HEADERS })).json();
  return { title: j.title as string, about: (j.extract as string).split(". ").slice(0, 2).join(". ") };
});

const local = await createDecider();
const vecs = new Map<string, number[]>();
async function embed(texts: string[]) {
  const fresh = [...new Set(texts.filter((t) => !vecs.has(t)))];
  if (fresh.length) (await local.embed(fresh)).forEach((v, i) => vecs.set(fresh[i], v));
  return texts.map((t) => vecs.get(t)!);
}
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);

let rerank: ((query: string, titles: string[]) => Promise<number[]>) | undefined;
if (STRATEGY === "rerank") {
  const id = "Xenova/ms-marco-MiniLM-L-6-v2";
  const tokenizer = await AutoTokenizer.from_pretrained(id);
  const model = await AutoModelForSequenceClassification.from_pretrained(id, { dtype: "q8" });
  rerank = async (query, titles) => {
    const inputs = tokenizer(new Array(titles.length).fill(query), { text_pair: titles, padding: true, truncation: true });
    const { logits } = await model(inputs);
    return Array.from(logits.data as Float32Array);
  };
}

const same = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();
const results: { race: string; won: boolean; hops: number; path: string[]; ms: number }[] = [];
for (const [start, goal] of RACES) {
  const target = await summary(goal);
  const [q] = await embed([target.about]);
  const path = [start];
  let ms = 0;
  while (path.length <= MAX_HOPS) {
    const page = await links(path.at(-1)!);
    path[path.length - 1] = page.title;
    if (same(page.title, target.title)) break;
    const options = page.links.filter((l) => !path.some((p) => same(p, l)));
    if (!options.length) break;
    // The target among the links wins outright, as it would for any model reading the links.
    let next = options.find((l) => same(l, target.title));
    const t0 = performance.now();
    if (!next) {
      const sims = (await embed(options)).map((v) => dot(q, v));
      const ranked = options.map((l, i) => [l, sims[i]] as const).sort((a, b) => b[1] - a[1]);
      next = ranked[0][0];
      if (rerank) {
        const top = ranked.slice(0, RERANK).map(([l]) => l);
        const scores = await rerank(target.about, top);
        next = top[scores.indexOf(Math.max(...scores))];
      } else if (STRATEGY === "look") {
        let best = -Infinity;
        for (const [l, own] of ranked.slice(0, LOOK)) {
          const grand = (await links(l)).links;
          if (grand.some((g) => same(g, target.title))) { next = l; break; }
          const g = grand.length ? Math.max(...(await embed(grand)).map((v) => dot(q, v))) : -1;
          const value = Math.max(own, DISCOUNT * g);
          if (value > best) [best, next] = [value, l];
        }
      }
    }
    ms += performance.now() - t0;
    path.push(next);
  }
  const won = same(path.at(-1)!, target.title);
  results.push({ race: `${start} -> ${goal}`, won, hops: path.length - 1, path, ms: Math.round(ms) });
  console.log(`${won ? "won " : "lost"} ${start} -> ${goal}: ${path.length - 1} hops (${path.join(" > ")})`);
}

const wins = results.filter((r) => r.won), hops = wins.map((r) => r.hops).sort((a, b) => a - b);
const line = { strategy: STRATEGY, date: new Date().toISOString().slice(0, 10), wins: wins.length, races: results.length, medianHops: hops[Math.floor(hops.length / 2)] ?? null, ms: results.reduce((s, r) => s + r.ms, 0), results };
console.log(`${STRATEGY}: won ${wins.length}/${results.length}, median ${line.medianHops} hops, ${line.ms} ms deciding`);
await appendFile(new URL("./results-wikirace.jsonl", import.meta.url), JSON.stringify(line) + "\n");
