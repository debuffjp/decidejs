// WikiRace with decidekit as the decision model, through the switch adapter: the race TypeSafe launched Jev with, and
// one of switch's example domains. Each step is one switch choice question over the current article's links; decidekit
// reads the target and its summary. Unlike Jev (255 answers at most), it sees every link of the article. Reads English Wikipedia's public API. Run: node --experimental-strip-types eval/wikirace.ts
import { createDecider } from "../src/index.ts";
import { switchDecider } from "../src/switch.ts";

const RACES = [
  { start: "DNA", target: "Manipuri pony" },
  { start: "Rubber duck", target: "Moon" },
  { start: "Baseball", target: "Sun" },
];
const MAX_HOPS = 12;
const API = "https://en.wikipedia.org/w/api.php";
const HEADERS = { "user-agent": "decidekit-eval/0.1 (https://github.com/debuffjp/decidekit)" };

async function links(title: string): Promise<{ title: string; links: string[] }> {
  const out: string[] = [];
  let cont: Record<string, string> = {};
  let resolved = title;
  for (;;) {
    const q = new URLSearchParams({ action: "query", format: "json", formatversion: "2", prop: "links", plnamespace: "0", pllimit: "max", redirects: "1", titles: title, ...cont });
    const j = await (await fetch(`${API}?${q}`, { headers: HEADERS })).json();
    const page = j.query.pages[0];
    resolved = page.title;
    out.push(...(page.links ?? []).map((l: { title: string }) => l.title));
    if (!j.continue) break;
    cont = j.continue;
  }
  return { title: resolved, links: out };
}

async function summary(title: string) {
  const j = await (await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}?redirect=true`, { headers: HEADERS })).json();
  return { title: j.title as string, about: (j.extract as string).split(". ").slice(0, 2).join(". ") };
}

const same = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();
const local = await createDecider();
const decider = switchDecider(local, { text: (state) => String(state.about_the_target) });

for (const race of RACES) {
  const target = await summary(race.target);
  const path = [race.start];
  let at = race.start, ms = 0;
  while (path.length <= MAX_HOPS) {
    const page = await links(at);
    if (same(page.title, target.title)) break;
    // The target among the links wins outright, as it would for any model reading the links.
    const options = [...new Set(page.links)].filter((l) => !path.some((p) => same(p, l)));
    const direct = options.find((l) => same(l, target.title));
    let next = direct;
    if (!next && options.length) {
      const criteria = Object.fromEntries(options.map((l, i) => [`l${i}`, l]));
      const r = await decider.decide({
        decisionState: { target: target.title, about_the_target: target.about, at: page.title, path },
        decisionQuestions: { pick: { type: "choice", instructions: "Pick the link that brings you closer to the target.", criteria } },
      });
      if (r.stopReason !== "stop") throw new Error(r.errorMessage);
      ms += r.milliseconds;
      const a = r.decisionAnswers.pick;
      next = a.type === "choice" ? criteria[a.choice] : undefined;
    }
    if (!next) break;
    path.push(next);
    at = next;
  }
  const won = same(path.at(-1)!, target.title);
  console.log(`${race.start} -> ${race.target}: ${won ? `${path.length - 1} hops` : "lost"} (${path.join(" > ")}) ${Math.round(ms)} ms of decisions`);
}
