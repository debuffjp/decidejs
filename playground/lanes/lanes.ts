// "Two lanes": decidejs answers what it's sure of in milliseconds, sends the rest to a slow LLM, and learns from
// every answer it gets back. Drawn with spryte. Messages are BANKING77 test queries (CC BY 4.0, PolyAI): twelve card
// intents close enough to each other that a gut feeling needs help at first. The LLM's answers to them were recorded
// once from the live endpoint (`llm` in messages.json); only messages typed into the page ask it live.
import { createRenderer } from "spryte";
import { createWorkerDecider, withFallback, type Head } from "../../src/browser.ts";
import messages from "./messages.json" with { type: "json" };

const LABELS = ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card", "card_not_working", "declined_card_payment", "card_payment_fee_charged", "exchange_rate", "top_up_failed", "pending_transfer", "cash_withdrawal_charge", "activate_my_card", "change_pin"] as const;
type Label = (typeof LABELS)[number];
type Message = { text: string; label: Label | null; llm?: Label | null };
const MESSAGES = messages as Message[];
const NAMES = Object.fromEntries(LABELS.map((l) => [l, l.replace(/_/g, " ")])) as Record<Label, string>;
// One sentence per label. Both sides read them: decidejs scores each label by the better of its name and this, and
// the LLM sees them in its prompt (they raised Llama 3.1 8B from 84% to 90% on these messages).
const DESCRIPTIONS: Record<Label, string> = {
  card_arrival: "My card still hasn't arrived in the post.",
  card_delivery_estimate: "How long will it take for my card to be delivered?",
  lost_or_stolen_card: "I lost my card or it was stolen.",
  card_not_working: "My card isn't working.",
  declined_card_payment: "My card payment was declined.",
  card_payment_fee_charged: "I was charged a fee for paying by card.",
  exchange_rate: "What exchange rate do you use?",
  top_up_failed: "My top-up failed.",
  pending_transfer: "My transfer is still pending.",
  cash_withdrawal_charge: "I was charged a fee for withdrawing cash.",
  activate_my_card: "I got my new card and want to activate it.",
  change_pin: "How do I change my PIN?",
};
const W = 960, H = 560, EVERY = 8, THRESHOLD = 0.9, CADENCE = 1100;
const C = {
  bg: "#0b0b0d", panel: "#17181c", line: "#26272d", text: "#ececef", muted: "#8b8d97", faint: "#4a4c55",
  gut: "#cffb2d", llm: "#ff7ab6", wrong: "#ff5c5c",
};
const F = { s: "11px Geist", m: "12px Geist", b: "600 13px Geist", big: "600 22px Geist", mono: "11px 'Geist Mono'" };
const GUT = { x: 400, y: 150, r: 54 }, LLM = { x: 400, y: 392, r: 46 };
const QUEUE = { x: 16, y: 56, w: 236, h: 50, gap: 8, shown: 3 };
const BIN = { x: 600, y: 64, step: 33, dots: 26 };
const binY = (i: number) => BIN.y + i * BIN.step;
const hex = (c: string, a: number) => c + Math.round(Math.max(0, Math.min(1, a)) * 255).toString(16).padStart(2, "0");
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// The simulation clock drives everything, the LLM's latency included, so Pause and Faster are exact.
let clock = 0, speed = 1, paused = false;
const timers: { at: number; done: () => void }[] = [];
const wait = (ms: number) => new Promise<void>((done) => timers.push({ at: clock + ms, done }));

type Path = { x0: number; y0: number; cx: number; cy: number; x1: number; y1: number; t0: number; dur: number };
type Dot = {
  m: Message; idx: number; again: boolean; x: number; y: number; path?: Path; colour: string;
  phase: "queued" | "toGut" | "toLlm" | "thinking" | "toBin" | "done";
  via?: "local" | "llm"; choice?: Label | null; thinkFrom?: number;
};
const dots: Dot[] = [], sparks: { label: Label; path: Path }[] = [];
const bins: string[][] = LABELS.map(() => []);
const taught = LABELS.map(() => 0);
let queue = 0, departAt = 0, replayed = false;
let local = 0, sent = 0, total = 0, labelled = 0, right = 0, firstLocal = 0, againLocal = 0, againTotal = 0;
let lessons = 0, retrains = 0, flashAt = -1e9;
const recent: number[] = [];
let last: { text: string; top: [Label, number][]; confidence: number } | null = null;
let status = "Loading the model (first visit: about 34 MB of model)...", liveFailed = false;

function path(x0: number, y0: number, x1: number, y1: number, dur: number, bend = 0): Path {
  const nx = -(y1 - y0), ny = x1 - x0, len = Math.hypot(nx, ny) || 1;
  return { x0, y0, x1, y1, cx: (x0 + x1) / 2 + (nx / len) * bend, cy: (y0 + y1) / 2 + (ny / len) * bend, t0: clock, dur };
}
function at(p: Path) {
  const t = Math.min(1, (clock - p.t0) / p.dur), e = ease(t), u = 1 - e;
  return { x: u * u * p.x0 + 2 * u * e * p.cx + e * e * p.x1, y: u * u * p.y0 + 2 * u * e * p.cy + e * e * p.y1, t };
}

// --- decisions -------------------------------------------------------------------------------------------------
const inFlight = new Map<string, Dot>();
const topOf = (probs: Record<Label, number>) => (Object.entries(probs) as [Label, number][]).sort((a, b) => b[1] - a[1]);

// Called by withFallback when decidejs isn't sure. The dot learns its answer as soon as it's known, but the promise
// resolves only when the lesson chip reaches decidejs, so the retrain lands on screen when the lesson does.
async function ask(text: string, labels: readonly Label[], localDecision: { probs: Record<Label, number>; confidence: number }) {
  const dot = inFlight.get(text)!;
  dot.via = "llm";
  last = { text, top: topOf(localDecision.probs).slice(0, 3), confidence: localDecision.confidence };
  const lands = wait(3000);
  if (dot.m.llm !== undefined) dot.choice = dot.m.llm;
  else {
    try {
      const r = await fetch("/api/decidejs/ask", { method: "POST", body: JSON.stringify({ text, labels, descriptions: DESCRIPTIONS }) });
      if (!r.ok) throw new Error(String(r.status));
      dot.choice = (await r.json()).label;
    } catch {
      liveFailed = true;
      dot.choice = null;
    }
  }
  await lands;
  return dot.choice ?? null;
}

const d = await createWorkerDecider(
  { onProgress: ({ loaded, total }) => (status = `Downloading the model (first visit only): ${Math.round((100 * loaded) / total)}%`) },
  new Worker(new URL("../worker.js", import.meta.url), { type: "module" }),
);
const route = withFallback(d, DESCRIPTIONS, { minConfidence: THRESHOLD, ask, retrainEvery: EVERY });
let head: Head<Label> | null = route.head;
status = "";

function depart(dot: Dot) {
  dot.phase = "toGut";
  dot.path = path(dot.x + QUEUE.w / 2, dot.y + QUEUE.h / 2, GUT.x - GUT.r + 4, GUT.y, 650, -40);
  inFlight.set(dot.m.text, dot);
  void route(dot.m.text).then((r) => {
    if (r.source !== "local") return;
    dot.via = "local";
    dot.choice = r.choice ?? topOf(r.probs)[0][0];
    last = { text: dot.m.text, top: topOf(r.probs).slice(0, 3), confidence: r.confidence };
  });
}

// --- input -----------------------------------------------------------------------------------------------------
document.getElementById("say")!.addEventListener("submit", (e) => {
  e.preventDefault();
  const input = document.getElementById("msg") as HTMLInputElement;
  const text = input.value.trim();
  input.value = "";
  if (!text || inFlight.has(text)) return;
  // Typed messages jump the queue.
  const first = dots.findIndex((d) => d.phase === "queued");
  const dot: Dot = { m: { text, label: null }, idx: -1, again: false, x: QUEUE.x, y: QUEUE.y - 60, colour: C.muted, phase: "queued" };
  dots.splice(first < 0 ? dots.length : first, 0, dot);
});
document.getElementById("speed")!.addEventListener("click", (e) => {
  speed = speed === 1 ? 3 : 1;
  (e.target as HTMLButtonElement).textContent = speed === 1 ? "Faster" : "Slower";
});
document.getElementById("pause")!.addEventListener("click", (e) => {
  paused = !paused;
  (e.target as HTMLButtonElement).textContent = paused ? "Play" : "Pause";
});

// --- update ----------------------------------------------------------------------------------------------------
function settle(dot: Dot) {
  inFlight.delete(dot.m.text);
  total++;
  recent.push(dot.via === "llm" ? 1 : 0);
  if (dot.via === "llm") sent++;
  else {
    local++;
    if (!dot.again && dot.idx >= 0 && dot.idx < 20) firstLocal++;
    if (dot.again) againLocal++;
  }
  if (dot.again) againTotal++;
  if (dot.m.label) {
    labelled++;
    if (dot.choice === dot.m.label) right++;
  }
  if (dot.choice) bins[LABELS.indexOf(dot.choice)].push(dot.colour);
}

function update(dt: number) {
  clock += dt;
  for (let i = timers.length - 1; i >= 0; i--) if (clock >= timers[i].at) timers.splice(i, 1)[0].done();

  // Keep the queue topped up, then let the front card go every CADENCE ms.
  let queued = dots.filter((d) => d.phase === "queued").length;
  while (queued < QUEUE.shown && queue < MESSAGES.length) {
    dots.push({ m: MESSAGES[queue], idx: queue, again: false, x: QUEUE.x, y: QUEUE.y + (QUEUE.shown - 1) * 58 + 40, colour: C.muted, phase: "queued" });
    queue++;
    queued++;
  }
  if (!replayed && queue >= MESSAGES.length && dots.length === 0) {
    replayed = true;
    for (let i = 0; i < 20; i++) dots.push({ m: MESSAGES[i], idx: i, again: true, x: QUEUE.x, y: QUEUE.y + (QUEUE.shown + i) * 60, colour: C.muted, phase: "queued" });
  }
  const front = dots.find((d) => d.phase === "queued");
  if (front && clock >= departAt && !inFlight.has(front.m.text)) {
    depart(front);
    departAt = clock + CADENCE;
  }

  let slot = 0;
  for (const dot of dots) {
    if (dot.phase === "queued") {
      const ty = QUEUE.y + slot++ * (QUEUE.h + QUEUE.gap);
      const k = 1 - Math.exp(-dt / 90);
      dot.x += (QUEUE.x - dot.x) * k;
      dot.y += (ty - dot.y) * k;
      continue;
    }
    if (dot.phase === "thinking") {
      const a = (clock - dot.thinkFrom!) / 240 - Math.PI / 4;
      dot.x = LLM.x + Math.cos(a) * (LLM.r + 3);
      dot.y = LLM.y + Math.sin(a) * (LLM.r + 3);
      if (clock - dot.thinkFrom! > 900 && dot.choice !== undefined) {
        dot.phase = "toBin";
        const bin = dot.choice ? LABELS.indexOf(dot.choice) : -1;
        dot.path = path(dot.x, dot.y, BIN.x - 12, bin >= 0 ? binY(bin) + 16 : H - 24, 850, 30);
        if (dot.choice) sparks.push({ label: dot.choice, path: path(LLM.x - LLM.r - 4, LLM.y, GUT.x - GUT.r * 0.75, GUT.y + GUT.r * 0.75, 1100, -60) });
      }
      continue;
    }
    if (dot.phase === "done" || !dot.path) continue;
    const p = at(dot.path);
    dot.x = p.x;
    dot.y = p.y;
    if (p.t < 1) continue;
    if (dot.phase === "toGut") {
      // Normally decided long before it arrives; during a retrain it waits here, which is the real cost.
      if (dot.via === "llm") {
        dot.colour = C.llm;
        dot.phase = "toLlm";
        dot.path = path(GUT.x, GUT.y + GUT.r, LLM.x + LLM.r * 0.7, LLM.y - LLM.r * 0.7, 600, -30);
      } else if (dot.via === "local" && dot.choice) {
        dot.colour = dot.m.label && dot.choice !== dot.m.label ? C.wrong : C.gut;
        dot.phase = "toBin";
        dot.path = path(GUT.x + GUT.r, GUT.y, BIN.x - 12, binY(LABELS.indexOf(dot.choice)) + 16, 750, 50);
      }
    } else if (dot.phase === "toLlm") {
      dot.phase = "thinking";
      dot.thinkFrom = clock;
    } else if (dot.phase === "toBin") {
      dot.phase = "done";
      settle(dot);
    }
  }
  for (let i = dots.length - 1; i >= 0; i--) if (dots[i].phase === "done") dots.splice(i, 1);
  for (let i = sparks.length - 1; i >= 0; i--) {
    if (at(sparks[i].path).t >= 1) {
      taught[LABELS.indexOf(sparks[i].label)]++;
      lessons++;
      sparks.splice(i, 1);
    }
  }
  // A new head object means withFallback really retrained.
  if (route.head !== head) {
    head = route.head;
    retrains++;
    flashAt = clock;
  }
}

// --- drawing ---------------------------------------------------------------------------------------------------
const canvas = document.getElementById("c") as HTMLCanvasElement;
const draw = createRenderer(canvas);
function fit() {
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(canvas.clientWidth * scale);
  canvas.height = Math.round(((canvas.clientWidth * H) / W) * scale);
  draw.resize(canvas.width, canvas.height);
}
addEventListener("resize", fit);
await Promise.all([F.s, F.m, F.b, F.big, F.mono, "600 16px Geist"].map((f) => document.fonts.load(f))).catch(() => {});
fit();

const wrapped = new Map<string, string[]>();
function wrap(text: string, font: string, width: number, lines: number) {
  const key = `${font}|${width}|${lines}|${text}`;
  const hit = wrapped.get(key);
  if (hit) return hit;
  const out: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (!line || draw.measure(next, font).w <= width) line = next;
    else {
      out.push(line);
      line = word;
    }
  }
  out.push(line);
  if (out.length > lines) {
    out.length = lines;
    let l = out[lines - 1];
    while (l && draw.measure(l + "…", font).w > width) l = l.slice(0, -1);
    out[lines - 1] = l.trimEnd() + "…";
  }
  wrapped.set(key, out);
  return out;
}

function roundRect(x: number, y: number, w: number, h: number, r: number, colour: string) {
  r = Math.min(r, w / 2, h / 2);
  draw.rect(x + r, y, w - 2 * r, h, colour);
  draw.rect(x, y + r, r, h - 2 * r, colour);
  draw.rect(x + w - r, y + r, r, h - 2 * r, colour);
  for (const [cx, cy] of [[x + r, y + r], [x + w - r, y + r], [x + r, y + h - r], [x + w - r, y + h - r]]) draw.circle(cx, cy, r, colour);
}

function card(dot: Dot, x: number, y: number, alpha: number) {
  roundRect(x, y, QUEUE.w, QUEUE.h, 8, hex(dot.m.label ? C.panel : "#2a1d25", alpha));
  wrap(dot.m.text, F.m, QUEUE.w - 24, 2).forEach((l, i, a) =>
    draw.text(l, x + 12, y + QUEUE.h / 2 + (i - (a.length - 1) / 2) * 16, { font: F.m, color: hex(C.text, alpha), baseline: "middle" }),
  );
}

function node(x: number, y: number, r: number, colour: string, title: string, sub: string, glow: number) {
  if (glow > 0) draw.circle(x, y, r + 8 + glow * 10, hex(colour, 0.12 * glow));
  draw.circle(x, y, r, C.panel);
  draw.circle(x, y, r - 5, hex(colour, 0.16 + 0.2 * glow));
  draw.text(title, x, y - 7, { font: "600 16px Geist", color: C.text, align: "center", baseline: "middle" });
  draw.text(sub, x, y + 13, { font: F.s, color: C.muted, align: "center", baseline: "middle" });
}

function render() {
  draw.begin(C.bg, W, H, canvas.width / W);
  const share = total ? Math.round((100 * sent) / total) : 0;
  draw.text(`${total} message${total === 1 ? "" : "s"} · ${local} answered on device · ${sent} sent to the LLM (${share}%)`, BIN.x - 180, 28, { font: F.m, color: C.muted, baseline: "middle" });
  draw.text("Incoming", QUEUE.x, 34, { font: F.s, color: C.muted, baseline: "middle" });

  // lanes
  draw.line(GUT.x, GUT.y + GUT.r + 18, LLM.x, LLM.y - LLM.r - 4, 2, hex(C.llm, 0.22));
  draw.line(QUEUE.x + QUEUE.w + 8, GUT.y, GUT.x - GUT.r - 18, GUT.y, 2, C.line);
  draw.line(GUT.x + GUT.r + 18, GUT.y, BIN.x - 20, BIN.y + 180, 2, hex(C.gut, 0.16));
  draw.line(LLM.x + LLM.r + 4, LLM.y, BIN.x - 20, BIN.y + 220, 2, hex(C.llm, 0.14));

  // decidejs, ringed by the lessons since its last retrain
  const flash = Math.max(0, 1 - (clock - flashAt) / 1000);
  node(GUT.x, GUT.y, GUT.r, C.gut, "decidejs", "~5 ms · free", flash);
  const pendingLessons = Math.max(0, Math.min(EVERY, lessons - retrains * EVERY));
  for (let i = 0; i < EVERY; i++) {
    const a = -Math.PI / 2 + (i / EVERY) * Math.PI * 2;
    const lit = flash > 0 || i < pendingLessons;
    draw.circle(GUT.x + Math.cos(a) * (GUT.r + 13), GUT.y + Math.sin(a) * (GUT.r + 13), 4, lit ? C.gut : C.line);
  }
  draw.text(retrains ? `retrained ${retrains}× on the LLM's answers` : `retrains every ${EVERY} lessons`, GUT.x, GUT.y - GUT.r - 30, { font: F.s, color: C.muted, align: "center", baseline: "middle" });
  if (flash > 0) draw.text("retrained", GUT.x + GUT.r + 24, GUT.y - GUT.r + 2, { font: F.b, color: hex(C.gut, flash), baseline: "middle" });

  const thinking = dots.some((d) => d.phase === "thinking");
  node(LLM.x, LLM.y, LLM.r, C.llm, "LLM", "~1 s · paid", thinking ? (Math.sin(clock / 140) + 1) / 2 : 0);
  draw.text("unsure? ask", GUT.x + 8, (GUT.y + LLM.y) / 2, { font: F.s, color: hex(C.llm, 0.8), baseline: "middle" });
  draw.text("answers teach decidejs", LLM.x - 22, (GUT.y + LLM.y) / 2 + 12, { font: F.s, color: hex(C.gut, 0.7), align: "right", baseline: "middle" });

  // bins
  LABELS.forEach((l, i) => {
    const y = binY(i), inBin = bins[i];
    draw.text(NAMES[l], BIN.x, y + 2, { font: F.m, color: C.muted, baseline: "middle" });
    if (taught[i]) draw.text(`+${taught[i]} learned`, BIN.x + draw.measure(NAMES[l], F.m).w + 8, y + 2, { font: F.s, color: hex(C.gut, 0.75), baseline: "middle" });
    inBin.slice(-BIN.dots).forEach((c, j) => draw.circle(BIN.x + 4 + j * 11, y + 16, 3.5, c));
    if (inBin.length) draw.text(String(inBin.length), W - 14, y + 2, { font: F.mono, color: C.faint, align: "right", baseline: "middle" });
  });

  // lessons travelling back to decidejs
  for (const s of sparks) {
    const p = at(s.path), w = draw.measure(NAMES[s.label], F.s).w + 14, a = Math.min(1, (1 - p.t) * 6);
    roundRect(p.x - w / 2, p.y - 9, w, 18, 9, hex(C.gut, 0.92 * a));
    draw.text(NAMES[s.label], p.x, p.y, { font: F.s, color: hex(C.bg, a), align: "center", baseline: "middle" });
  }

  // messages
  for (const dot of dots) {
    if (dot.phase === "queued") {
      const bottom = QUEUE.y + (QUEUE.shown - 1) * (QUEUE.h + QUEUE.gap);
      if (dot.y < bottom + 40) card(dot, dot.x, dot.y, Math.min(1, (bottom + 40 - dot.y) / 40));
    } else if (dot.phase === "toGut") {
      const t = (clock - dot.path!.t0) / dot.path!.dur;
      if (t < 0.4) card(dot, dot.x - QUEUE.w / 2, dot.y - QUEUE.h / 2, 1 - t / 0.4);
      draw.circle(dot.x, dot.y, 2 + 4 * Math.min(1, t * 2), C.muted);
    } else draw.circle(dot.x, dot.y, 6, dot.colour);
  }

  // the gut's last call
  if (last) {
    const x = 16, y = 300, w = 236;
    draw.text("How sure decidejs was", x, y, { font: F.s, color: C.muted, baseline: "middle" });
    wrap(`“${last.text}”`, F.m, w, 2).forEach((l, i) => draw.text(l, x, y + 18 + i * 16, { font: F.m, color: C.text, baseline: "middle" }));
    const sure = last.confidence >= THRESHOLD;
    last.top.forEach(([l, p], i) => {
      const yy = y + 60 + i * 24, colour = i ? C.faint : sure ? C.gut : C.llm;
      draw.text(NAMES[l], x, yy, { font: F.s, color: i ? C.muted : C.text, baseline: "middle" });
      draw.text(`${Math.round(p * 100)}%`, x + w, yy, { font: F.mono, color: i ? C.muted : colour, align: "right", baseline: "middle" });
      roundRect(x, yy + 8, w, 5, 2, C.panel);
      if (p > 0.02) roundRect(x, yy + 8, w * p, 5, 2, colour);
    });
    draw.rect(x + w * THRESHOLD, y + 64, 1, 66, hex(C.text, 0.45));
    draw.text("90%", x + w * THRESHOLD, y + 140, { font: F.mono, color: C.muted, align: "center", baseline: "middle" });
    draw.text(sure ? "Sure, so it answers" : "Torn, so it asks", x, y + 140, { font: F.b, color: sure ? C.gut : C.llm, baseline: "middle" });
  }

  // share sent to the LLM, rolling 20, across the whole run
  const cx = 16, cy = 480, cw = 236, ch = 60;
  draw.text("Sent to the LLM, last 20 messages", cx, cy - 12, { font: F.s, color: C.muted, baseline: "middle" });
  roundRect(cx, cy, cw, ch, 6, C.panel);
  const n = MESSAGES.length + 20;
  let prevX = 0, prevY = 0;
  for (let i = 0; i < recent.length; i++) {
    const win = recent.slice(Math.max(0, i - 19), i + 1), p = win.reduce((a, b) => a + b, 0) / win.length;
    const px = cx + 4 + ((cw - 8) * i) / n, py = cy + ch - 4 - p * (ch - 8);
    draw.line(px, cy + ch - 4, px, py, 1, hex(C.llm, 0.16));
    if (i) draw.line(prevX, prevY, px, py, 2, C.llm);
    prevX = px;
    prevY = py;
  }

  // LLM calls
  draw.text("LLM calls", 300, cy - 12, { font: F.s, color: C.muted, baseline: "middle" });
  draw.text(String(sent), 300, cy + 14, { font: F.big, color: C.llm, baseline: "middle" });
  draw.text(`for ${total} messages`, 306 + draw.measure(String(sent), F.big).w, cy + 16, { font: F.m, color: C.muted, baseline: "middle" });
  if (total) draw.text(`${local} fewer than asking every time`, 300, cy + 42, { font: F.m, color: C.gut, baseline: "middle" });

  // before and after
  if (replayed) {
    draw.text("The first 20 messages, again", BIN.x, cy - 12, { font: F.s, color: C.muted, baseline: "middle" });
    draw.text(`At the start: ${firstLocal} of 20 answered on device`, BIN.x, cy + 14, { font: F.m, color: C.text, baseline: "middle" });
    draw.text(`Now: ${againLocal} of ${againTotal}`, BIN.x, cy + 40, { font: F.b, color: C.gut, baseline: "middle" });
  } else if (labelled) {
    draw.text("Agreement with BANKING77's labels", BIN.x, cy - 12, { font: F.s, color: C.muted, baseline: "middle" });
    draw.text(`${Math.round((100 * right) / labelled)}%`, BIN.x, cy + 14, { font: F.big, color: C.text, baseline: "middle" });
    draw.text("on device and LLM answers together", BIN.x, cy + 42, { font: F.m, color: C.muted, baseline: "middle" });
  }
  if (liveFailed) draw.text("The live LLM isn't reachable from here, so typed messages get no answer", W / 2, H - 8, { font: F.s, color: C.llm, align: "center", baseline: "middle" });
  if (status) draw.text(status, W / 2, H / 2, { font: "15px Geist", color: C.text, align: "center", baseline: "middle" });
  draw.end();
}

let prev = performance.now();
function frame(now: number) {
  const dt = Math.min(50, now - prev);
  prev = now;
  if (!paused) update(dt * speed);
  render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
