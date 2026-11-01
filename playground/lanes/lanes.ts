// "Two lanes": decidejs answers what it's sure of in milliseconds, sends the rest to a slow LLM, and learns from
// every answer it gets back. Drawn with spryte. Messages are BANKING77 test queries (CC BY 4.0, PolyAI): twelve card
// intents close enough to each other that a gut feeling needs help at first.
import { createRenderer } from "spryte";
import { createWorkerDecider, withFallback } from "../../src/browser.ts";
import messages from "./messages.json" with { type: "json" };

const LABELS = ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card", "card_not_working", "declined_card_payment", "card_payment_fee_charged", "exchange_rate", "top_up_failed", "pending_transfer", "cash_withdrawal_charge", "activate_my_card", "change_pin"] as const;
type Label = (typeof LABELS)[number];
const NAMES = Object.fromEntries(LABELS.map((l) => [l, l.replace(/_/g, " ")])) as Record<Label, string>;
const W = 960, H = 540;
const C = {
  bg: "#101114", panel: "#1a1c21", line: "#2a2d34", text: "#e8e8ea", muted: "#8a8d96",
  gut: "#cffb2d", llm: "#ff7ab6", wrong: "#ff5c5c",
};
const GUT = { x: 330, y: 170 }, LLM = { x: 330, y: 400 }, BIN_X = 640;
const binY = (i: number) => 66 + i * 36;

// --- state -----------------------------------------------------------------------------------------------------
type Dot = {
  text: string; label: Label | null; idx: number; again?: boolean; x: number; y: number; tx: number; ty: number; speed: number;
  phase: "in" | "gut" | "llm" | "thinking" | "out" | "done"; colour: string; bin?: number; wait?: number;
};
const dots: Dot[] = [];
const binned: { colour: string }[][] = LABELS.map(() => []);
const sparks: { x: number; y: number; tx: number; ty: number; t: number }[] = [];
let local = 0, sent = 0, right = 0, total = 0, labelled = 0, learned = 0, firstLocal = 0, againLocal = 0, againTotal = 0, replayed = false, llmMode: "live" | "simulated" = "live";
const recent: number[] = []; // 1 = sent to the LLM, per message, for the chart
let status = "Loading the model (first visit: about 36 MB)...";
let last1: { text: string; top: [string, number][]; confidence: number } | null = null;
let speed = 1, next = 0, paused = false;

// --- decisions -------------------------------------------------------------------------------------------------
async function askLlm(text: string, labels: readonly Label[], truth: Label | null): Promise<Label | null> {
  if (llmMode === "live") {
    try {
      const r = await fetch("/api/decidejs/ask", { method: "POST", body: JSON.stringify({ text, labels }) });
      if (r.ok) return (await r.json()).label;
    } catch {}
    llmMode = "simulated"; // no endpoint (local run) or over its limit
  }
  await new Promise((r) => setTimeout(r, 900));
  return truth;
}

const truthOf = new Map<string, Label | null>();
// The worker bundle sits one level up, next to the playground's.
const d = await createWorkerDecider(
  { onProgress: ({ loaded, total }) => (status = `Downloading the model: ${Math.round((100 * loaded) / total)}%`) },
  new Worker(new URL("../worker.js", import.meta.url), { type: "module" }),
);
const route = withFallback(d, LABELS, {
  minConfidence: 0.9,
  ask: (text, labels) => askLlm(text, labels, truthOf.get(text) ?? null),
  retrainEvery: 8,
});
status = "";

async function decide(dot: Dot) {
  truthOf.set(dot.text, dot.label);
  const t0 = performance.now();
  const r = await route(dot.text);
  total++;
  const top = (Object.entries(r.probs) as [Label, number][]).sort((a, b) => b[1] - a[1]);
  last1 = { text: dot.text, top: top.slice(0, 3).map(([l, p]) => [NAMES[l], p]), confidence: r.confidence };
  const viaLlm = r.source === "fallback";
  recent.push(viaLlm ? 1 : 0);
  if (viaLlm) {
    sent++;
    dot.phase = "llm";
    Object.assign(dot, { tx: LLM.x, ty: LLM.y, speed: 0.05 });
    dot.wait = performance.now() + Math.max(0, 900 - (performance.now() - t0)); // the LLM lane always looks slow
  } else {
    local++;
    dot.phase = "out";
    if (!dot.again && dot.idx < 20) firstLocal++;
    if (dot.again) againLocal++;
  }
  if (dot.again) againTotal++;
  const choice = (r.choice ?? top[0][0]) as Label;
  const ok = dot.label === null || choice === dot.label;
  if (dot.label !== null) {
    labelled++;
    if (ok) right++;
  }
  dot.bin = LABELS.indexOf(choice);
  dot.colour = viaLlm ? C.llm : ok ? C.gut : C.wrong;
  if (!viaLlm) Object.assign(dot, { tx: BIN_X, ty: binY(dot.bin), speed: 0.25 });
}

// --- input -----------------------------------------------------------------------------------------------------
function spawn(text: string, label: Label | null, idx = -1, again = false) {
  const dot: Dot = { text, label, idx, again, x: -40, y: 40 + Math.random() * 40, tx: GUT.x - 140, ty: GUT.y, speed: 0.08, phase: "in", colour: C.muted };
  dots.push(dot);
}
let queue = 0;
document.getElementById("say")!.addEventListener("submit", (e) => {
  e.preventDefault();
  const input = document.getElementById("msg") as HTMLInputElement;
  if (input.value.trim()) spawn(input.value.trim(), null);
  input.value = "";
});
document.getElementById("speed")!.addEventListener("click", (e) => {
  speed = speed === 1 ? 4 : 1;
  (e.target as HTMLButtonElement).textContent = speed === 1 ? "Faster" : "Slower";
});
document.getElementById("pause")!.addEventListener("click", (e) => {
  paused = !paused;
  (e.target as HTMLButtonElement).textContent = paused ? "Play" : "Pause";
});

await Promise.all(["400 12px Geist", "600 16px Geist", "12px 'Geist Mono'"].map((f) => document.fonts.load(f))).catch(() => {});

// --- drawing ---------------------------------------------------------------------------------------------------
const canvas = document.getElementById("c") as HTMLCanvasElement;
const draw = createRenderer(canvas);
function fit() {
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(canvas.clientWidth * scale);
  canvas.height = Math.round((canvas.clientWidth * H) / W * scale);
  draw.resize(canvas.width, canvas.height);
}
addEventListener("resize", fit);
fit();

const pill = (x: number, y: number, w: number, h: number, colour: string) => {
  draw.rect(x + h / 2, y, w - h, h, colour);
  draw.circle(x + h / 2, y + h / 2, h / 2, colour);
  draw.circle(x + w - h / 2, y + h / 2, h / 2, colour);
};
const clip = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

function box(x: number, y: number, title: string, sub: string, colour: string, pulse = 0) {
  if (pulse) draw.circle(x, y, 58 + pulse * 6, colour + "22");
  draw.circle(x, y, 52, C.panel);
  draw.circle(x, y, 46, colour + "33");
  draw.text(title, x, y - 8, { font: "600 16px Geist", color: C.text, align: "center", baseline: "middle" });
  draw.text(sub, x, y + 12, { font: "11px Geist", color: C.muted, align: "center", baseline: "middle" });
}

let last = performance.now();
function frame(now: number) {
  const dt = Math.min(50, now - last);
  last = now;
  if (!paused && !status) {
    next -= dt * speed;
    if (next <= 0 && queue < messages.length) {
      const m = messages[queue] as { text: string; label: Label };
      spawn(m.text, m.label, queue++);
      next = 900;
    } else if (next <= 0 && !replayed && !dots.length) {
      // Same first 20 messages, with everything learned since.
      replayed = true;
      for (let i = 0; i < 20; i++) setTimeout(() => spawn(messages[i].text, messages[i].label as Label, i, true), (i * 600) / speed);
    }
  }
  const k = speed * dt / 16;
  for (const dot of dots) {
    dot.x += (dot.tx - dot.x) * Math.min(1, dot.speed * k);
    dot.y += (dot.ty - dot.y) * Math.min(1, dot.speed * k);
    const close = Math.hypot(dot.tx - dot.x, dot.ty - dot.y) < 4;
    if (dot.phase === "in" && close) { dot.phase = "gut"; void decide(dot); }
    else if (dot.phase === "llm" && close) dot.phase = "thinking";
    else if (dot.phase === "thinking" && now >= (dot.wait ?? 0)) {
      dot.phase = "out";
      Object.assign(dot, { tx: BIN_X, ty: binY(dot.bin!), speed: 0.12 });
      sparks.push({ x: LLM.x, y: LLM.y, tx: GUT.x, ty: GUT.y, t: 0 });
    } else if (dot.phase === "out" && close) {
      dot.phase = "done";
      binned[dot.bin!].push({ colour: dot.colour });
    }
  }
  for (let i = dots.length - 1; i >= 0; i--) if (dots[i].phase === "done") dots.splice(i, 1);
  for (const s of sparks) s.t += 0.02 * k;
  for (let i = sparks.length - 1; i >= 0; i--) if (sparks[i].t >= 1) { sparks.splice(i, 1); learned++; }

  draw.begin(C.bg, W, H, canvas.width / W);
  // lanes
  draw.line(160, GUT.y, GUT.x, GUT.y, 2, C.line);
  draw.line(GUT.x, GUT.y, BIN_X - 40, H / 2, 2, C.gut + "55");
  draw.line(GUT.x, GUT.y + 52, LLM.x, LLM.y - 52, 2, C.llm + "55");
  draw.line(LLM.x, LLM.y, BIN_X - 40, H / 2, 2, C.llm + "33");
  const thinking = dots.some((d) => d.phase === "thinking");
  box(GUT.x, GUT.y, "decidejs", "gut: ~5 ms, free", C.gut);
  box(LLM.x, LLM.y, "LLM", "thinks: ~1 s, per call", C.llm, thinking ? (Math.sin(now / 120) + 1) / 2 : 0);
  draw.text(`learned ${learned}`, GUT.x, GUT.y - 70, { font: "13px Geist", color: C.gut, align: "center" });
  // the gut's last call: how sure it was, against the line where it asks for help
  if (last1) {
    const gx = 40, gy = 250, gw = 200, sure = last1.confidence >= 0.9;
    draw.text("last message", gx, gy, { font: "11px Geist", color: C.muted, baseline: "middle" });
    draw.text(clip(last1.text, 30), gx, gy + 18, { font: "12px Geist", color: C.text, baseline: "middle" });
    last1.top.forEach(([name, p], i) => {
      const y = gy + 36 + i * 22, colour = i ? C.muted : sure ? C.gut : C.llm;
      draw.text(clip(name, 26), gx, y, { font: "11px Geist", color: i ? C.muted : C.text, baseline: "middle" });
      draw.text(`${Math.round(p * 100)}%`, gx + gw, y, { font: "11px 'Geist Mono'", color: colour, align: "right", baseline: "middle" });
      draw.rect(gx, y + 8, gw, 5, C.panel);
      draw.rect(gx, y + 8, gw * p, 5, colour);
    });
    draw.rect(gx + gw * 0.9, gy + 40, 1, 66, C.text + "88");
    draw.text("90%", gx + gw * 0.9, gy + 112, { font: "10px 'Geist Mono'", color: C.muted, align: "center", baseline: "middle" });
    draw.text(sure ? "sure: answers itself" : "torn: asks the LLM", gx, gy + 126, { font: "12px Geist", color: sure ? C.gut : C.llm, baseline: "middle" });
  }
  // bins
  LABELS.forEach((l, i) => {
    const y = binY(i);
    draw.text(NAMES[l], BIN_X, y - 9, { font: "12px Geist", color: C.muted, baseline: "middle" });
    binned[i].slice(-30).forEach((b, j) => draw.circle(BIN_X + 5 + j * 10, y + 7, 4, b.colour));
  });
  // learning sparks, LLM to decidejs
  for (const s of sparks) {
    const x = s.x + (s.tx - s.x) * s.t, y = s.y + (s.ty - s.y) * s.t;
    draw.circle(x - 40 * Math.sin(Math.PI * s.t), y, 5, C.gut);
  }
  // messages
  for (const dot of dots) {
    if (dot.phase === "in" || dot.phase === "gut") {
      const text = clip(dot.text, 26), w = draw.measure(text, "12px Geist").w + 18;
      pill(dot.x - w / 2, dot.y - 11, w, 22, C.panel);
      draw.text(text, dot.x, dot.y, { font: "12px Geist", color: C.text, align: "center", baseline: "middle" });
    } else draw.circle(dot.x, dot.y, 6, dot.colour);
  }
  // counters and chart
  const share = total ? Math.round((100 * sent) / total) : 0;
  draw.text(`${total} message${total === 1 ? "" : "s"} · ${local} answered on device · ${sent} sent to the LLM (${share}%) · ${labelled ? Math.round((100 * right) / labelled) : 0}% right`, 20, 26, { font: "14px Geist", color: C.text });
  draw.text("Sent to the LLM, last 20 messages", 20, 462, { font: "12px Geist", color: C.muted, baseline: "middle" });
  draw.rect(20, 476, 220, 48, C.panel);
  const pts: number[] = [];
  for (let i = 20; i <= recent.length; i += 2) pts.push(recent.slice(i - 20, i).reduce((a, b) => a + b, 0) / 20);
  pts.slice(-55).forEach((p, i, a) => i && draw.line(20 + (i - 1) * 4, 520 - a[i - 1] * 40, 20 + i * 4, 520 - p * 40, 2, C.llm));
  draw.text("100%", 244, 480, { font: "10px Geist", color: C.muted, baseline: "middle" });
  draw.text("0%", 244, 520, { font: "10px Geist", color: C.muted, baseline: "middle" });
  // what it saved
  draw.text("LLM calls", 300, 476, { font: "11px Geist", color: C.muted, baseline: "middle" });
  draw.text(`${sent} of ${total}`, 300, 496, { font: "600 18px Geist", color: C.llm, baseline: "middle" });
  draw.text(total ? `${Math.round((100 * local) / total)}% fewer than LLM-only` : "", 300, 516, { font: "11px Geist", color: C.gut, baseline: "middle" });
  // before and after: the first 20 messages again
  if (replayed) {
    draw.text("The first 20 messages again", 430, 476, { font: "11px Geist", color: C.muted, baseline: "middle" });
    draw.text(`at the start: ${firstLocal}/20 on device`, 430, 496, { font: "13px Geist", color: C.text, baseline: "middle" });
    draw.text(`now: ${againLocal}/${againTotal} on device`, 430, 516, { font: "600 13px Geist", color: C.gut, baseline: "middle" });
  }
  if (llmMode === "simulated") draw.text("LLM simulated here: the true label stands in", 20, 48, { font: "11px Geist", color: C.llm, baseline: "middle" });
  if (status) draw.text(status, W / 2, H / 2, { font: "16px Geist", color: C.text, align: "center" });
  draw.end();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
