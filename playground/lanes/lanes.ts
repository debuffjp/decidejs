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
  text: string; label: Label; x: number; y: number; tx: number; ty: number; speed: number;
  phase: "in" | "gut" | "llm" | "thinking" | "out" | "done"; colour: string; bin?: number; wait?: number;
};
const dots: Dot[] = [];
const binned: { colour: string }[][] = LABELS.map(() => []);
const sparks: { x: number; y: number; tx: number; ty: number; t: number }[] = [];
let local = 0, sent = 0, right = 0, total = 0, learned = 0, llmMode: "live" | "simulated" = "live";
const recent: number[] = []; // 1 = sent to the LLM, per message, for the chart
let status = "Loading the model (first visit: about 36 MB)...";
let speed = 1, next = 0, paused = false;

// --- decisions -------------------------------------------------------------------------------------------------
async function askLlm(text: string, labels: readonly Label[], truth: Label): Promise<Label | null> {
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

const truthOf = new Map<string, Label>();
// The worker bundle sits one level up, next to the playground's.
const d = await createWorkerDecider(
  { onProgress: ({ loaded, total }) => (status = `Downloading the model: ${Math.round((100 * loaded) / total)}%`) },
  new Worker(new URL("../worker.js", import.meta.url), { type: "module" }),
);
const route = withFallback(d, LABELS, {
  minConfidence: 0.9,
  ask: (text, labels) => askLlm(text, labels, truthOf.get(text)!),
  retrainEvery: 8,
});
status = "";

async function decide(dot: Dot) {
  truthOf.set(dot.text, dot.label);
  const t0 = performance.now();
  const r = await route(dot.text);
  total++;
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
  }
  const choice = (r.choice ?? dot.label) as Label;
  const ok = choice === dot.label;
  if (ok) right++;
  dot.bin = LABELS.indexOf(choice);
  dot.colour = viaLlm ? C.llm : ok ? C.gut : C.wrong;
  if (!viaLlm) Object.assign(dot, { tx: BIN_X, ty: binY(dot.bin), speed: 0.25 });
}

// --- input -----------------------------------------------------------------------------------------------------
function spawn(text: string, label: Label) {
  const dot: Dot = { text, label, x: -40, y: 40 + Math.random() * 40, tx: GUT.x - 60, ty: GUT.y, speed: 0.08, phase: "in", colour: C.muted };
  dots.push(dot);
}
let queue = 0;
document.getElementById("speed")!.addEventListener("click", (e) => {
  speed = speed === 1 ? 4 : 1;
  (e.target as HTMLButtonElement).textContent = speed === 1 ? "Faster" : "Slower";
});
document.getElementById("pause")!.addEventListener("click", (e) => {
  paused = !paused;
  (e.target as HTMLButtonElement).textContent = paused ? "Play" : "Pause";
});

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
  draw.text(title, x, y - 8, { font: "600 16px system-ui", color: C.text, align: "center", baseline: "middle" });
  draw.text(sub, x, y + 12, { font: "11px system-ui", color: C.muted, align: "center", baseline: "middle" });
}

let last = performance.now();
function frame(now: number) {
  const dt = Math.min(50, now - last);
  last = now;
  if (!paused && !status) {
    next -= dt * speed;
    if (next <= 0 && queue < messages.length) {
      const m = messages[queue++] as { text: string; label: Label };
      spawn(m.text, m.label);
      next = 900;
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
  draw.text(`learned ${learned}`, GUT.x, GUT.y - 70, { font: "13px system-ui", color: C.gut, align: "center" });
  // bins
  LABELS.forEach((l, i) => {
    const y = binY(i);
    draw.text(NAMES[l], BIN_X, y - 9, { font: "12px system-ui", color: C.muted, baseline: "middle" });
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
      const text = clip(dot.text, 26), w = draw.measure(text, "12px system-ui").w + 18;
      pill(dot.x - w / 2, dot.y - 11, w, 22, C.panel);
      draw.text(text, dot.x, dot.y, { font: "12px system-ui", color: C.text, align: "center", baseline: "middle" });
    } else draw.circle(dot.x, dot.y, 6, dot.colour);
  }
  // counters and chart
  const share = total ? Math.round((100 * sent) / total) : 0;
  draw.text(`${total} messages · ${local} answered on device · ${sent} sent to the LLM (${share}%) · ${total ? Math.round((100 * right) / total) : 0}% right`, 20, 26, { font: "14px system-ui", color: C.text });
  draw.text("Sent to the LLM, last 20 messages", 20, 462, { font: "12px system-ui", color: C.muted, baseline: "middle" });
  draw.rect(20, 476, 220, 48, C.panel);
  const pts: number[] = [];
  for (let i = 20; i <= recent.length; i += 2) pts.push(recent.slice(i - 20, i).reduce((a, b) => a + b, 0) / 20);
  pts.slice(-55).forEach((p, i, a) => i && draw.line(20 + (i - 1) * 4, 520 - a[i - 1] * 40, 20 + i * 4, 520 - p * 40, 2, C.llm));
  draw.text("100%", 244, 480, { font: "10px system-ui", color: C.muted, baseline: "middle" });
  draw.text("0%", 244, 520, { font: "10px system-ui", color: C.muted, baseline: "middle" });
  if (llmMode === "simulated") draw.text("LLM simulated here: the true label stands in", 20, 446, { font: "11px system-ui", color: C.llm, baseline: "middle" });
  if (status) draw.text(status, W / 2, H / 2, { font: "16px system-ui", color: C.text, align: "center" });
  draw.end();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
