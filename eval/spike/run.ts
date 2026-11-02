// One candidate, one measurement. Appends a line per (dataset, k, seed) to eval/spike/results.jsonl.
//   node --experimental-strip-types eval/spike/run.ts <id> '<json cfg>' [datasets=banking77,clinc150] [ks=5,10,all] [split=dev]
// cfg {"solver":"base", ...TrainOptions} calls the SGD path of src/head.ts, which was the whole of it during the spike; otherwise eval/spike/fit.ts.
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { fitHead, logits, softmax } from "../../src/head.ts";
import { dataset, ece, embeddings, kShot, readable, type Split } from "./lib.ts";
import { fitCandidate } from "./fit.ts";

const [id, cfgS, ds = "banking77,clinc150", ks = "5,10,all", split = "dev"] = process.argv.slice(2);
const cfg = JSON.parse(cfgS);
const model = process.env.MODEL ?? "debuffjp/decide-small";
const E = await embeddings(model);
const FLOOR = Number(process.env.FLOOR ?? (model.includes("decide-small") ? 0.5 : 1));
mkdirSync("data/spike/preds", { recursive: true });
for (const name of ds.split(",")) {
  const data = await dataset(name, split as Split, id);
  for (const kk of ks.split(",")) {
    for (const seed of kk === "all" ? [7] : [7, 8, 9]) {
      const rows = kk === "all" ? data.train : kShot(data.train, Number(kk), seed);
      const labels = [...new Set(rows.map((r) => r.label))].sort();
      const X = rows.map((r) => E.vec(r.text));
      const y = rows.map((r) => labels.indexOf(r.label));
      const pri = cfg.labelNames === false ? null : labels.map((l) => E.vec(readable(l)));
      const t0 = performance.now();
      let head: { weights: Float32Array; bias: Float32Array; temperature: number }, info: unknown;
      if (cfg.solver === "base" || rows.length / labels.length < (cfg.fewBelow ?? 0)) {
        const { solver, centroidScale, fewBelow, prior, bestEpoch, iters, lr, ...o } = cfg.solver === "base" ? cfg : { solver: 0 };
        head = await fitHead(X, y, labels, { solver: "sgd", seed, ...o }, pri ?? undefined, 1, centroidScale, FLOOR);
      } else {
        const fx = new Float32Array(rows.length * E.dim);
        X.forEach((x, i) => fx.set(x, i * E.dim));
        const fp = pri ? new Float32Array(labels.length * E.dim) : null;
        pri?.forEach((x, i) => fp!.set(x, i * E.dim));
        const r = fitCandidate(fx, E.dim, Int32Array.from(y), labels.length, cfg, seed, fp, FLOOR);
        head = r;
        info = r.info;
      }
      const seconds = (performance.now() - t0) / 1000;
      const h = { labels, dim: E.dim, weights: head.weights, bias: head.bias };
      let nll = 0;
      const right = new Uint8Array(data.eval.length);
      const s = data.eval.map((r, n) => {
        const pr = softmax(logits(h, E.vec(r.text)), head.temperature);
        let b = 0;
        for (let c = 1; c < pr.length; c++) if (pr[c] > pr[b]) b = c;
        right[n] = labels[b] === r.label ? 1 : 0;
        const at = labels.indexOf(r.label);
        nll -= Math.log((at >= 0 ? pr[at] : 0) + 1e-12);
        return { right: labels[b] === r.label, confidence: pr[b] };
      });
      writeFileSync(`data/spike/preds/${id}.${name}.${kk}.${seed}.${split}.bin`, right);
      const out = { id, model, cfg, dataset: name, k: kk, seed, split, n: s.length, errors: s.filter((x) => !x.right).length, accuracy: +(s.filter((x) => x.right).length / s.length).toFixed(5), ece: +ece(s).toFixed(4), nll: +(nll / s.length).toFixed(4), temperature: head.temperature, seconds: +seconds.toFixed(1), info };
      appendFileSync("eval/spike/results.jsonl", JSON.stringify(out) + "\n");
      console.log(JSON.stringify({ ...out, cfg: undefined, info: undefined, model: undefined }));
    }
  }
}
