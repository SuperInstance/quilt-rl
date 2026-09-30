#!/usr/bin/env node
// scripts/live-run.mjs — the LIVE certified run (never run by tests).
//
//   node scripts/live-run.mjs live    # 1 moth job -> seed -> 1500-ep run -> receipts/
//   node scripts/live-run.mjs replay  # offline reproduction from seeds/live-seed-001.json (no network)
//
// Budget doctrine: ONE moth job (POST) per invocation. Receipts carry
// job_id + bits_sha256, never key material.
import { writeFileSync, readFileSync, existsSync } from "node:fs";
import { Buffer } from "node:buffer";
import { fetchMothBits, degradedSeedBytes, qrngReceipt, sha256hex } from "../src/qrng.mjs";
import { train, verifyWhy, epsilonSchedule, alphaSchedule, gammaSchedule } from "../src/qlearn.mjs";
import { certifyConvergence, firstConvergenceEpisode } from "../src/shape.mjs";

const mode = process.argv[2] ?? "live";
const EPISODES = 1500;
const WINDOW = 100;
const TOLERANCE = 0.01;

let channel, seedHex, receipt;
if (mode === "live") {
  if (!process.env.MOTH_KEY) throw new Error("live: MOTH_KEY not set");
  console.error("[live] POSTing ONE moth job: comet-qrng-v1, 256 bits ...");
  const res = await fetchMothBits({ numBits: 256 });
  channel = "moth-live";
  seedHex = res.seedHex;
  receipt = qrngReceipt({
    channel, job: res.job, cert: res.cert, seedHex, bitsSha256: res.bitsSha256,
    note: "1 earlier POST to /api/v1/jobs (brief's guessed shape) returned 405 — no job created, no credits; adopted fleet wave-45-e proven shape engines/comet-qrng-v1/process",
  });
  console.error(`[live] job ${res.job.job_id} status=${res.job.status} bits_sha256=${res.bitsSha256}`);
  writeFileSync(new URL("../receipts/qrng-job-001.json", import.meta.url), JSON.stringify(receipt, null, 2) + "\n");
  writeFileSync(
    new URL("../seeds/live-seed-001.json", import.meta.url),
    JSON.stringify({ v: 1, channel, seed_hex: seedHex, bits_sha256: res.bitsSha256, job: res.job, note: "seed bytes are the run's root of determinism — publish them so the chain reproduces offline; they are random data, not a credential" }, null, 2) + "\n"
  );
} else if (mode === "replay") {
  const f = new URL("../seeds/live-seed-001.json", import.meta.url);
  if (!existsSync(f)) throw new Error("replay: seeds/live-seed-001.json missing");
  const meta = JSON.parse(readFileSync(f, "utf8"));
  channel = meta.channel;
  seedHex = meta.seed_hex;
  receipt = { ...meta, replayed: true };
} else {
  throw new Error(`unknown mode ${mode}`);
}

const seedBytes = Buffer.from(seedHex, "hex");
console.error(`[run] channel=${channel} seed_sha256=${sha256hex(seedBytes)} episodes=${EPISODES}`);
const t0 = Date.now();
const { chain, optimalReturn, optimalSteps, gammaReport } = train({ episodes: EPISODES, seedBytes });
const wallMs = Date.now() - t0;

const v = verifyWhy(chain);
if (!v.ok) throw new Error(`chain failed self-verification: ${v.why}`);
const returns = chain.map((e) => e.return);
const last100 = returns.slice(-WINDOW);
const meanLast100 = last100.reduce((a, b) => a + b, 0) / WINDOW;
const bar = 0.95 * optimalReturn;
const pin = firstConvergenceEpisode(returns, { window: WINDOW, tolerance: TOLERANCE, bar });
const finalCert = certifyConvergence(returns, { window: WINDOW, tolerance: TOLERANCE, bar });

const convergence = {
  v: 1,
  channel,
  seed_sha256: sha256hex(seedBytes),
  episodes: EPISODES,
  gamma_report: gammaReport,
  optimal_steps: optimalSteps,
  optimal_return: optimalReturn,
  bar: bar,
  mean_last_100: meanLast100,
  shortfall_vs_optimal: optimalReturn - meanLast100,
  t_conv: pin,
  final_certificate: finalCert,
  chain_tip: v.tip,
  chain_verified: v.ok,
  wall_ms: wallMs,
  schedules: { epsilon: "max(0, 1 - ep/600)", alpha: "max(0.1, 0.5 - ep*0.4/500)", gamma: "min(0.98, 0.9 + ep*0.08/500)" },
};

writeFileSync(
  new URL("../receipts/chain-live.jsonl", import.meta.url),
  chain.map((e) => JSON.stringify(e)).join("\n") + "\n"
);
writeFileSync(
  new URL("../receipts/convergence-live.json", import.meta.url),
  JSON.stringify(convergence, null, 2) + "\n"
);

console.log(JSON.stringify({
  channel,
  episodes: EPISODES,
  optimal_return: optimalReturn,
  bar,
  mean_last_100: meanLast100,
  pinned: pin.pinned,
  pinned_at_episode: pin.episode,
  chain_tip: v.tip,
  wall_ms: wallMs,
}, null, 2));
