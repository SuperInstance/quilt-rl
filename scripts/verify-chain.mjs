#!/usr/bin/env node
// scripts/verify-chain.mjs — verify the SHIPPED receipts/chain-live.jsonl
// against the receipted seed (no network). CI step: proves the chain on disk
// is intact and its tip matches receipts/convergence-live.json.
import { readFileSync } from "node:fs";
import { Buffer } from "node:buffer";
import { verifyWhy } from "../src/qlearn.mjs";
import { certifyConvergence } from "../src/shape.mjs";
import { sha256hex } from "../src/qrng.mjs";

const seedMeta = JSON.parse(readFileSync(new URL("../seeds/live-seed-001.json", import.meta.url), "utf8"));
const chain = readFileSync(new URL("../receipts/chain-live.jsonl", import.meta.url), "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l));
const cert = JSON.parse(readFileSync(new URL("../receipts/convergence-live.json", import.meta.url), "utf8"));

const v = verifyWhy(chain);
if (!v.ok) {
  console.error(`CHAIN TAMPERED: ${v.why}`);
  process.exit(1);
}
if (v.tip !== cert.chain_tip) {
  console.error(`TIP MISMATCH: chain ${v.tip} vs certificate ${cert.chain_tip}`);
  process.exit(1);
}
// the shipped seed must be the one the certificate was computed from
if (sha256hex(Buffer.from(seedMeta.seed_hex, "hex")) !== cert.seed_sha256) {
  console.error("SEED MISMATCH: seeds/live-seed-001.json does not match the certificate");
  process.exit(1);
}
// re-certify the return series from the chain itself
const returns = chain.map((e) => e.return);
const final = certifyConvergence(returns, { window: 100, tolerance: 0.01, bar: cert.bar });
if (!final.converged) {
  console.error(`RE-CERTIFICATION REFUSED: ${final.refused} (mean ${final.window_mean} vs bar ${cert.bar})`);
  process.exit(1);
}
console.log(`CHAIN VALID: ${chain.length} entries, tip ${v.tip}`);
console.log(`CONVERGENCE RE-CERTIFIED: mean_last100 ${final.window_mean} >= bar ${cert.bar}, flat tail ok (maxDev ${final.flat_tail.maxDev})`);
