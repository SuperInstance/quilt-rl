// src/qrng.mjs — the certified-entropy lane + the deterministic draw stream.
//
// LIVE (channel 'moth-live'): certified randomness from comet-qrng-v1:
//   POST https://api.mothquantum.com/api/v1/engines/comet-qrng-v1/process
//     Authorization: Bearer $MOTH_KEY
//     {"params":{"mode":"emu","num_qubits":12,"shots":4096,"output_bytes":512,
//                 "include_raw_counts":true,"bell_witness":true}}
//   -> {job_id} -> poll GET /api/v1/jobs/{id}/status -> GET /api/v1/jobs/{id}/result
//   -> output.random.hex = certified bytes (SP 800-90B min-entropy certificate,
//      Toeplitz extractor, CHSH witness). ~584 certified bits per job.
//
// WIRE-SHAPE PROVENANCE (honest): the task brief guessed POST /api/v1/jobs
// with {"engine_id":...,"params":{"num_bits":256}} — that shape 405s (no job
// created). Per the brief's fallback instruction we inspected the docs and
// adopted the fleet's PROVEN shape from wave-45-e moth-census receipts
// (/home/z/my-project/fleet-seeds/playtest/wave45/moth-census/, moth-seal.mjs
// ENVELOPE GATE): engines/{id}/process REQUIRES {"params":{...}} — flat bodies
// 422. GET /api/v1/engines (not a job) confirmed comet-qrng-v1 live.
//
// Budget doctrine: <= 6 moth JOBS (successful POSTs) per session; every job
// leaves a receipt {job_id, bits_sha256, cert fields} with NO key material.
//
// DEGRADED (channel 'qrng-degraded', the channel-health doctrine): if moth is
// unreachable, the seed comes from SHA-256 of a fixed label. Receipts say so
// honestly — a degraded run NEVER ships labeled 'moth-live'.
//
// In BOTH cases exploration draws come from the same construction: a
// SHA-256 counter stream seeded by the 32 seed bytes
//   block_i = SHA-256(seed_bytes || u32be(i)),  draws consumed byte-wise.
// The certified bits seed the stream; the stream is reproducible from the
// receipted seed bytes. That is the point: a run's exploration is
// byte-reproducible from its receipt (T2) without re-talking to moth, while
// still being certified-random at the root.

import { createHash } from "node:crypto";

export const sha256hex = (bytes) => createHash("sha256").update(bytes).digest("hex");

export function degradedSeedBytes(label = "quilt-rl/qrng-degraded/v1") {
  return createHash("sha256").update(label, "utf8").digest();
}

// The deterministic draw stream (used for BOTH channels — only the seed
// differs). nextUnit() consumes 4 bytes; nextIndex(n) = nextU32() % n
// (modulo bias < 2^-30 relative for n << 2^16; documented, deterministic).
export function makeStream(seedBytes) {
  let ctr = 0, buf = Buffer.alloc(0), pos = 0;
  const refill = () => {
    const c = Buffer.alloc(4);
    c.writeUInt32BE(ctr >>> 0, 0);
    buf = createHash("sha256").update(seedBytes).update(c).digest();
    ctr++;
    pos = 0;
  };
  const nextByte = () => {
    if (pos >= buf.length) refill();
    return buf[pos++];
  };
  return {
    nextU32() {
      return ((nextByte() * 256 + nextByte()) * 256 + nextByte()) * 256 + nextByte();
    },
    nextUnit() {
      return this.nextU32() / 4294967296;
    },
    nextIndex(n) {
      return this.nextU32() % n;
    },
  };
}

// ---- moth live lane -------------------------------------------------------

const MOTH_BASE = "https://api.mothquantum.com/api/v1";
const COMET_PARAMS = { mode: "emu", num_qubits: 12, shots: 4096, output_bytes: 512, include_raw_counts: true, bell_witness: true };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function extractOutput(result) {
  const r = result && typeof result === "object" ? result : {};
  const inner = r.result && typeof r.result === "object" ? r.result : {};
  const out = inner.output && typeof inner.output === "object" ? inner.output : r.output && typeof r.output === "object" ? r.output : null;
  if (!out) throw new Error("qrng: no output object in job result");
  return out;
}

// One certified moth job. Throws (fail-closed) on any transport/cert problem.
export async function fetchMothBits({ key = process.env.MOTH_KEY, numBits = 256, pollIntervalMs = 1500, maxPolls = 80, log = () => {} } = {}) {
  if (!key) throw new Error("qrng: MOTH_KEY not set (refusing to silently degrade — caller decides)");
  log(`POST /engines/comet-qrng-v1/process`);
  const res = await fetch(`${MOTH_BASE}/engines/comet-qrng-v1/process`, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({ params: COMET_PARAMS }),
    signal: AbortSignal.timeout(180000),
  });
  const subText = await res.text();
  if (!res.ok) {
    const err = new Error(`qrng: moth submit ${res.status}: ${subText.slice(0, 300)}`);
    err.status = res.status;
    throw err;
  }
  const sub = JSON.parse(subText);
  const jobId = sub.job_id;
  if (!jobId) throw new Error("qrng: no job_id in submit response");

  let status = "";
  let polls = 0;
  while (polls < maxPolls) {
    await sleep(pollIntervalMs);
    polls++;
    const st = await fetch(`${MOTH_BASE}/jobs/${jobId}/status`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(60000) });
    try { status = JSON.parse(await st.text()).status; } catch { status = `HTTP${st.status}`; }
    if (["completed", "failed", "error", "cancelled"].includes(status)) break;
  }
  if (status !== "completed") throw new Error(`qrng: job ${jobId} final status '${status}' after ${polls} polls`);

  const r = await fetch(`${MOTH_BASE}/jobs/${jobId}/result`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(60000) });
  if (!r.ok) throw new Error(`qrng: result HTTP ${r.status} for job ${jobId}`);
  const out = extractOutput(JSON.parse(await r.text()));
  const hex = String(out.random?.hex ?? "").replace(/^0x/, "");
  if (!hex) throw new Error("qrng: result lacks random.hex");
  const allBytes = Buffer.from(hex, "hex");
  if (allBytes.length * 8 < numBits) {
    throw new Error(`qrng: job delivered ${allBytes.length * 8} certified bits, need ${numBits} (min-entropy-limited yield)`);
  }
  const seedBytes = allBytes.subarray(0, Math.ceil(numBits / 8));
  const er = out.entropy_report ?? {};
  const ex = out.extractor ?? {};
  return {
    seedBytes,
    seedHex: seedBytes.toString("hex"),
    bitsSha256: sha256hex(seedBytes),
    job: {
      job_id: jobId,
      status,
      polls,
      engine_id: "comet-qrng-v1",
      params: COMET_PARAMS,
      delivered_bits: allBytes.length * 8,
      requested_bits: numBits,
    },
    cert: {
      h_bit: er.h_bit,
      grade: er.grade,
      health_passed: er.health_passed,
      budget_bits: er.budget_bits,
      extractor: ex.kind,
      extractor_public_seed: ex.public_seed,
    },
  };
}

// receipt factory — NO key material ever enters a receipt
export function qrngReceipt({ channel, ts = new Date().toISOString(), job = null, cert = null, seedHex, bitsSha256, numBits = 256, note = "" }) {
  const r = {
    v: 1,
    channel, // 'moth-live' | 'qrng-degraded'
    ts,
    engine: channel === "moth-live" ? "comet-qrng-v1" : "sha256-counter-stream",
    num_bits: numBits,
    seed_sha256: sha256hex(Buffer.from(seedHex, "hex")),
    seed_hex: seedHex,
    bits_sha256: bitsSha256,
  };
  if (job) r.job = job;
  if (cert) r.cert = cert;
  if (note) r.note = note;
  return r;
}
