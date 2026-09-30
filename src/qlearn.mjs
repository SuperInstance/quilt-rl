// src/qlearn.mjs — tabular Q-learning over sheet states, with a sha256-linked
// per-episode receipt chain.
//
// WHY THIS CONVERGES WHERE D13 DID NOT (see README for the full receipts):
//   D13/D13b/D13c were gradient-free or underscaled online updates chasing a
//   NON-STATIONARY target: a per-sender preference updated by an asymmetric
//   Hebbian rule (decay 0.98 vs boost 0.02 — net negative pressure), or a
//   REINFORCE logit getting ~32 reward samples per edge over a 0.5 signal
//   buried in coin-flip noise. D18 later showed reward learning DOES work
//   when the relational target is consistent. The countermeasures here:
//   1. STATIONARY MDP — fixed grid, fixed reward, fixed transitions; the Q
//      target r + gamma*maxQ(s') is consistent (D18's precondition).
//   2. SCALED off-policy TD update — Q(s,a) += alpha*(target - Q(s,a)),
//      alpha in [0.1, 0.5]; no decay/boost asymmetry, bootstrapped credit.
//   3. SAMPLE ABUNDANCE — 1500 episodes x <=100 steps over a 16-state x
//      4-action table (64 numbers) = thousands of updates per cell.
//   4. PINNED EXPLORATION — epsilon-greedy, epsilon annealed 1 -> 0 from a
//      certified-random stream; NC1 proves epsilon=0 provably stalls.
//   5. PINNED CONVERGENCE — T-CONV (shape.mjs) certifies the return series
//      numerically; the receipt chain makes the claim tamper-evident.

import { createHash } from "node:crypto";
import { makeSheetEnv, ACTIONS } from "./sheet-env.mjs";
import { makeStream } from "./qrng.mjs";

// ---- canonical bytes ------------------------------------------------------

// recursively key-sorted JSON — the canonical form every hash commits to
export function canon(v) {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(canon).join(",") + "]";
  const keys = Object.keys(v).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + canon(v[k])).join(",") + "}";
}

export function f64hex(x) {
  if (x === 0) x = 0; // normalize -0 -> +0
  const b = Buffer.alloc(8);
  b.writeDoubleBE(x, 0);
  return b.toString("hex");
}

export const sha256hex = (s) => createHash("sha256").update(s, "utf8").digest("hex");

// content root of the Q table: every state's every action value, f64 big-endian
export function qRootSha(Q) {
  const parts = [];
  for (const k of [...Q.keys()].sort()) {
    parts.push(`${k}=${Q.get(k).map(f64hex).join(",")}`);
  }
  return sha256hex(parts.join(";"));
}

export const GENESIS = "0".repeat(64);

// ---- pinned schedules (spec'd in TEST-RECEIPT.md; floors are load-bearing) -

export function epsilonSchedule(ep) { return Math.max(0, 1 - ep / 600); }        // 1 -> 0 by ep 600
export function alphaSchedule(ep) { return Math.max(0.1, 0.5 - ep * (0.4 / 500)); } // 0.5 -> 0.1 floor
export function gammaSchedule(ep) { return Math.min(0.98, 0.9 + ep * (0.08 / 500)); } // 0.90 -> 0.98 ceiling
// NOTE on gamma: spec said "alpha/gamma decays". alpha decays as specified.
// gamma is annealed UPWARD 0.90 -> 0.98 (documented deviation): a shrinking
// horizon would progressively break 6-step credit assignment on the very
// task the predicate certifies; a bounded rising gamma keeps the target
// consistent while values stabilize.

// ---- the learner ----------------------------------------------------------

export function makeQ() {
  return new Map(); // "x,y" -> [q_up, q_down, q_left, q_right] (ACTIONS order)
}

function qRow(Q, s) {
  let row = Q.get(s);
  if (!row) { row = [0, 0, 0, 0]; Q.set(s, row); }
  return row;
}

export function argmaxLegal(row, legal) {
  let best = null, bestV = -Infinity;
  for (const a of legal) {
    const i = ACTIONS.indexOf(a);
    if (row[i] > bestV) { bestV = row[i]; best = a; } // ties -> first in ACTIONS order (deterministic)
  }
  return best;
}

export function chooseAction(Q, stateKey, legal, stream, epsilon) {
  const u = stream.nextUnit(); // one stream draw per step, ALWAYS consumed (stream alignment invariant)
  if (u < epsilon) return legal[stream.nextIndex(legal.length)];
  return argmaxLegal(qRow(Q, stateKey), legal);
}

export function qUpdate(Q, s, a, r, s2, done2, legal2, alpha, gamma) {
  const row = qRow(Q, s);
  const i = ACTIONS.indexOf(a);
  // target = r + gamma * max over s2's LEGAL actions (router-masked), 0 at terminal
  const nextBest = done2 ? 0 : Math.max(...legal2.map((x) => qRow(Q, s2)[ACTIONS.indexOf(x)]));
  const target = r + gamma * nextBest;
  row[i] += alpha * (target - row[i]);
}

// ---- training run ---------------------------------------------------------

export function train({
  episodes = 1500,
  seedBytes,                    // 32 bytes (from moth bits or degraded stream)
  envOpts = {},
  gammaReport = 0.9,            // FIXED reporting discount (independent of the annealed learning gamma)
  schedules = { epsilon: epsilonSchedule, alpha: alphaSchedule, gamma: gammaSchedule },
  maxSteps = 100,
} = {}) {
  if (!seedBytes || seedBytes.length !== 32) throw new Error("qlearn: seedBytes must be 32 bytes");
  const env = makeSheetEnv(envOpts);
  const Q = makeQ();
  const chain = [];
  let prev = GENESIS;
  const stream = makeStream(seedBytes); // ONE stream across the whole run
  const optimalReturn = env.goalReward * Math.pow(gammaReport, env.optimalSteps - 1);

  for (let ep = 0; ep < episodes; ep++) {
    const epsilon = schedules.epsilon(ep);
    const alpha = schedules.alpha(ep);
    const gamma = schedules.gamma(ep);
    let st = env.reset();
    let s = env.stateKey();
    let ret = 0, steps = 0, done = false;
    while (!done) {
      const legal = st.legal;
      const a = chooseAction(Q, s, legal, stream, epsilon);
      const { reward, done: d, state: st2 } = env.step(a);
      const s2 = env.stateKey();
      qUpdate(Q, s, a, reward, s2, d, st2.legal, alpha, gamma);
      ret += Math.pow(gammaReport, steps) * reward;
      steps++;
      s = s2; st = st2; done = d;
    }
    const entry = {
      seq: ep,
      episode: ep,
      return: ret,
      epsilon,
      alpha,
      gamma,
      steps,
      q_root_sha: qRootSha(Q),
      prev,
    };
    entry.sha = sha256hex(canon(entry)); // canonical form = entry WITHOUT sha
    chain.push(entry);
    prev = entry.sha;
  }
  return { chain, Q, optimalReturn, optimalSteps: env.optimalSteps, gammaReport };
}

// ---- chain verification ---------------------------------------------------

export function verifyWhy(chain) {
  if (!Array.isArray(chain)) return { ok: false, why: "not_an_array", tip: GENESIS };
  let prev = GENESIS;
  for (let i = 0; i < chain.length; i++) {
    const e = chain[i];
    if (e.seq !== i) return { ok: false, why: `seq_drift_at_${i}`, tip: prev };
    if (e.prev !== prev) return { ok: false, why: `prev_link_broken_at_${i}`, tip: prev };
    const { sha, ...rest } = e; // canonical form = entry WITHOUT sha
    const expect = sha256hex(canon(rest));
    if (e.sha !== expect) return { ok: false, why: `sha_mismatch_at_${i}`, tip: prev };
    if (typeof e.return !== "number" || Number.isNaN(e.return)) return { ok: false, why: `bad_return_at_${i}`, tip: prev };
    if (typeof e.q_root_sha !== "string" || !/^[0-9a-f]{64}$/.test(e.q_root_sha)) {
      return { ok: false, why: `bad_q_root_sha_at_${i}`, tip: prev };
    }
    prev = e.sha;
  }
  return { ok: true, why: null, tip: chain.length ? chain[chain.length - 1].sha : GENESIS };
}
export const verify = (chain) => verifyWhy(chain).ok;
