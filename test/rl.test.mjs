// test/rl.test.mjs — the pinned test battery for quilt-rl.
// T1 convergence · T2 byte-determinism · T3 greedy policy optimal ·
// T4 chain tamper detection · NC1 exploration-off fails the bar ·
// NC2 shape predicate refuses non-converging series · T-env sheet semantics ·
// T-shape delta-shape concept parity.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { train, verify, verifyWhy, qRootSha, argmaxLegal, canon, sha256hex, GENESIS, epsilonSchedule, alphaSchedule, gammaSchedule } from "../src/qlearn.mjs";
import { makeSheetEnv, ACTIONS } from "../src/sheet-env.mjs";
import { degradedSeedBytes, makeStream } from "../src/qrng.mjs";
import { flatTail, changePoints, shapeOf, firstCrossing, certifyConvergence, firstConvergenceEpisode } from "../src/shape.mjs";

const SEED = degradedSeedBytes("quilt-rl/test/v1");
const SEED2 = degradedSeedBytes("quilt-rl/test/v2-other");

let _big = null;
function bigRun() {
  if (!_big) _big = train({ episodes: 1500, seedBytes: SEED });
  return _big;
}

// ---- T1: convergence within 1500 episodes ---------------------------------

test("T1: mean return of last 100 episodes >= 0.95 * optimal, and T-CONV pins inside 1500", () => {
  const { chain, optimalReturn, optimalSteps } = bigRun();
  assert.equal(optimalSteps, 6, "4x4 corner-to-corner shortest path is 6");
  assert.ok(verify(chain), "receipt chain verifies");
  const last100 = chain.slice(-100).map((e) => e.return);
  const mean = last100.reduce((a, b) => a + b, 0) / 100;
  const bar = 0.95 * optimalReturn;
  assert.ok(mean >= bar - 1e-12, `mean ${mean} >= bar ${bar} (0.95*optimal ${optimalReturn})`);
  const pin = firstConvergenceEpisode(chain.map((e) => e.return), { window: 100, tolerance: 0.01, bar });
  assert.ok(pin.pinned, "T-CONV pins inside the 1500-episode run");
  assert.ok(pin.episode <= 1499);
  // the pinned certificate must agree with the numbers it states
  assert.ok(pin.receipt.window_mean >= pin.receipt.bar);
  assert.ok(pin.receipt.flat_tail.ok);
  console.log(`    [T1] optimal=${optimalReturn} bar=${bar} mean_last100=${mean} pinned_at_episode=${pin.episode}`);
});

// ---- T2: byte-determinism --------------------------------------------------

test("T2: same seed -> identical receipt chains; different seed -> different chains", () => {
  const a = train({ episodes: 200, seedBytes: SEED });
  const b = train({ episodes: 200, seedBytes: SEED });
  assert.equal(JSON.stringify(a.chain), JSON.stringify(b.chain), "chains byte-identical");
  assert.equal(a.chain[a.chain.length - 1].sha, b.chain[b.chain.length - 1].sha, "tips equal");
  assert.equal(a.chain[0].q_root_sha, b.chain[0].q_root_sha);
  const c = train({ episodes: 200, seedBytes: SEED2 });
  assert.notEqual(JSON.stringify(a.chain), JSON.stringify(c.chain), "different seed -> different chain");
  assert.notEqual(a.chain[a.chain.length - 1].sha, c.chain[c.chain.length - 1].sha);
});

// ---- T3: greedy policy is optimal ------------------------------------------

test("T3: argmax greedy walk reaches the goal in minimal steps (6)", () => {
  const { Q, optimalSteps } = bigRun();
  const env = makeSheetEnv();
  let st = env.reset();
  let steps = 0;
  while (!st.terminal && steps <= 2 * optimalSteps + 10) {
    const row = Q.get(env.stateKey());
    const a = argmaxLegal(row, st.legal); // pure greedy, no stream
    st = env.step(a).state;
    steps++;
  }
  assert.ok(st.terminal, "greedy walk reaches the goal");
  assert.equal(steps, optimalSteps, `greedy walk takes ${steps} steps, optimal is ${optimalSteps}`);
});

// ---- T4: chain tamper detection --------------------------------------------

function sixtyChain() {
  return train({ episodes: 60, seedBytes: SEED }).chain;
}

test("T4: verify() detects every tamper class", () => {
  const base = sixtyChain();

  // (a) flip one byte of one return
  let c = structuredClone(base);
  c[30].return += 1.5;
  let v = verifyWhy(c);
  assert.equal(v.ok, false);
  assert.equal(v.why, "sha_mismatch_at_30");

  // (b) re-hash the tampered entry (sophisticated forgery) -> prev-link breaks at 31
  c = structuredClone(base);
  c[30].return += 1.5;
  const { sha, ...rest } = c[30];
  c[30].sha = createHash("sha256").update(canon(rest)).digest("hex");
  v = verifyWhy(c);
  assert.equal(v.ok, false);
  assert.equal(v.why, "prev_link_broken_at_31");

  // (c) corrupt q_root_sha
  c = structuredClone(base);
  c[10].q_root_sha = "f".repeat(64);
  v = verifyWhy(c);
  assert.equal(v.ok, false);
  assert.equal(v.why, "sha_mismatch_at_10");

  // (d) drop a middle entry
  c = structuredClone(base);
  c.splice(20, 1);
  v = verifyWhy(c);
  assert.equal(v.ok, false);
  assert.match(v.why, /seq_drift_at_20|prev_link_broken_at_20/);

  // (e) break the genesis link
  c = structuredClone(base);
  c[0].prev = "1".repeat(64);
  v = verifyWhy(c);
  assert.equal(v.ok, false);
  assert.equal(v.why, "prev_link_broken_at_0");

  // (f) empty chain verifies with genesis tip
  v = verifyWhy([]);
  assert.equal(v.ok, true);
  assert.equal(v.tip, GENESIS);
});

// ---- NC1: exploration off from episode 0 must FAIL T1's bar -----------------

test("NC1: epsilon=0 from start provably stalls (2-cycle, zero reward) and fails the bar", () => {
  const { chain, optimalReturn } = train({
    episodes: 300,
    seedBytes: SEED,
    schedules: { epsilon: () => 0, alpha: alphaSchedule, gamma: gammaSchedule },
  });
  assert.ok(verify(chain), "the NC1 chain is itself intact (it converges-nowhere honestly)");
  const last100 = chain.slice(-100).map((e) => e.return);
  const mean = last100.reduce((a, b) => a + b, 0) / 100;
  assert.ok(mean < 0.95 * optimalReturn, `exploration-off mean ${mean} must fail bar ${0.95 * optimalReturn}`);
  // the failure is the predicted limit cycle: cap-length episodes, zero reward ever
  assert.equal(chain[0].steps, 100, "episode 0 burns the full 100-step cap");
  assert.equal(chain[5].return, 0, "no reward is ever observed without exploration");
  const roots = new Set(chain.map((e) => e.q_root_sha));
  assert.ok(roots.size <= 2, `Q frozen on the cycle (roots seen: ${roots.size})`);
  // certify refuses it for the right reason
  const cert = certifyConvergence(chain.map((e) => e.return), { window: 100, tolerance: 0.01, bar: 0.95 * optimalReturn });
  assert.equal(cert.converged, false);
  assert.equal(cert.refused, "bar");
});

// ---- NC2: shape predicate refuses non-converging synthetic series ----------

test("NC2: certification refuses oscillating and flat-but-below-bar series", () => {
  const optimal = 5.9049;
  const bar = 0.95 * optimal;
  // (a) oscillating ABOVE the bar: right mean, wrong shape
  const osc = [];
  for (let i = 0; i < 300; i++) osc.push(i % 2 === 0 ? optimal : optimal * 1.1);
  let cert = certifyConvergence(osc, { window: 100, tolerance: 0.01, bar });
  assert.equal(cert.converged, false);
  assert.equal(cert.refused, "not_flat");
  assert.ok(cert.n_change_points > 100, "oscillation shows as many change-points");

  // (b) perfectly flat, below the bar: right shape, wrong level
  const flatBad = new Array(300).fill(optimal * 0.5);
  cert = certifyConvergence(flatBad, { window: 100, tolerance: 0.01, bar });
  assert.equal(cert.converged, false);
  assert.equal(cert.refused, "bar");

  // (c) firstConvergenceEpisode never pins on a drifting series
  const drift = [];
  for (let i = 0; i < 300; i++) drift.push(optimal * (0.3 + 0.5 * (i / 300)));
  const scan = firstConvergenceEpisode(drift, { window: 100, tolerance: 0.01, bar });
  assert.equal(scan.pinned, false);
  assert.equal(scan.episode, null);
});

// ---- T-env: the sheet IS the MDP -------------------------------------------

test("T-env: cells, formulas, router, listener, rewinds", () => {
  const env = makeSheetEnv({ shapingK: 0.25 });
  let st = env.reset();

  // sensor cells + router cell at the corner
  const cv = env.cells();
  assert.equal(cv["sensor.x"], 0);
  assert.equal(cv["sensor.y"], 0);
  assert.deepEqual(cv["router.legalMoves"], ["down", "right"]);
  assert.equal(cv["listener.tick"], 0);
  assert.equal(cv["formula.dist"], 6);

  // formula shaping: one step toward the goal, k=0.25, distPrev=6, dist=5 -> reward 0.25
  st = env.step("down").state;
  let cv2 = env.cells();
  assert.equal(cv2["formula.distPrev"], 6);
  assert.equal(cv2["formula.dist"], 5);
  assert.equal(cv2["formula.shaping"], 0.25);
  assert.equal(cv2["listener.tick"], 1);
  assert.equal(cv2["router.lastAction"], "down");

  // goal reward formula + shaping telescoping over a full monotone path
  env.reset();
  let total = 0, lastReward = null;
  for (const a of ["right", "right", "right", "down", "down", "down"]) {
    const r = env.step(a);
    total += r.reward;
    lastReward = r.reward;
  }
  assert.equal(env.cells()["listener.terminal"], true);
  assert.equal(lastReward, 10 + 0.25, "goal-step reward = goalReward + last shaping increment");
  assert.equal(total, 10 + 0.25 * 6, "total reward telescopes to goalReward + k*dist(start)");

  // illegal action refused loudly (router is the law)
  assert.throws(() => env.step("up"));

  // rewind: a restored sheet reproduces the original future, cell for cell
  const e2 = makeSheetEnv({ shapingK: 0.25 });
  e2.reset();
  e2.step("down");
  const snap = e2.snapshot();
  const futureA = [e2.step("right"), e2.step("down")];
  const cellsA = e2.cells();
  e2.restore(snap);
  const futureB = [e2.step("right"), e2.step("down")];
  const cellsB = e2.cells();
  assert.deepEqual(cellsB, cellsA, "rewound sheet reproduces the future exactly");
  assert.deepEqual(futureB.map((f) => f.reward), futureA.map((f) => f.reward));
  assert.throws(() => e2.restore({ "nope.cell": 1 }));
});

// ---- T-shape: delta-shape concept parity on integer series ------------------

test("T-shape: shapeOf/changePoints/firstCrossing match delta-shape semantics", () => {
  assert.equal(shapeOf([1, 2, 1, 2]), "+-+");
  assert.deepEqual(changePoints([1, 2, 1, 2]), [1, 2]);
  assert.equal(shapeOf([5, 5, 5]), "00");
  assert.deepEqual(changePoints([5, 5, 5]), []);
  assert.equal(firstCrossing([1, 2, 3], 3), 2);
  assert.equal(firstCrossing([1, 2, 3], 9), null);
  const flat = flatTail([9, 9, 9], { window: 100, tolerance: 0 });
  assert.equal(flat.ok, true);
  assert.equal(flat.n, 3);
});

// ---- schedules sanity -------------------------------------------------------

test("schedules: epsilon hits 0, alpha floors at 0.1, gamma ceilings at 0.98", () => {
  assert.equal(epsilonSchedule(0), 1);
  assert.equal(epsilonSchedule(600), 0);
  assert.equal(epsilonSchedule(1499), 0);
  assert.ok(alphaSchedule(0) === 0.5 && alphaSchedule(500) === 0.1 && alphaSchedule(1499) === 0.1);
  assert.ok(gammaSchedule(0) === 0.9 && gammaSchedule(500) === 0.98 && gammaSchedule(1499) === 0.98);
});

// ---- stream sanity ----------------------------------------------------------

test("stream: deterministic from seed bytes, uniform consumption", () => {
  const s1 = makeStream(SEED);
  const s2 = makeStream(SEED);
  const xs = [s1.nextUnit(), s1.nextUnit(), s1.nextU32()];
  const ys = [s2.nextUnit(), s2.nextUnit(), s2.nextU32()];
  assert.deepEqual(xs, ys);
  for (let i = 0; i < 1000; i++) {
    const u = s1.nextUnit();
    assert.ok(u >= 0 && u < 1);
  }
  assert.equal(makeStream(SEED).nextIndex(4), makeStream(SEED).nextIndex(4));
});
