# quilt-rl

**RL that converges, in a quilt sheet.** The MDP *is* the sheet: a 4x4 gridworld
whose cells carry the sensors, the reward formulas, the legal-move router and
the episode listener. Tabular Q-learning runs over cell states, exploration is
seeded by **certified quantum-random bits** (moth `comet-qrng-v1`), every
episode lands in a sha256-linked receipt chain, and convergence is certified by
**pinned shape predicates** — not by eyeball.

Node >= 18. **Zero dependencies.** `node --test test/rl.test.mjs`.

---

## 1. Lineage: the account's RL attempts all KILLed. We read the bodies first.

`SuperInstance/quilt-gpu-lab` experiments D13 / D13b / D13c tried to make
relational cells *learn* whom to message (pre-registered bar: reconstruction
accuracy >= 0.80, edge concentration >= 0.5). All three KILLed
(`RESULTS.md`, verbatim):

| run | mechanism | outcome (pre-registered bar unmet) |
|---|---|---|
| **D13** | binary-reward Hebbian (decay 0.98 / boost 0.02) | final accuracy **0.467** — *below* the isolated baseline 0.558; concentration 0.233 vs chance 0.143. "the rule is too slow/weak to climb in 30 epochs over only ~30 updates/edge" |
| **D13b** | confidence-weighted signed reward | final accuracy **0.533**; concentration 0.30; mid-run peak 0.717 **then decayed back** — no convergence |
| **D13c** | REINFORCE with learned baseline (LR 0.5) | concentration **0.217**, final acc 0.65 — *worse* than D13b |
| D13d | shared-key correlation (no reward at all) | 1.0 KEEP — and D18's re-run showed REINFORCE **also** reaches concentration 1.0 when the relational target is *consistent* |

The arc's own conclusion (D18): *"the primitive is NOT 'correlation vs reward'
— it is 'consistent relational target'"*. D13's failure mode decomposes into:

1. **Non-stationary target** — one policy chasing ~7 conflicting per-fact
   partners; the target moved under the learner.
2. **Gradient-free / underscaled updates** — an asymmetric decay(0.98)/boost(0.02)
   Hebbian rule and ~30-32 reward updates per edge: a 0.5 signal buried in
   coin-flip noise, with no bootstrapped credit assignment.
3. **No exploration discipline** — nothing pinned, nothing annealed, nothing
   negative-controlled.
4. **No pinned convergence predicate** — "did it converge?" was judged at a
   fixed epoch count, on numbers nobody could tamper-check.

## 2. Why quilt-rl converges — the countermeasure for each failure

| D13 failure mode | quilt-rl countermeasure |
|---|---|
| non-stationary target | **stationary MDP**: fixed grid, fixed reward, fixed transitions. The Q target `r + gamma·maxQ(s',·)` is consistent by construction (D18's precondition). The sheet's `listener.tick` / `router.lastAction` are visible but *deliberately excluded* from Q's state key |
| unscaled online updates | proper off-policy TD: `Q(s,a) += α(r + γ·maxQ(s',·) − Q(s,a))`, α ∈ [0.1, 0.5], bootstrapped, router-masked. No decay/boost asymmetry |
| ~30 updates/edge, tiny signal | **sample abundance**: 1500 episodes × up to 100 steps over a 16-state × 4-action table — thousands of updates per Q cell |
| no exploration discipline | ε-greedy with a pinned anneal (ε: 1 → 0 by ep 600), draws from a certified-random-seeded stream. **NC1 proves the point**: ε ≡ 0 provably stalls in a 2-cycle and never sees a reward |
| convergence by eyeball | **T-CONV pin** (`src/shape.mjs`): last-100 mean ≥ 0.95·optimal **AND** flatTail(100, 0.01) verified, receipted with the numbers, inside a tamper-evident chain. **NC2** refuses non-converging series |

## 3. The sheet IS the MDP (`src/sheet-env.mjs`)

No external gym. Ten cells, all visible as values:

| cell | kind | carries |
|---|---|---|
| `sensor.x`, `sensor.y` | sensor | agent position (written by `step()`) |
| `formula.dist` | formula | manhattan distance to goal |
| `formula.distPrev` | latch | dist at the previous step (the shaping potential) |
| `formula.shaping` | formula | `k·(distPrev − dist)` — potential-based shaping (Ng et al. 1999): telescopes to `k·dist(start)` over any reaching trajectory, so it can speed learning but never bend the optimal policy |
| `formula.reward` | formula | `(at goal ? goalReward : stepCost) + shaping` — THE reward, in a cell |
| `router.legalMoves` | router | legal moves at the current position — **actions are a cell value**; illegal moves throw (fail loudly, nothing silently absorbed) |
| `router.lastAction` | latch | the action just taken |
| `listener.tick` | listener | episode step counter |
| `listener.terminal` | listener | at-goal flag |

**State/actions/rewinds are all cell values.** `snapshot()` captures every cell;
`restore()` rewinds; a rewound sheet reproduces the original future bit-for-bit
(T-env asserts this).

Default reward: `goalReward=10`, `stepCost=0`, `shapingK=0` — a purely sparse
reward chosen so the convergence bars are crisp (see §5); the shaping formula
ships and is exercised with k=0.25 in tests.

## 4. The QRNG lane (`src/qrng.mjs`)

- **LIVE** (`moth-live`): one moth job POSTs
  `engines/comet-qrng-v1/process` `{"params":{mode:"emu",num_qubits:12,shots:4096,output_bytes:512,include_raw_counts:true,bell_witness:true}}`,
  polls `jobs/{id}/status`, fetches `jobs/{id}/result`, and takes 256 certified
  bits (SP 800-90B min-entropy certificate, Toeplitz extractor, CHSH witness).
  Budget: **<= 6 moth jobs per session**; every job receipts `job_id` +
  `bits_sha256` + cert fields, never key material.
  Wire-shape provenance: the brief's guessed `POST /api/v1/jobs` 405s (no job
  created); per the brief we inspected the docs and adopted the fleet's proven
  wave-45-e census shape (see receipts/qrng-job-001.json note).
- **DEGRADED** (`qrng-degraded`): if moth is down, the seed is `SHA-256` of a
  fixed label — receipted honestly as degraded, never shipped as live
  (channel-health doctrine).
- **Both channels feed the same stream**: `block_i = SHA-256(seedBytes ‖ u32be(i))`,
  consumed byte-wise; ε-draws and random action indices come from it. One
  `nextUnit()` per step is consumed *unconditionally* (stream-alignment
  invariant), so ε ≡ 0 and ε = 1 runs stay reproducible. 256 certified bits
  seed tens of thousands of draws — the receipt carries `seed_hex`, so any run
  replays offline byte-exactly (T2) without re-talking to moth.

## 5. The convergence predicate — the crown (`src/shape.mjs`)

A zero-dep re-implementation of `SuperInstance/delta-shape`'s concepts
(`shapeOf` / `changePoints` / `firstCrossing` / `flatTail` / `shapeHash`),
adapted for float return series: upstream's exact-equality `flatTail` becomes
**pinned-window flatTail** — every value in the trailing window within
tolerance of the window mean.

**T-CONV pin**: a return series is *converged* iff

1. the trailing window (100 episodes) is full,
2. `window_mean >= bar` (bar = **0.95 × optimal**, optimal = `10·0.9⁵ = 5.9049`
   at the fixed report discount γ_report = 0.9 — a 7-step path returns 5.314,
   strictly under the bar), and
3. `flatTail(100, 0.01)` verifies (spread ≤ 0.01).

The certificate states its numbers (mean, bar, shortfall, maxDev, change
points, 16-hex shape hash) and `refused` names the failed leg
(`bar` / `not_flat` / `window_not_full`). Shape is necessary, not sufficient —
a uniformly bad policy is perfectly flat too; the bar and the shape must both
hold. `firstConvergenceEpisode` scans the series for the first pin.

## 6. The receipt chain (`src/qlearn.mjs`)

Every episode appends `{seq, episode, return, epsilon, alpha, gamma, steps,
q_root_sha, prev, sha}` — `q_root_sha` = sha256 over the whole Q table (f64
big-endian hex, canonical), `sha` = sha256 over the canonical key-sorted entry,
`prev` = previous `sha`, genesis = 64 zeros. `verify(chain)` re-hashes every
entry and every link; `verifyWhy` names the breach. T4 tampers six ways
(flip-a-byte, **re-hash-the-forgery** → prev-link break, q_root corruption,
entry deletion, genesis break) — all detected.

## 7. Quickstart

```bash
node --test test/rl.test.mjs     # the full battery (T1-T4, NC1-NC2, T-env, T-shape)
node scripts/live-run.mjs live   # 1 moth job -> certified seed -> 1500-ep run -> receipts/
node scripts/live-run.mjs replay # reproduce the live chain offline from seeds/live-seed-001.json
```

```js
import { train, verifyWhy } from "./src/qlearn.mjs";
import { firstConvergenceEpisode } from "./src/shape.mjs";
const { chain, optimalReturn } = train({ episodes: 1500, seedBytes }); // 32 bytes
const pin = firstConvergenceEpisode(chain.map(e => e.return), { bar: 0.95 * optimalReturn });
verifyWhy(chain); // { ok: true, tip: "…" }
```

Pinned schedules: ε = `max(0, 1 − ep/600)`; α = `max(0.1, 0.5 − ep·0.4/500)`;
γ = `min(0.98, 0.9 + ep·0.08/500)`. **Documented deviation from the brief's
"alpha/gamma decays"**: α decays exactly as specified; γ is annealed *upward*
0.90 → 0.98 (ceiling) because a shrinking horizon would progressively break
6-step credit assignment on the very task the predicate certifies.

## 8. Test receipt (actual numbers — see TEST-RECEIPT.md for the full table)

10/10 green. Live run: T-CONV pins at **episode 692**, last-100 mean
**5.904899999999999** vs optimal **5.9049000000000005** (shortfall 0.0,
flat-tail maxDev 1.8e-15), chain tip `119a13f3…43ea0`, seeded by 1 certified
moth job, replay byte-identical offline. NC1: ε ≡ 0 → 2-cycle, mean 0.0, bar
refused. NC2: oscillating and flat-below-bar series refused (`not_flat`,
`bar`).

## 9. Honest limitations

- Tabular Q on a 16-state world: the convergence *mechanism* (stationary target
  + scaled TD + pinned exploration + pinned predicate) is the transferable
  part, not the scale. D13-scale relational learning is a different, harder
  credit-assignment problem — this repo does not claim it.
- Tamper-EVIDENT, not tamper-PROOF: a bare chain cannot detect tail truncation
  without an externally anchored tip (the tip is anchored in TEST-RECEIPT.md,
  the git tag message and the release body).
- Receipt `return` uses a fixed report discount (γ_report = 0.9) independent of
  the annealed learning γ — deliberate, documented, so the series is comparable
  across episodes.
- `nextIndex(n) = nextU32() % n` has modulo bias (< 2⁻³⁰ relative for small n);
  accepted for determinism, documented.
- moth `grade: "simulator-baseline"` — the certified bits came from the Aer
  baseline lane, not IBM hardware (device-grade honest fact from the cert).

## 10. Credit

- **quilt-gpu-lab D13/D13b/D13c/D13d + D18** — the failure receipts this whole
  design is a reply to; D18's "consistent relational target" correction is the
  design's spine.
- **SuperInstance/delta-shape** — flatTail / changePoints / firstCrossing /
  shapeHash concepts, re-implemented here zero-dep with float tolerance.
- **fleet-seeds wave-45-e moth census** — the proven comet-qrng-v1 wire shape.
- Receipt-chain doctrine: quilt-neighbourhood, slackwater-quilt, quilt-nn.
