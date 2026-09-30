# TEST-RECEIPT — quilt-rl v0.1.0

All numbers below are from the actual runs committed in this repo. Nothing is
projected. Channels are labeled honestly (`moth-live` / `qrng-degraded`).

## 1. Test battery

```
node --test test/rl.test.mjs
ℹ tests 10  ℹ pass 10  ℹ fail 0  ℹ cancelled 0  ℹ skipped 0
ℹ duration_ms 515 (degraded-seed machine run; variance across machines is float-dust only)
```

| id | claim | verdict |
|---|---|---|
| T1 | mean return of last 100 eps ≥ 0.95·optimal within 1500 eps, T-CONV pins | PASS — mean 5.904899999999999 ≥ bar 5.609655; pinned inside run |
| T2 | same seed → byte-identical receipt chains; different seed → different | PASS (JSON-identical chains, equal tips; cross-seed differs) |
| T3 | greedy argmax walk reaches goal in minimal steps | PASS — 6 steps = manhattan(0,0→3,3) |
| T4 | chain tamper detection (6 classes) | PASS — flip-byte→`sha_mismatch_at_30`; re-hashed forgery→`prev_link_broken_at_31`; q_root corruption→`sha_mismatch_at_10`; deletion→seq drift; genesis break→link break; empty chain → genesis tip |
| NC1 | exploration off (ε≡0 from ep 0) FAILS T1's bar | PASS-as-failure — mean 0.0 < 5.609655; ep-0 burns the 100-step cap; zero reward ever observed; Q frozen on the 2-cycle; certify refuses with `bar` |
| NC2 | shape predicate refuses non-converging series | PASS — oscillating-above-bar → `not_flat` (298 change points); flat-below-bar → `bar`; drifting series never pins |
| T-env | sheet cells, formulas, router, listener, rewinds | PASS — shaping k=0.25 telescopes to k·dist(start) = 1.5; rewound sheet reproduces the future exactly; illegal action throws |
| T-shape | delta-shape concept parity (integer series) | PASS — `shapeOf([1,2,1,2])="+-+"`, changePoints [1,2], firstCrossing, exact flatTail |
| sched | pinned schedules hit floors/ceilings | PASS |
| stream | seed-deterministic draws | PASS |

## 2. The LIVE certified run (moth channel)

- **Channel: `moth-live`. Moth jobs used: 1 of ≤ 6.**
  (`5f4dc0c0-354a-4182-99aa-8f157d807af8`, status `completed` after 6 polls,
  1032 certified bits delivered, first 256 taken.)
- One earlier POST to `/api/v1/jobs` (the brief's guessed wire shape) returned
  **405 — no job created, no credits burned**; the proven wave-45-e shape
  (`engines/comet-qrng-v1/process` + `{"params":{…}}`) was adopted. Declared in
  the job receipt note.
- Cert (verbatim from the job): `h_bit 0.9036785864868464`, `grade
  "simulator-baseline"`, `health_passed true`, `budget_bits 1167.56`,
  extractor `toeplitz` (public seed `toeplitz-v1`).
- Seed bytes (published so the chain reproduces offline): `5c8ebd1ae84514ac7f705c80226fda978f7f598675240876e9b1471c1ea95c48`
  (`seed_sha256 921c4aff…431324`; the seed is random data, not a credential).
- Run receipt: `receipts/qrng-job-001.json`; chain: `receipts/chain-live.jsonl`
  (1500 entries); certificate: `receipts/convergence-live.json`.

### Convergence numbers (the deliverable)

| quantity | value |
|---|---|
| episodes run | 1500 |
| optimal return (10 · 0.9⁵ at fixed γ_report = 0.9) | **5.9049000000000005** |
| T-CONV bar (0.95 · optimal) | 5.609655 |
| **episodes to convergence (first T-CONV pin)** | **692** |
| pinned certificate | window_mean 5.904899999999999 ≥ bar; shortfall **0**; flatTail(100, 0.01) ok, maxDev 1.78e-15 |
| final mean return (last 100 eps) | **5.904899999999999** = 100.000% of optimal (float dust −1.8e-15) |
| greedy policy (T3) | goal reached in **6 steps** = minimal |
| change points across the noisy learning phase | 420 (shape_hash `1c932e2e65c2d978`) |
| last-500 episodes distinct return values | **1** (5.9049000000000005 every episode) |
| chain tip (1500 entries, genesis-linked) | `119a13f391b3d4045da55353ae4073943c6abc1e18096b4024321f99bbe43ea0` |
| chain self-verification | ok |
| wall time | 97 ms |

Episode 0 (ε = 1.0) already lucked into the goal (return 1.35, 20 steps);
episode 14 was the first individual optimal-return episode — but T-CONV does
not pin until **692**, when a full trailing window is simultaneously
above-bar and flat. That is exactly why the predicate is windowed: one lucky
episode is not convergence.

**Offline reproduction** (`node scripts/live-run.mjs replay`): re-run from the
committed seed bytes, no network — identical chain tip
`119a13f3…43ea0`, identical pin at 692. The live run is byte-reproducible from
its receipt.

## 3. Budget & channel health

- moth jobs: **1 / 6** (successful POSTs; the 405 shape-probe created no job).
- Key material: never printed, never receipted, never committed (key-scan
  pattern check run pre-push; clean).
- No degraded-channel run shipped. The NC1/T2-style test runs use the
  `qrng-degraded` seed (`degradedSeedBytes("quilt-rl/test/v1")`) — labeled as
  such in code and used ONLY inside tests, never presented as a live run.

## 4. Deviations (all honest, none silent)

1. **Wire shape**: brief guessed `POST /api/v1/jobs` `{"engine_id":…,
   "params":{"num_bits":256}}` → 405. Adopted the fleet's proven
   `engines/{id}/process` envelope (wave-45-e census) per the brief's own
   inspect-on-4xx instruction.
2. **γ anneals up, not down** (0.90 → 0.98 ceiling): "alpha/gamma decays" is
   implemented for α (0.5 → 0.1 floor); a shrinking γ-horizon would break
   6-step credit assignment. Documented in code and README.
3. **Report discount**: episode `return` in the chain is the discounted return
   at a fixed γ_report = 0.9 (independent of the annealed learning γ) — this is
   what makes 7-step paths strictly sub-bar and the series comparable across
   episodes.
4. **Reward design**: sparse 0-per-step / +10 goal (shaping formula shipped
   with default k = 0, exercised in tests at k = 0.25). Chosen so the 0-init
   has no optimistic-exploration artifact and NC1's failure is the genuine
   no-exploration limit cycle, not a strawman.
5. **Superset receipt fields**: chain entries carry α/γ/steps alongside the
   spec'd {episode, return, epsilon, q_root_sha}.

## 5. Chain tip anchoring

Tip of record: `119a13f391b3d4045da55353ae4073943c6abc1e18096b4024321f99bbe43ea0`
— anchored here, in the v0.1.0 tag message, and in the release body.
