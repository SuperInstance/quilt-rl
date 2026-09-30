// src/sheet-env.mjs — the environment AS a quilt sheet.
//
// Doctrine: there is no external gym. The 4x4 gridworld IS a sheet of typed
// cells, and every quantity the MDP touches is visible as a cell value:
//
//   sensor.x, sensor.y        sensor cells  — the agent's position (written
//                             by step(); the only externally-driven cells)
//   formula.dist              formula cell  — manhattan distance to the goal
//   formula.distPrev          latch cell    — dist at the previous step (the
//                             potential that reward shaping is measured against)
//   formula.shaping           formula cell  — k * (distPrev - dist):
//                             potential-based shaping (Ng et al. 1999) —
//                             telescopes to k*dist(start) over any reaching
//                             trajectory, so it can only speed learning,
//                             never change the optimal policy
//   formula.reward            formula cell  — (at goal ? goalReward : stepCost)
//                             + shaping. THE reward, visible in a cell.
//   router.legalMoves         router cell   — legal moves at the current
//                             position (bounds). Actions are a cell value.
//   router.lastAction         latch cell    — the action just taken
//   listener.tick             listener cell — episode step counter
//   listener.terminal         listener cell — at-goal flag
//
// REWINDS: sheet.snapshot() captures every cell value as plain JSON;
// sheet.restore() rewinds to it. A rewind is a cell-value restore, not
// interpreter magic — T-env asserts a rewound sheet reproduces the original
// future trajectory bit-for-bit.
//
// STATIONARITY (the D13 lesson): the Q-learner keys on (sensor.x, sensor.y)
// only. listener.tick and router.lastAction are VISIBLE on the sheet but are
// deliberately NOT part of the MDP state — the transition kernel and reward
// are fixed functions of (position, action), so the target the learner
// chases is consistent. D13/D13b/D13c chased a per-fact partner target that
// effectively moved under the policy; D18's correction showed reward
// learning works exactly when the target is consistent. The sheet makes the
// split explicit: observation superset on the sheet, stationary state for Q.

export const ACTIONS = ["up", "down", "left", "right"];
const DELTA = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };

export function makeSheetEnv({
  width = 4,
  height = 4,
  start = { x: 0, y: 0 },
  goal = { x: 3, y: 3 },
  goalReward = 10,
  stepCost = 0,      // per-step reward for NON-goal arrivals (0 -> clean bars)
  shapingK = 0,      // potential-based shaping coefficient (0 -> pure sparse)
  maxSteps = 100,
} = {}) {
  const cells = {
    "sensor.x": { kind: "sensor", value: start.x },
    "sensor.y": { kind: "sensor", value: start.y },
    "formula.dist": { kind: "formula", value: null },
    "formula.distPrev": { kind: "latch", value: null },
    "formula.shaping": { kind: "formula", value: null },
    "formula.reward": { kind: "formula", value: null },
    "router.legalMoves": { kind: "router", value: null },
    "router.lastAction": { kind: "latch", value: null },
    "listener.tick": { kind: "listener", value: 0 },
    "listener.terminal": { kind: "listener", value: false },
  };

  const manhattan = (x, y) => Math.abs(x - goal.x) + Math.abs(y - goal.y);
  const legalMoves = (x, y) =>
    ACTIONS.filter((a) => {
      const [dx, dy] = DELTA[a];
      const nx = x + dx, ny = y + dy;
      return nx >= 0 && nx < width && ny >= 0 && ny < height;
    });

  // formulas — pure functions over current cell values, evaluated in the
  // declared order after the sensors are written.
  function evaluateFormulas() {
    const x = cells["sensor.x"].value, y = cells["sensor.y"].value;
    cells["formula.dist"].value = manhattan(x, y);
    cells["formula.shaping"].value = shapingK * (cells["formula.distPrev"].value - cells["formula.dist"].value);
    cells["listener.terminal"].value = x === goal.x && y === goal.y;
    cells["formula.reward"].value =
      (cells["listener.terminal"].value ? goalReward : stepCost) + cells["formula.shaping"].value;
    cells["router.legalMoves"].value = legalMoves(x, y);
  }

  let done = false;

  function reset() {
    cells["sensor.x"].value = start.x;
    cells["sensor.y"].value = start.y;
    cells["listener.tick"].value = 0;
    cells["router.lastAction"].value = null;
    cells["formula.distPrev"].value = manhattan(start.x, start.y);
    done = false;
    evaluateFormulas();
    return state();
  }

  function state() {
    return {
      x: cells["sensor.x"].value,
      y: cells["sensor.y"].value,
      dist: cells["formula.dist"].value,
      tick: cells["listener.tick"].value,
      legal: cells["router.legalMoves"].value,
      terminal: cells["listener.terminal"].value,
    };
  }

  // THE stationary MDP state for Q: the sensor pair. Everything else on the
  // sheet is observation, not state. (See STATIONARITY note above.)
  function stateKey() {
    return `${cells["sensor.x"].value},${cells["sensor.y"].value}`;
  }

  function step(action) {
    if (done) throw new Error("sheet-env: step() on terminal episode — reset() first");
    const legal = cells["router.legalMoves"].value;
    if (!legal.includes(action)) {
      throw new Error(`sheet-env: illegal action ${action} at (${cells["sensor.x"].value},${cells["sensor.y"].value}); legal=${JSON.stringify(legal)}`);
    }
    // latch: the potential the next shaping is measured against
    cells["formula.distPrev"].value = cells["formula.dist"].value;
    const [dx, dy] = DELTA[action];
    cells["sensor.x"].value += dx;
    cells["sensor.y"].value += dy;
    cells["listener.tick"].value += 1;
    cells["router.lastAction"].value = action;
    evaluateFormulas();
    const atGoal = cells["listener.terminal"].value;
    done = atGoal || cells["listener.tick"].value >= maxSteps;
    return {
      reward: cells["formula.reward"].value,
      done,
      state: state(),
      shaping: cells["formula.shaping"].value,
    };
  }

  // rewinds — the whole visible sheet as plain JSON
  function snapshot() {
    const out = {};
    for (const [k, c] of Object.entries(cells)) out[k] = c.value;
    out.__done__ = done;
    return out;
  }

  function restore(snap) {
    for (const [k, v] of Object.entries(snap)) {
      if (k === "__done__") { done = v; continue; }
      if (!cells[k]) throw new Error(`sheet-env: restore() of unknown cell ${k}`);
      cells[k].value = v;
    }
    return state();
  }

  function cellValues() {
    const out = {};
    for (const [k, c] of Object.entries(cells)) out[k] = c.value;
    return out;
  }

  reset();
  return {
    width, height, start, goal, goalReward, stepCost, shapingK, maxSteps,
    cells: cellValues,
    reset, state, stateKey, step, snapshot, restore,
    optimalSteps: manhattan(start.x, start.y),
  };
}
