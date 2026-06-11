# Reinforcement Learning in the Swarm Simulation

A complete reference for the RL layer of this project — from first principles to
the exact implementation, the orchestration across classes, the fixes applied, and
what remains open. This consolidates everything in one place.

> **Scope.** The project is two control systems stacked together: a reactive
> **swarm / boids** layer (`Vehicle`) and a deliberative **Q-learning** layer
> (`SwarmQLearning`). This document covers the RL layer and exactly where it touches
> the swarm layer. The boids math is out of scope except at the one blend point.

---

## Table of contents

1. [RL foundations](#1-rl-foundations)
2. [How RL is modelled in this project](#2-how-rl-is-modelled-in-this-project)
3. [The learning methods, one by one](#3-the-learning-methods-one-by-one)
4. [Acting on the table](#4-acting-on-the-table)
5. [Orchestration: how the classes work together](#5-orchestration-how-the-classes-work-together)
6. [The moving target and target-relative state](#6-the-moving-target-and-target-relative-state)
7. [Changes applied (changelog)](#7-changes-applied-changelog)
8. [Open items / recommended improvements](#8-open-items--recommended-improvements)
9. [Quick reference](#9-quick-reference)

---

## 1. RL foundations

**Reinforcement learning** is learning by trial and reward. An **agent** observes a
**state**, picks an **action**, receives a **reward** and a new state, and adjusts
its strategy from that feedback — repeated thousands of times. Nobody supplies
correct answers; the agent infers them from consequences.

Key vocabulary:

- **State (s):** a snapshot of the situation the agent must react to.
- **Action (a):** one of the available choices.
- **Reward (r):** immediate numeric feedback after an action.
- **Policy (π):** the strategy mapping states to actions.
- **Return:** the *total future* reward, not just the next one — RL optimises the long run.

### Q-learning and the Q-table

Q-learning estimates `Q(s, a)` = "the total future reward I expect if I take action
`a` in state `s`." Store these in a lookup table (the **Q-table**), all zeros at
first. Once the estimates are accurate, the optimal policy is trivial: in any state
take the action with the largest Q-value.

### The temporal-difference (TD) update

Every experience nudges one cell toward a better estimate:

```
Q(s,a) ← Q(s,a) + α · [ r + γ · maxₐ' Q(s',a') − Q(s,a) ]
         └ old ┘       └────────────── correction ──────────────┘
```

- `r + γ · maxₐ' Q(s',a')` is the **target**: the reward just received plus the best
  achievable value from the next state (this is **bootstrapping** — using a current
  guess about the future instead of waiting for it).
- `γ` (**discount**, 0.9) shrinks future reward so sooner is preferred and the math
  stays stable.
- `[ target − Q(s,a) ]` is the **TD error** — how wrong the old estimate was. This is
  the learning signal.
- `α` (**learning rate**, 0.15) moves the estimate only a fraction of the way to the
  target, so learning is smooth rather than jumpy.

Because each cell's value depends on the cell ahead of it, the large reward at the
goal **propagates backward** over many updates, building a value gradient that points
toward the target.

### Exploration vs. exploitation (ε-greedy)

Always taking the best-known action (**exploit**) can lock onto a mediocre route;
always acting randomly (**explore**) never uses what was learned. **ε-greedy** picks
a random action with probability `ε`, otherwise the best one, and **decays ε** over
time — explore early, exploit later.

---

## 2. How RL is modelled in this project

### State — target-relative, 5-dimensional

Continuous positions can't index an array, so `worldToGrid` snaps any world position
into a **20 × 14** grid:

```
gx = floor( (x / worldWidth)  · 20 ),  clamped to [0,19]      // worldWidth  = WIDTH·pix  = 590
gy = floor( (y / worldHeight) · 14 ),  clamped to [0,13]      // worldHeight = HEIGHT·pix = 320
```

The state is the **pair** (vehicle cell, target cell), so the table is 5-D:

```
Q[vx][vy][tx][ty][action]      ≈ 20 · 14 · 20 · 14 · 4 ≈ 313,600 entries
```

Including the target's cell is the defining design choice: the table learns
*direction toward the target* rather than a fixed per-cell bias, which is what lets
it cope with a target that moves (see §6).

### Actions — four cardinal directions

`0 = up, 1 = down, 2 = left, 3 = right`. Up is `y − 1` because grid rows grow
downward. Kept tiny so the table stays tractable and learns fast.

### Rewards

| Condition | Reward | Where applied |
| :-- | :-- | :-- |
| Reach the target (arriving hop) | `+10.0` | `recordTargetCapture` |
| Moved closer to target this tick | `+0.2` | `stepUpdate` |
| Moved further this tick | `−0.1` | `stepUpdate` |
| Per-step cost (prefer short paths) | `−0.02` | `stepUpdate` |
| Inside a black hole's danger zone (`2.0×` radius), scaled by depth | up to `−5.0` | `stepUpdate` |
| Inside an obstacle's sensing buffer, scaled by depth | up to `−2.0` | `stepUpdate` |
| Intermediate hop of the synthetic capture path | `−0.05` | `recordTargetCapture` |

### Hyperparameters

`ALPHA = 0.15` (learning rate), `GAMMA = 0.9` (discount), `EPSILON_START = 0.10`,
`EPSILON_FLOOR = 0.02`, `EPSILON_DECAY = 0.995` per episode, `EXPLORE_MAGNITUDE = 0.5`
(strength of an exploratory nudge — kept low so exploration doesn't disrupt the swarm).

### The two learning channels + the actor

- **`stepUpdate`** — learns *every tick* from the vehicles' real motion (writer).
- **`recordTargetCapture`** — at capture, injects the `+10` along a synthetic path (writer).
- **`getQLearningBias`** — *reads* the table each tick to steer the swarm (the only reader).
- **`endEpisode`** — once per episode: capture-time stats + ε decay (bookkeeping).
- **`updateQ`** — the single method all writers funnel through to change the table.

---

## 3. The learning methods, one by one

### 3.1 `updateQ` — the only writer

```java
private void updateQ(int vx, int vy, int tx, int ty, int action,
                     double reward, int nvx, int nvy) {
    double oldQ    = Q[vx][vy][tx][ty][action];
    double maxNext = getMaxQ(nvx, nvy, tx, ty);
    Q[vx][vy][tx][ty][action] = oldQ + ALPHA * (reward + GAMMA * maxNext - oldQ);
}
```

Parameters describe one transition: `(vx,vy)` the *from* cell, `(tx,ty)` the target
cell (same for both lookups — the target doesn't move within a step), `action`,
`reward`, and `(nvx,nvy)` the *next* cell. `getMaxQ` returns the best Q over the four
actions of the next state — the bootstrap term.

**Worked example.** `oldQ = 0.50`, `reward = 0.18`, `maxNext = 1.20`:

```
target  = 0.18 + 0.9 · 1.20 = 1.26
tdError = 1.26 − 0.50       = 0.76
newQ    = 0.50 + 0.15 · 0.76 = 0.614
```

**Two depth points.**

- **Off-policy (Q-learning, not SARSA).** The `max` uses the *best* next action
  regardless of what the agent will actually do next, so the table learns the optimal
  value function even while the swarm behaves sub-optimally (dragged by boids or
  exploring). This is exactly what makes "learn from whatever motion happens" valid.
- **No terminal special-case.** Strictly, the arriving hop should use `reward` only
  (no `γ · maxNext`), i.e. `Q ← oldQ + α(reward − oldQ)`. The code always adds the
  discounted future; since the goal cell's `maxNext` is near zero early on the error
  is small, but it is a textbook deviation worth noting.

### 3.2 `stepUpdate` — per-tick learning from real motion

Called every tick for each vehicle **while searching** (`!isConsuming && !isDispersing`).
Signature: `stepUpdate(id, pos, target, blackHoles, obstacles)`.

Steps:

1. `worldToGrid` the vehicle and the target → `(vx,vy)`, `(tx,ty)`.
2. Read the cached previous cell `prevVehicleCell[id]` → `(pvx,pvy)`.
   **Same-cell guard:** if `(pvx,pvy) == (vx,vy)` return immediately — a cell is
   ~25–30 ticks wide, and learning from self-transitions wrote spurious
   "not closer" penalties to a defaulted action (RIGHT) every tick.
3. Build the **shaped reward**: `STEP_PENALTY(−0.02)` + (`+0.2` if closer else `−0.1`),
   then add the **black-hole penalty** (up to `−5 · depth` inside `2.0×` the radius)
   and the **obstacle penalty** (up to `−2 · depth` inside the sensing buffer).
4. `inferAction(pvx,pvy,vx,vy)` — guess which action best explains the observed move.
5. `updateQ(pvx,pvy, tx,ty, action, reward, vx,vy)` — one TD write.
6. Refresh `prevVehicleCell[id] = (vx,vy)` for next tick.

**Scenario.** Target cell `(6,10)`; previous cell `(10,6)`; current cell `(9,7)`:

```
prevDist = √[(10−6)²+(6−10)²] = 5.657 ;  currDist = √[(9−6)²+(7−10)²] = 4.243  → CLOSER
reward   = −0.02 + 0.2 = 0.18
action   = inferAction(10,6 → 9,7) = LEFT (2)
updateQ( (10,6),(6,10), LEFT, 0.18, (9,7) )   → e.g. 0.50 → 0.614
```

**Known nuance (by design).** The action is *inferred* after the fact rather than
committed-and-executed, because the Q-bias is only 35% of steering. So `stepUpdate`
learns from observed transitions — a loose, off-policy coupling rather than textbook
control.

### 3.3 `recordTargetCapture` — terminal synthetic rollout

Called once per vehicle at capture (after the 3-second consume window).
Signature: `recordTargetCapture(pos, target, blackHoles, obstacles)`.

It fabricates an **idealised, hazard-aware** path from the vehicle's cell to the
target's cell and lays TD updates along it, dropping `+10` at the arriving step and
`−0.05` on intermediate steps.

```java
boolean[][] visited = new boolean[GRID_W][GRID_H];
visited[current[0]][current[1]] = true;
while (current != goal && steps < maxSteps) {
    int action = chooseSafeStep(current, goal, visited, blackHoles, obstacles);
    if (action < 0) break;                              // boxed in → stop (lay no harmful reward)
    int[] next = applyAction(action, current[0], current[1]);
    double reward = arriving ? REWARD_CAPTURE : -0.05;
    updateQ(current[0],current[1], tc[0],tc[1], action, reward, next[0],next[1]);
    current = next; visited[current[0]][current[1]] = true; steps++;
}
```

Supporting helpers:

- **`chooseSafeStep`** — prefers the greedy direction toward the goal (larger axis
  first), but skips any next cell that does not move (clamped at a boundary), is
  already visited, or is a hazard; returns `-1` if boxed in.
- **`applyAction(action, x, y)`** — pure function: returns the next cell after a
  cardinal move, **clamped** to the grid (an edge move returns the same cell) and
  **without mutating** its inputs (returns a fresh `int[]{x,y}`). Called both inside
  `chooseSafeStep` (to *test* candidates) and in the loop (to *commit* the step).
- **`isHazardCell`** — true if the cell centre (`gridCellToWorld`) lies inside a black
  hole's `2.0×` zone or an obstacle rectangle.

**Scenario.** `current = (8,10)`, `goal = (6,10)`:

```
iter 1: chooseSafeStep → LEFT;  applyAction(2,8,10) → (7,10);  not goal → −0.05
iter 2: chooseSafeStep → LEFT;  applyAction(2,7,10) → (6,10);  arriving → +10
iter 3: current == goal → stop
```

**Two subtleties.**

- **Forward propagation is slow.** Updates are written start→goal, so within one call
  the `+10` only reaches the cell adjacent to the goal; earlier cells fill in **one
  cell per episode**. The full gradient (≈10 near goal, decaying by ×0.9 outward) is a
  *multi-episode* phenomenon — it is **not** a row of `+10` cells after a single pass.
- **Per-target slices.** Because the target spawns at a random cell each episode, each
  capture trains a different `[*][*][tx][ty][*]` slice (see §6).

---

## 4. Acting on the table

### `getQLearningBias` — the only reader (ε-greedy + confidence)

Called every tick per vehicle, right after `stepUpdate`. Its output `qBias` is passed
to `Vehicle.move`.

```java
if (rand.nextDouble() < epsilon) {
    action    = rand.nextInt(ACTIONS);                 // EXPLORE: random direction
    magnitude = EXPLORE_MAGNITUDE;                      // 0.5 — a gentle nudge, not a full slam
} else {
    action    = getBestAction(vx, vy, tx, ty);         // EXPLOIT: best learned action
    magnitude = clamp( (getMaxQ(…) − getMinQ(…)) / CONFIDENCE_GAP_SCALE, 0, 1);  // advantage-gap confidence
}
bias = directionOf(action) · magnitude;                // a 2-D vector
```

- **Exploit** scales by **confidence** = the **advantage gap** `(maxQ − minQ) / 1.0`:
  how decisively the best action beats the worst. (Previously this was raw
  `maxQ / 10`, which clamps to **zero** whenever `maxQ ≤ 0` — and since step costs
  and hazard penalties make most visited states net-negative, the exploit branch
  was effectively silent almost everywhere.) The gap stays informative when all
  values are negative ("least bad" is still a direction), an unlearned cell
  (all zeros, gap 0) still contributes nothing and lets the boids lead, and a
  4-way tie no longer actuates `getBestAction`'s default-to-UP bias.
- **Explore** uses a fixed gentle magnitude (`EXPLORE_MAGNITUDE = 0.5`) so `epsilon`
  does real work without overwhelming the flocking — `EPSILON_START` is kept at `0.10`
  for the same reason (full-strength exploration on 30% of ticks made the swarm jittery).

### The blend in `Vehicle.move`

```java
acc = baseAcc + qBias · 0.35 · max_acc;     // qBiasWeight = 0.35
acc = truncate(acc, max_acc);
```

The RL term is capped at **35% of max acceleration** — it advises the steering but
can never override the boid obstacle/black-hole safety forces. Example: confidence
0.35 → leftward push `0.35 · 0.35 · 0.2 ≈ 0.0245` (~12% of max accel); a full-strength
explore push is `1 · 0.35 · 0.2 = 0.07` (the 35% cap).

### `endEpisode` — once-per-episode bookkeeping

```java
public void endEpisode(long searchMillis) {
    captureHistory[historyHead % HISTORY_SIZE] = searchMillis;  historyHead++;
    episodeCount++;
    epsilon = Math.max(EPSILON_FLOOR, epsilon * EPSILON_DECAY);  // decay once per real episode
}
```

---

## 5. Orchestration: how the classes work together

### The key fact: Simulation is the mediator

`Vehicle` and `SwarmQLearning` **never call each other** — neither holds a reference
to the other. `Simulation` is the only class that owns the `SwarmQLearning` instance.
It reads `v.id` / `v.pos` and feeds them to the brain, then takes the brain's `qBias`
and feeds it to `v.move`. `Vehicle` is RL-agnostic; the brain only ever sees
primitives (`int id`, `double[] pos`), never a `Vehicle`.

### `checkTargetStatus` — the episode state machine

Runs first on every tick. One boolean drives it: `isConsuming`
(`false` = SEARCHING, `true` = CONSUMING).

```
SEARCHING ──(nearest vehicle dist < targetDetectionRadius)──▶ CONSUMING
CONSUMING ──(3 s elapsed)──▶ recordTargetCapture ×30 → endEpisode ×1 → respawn → spawnNextTarget ──▶ SEARCHING
```

This capture check is the **terminal-state detector** and is essential: it is the
only thing that ends an episode and thus the only trigger for the terminal `+10`, the
ε decay, the respawn, and the next target. Note that during SEARCHING the explicit
target-seek weight is `0` (`f_target = isConsuming ? 1.2 : 0.00`), so the only
intelligent pull toward the target during the hunt is the learned `qBias`.

### The per-tick sequence (while searching, per vehicle)

1. `qLearning.stepUpdate(v.id, v.pos, currentTarget, allBlackHoles, allObstacles)` — **LEARN** (writes `Q`).
2. `double[] qBias = qLearning.getQLearningBias(v.pos, currentTarget)` — **ACT** (reads `Q`).
3. `v.move(..., qBias)` — **BLEND** (`acc += 0.35·qBias`; updates `v.pos`).

The chaining across ticks works through `prevVehicleCell` (the brain's one-tick
memory) and the shared `v.pos`: `move` changes `v.pos` at the end of a tick, and next
tick `stepUpdate` reads it and learns the transition. `Simulation` never tells the
brain "the vehicle moved" — the brain infers it by comparing the cached cell to the
new position.

### The boundary contract

| Call (Simulation →) | Passes | Returns | Effect |
| :-- | :-- | :-- | :-- |
| `new SwarmQLearning` | world size, vehicle count | — | allocate `Q`, caches |
| `initVehicle(id, pos)` | `v.id`, `v.pos` | — | seed `prevVehicleCell[id]` |
| `stepUpdate(id, pos, target, holes, obs)` | ids, positions, hazards | — | **write `Q`**, refresh cache |
| `getQLearningBias(pos, target)` | `v.pos`, `currentTarget` | `double[2]` qBias | read `Q` (ε-greedy) |
| `move(…, qBias)` (→ Vehicle) | the qBias vector | — | `acc += 0.35·qBias`; update `pos` |
| `recordTargetCapture(pos, target, holes, obs)` | positions, hazards | — | write `+10` path |
| `endEpisode(ms)` | `lastCaptureMillis` | — | stats + ε decay |

---

## 6. The moving target and target-relative state

Because the state includes the target's cell, the table is effectively **one gradient
per target location** — ~280 separate "maps", one for each target cell. When the
target moves (new spawn, or the "New Target" button), `stepUpdate` and
`getQLearningBias` recompute `worldToGrid(target)` every tick, so the lookup
automatically switches to the slice for the new target location. The design therefore
**handles a moving target correctly** — that is its whole purpose.

**The caveat:** a target in a fresh cell lands on a barely-trained slice
(`maxQ ≈ 0` → confidence ≈ 0 → the boids do the work), and since targets spawn at
random cells, experience is spread thin across ~280 slices. So learned steering is
sparse and helps most when a target reappears near a practised region.

**The high-value fix:** encode the state as the **relative offset**
`(tx − vx, ty − vy)` instead of absolute cells. Then "target is down-left of me" is
one state regardless of position; every episode reinforces the same compact set of
states, the table shrinks dramatically, and a transferable "steer toward target"
policy is learned in a handful of episodes. (Not yet applied — see §8.)

---

## 7. Changes applied (changelog)

| Area | Change | Why |
| :-- | :-- | :-- |
| `stepUpdate` | Added an **obstacle penalty** (`−2·depth` in the sensing buffer); widened the signature + call to pass `allObstacles` | The table was blind to obstacles and relied entirely on boid steering; now it learns to route around them. Kept mild/non-terminal since obstacles aren't lethal. |
| `recordTargetCapture` / `SwarmQLearning` | Split off **`endEpisode`** so capture-time stats and ε decay run **once per episode**, not once per vehicle | The bookkeeping ran 30× per capture, decaying ε ~30× too fast (floor in ~12 captures instead of ~357) and inflating the episode count 30×. |
| `recordTargetCapture` | Made the synthetic path **hazard-aware** via `chooseSafeStep` / `isHazardCell` / `gridCellToWorld` | The old straight-line path could reward driving through a black hole — contradicting what `stepUpdate` penalises. The two channels now agree. |
| `getQLearningBias` | Wired in **ε-greedy** (explore at full magnitude with prob ε, else exploit scaled by confidence) | `epsilon` was decayed but never consulted — the exploration was dead code. Now it actually affects behaviour. |
| `stepUpdate` | Added a **same-cell guard**: return early when the vehicle's grid cell has not changed since the last tick | A cell is ~25–30 ticks wide at `max_vel`, so most ticks were self-transitions. Each one wrote a spurious update: shaping read "not closer" (`−0.1`) and `inferAction(0,0)` defaulted to RIGHT, so action 3 absorbed ~25 bogus penalties per real cell crossing in every visited state — corrupting the table and drowning real transitions ~25:1. Now exactly one TD update is written per actual cell transition. |
| `getQLearningBias` | Replaced raw-`maxQ` confidence with the **advantage gap** `(maxQ − minQ) / CONFIDENCE_GAP_SCALE`; added `getMinQ` | `maxQ / REWARD_CAPTURE` clamps to 0 whenever `maxQ ≤ 0`, and step costs / hazard penalties make most visited states net-negative — so the exploit branch output a zero vector almost everywhere and the learned policy never actuated. The gap preserves direction information in all-negative states, is 0 for untrained states (boids still lead), and neutralises the UP tie-break bias. |

**Files touched:** `SwarmQLearning.java`, `Simulation.java`.

---

## 8. Open items / recommended improvements

- **Reverse the capture-path walk (goal → start).** Propagate the `+10` along the whole
  route in one pass instead of one cell per episode. (`recordTargetCapture`.)
- **Relative-offset state encoding.** `(tx−vx, ty−vy)` instead of absolute cells —
  the biggest learning-speed win (see §6).
- **Terminal-correct capture update.** Drop the `γ·maxNext` term on the arriving hop
  so the `+10` isn't discounted by a bootstrapped future. (`updateQ`.)
- **Make the black hole terminal in `stepUpdate`.** A large one-off penalty + early
  return matches the "lethal" design intent rather than a soft per-tick penalty.
- **Unify the danger radius.** The learning penalty uses `2.0×` the hole radius while
  the steering layer avoids at `2.5×`; pull both from one shared constant.
- **Tighten ε-greedy coupling.** Have `stepUpdate` learn the action `getQLearningBias`
  actually chose rather than inferring one from motion (textbook on-action learning).

---

## 9. Quick reference

### Constants

| Name | Value | Meaning |
| :-- | :-- | :-- |
| `GRID_W × GRID_H` | 20 × 14 | state grid resolution |
| `ACTIONS` | 4 | up / down / left / right |
| `ALPHA` | 0.15 | learning rate |
| `GAMMA` | 0.9 | discount factor |
| `EPSILON_START / FLOOR / DECAY` | 0.10 / 0.02 / 0.995 | exploration schedule (per episode) |
| `EXPLORE_MAGNITUDE` | 0.5 | strength of an exploratory nudge (gentle) |
| `CONFIDENCE_GAP_SCALE` | 1.0 | advantage gap (`maxQ − minQ`) that yields full exploit strength |
| `REWARD_CAPTURE` | +10 | reach the target |
| `REWARD_CLOSER / FURTHER` | +0.2 / −0.1 | per-tick shaping |
| `REWARD_STEP_PENALTY` | −0.02 | per-tick cost |
| `PENALTY` (black hole) | −5 | scaled by depth inside `2.0×` radius |
| obstacle penalty | −2 | scaled by depth in sensing buffer |
| `qBiasWeight` (in `Vehicle`) | 0.35 | RL share of max acceleration |
| `targetDetectionRadius` | 15 | capture threshold (slider-tunable) |
| consume window | 3000 ms | dwell before episode reset |
| world size | 590 × 320 | `WIDTH·pix × HEIGHT·pix` |

### Method map (`SwarmQLearning`)

| Method | Role |
| :-- | :-- |
| `worldToGrid` | continuous position → grid cell |
| `stepUpdate` | per-tick learning from real motion (writer) |
| `recordTargetCapture` | terminal hazard-aware rollout, injects `+10` (writer) |
| `endEpisode` | once-per-episode stats + ε decay |
| `getQLearningBias` | ε-greedy read of the table → steering nudge (reader) |
| `updateQ` | the single TD write |
| `getBestAction` / `getMaxQ` | argmax / max over a state's actions |
| `applyAction` | one cardinal grid hop (clamped, immutable) |
| `inferAction` | guess the action behind an observed move |
| `chooseSafeStep` | hazard-aware step chooser for the capture path |
| `isHazardCell` / `gridCellToWorld` | hazard test / cell-centre helper |
| `initVehicle` | seed a vehicle's previous-cell cache |