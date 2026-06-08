package com.swarm_reinforcement_learning;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Enhanced Q-Learning layer for swarm path optimisation.
 *
 * Key improvements over the previous version:
 *
 *  1. TARGET-RELATIVE STATE  — state is (vehicleCell, targetCell), not just
 *     vehicleCell.  The Q-table now knows where the target is, so it learns
 *     direction-to-target rather than a fixed per-cell bias that is meaningless
 *     when the target changes position.
 *
 *  2. PER-STEP Q-UPDATE via stepUpdate() — called every timer tick so the
 *     table receives continuous feedback (observe → act → reward → next state)
 *     instead of only learning at the rare capture event.
 *
 *  3. DECAYING EPSILON — exploration starts high (0.3) and decays toward a
 *     floor (0.05) as episodes accumulate, matching standard RL practice.
 *
 *  4. EPISODE STATISTICS — captureHistory[] records the milliseconds taken to
 *     find each target so you can observe learning progress over time.
 *
 *  5. REWARD SHAPING — step update gives a small positive reward when the
 *     vehicle moves closer to the target and a small penalty when it moves
 *     further away, in addition to the large terminal reward on capture.
 *
 *  State space size: GRID_W × GRID_H × GRID_W × GRID_H × ACTIONS
 *                  = 20 × 14 × 20 × 14 × 4  ≈ 313 600 entries  (tractable)
 *
 *  Grid is intentionally coarser than before (20×14 vs 30×20) to keep the
 *  table a manageable size given the extra target dimensions.
 */
public class SwarmQLearning {

    // ── Grid resolution ───────────────────────────────────────────────────────
    private static final int GRID_W = 20;
    private static final int GRID_H = 14;
    private static final int ACTIONS = 4; // 0=up 1=down 2=left 3=right

    // ── Hyperparameters ───────────────────────────────────────────────────────
    private static final double ALPHA         = 0.15;  // learning rate
    private static final double GAMMA         = 0.9;   // discount factor
    private static final double EPSILON_START = 0.10;  // initial exploration (kept low so the swarm stays smooth)
    private static final double EPSILON_FLOOR = 0.02;  // minimum exploration
    private static final double EPSILON_DECAY = 0.995; // per-episode multiplier
    private static final double EXPLORE_MAGNITUDE = 0.5; // strength of an exploratory nudge (0..1), gentle on purpose

    // ── Rewards ───────────────────────────────────────────────────────────────
    private static final double REWARD_CAPTURE     =  10.0;
    private static final double REWARD_CLOSER      =   0.2;
    private static final double REWARD_FURTHER     =  -0.1;
    private static final double REWARD_STEP_PENALTY = -0.02;

    // Penalty
    private static final double PENALTY = -5.0;

    // ── State ─────────────────────────────────────────────────────────────────
    /**
     * Q[vx][vy][tx][ty][action]
     * vx,vy = vehicle grid cell   tx,ty = target grid cell
     */
    private final double[][][][][] Q;
    private final Random rand;
    private final double worldWidth;
    private final double worldHeight;

    // ── Episode tracking ──────────────────────────────────────────────────────
    private double epsilon = EPSILON_START;
    private int    episodeCount = 0;
    private static final int HISTORY_SIZE = 200;
    private final long[] captureHistory = new long[HISTORY_SIZE];
    private int    historyHead = 0;

    // ── Previous-step cache (per vehicle id → last grid state) ────────────────
    // Used by stepUpdate() to compute the before/after distance reward.
    // Stored as flat index: (vx*GRID_H + vy)*2 where [0]=vx [1]=vy
    private final int[][] prevVehicleCell; // [vehicleId][0=gx, 1=gy]
    private double[] cachedTargetPos = null;

    public SwarmQLearning(double worldWidth, double worldHeight, int numVehicles) {
        this.worldWidth  = worldWidth;
        this.worldHeight = worldHeight;
        this.Q = new double[GRID_W][GRID_H][GRID_W][GRID_H][ACTIONS];
        this.rand = new Random();
        this.prevVehicleCell = new int[numVehicles + 1][2]; // +1: ids are 1-based
    }

    // ── Coordinate helpers ────────────────────────────────────────────────────

    /** World position → [gridX, gridY] */
    public int[] worldToGrid(double[] worldPos) {
        int gx = (int) ((worldPos[0] / worldWidth)  * GRID_W);
        int gy = (int) ((worldPos[1] / worldHeight) * GRID_H);
        return new int[]{
                Math.max(0, Math.min(GRID_W - 1, gx)),
                Math.max(0, Math.min(GRID_H - 1, gy))
        };
    }

    // ── Core Q-Learning ───────────────────────────────────────────────────────

    /** Returns the action (0–3) with the highest Q-value for state (vehicle cell, target cell) — the argmax. */
    private int getBestAction(int vx, int vy, int tx, int ty) {
        double best = -1e18;
        int    bestA = 0;
        for (int a = 0; a < ACTIONS; a++) {
            if (Q[vx][vy][tx][ty][a] > best) {
                best  = Q[vx][vy][tx][ty][a];
                bestA = a;
            }
        }
        return bestA;
    }

    /** Returns the highest Q-value over the four actions of a state — the bootstrap term in the TD update. */
    private double getMaxQ(int vx, int vy, int tx, int ty) {
        double max = Q[vx][vy][tx][ty][0];
        for (int a = 1; a < ACTIONS; a++) max = Math.max(max, Q[vx][vy][tx][ty][a]);
        return max;
    }

    /**
     * The single Q-table writer: applies one temporal-difference (Bellman) update for
     * a transition from cell {@code (vx,vy)} to {@code (nvx,nvy)} with the target at
     * {@code (tx,ty)}:
     * {@code Q ← Q + ALPHA * (reward + GAMMA * maxNext - Q)}, where {@code maxNext} is
     * the best Q of the next cell. The {@code max} makes this off-policy Q-learning —
     * it learns the optimal value regardless of what the swarm actually does next.
     *
     * @param vx,vy   the "from" cell (state being updated)
     * @param tx,ty   the target cell (part of the state; same for both lookups)
     * @param action  the action taken (0–3)
     * @param reward  the reward earned by this transition
     * @param nvx,nvy the "next" cell the action led to
     */
    private void updateQ(int vx, int vy, int tx, int ty, int action,
                         double reward, int nvx, int nvy) {
        double oldQ    = Q[vx][vy][tx][ty][action];
        double maxNext = getMaxQ(nvx, nvy, tx, ty);
        Q[vx][vy][tx][ty][action] = oldQ + ALPHA * (reward + GAMMA * maxNext - oldQ);
    }

    /** Returns int[]{x, y} after applying action — does NOT mutate inputs. */
    private int[] applyAction(int action, int x, int y) {
        switch (action) {
            case 0: y = Math.max(0,        y - 1); break; // up
            case 1: y = Math.min(GRID_H-1, y + 1); break; // down
            case 2: x = Math.max(0,        x - 1); break; // left
            case 3: x = Math.min(GRID_W-1, x + 1); break; // right
        }
        return new int[]{x, y};
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Called every timer tick for each vehicle.
     * Performs a TD update using the vehicle's movement since the last tick.
     * Applies a heavy penalty if the vehicle is inside a black hole's danger zone.
     *
     * @param vehicleId    Vehicle.id (1-based)
     * @param vehiclePos   current world position of the vehicle
     * @param targetPos    current world position of the target
     * @param blackHoles   list of black holes in the world
     * @param allObstacles list of all obstacles in the world
     */
    public void stepUpdate(int vehicleId, double[] vehiclePos, double[] targetPos,
                           List<BlackHole> blackHoles, ArrayList<Obstacle> allObstacles) {
        if (targetPos == null) return;

        int[] vc  = worldToGrid(vehiclePos);
        int[] tc  = worldToGrid(targetPos);
        int   vx  = vc[0], vy = vc[1];
        int   tx  = tc[0], ty = tc[1];

        int[] prev = prevVehicleCell[vehicleId];
        int   pvx  = prev[0], pvy = prev[1];

        // Distance reward: positive if moved closer to target grid cell
        double prevDist = Math.sqrt(Math.pow(pvx - tx, 2) + Math.pow(pvy - ty, 2));
        double currDist = Math.sqrt(Math.pow(vx  - tx, 2) + Math.pow(vy  - ty, 2));
        double shaping  = (currDist < prevDist) ? REWARD_CLOSER : REWARD_FURTHER;
        double reward   = REWARD_STEP_PENALTY + shaping;

        // Heavy penalty if the vehicle is inside any black hole's danger zone —
        // this is what teaches the Q-table to route around black holes.
        if (blackHoles != null) {
            for (BlackHole bh : blackHoles) {
                double dx   = vehiclePos[0] - bh.position[0];
                double dy   = vehiclePos[1] - bh.position[1];
                double dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < bh.getHole_radius() * 2.0) {
                    // Scale penalty by how deep inside the danger zone the vehicle is
                    double depth = 1.0 - (dist / (bh.getHole_radius() * 2.0));
                    reward += PENALTY * depth;
                    break; // one black hole penalty per tick is enough
                }
            }
        }

        if (allObstacles != null) {
            for (Obstacle obs : allObstacles) {
                double ox = obs.position[0], oy = obs.position[1];
                double ow = obs.getObstacle_width(), oh = obs.getObstacle_height();
                double cx = Math.max(ox, Math.min(vehiclePos[0], ox + ow));
                double cy = Math.max(oy, Math.min(vehiclePos[1], oy + oh));
                double ddx = vehiclePos[0] - cx, ddy = vehiclePos[1] - cy;
                double odist = Math.sqrt(ddx*ddx + ddy*ddy);
                double buffer = Vehicle.BASE_AVOIDANCE_RADIUS;   // sensing distance
                if (odist < buffer) {
                    double depth = 1.0 - (odist / buffer);
                    reward += -2.0 * depth;   // milder than a black hole
                    break;
                }
            }
        }

        // Choose the action that most plausibly caused the transition
        int action = inferAction(pvx, pvy, vx, vy);

        // TD update: previous cell → current cell
        updateQ(pvx, pvy, tx, ty, action, reward, vx, vy);

        // Store current cell for next tick
        prev[0] = vx;
        prev[1] = vy;
    }

    /**
     * Called once per VEHICLE when a target is captured.
     * Traces the direct path from vehiclePos to targetPos and applies the
     * terminal reward (+10) at the final step.
     *
     * NOTE: this method now performs ONLY the per-vehicle Q-table updates.
     * The episode-level bookkeeping (capture-time statistics + epsilon decay)
     * has moved to {@link #endEpisode(long)}, which must be called exactly ONCE
     * per episode after every vehicle has recorded its path.
     *
     * Why: this method is invoked once per vehicle (e.g. 30 times per captured
     * target). When the epsilon decay and episode counter lived here, exploration
     * decayed and the episode count advanced ~30x faster than intended —
     * collapsing epsilon to its floor after only a handful of real captures and
     * over-counting episodes by 30x. Separating the two responsibilities restores
     * the intended once-per-episode decay schedule.
     *
     * HAZARD-AWARE PATH: the imagined path now routes AROUND black holes and
     * obstacles instead of walking a blind straight line at the goal. Each step is
     * chosen by {@link #chooseSafeStep} which prefers the greedy direction toward
     * the target but skips any next cell that is a hazard or already visited. This
     * removes the previous contradiction where this method could reward driving
     * straight through a black hole that {@code stepUpdate} is simultaneously
     * penalising. If the path gets boxed in by hazards, it simply stops (laying no
     * harmful reward) rather than forcing a step into danger.
     *
     * @param vehiclePos  vehicle position at time of capture
     * @param targetPos   target position
     * @param blackHoles  black holes to route around (may be null)
     * @param obstacles   obstacles to route around (may be null)
     */
    public void recordTargetCapture(double[] vehiclePos, double[] targetPos,
                                    List<BlackHole> blackHoles,
                                    ArrayList<Obstacle> obstacles) {
        int[] current = worldToGrid(vehiclePos);
        int[] goal    = worldToGrid(targetPos);

        int[] tc      = goal;
        boolean[][] visited = new boolean[GRID_W][GRID_H];
        visited[current[0]][current[1]] = true;

        int maxSteps = GRID_W + GRID_H;
        int steps    = 0;

        while ((current[0] != goal[0] || current[1] != goal[1]) && steps < maxSteps) {
            int action = chooseSafeStep(current, goal, visited, blackHoles, obstacles);
            if (action < 0) break; // boxed in by hazards/visited cells — stop the path

            int[]   next     = applyAction(action, current[0], current[1]);
            boolean arriving = (next[0] == goal[0] && next[1] == goal[1]);
            double  reward   = arriving ? REWARD_CAPTURE : -0.05;
            updateQ(current[0], current[1], tc[0], tc[1], action, reward, next[0], next[1]);

            current = next;
            visited[current[0]][current[1]] = true;
            steps++;
        }
    }

    /**
     * Called exactly ONCE per episode — after every vehicle has recorded its
     * capture path — NOT once per vehicle.
     *
     * Handles the episode-level bookkeeping that must happen a single time per
     * captured target:
     *   1. records how long this episode took, for the rolling capture-time average
     *   2. advances the episode counter
     *   3. decays exploration (epsilon) by one step toward the floor
     *
     * @param searchMillis how long this episode took (for statistics)
     */
    public void endEpisode(long searchMillis) {
        // Record episode stats
        captureHistory[historyHead % HISTORY_SIZE] = searchMillis;
        historyHead++;

        // Decay epsilon once per real episode
        episodeCount++;
        epsilon = Math.max(EPSILON_FLOOR, epsilon * EPSILON_DECAY);
    }

    /**
     * Returns a direction-bias vector for the vehicle's current position.
     * Target position is part of the state lookup.
     *
     * EPSILON-GREEDY ACTION SELECTION: with probability {@code epsilon} the method
     * EXPLORES — it returns a random cardinal direction so the swarm occasionally
     * deviates from the learned route and can discover better paths. The rest of the
     * time it EXPLOITS — it returns the highest-Q action for the current state.
     * This is what finally gives the decaying epsilon a real effect on behaviour;
     * previously the method always exploited and epsilon was never consulted.
     *
     * MAGNITUDE:
     *  - Exploit: the bias is scaled by a confidence term {@code maxQ/REWARD_CAPTURE}
     *    in [0,1], so the table only steers as strongly as it trusts the cell — an
     *    unlearned cell contributes nothing and lets the flocking forces lead.
     *  - Explore: the random nudge is applied at full magnitude (confidence is
     *    bypassed), so exploration can actuate even in cells the table has not
     *    learned yet. It remains safe because {@code move()} caps the whole bias at
     *    35% of max acceleration and the boid obstacle/black-hole forces dominate.
     */
    public double[] getQLearningBias(double[] vehiclePos, double[] targetPos) {
        if (targetPos == null) return new double[]{0, 0};

        int[] vc = worldToGrid(vehiclePos);
        int[] tc = worldToGrid(targetPos);
        int vx = vc[0], vy = vc[1], tx = tc[0], ty = tc[1];

        int    action;
        double magnitude;
        if (rand.nextDouble() < epsilon) {
            // EXPLORE: random cardinal direction, but only a gentle nudge so it perturbs
            // the swarm without overwhelming the flocking behaviour.
            action    = rand.nextInt(ACTIONS);
            magnitude = EXPLORE_MAGNITUDE;
        } else {
            // EXPLOIT: best known action, scaled by how much we trust this cell
            action    = getBestAction(vx, vy, tx, ty);
            double maxQ = getMaxQ(vx, vy, tx, ty);
            magnitude = Math.max(0.0, Math.min(1.0, maxQ / REWARD_CAPTURE));
        }

        double[] bias = new double[2];
        switch (action) {
            case 0: bias[1] = -1; break; // up
            case 1: bias[1] =  1; break; // down
            case 2: bias[0] = -1; break; // left
            case 3: bias[0] =  1; break; // right
        }
        bias[0] *= magnitude;
        bias[1] *= magnitude;
        return bias;
    }

    /**
     * Initialises the previous-cell cache for a vehicle.
     * Call this when a vehicle is first created or after a reset.
     */
    public void initVehicle(int vehicleId, double[] vehiclePos) {
        int[] cell = worldToGrid(vehiclePos);
        prevVehicleCell[vehicleId][0] = cell[0];
        prevVehicleCell[vehicleId][1] = cell[1];
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /** Returns the average capture time (ms) over the last N episodes (max 200). */
    public double getAverageCaptureMs() {
        int count = Math.min(episodeCount, HISTORY_SIZE);
        if (count == 0) return 0;
        long sum = 0;
        for (int i = 0; i < count; i++) sum += captureHistory[i];
        return (double) sum / count;
    }

    public int getEpisodeCount() { return episodeCount; }
    public double getEpsilon()   { return epsilon; }

    // ── Visualisation accessors (read-only, for the grid / Q-value overlay) ─────

    /** Number of state-grid cells along X (for the overlay). */
    public int getGridW() { return GRID_W; }

    /** Number of state-grid cells along Y (for the overlay). */
    public int getGridH() { return GRID_H; }

    /**
     * Best (maximum) Q-value stored for vehicle cell {@code (gx,gy)} given the current
     * target — i.e. the number the grid overlay prints in that cell. Returns 0 when
     * there is no target, the cell is out of range, or it has not been learned yet.
     */
    public double getBestValueForCell(int gx, int gy, double[] targetPos) {
        if (targetPos == null) return 0.0;
        if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H) return 0.0;
        int[] tc = worldToGrid(targetPos);
        return getMaxQ(gx, gy, tc[0], tc[1]);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /** Action most likely responsible for moving from (px,py) to (cx,cy). */
    private int inferAction(int px, int py, int cx, int cy) {
        int dx = cx - px, dy = cy - py;
        if (Math.abs(dx) >= Math.abs(dy)) return dx >= 0 ? 3 : 2; // right/left
        else                               return dy >= 0 ? 1 : 0; // down/up
    }

    /** Action that moves (current) one step closer to (goal) along larger axis. */
    private int directionToward(int[] current, int[] goal) {
        int dx = goal[0] - current[0], dy = goal[1] - current[1];
        if (Math.abs(dx) >= Math.abs(dy)) return dx > 0 ? 3 : 2;
        else                               return dy > 0 ? 1 : 0;
    }

    /**
     * Hazard-aware step chooser for the synthetic capture path.
     *
     * Builds a preference order of cardinal moves — greedy toward the goal first
     * (larger axis), then the smaller axis, then the remaining directions as
     * detours — and returns the first one whose resulting cell actually moves, is
     * not already visited, and is not a hazard cell. Returns -1 if every option is
     * blocked (the path is boxed in), so the caller can stop without stepping into
     * danger.
     */
    private int chooseSafeStep(int[] current, int[] goal, boolean[][] visited,
                               List<BlackHole> blackHoles, ArrayList<Obstacle> obstacles) {
        int dx = goal[0] - current[0];
        int dy = goal[1] - current[1];

        int horiz = dx > 0 ? 3 : (dx < 0 ? 2 : -1); // right / left
        int vert  = dy > 0 ? 1 : (dy < 0 ? 0 : -1); // down  / up

        java.util.List<Integer> order = new java.util.ArrayList<>();
        if (Math.abs(dx) >= Math.abs(dy)) { addIf(order, horiz); addIf(order, vert); }
        else                              { addIf(order, vert);  addIf(order, horiz); }
        // Remaining cardinal directions allow detours around a hazard.
        for (int a = 0; a < ACTIONS; a++) if (!order.contains(a)) order.add(a);

        for (int a : order) {
            int[] n = applyAction(a, current[0], current[1]);
            if (n[0] == current[0] && n[1] == current[1]) continue;          // no movement (boundary)
            if (visited[n[0]][n[1]]) continue;                               // avoid loops
            boolean isGoal = (n[0] == goal[0] && n[1] == goal[1]);
            if (!isGoal && isHazardCell(n[0], n[1], blackHoles, obstacles)) continue; // avoid hazards
            return a;
        }
        return -1; // boxed in
    }

    private void addIf(java.util.List<Integer> order, int action) {
        if (action >= 0 && !order.contains(action)) order.add(action);
    }

    /** Centre of a grid cell in world coordinates (inverse of worldToGrid). */
    private double[] gridCellToWorld(int gx, int gy) {
        double x = (gx + 0.5) * worldWidth  / GRID_W;
        double y = (gy + 0.5) * worldHeight / GRID_H;
        return new double[]{x, y};
    }

    /**
     * True if the centre of grid cell (gx,gy) falls inside any black hole's danger
     * zone (2.0x the hole radius, matching stepUpdate) or inside any obstacle
     * rectangle. Used to keep the synthetic capture path away from hazards.
     */
    private boolean isHazardCell(int gx, int gy,
                                 List<BlackHole> blackHoles, ArrayList<Obstacle> obstacles) {
        double[] c = gridCellToWorld(gx, gy);

        if (blackHoles != null) {
            for (BlackHole bh : blackHoles) {
                double ddx = c[0] - bh.position[0];
                double ddy = c[1] - bh.position[1];
                if (Math.sqrt(ddx * ddx + ddy * ddy) < bh.getHole_radius() * 2.0) return true;
            }
        }
        if (obstacles != null) {
            for (Obstacle obs : obstacles) {
                double ox = obs.position[0], oy = obs.position[1];
                double ow = obs.getObstacle_width(), oh = obs.getObstacle_height();
                if (c[0] >= ox && c[0] <= ox + ow && c[1] >= oy && c[1] <= oy + oh) return true;
            }
        }
        return false;
    }
}