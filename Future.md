# Future Work: Swarm Reinforcement Learning with Shared Q-Table

The next step for the simulation is to integrate a collaborative reinforcement learning framework. This allows the vehicles to crowdsource a high-level "brain" to navigate safely around obstacles and black holes, augmenting their low-level steering rules.

## Core Architecture

- **Collective Intelligence**: All vehicles operate as independent agents but read from and write to a single, globally **shared 3D Q-Table**. This enables parallel, accelerated learning across the entire swarm.
- **Blending Brains and Boids**: The selected Q-learning actions do not erase your existing Reynolds flocking behaviors (`cohesion`, `separation`, `alignment`). Instead, the chosen action generates an additional steering force vector ($qForce$) that seamlessly influences the vehicle's total acceleration ($acc$).

---

## State Space Discretization

Because the vehicles move using continuous floating-point pixels/coordinates, standard array indexing breaks down. To utilize a structured Q-Table, the continuous canvas map is partitioned into an invisible grid of mathematical zones exclusively for state representation.

- **Grid Sizing**: A spatial cell step size (e.g., `CELL_SIZE = 25.0` world units) divides the arena.
- **Matrix Mapping**:
  - $\text{Grid X} = \lfloor \text{Vehicle Position X} / \text{CELL_SIZE} \rfloor$
  - $\text{Grid Y} = \lfloor \text{Vehicle Position Y} / \text{CELL_SIZE} \rfloor$
- **Array Dimension**: This results in a 3D table structure declared as `double[Q_WIDTH][Q_HEIGHT][ACTIONS]`.

---

## Action Space Design

The action space is kept discrete, small, and stable by mapping choices onto the 4 cardinal compass directions. Rather than overriding raw positions, each action applies a directional impulse vector matched to your vehicle's maximum capability (`max_acc`):

- **Action `0` (UP)**: Applies a vertical corrective steering force against the Y-axis (`qForce[1] = -max_acc`).
- **Action `1` (DOWN)**: Applies a vertical steering force pushing along the Y-axis (`qForce[1] = max_acc`).
- **Action `2` (LEFT)**: Applies a horizontal steering force pushing against the X-axis (`qForce[0] = -max_acc`).
- **Action `3` (RIGHT)**: Applies a horizontal steering force pushing along the X-axis (`qForce[0] = max_acc`).

---

## Swarm Reward Function ($R$)

To force the swarm to steer clear of deadly regions and hunt down the current target, the simulation loop applies distinct feedback markers based on object intersections:

| Vehicle Landing Condition        | Reward Value ($r$) | Terminate Step & Trigger Respawn?                          |
| :------------------------------- | :----------------- | :--------------------------------------------------------- |
| **Target Capture Zone**          | `+100.0`           | **Yes** (Successfully completes trajectory)                |
| **Black Hole Core Radius**       | `-100.0`           | **Yes** (Vehicle dies instantly)                           |
| **Obstacle Bounds Intersection** | `-10.0`            | **No** (Steering is penalized, vehicle bounces/stays put)  |
| **Open Safe Pathway**            | `-1.0`             | **No** (Standard step penalty to optimize for short paths) |

---

## Learning Update Rule

After every physical execution step, the individual transitions are saved to the global table using the Temporal Difference model:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \left[ r + \gamma \cdot \max_{a'} Q(s',a') - Q(s,a) \right]$$

Where:

- `\alpha` (**Learning Rate** = `0.1`): Dictates that the agent adjusts its table records incrementally by 10% per transition to avoid erratic updates.
- `\gamma` (**Discount Factor** = `0.9`): Forces vehicles to care deeply about future cumulative potential, assisting in long-term bypass navigation around obstacles.
- `\epsilon` (**Exploration Index** = `0.15`): Dictates an $\epsilon$-greedy balance where the vehicle explores randomly 15% of the time, and exploits its maximum Q-values the remaining 85%.

---

## Implementation Checklist

### 1. New Class: `BrainQLearning.java`

- - [ ] Define hyperparameter constants (`ALPHA`, `GAMMA`, `EPSILON`, `CELL_SIZE`).
- - [ ] Allocate the static multi-agent 3D array `Q` matrix.
- - [ ] Implement index conversion utilities `toGridX()` and `toGridY()`.
- - [ ] Write an $\epsilon$-greedy action selection method `chooseAction()`.
- - [ ] Write the mathematical learning updater `updateQ()`.

### 2. Vehicle Class Updates: `Vehicle.java`

- - [ ] Incorporate a `qForce` switch-case block inside `calculateWeightedAcc()`.
- - [ ] Factor the new `qForce` vectors into the composite `x` and `y` acceleration pools.
- - [ ] Assign a tuning weight configuration (`f_qlearning`) to balance the RL grid engine alongside the baseline boid forces.

### 3. Loop Interlocking: `Simulation.java`

- - [ ] Cache a vehicle's pre-movement `(oldX, oldY)` coordinates inside the `Timer` frame execution.
- - [ ] Query `BrainQLearning.chooseAction()` before updating velocity vectors.
- - [ ] Execute `v.move()` as normal.
- - [ ] Check distance loops for target capture or black hole destruction zones.
- - [ ] Submit state outputs and corresponding rewards to `BrainQLearning.updateQ()`.
- - [ ] Intercept vehicle deaths to invoke `placeVehicleInSpawnCircle()` instantly, keeping the active swarm headcount stable.
