# Swarm-Mind RL: Intelligent Multi-Agent Navigation System

An autonomous multi-agent navigation system built using Java Swing that demonstrates a hybrid control architecture. The framework combines localized biological swarm intelligence (Craig Reynolds' Boids paradigm) with global macro-spatial path optimization through an off-policy Tabular Q-Learning engine. The agents successfully optimize forage paths toward dynamic targets while safely bypassing static rectangular obstacles and hazardous gravitational black holes.

![SwarmSimulation v2.1](v2_0.png)

## 🚀 Key Features

### Swarm Behavior

* **Separation ($F_{sep}$):** Prevents overcrowding and localized physical collisions by pushing agents away from close neighbors.
* **Alignment ($F_{aus}$):** Nudges each vehicle's steering velocity vector to match the average orientation and heading of nearby neighbors.
* **Cohesion ($F_{zus}$):** Pulls individual vehicles toward the calculated local center of mass of the immediate flock.

### 2. Reinforcement Learning Brain (Global Pathing)
An autonomous navigation layer driven by an off-policy tabular Q-Learning system:
* **5D Target-Relative State Representation:** Operates on a coordinate-independent state space tracking relative positions to resolve paths when targets change locations dynamically:
  $$Q[vx][vy][tx][ty][action]$$
* **Reward Shaping & Hazard Avoidance:** Eliminates reward sparsity by applying an active step-level feedback loop:
  * Goal Capture (`REWARD_CAPTURE`): **$+10.0$**
  * Target Closeness Gradient (`REWARD_CLOSER`): **$+0.2$**
  * Divergent Step Penalty (`REWARD_FURTHER`): **$-0.1$**
  * Execution Time-Step Cost (`REWARD_STEP_PENALTY`): **$-0.02$**
  * Black Hole Threat Event (`PENALTY`): **$-5.0 \times \text{depth}$** (Resets vehicle to safety loop boundaries)
* **Advantage-Gap Confidence Scaling:** Evaluates the relative performance delta ($\max Q - \min Q$) between choices instead of relying on absolute values. This prevents action vector clamping when step costs drive absolute Q-values below zero, eliminating direction bias in unlearned state blocks.



### 3. Hazard & Arena Geometry
* **Black Holes:** Circular gravity hazards loaded dynamically from `src/main/resources/blackholes.txt`, rendered with smooth radial gradients, labeled index IDs, and a visual range toggle overlay.
* **Rectangular Obstacles:** Rigid environmental structures. The steering system calculates the exact perpendicular foot ($Lot$) to generate a repulsive counter-force.
* **Dynamic Target Foraging:** Targets are spawned at random positions across the active arena while keeping explicit distance buffers to avoid spawning directly inside obstacles or black hole gravity wells.

---


## Project Structure

- `src/main/java/com/dgx/Simulation.java`: application entry point, simulation loop, target spawning, and UI controls.
- `src/main/java/com/dgx/Canvas.java`: renders vehicles, obstacles, targets, overlays, and debug visualizations.
- `src/main/java/com/dgx/Vehicle.java`: vehicle movement, steering, obstacle avoidance, and swarm logic.
- `src/main/java/com/dgx/Obstacle.java`: rectangular obstacle model.
- `src/main/java/com/dgx/BlackHole.java`: circular black hole model.
- `src/main/java/com/dgx/VectorCalculation.java`: helper methods for 2D vector math.

## Requirements

- Java 21
- Maven 3.9+

## Run

From the project root:

```bash
mvn clean package
java -cp target/classes com.swarm_reinforcement_learning.Simulation
```

## Notes

- The simulation uses `pix` as a world-to-screen scaling factor.
- Targets are spawned within the active arena and rejected if they overlap an obstacle or fall too close to a black hole.
- The controls are arranged in two rows of five tiles to keep the panel compact.
