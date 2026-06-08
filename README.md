# SwarmSimulation

SwarmSimulation is a Java Swing project that simulates a swarm of autonomous vehicles navigating a 2D arena with static rectangular obstacles, black holes, and dynamic targets.

![SwarmSimulation v2.1](v2_0.png)

## Features

### Swarm Behavior

- **Separation** keeps vehicles from crowding each other.
- **Alignment** nudges vehicles toward the local average heading.
- **Cohesion** pulls vehicles toward the local center of mass.

### Target Foraging

- **Target spawning** places a new target in the world bounds while avoiding obstacle rectangles.
- **Consumption** starts when a vehicle reaches the target area.
- **Dispersal** briefly randomizes movement after a target is consumed so the swarm does not clump.
- **Target timer** tracks how long the swarm needs to find a target and stores the last capture time.

### Black Holes

- **Black hole spawning** loads circular black holes from `src/main/resources/blackholes.txt`.
- **Black hole rendering** draws each hole with a dark radial gradient and a centered label.
- **Black hole radius overlay** can draw a dashed radius outline around each black hole.
- **Target spawn protection** prevents targets from spawning too close to any black hole.

### Obstacle Handling

- **Obstacle avoidance** uses a repulsive force based on the closest point on each rectangular obstacle.
- **World boundary bounce** keeps vehicles inside the active arena.
- **Spawnability overlay** shows allowed cells in green and blocked cells in red.
- **Obstacle radius overlay** can draw a dashed black avoidance radius around each obstacle.

## Runtime Controls

The bottom control panel now uses a compact 2-row x 5-column layout so the controls stay readable while you tune the simulation live.

### Obstacle / Debug Controls

- **AvoidRadius**: changes the base sensing distance used for obstacle avoidance.
- **AvoidMult**: changes how strongly vehicles push away when near obstacles.
- **ObsWeight**: changes how much obstacle avoidance influences the final steering decision.
- **Show obstacle radius**: toggles the dashed black radius visualization around each obstacle.
- **Show black hole radius**: toggles the dashed radius visualization around each black hole.
- **Avoidance multiplier vs. obstacle weight**: the multiplier changes the strength of the repulsive push once an obstacle is sensed, while the obstacle weight controls how much that repulsive force counts compared with cohesion, separation, and alignment.

### Target / Detection Controls

- **Target DetectRadius**: controls the radius used to decide when the swarm has "detected" a target (used by the nearest vehicle). This updates at runtime.
- **Show target radius**: toggle to draw a dashed circle around the current target showing the detection radius.
- **Show timer**: toggles the target search timer display below the spawn point.

### Swarm Weight Controls

- **F_zus**: controls cohesion strength.
- **F_sep**: controls separation strength.
- **F_aus**: controls alignment strength.

### Notes on the Sliders

- All sliders update the simulation at runtime.
- Higher values generally make the corresponding behavior stronger.
- Lower values reduce that behavior's influence on the final vehicle motion.
- The obstacle radius overlay uses the same avoidance radius concept as the steering logic.

## Spawn & Initialization

- Vehicles are now initially spawned inside a circular zone near the top-right of the world instead of uniformly across the arena. This helps test behaviors starting from a compact launch area.
- **Show spawn area**: toggle in the control panel that draws the initial spawn circle as a red dotted outline (visual only).
- **Show type1 circle**: toggle that controls whether the type-1 vehicles draw their separation/cohesion circles (used for debugging individual vehicle ranges).

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
