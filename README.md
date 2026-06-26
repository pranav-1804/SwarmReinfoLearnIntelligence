# Swarm-Mind RL: Intelligent Multi-Agent Navigation System

An autonomous multi-agent navigation system built using Java Swing that demonstrates a hybrid control architecture. The framework combines localized biological swarm intelligence (Craig Reynolds' Boids paradigm) with global macro-spatial path optimization through an off-policy Tabular Q-Learning engine. The agents successfully optimize forage paths toward dynamic targets while safely bypassing static rectangular obstacles and hazardous gravitational black holes.


https://github.com/user-attachments/assets/9df0d042-e979-4c63-85ab-25d3f3c3b357



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


## 🛠️ Project Structure

The project code is organized under the `com.swarm_reinforcement_learning` package layout:

src/main/java/com/swarm_reinforcement_learning/
│
├── Simulation.java        # Application orchestrator, 8ms engine render loop, config map parser
├── Canvas.java            # Render core (vehicles, shapes, grids, Q-value heatmaps, overlays)
├── Vehicle.java           # Kinematic state actor (combines flocking forces and RL bias vectors)
├── SwarmQLearning.java    # Central 5D Q-tensor repository, Bellman equation updates, action selections
├── Obstacle.java          # Model definitions for rigid rectangular barriers
├── BlackHole.java         # Model definitions for circular gravitational threat zones
└── VectorCalculation.java # Stateless low-level geometry engine (2D vector truncate, length, normalize)

---

## 🎛️ Interactive Runtime Control Panel

The simulation window provides a compact 2-row $\times$ 5-column live dashboard to adjust steering weights and monitor the crowdsourced brain metrics in real-time:

### Row 1: Proximity & Target Adjustments
* **AvoidRadius:** Modifies the baseline sensory range used by vehicles to detect upcoming structural obstacle bounds.
* **AvoidMult:** Scales the magnitude of the repulsive push once an obstacle barrier is detected.
* **Obs Weight:** Balances the influence of obstacle avoidance relative to flocking forces.
* **Toggle Obstacle Radius:** Renders dashed boundary rings around rectangular obstacles.
* **Toggle Black Hole Radius:** Renders dashed boundary rings around hazardous black hole circles.

### Row 2: Sensory Field & Flocking Coefficients
* **Target DetectRadius:** Adjusts the minimum distance threshold required for a vehicle to successfully consume an active target.
* **F_zus (Cohesion):** Controls the structural strength pulling the flock together.
* **F_sep (Separation):** Adjusts the distance threshold vehicles maintain to prevent clustering.
* **F_aus (Alignment):** Varies how tightly vehicles match their neighbor's flight vectors.
* **Toggle Q-Grid:** Toggles an on-screen grid heatmap showing learned numerical Q-values.

<img width="734" height="74" alt="Screenshot 2026-06-24 at 1 16 18 PM" src="https://github.com/user-attachments/assets/ae45d73d-8221-4644-bc37-c3d192042902" />


---

## 📐 Mathematical Formulation

### 1. Bellman Optimality Equation
The temporal-difference matrix writes tracking updates into memory when a cell boundary is crossed via an off-policy bootstrapping pipeline:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]$$

Where:
* Learning Rate ($\alpha$) = `0.15`
* Discount Factor ($\gamma$) = `0.90`

### 2. Force Vector Assembly
The vehicle's position is not directly overridden by the reinforcement learning engine. Instead, total vehicle acceleration represents a linear blend of biological, physical, and neural force components:

$$A_{\text{Total}} = F_{\text{Boids}} + F_{\text{Avoidance}} + F_{\text{RL-Bias}}$$

This ensures the swarm retains its organic local cohesion while strictly adhering to global path planning instructions. All computed interaction vectors are continuously normalized and truncated inside `VectorCalculation.java` to respect physical speed ceilings.

---

## 📋 Requirements

* **Java Development Kit (JDK):** Version 21
* **Build Automation Tool:** Maven 3.9+

---

## 📦 Building and Running the System

Follow these steps to clone, compile, and execute the simulation on your local machine:

## 1. Clone the Repository
Clone the project repository and navigate into the root directory:
```bash
git clone [https://github.com/pranav-1804/SwarmReinfoLearnIntelligence.git](https://github.com/pranav-1804/SwarmReinfoLearnIntelligence.git)
cd SwarmReinfoLearnIntelligence
```
## 2. Build the Project
Compile the source classes and package the application artifact using Maven:
```bash
mvn clean package
```

## 3. Execute the Simulation
Run the compiled simulation class directly from the target build path:
```bash
java -cp target/classes com.swarm_reinforcement_learning.Simulation
```

## Notes

- The simulation uses `pix` as a world-to-screen scaling factor.
- Targets are spawned within the active arena and rejected if they overlap an obstacle or fall too close to a black hole.
- The controls are arranged in two rows of five tiles to keep the panel compact.



## 🎓 Institution
Hochschule für Technik Stuttgart (HFT Stuttgart) University of Applied Sciences Course: Software Technology Project Management

Term: Summer Semester 2026

Project Group: Darren Gonsalves | Pranav Gadhave | Abhishek Wagh
