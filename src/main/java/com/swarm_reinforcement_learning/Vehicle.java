package com.swarm_reinforcement_learning;

import java.util.ArrayList;

/**
 * A single autonomous agent in the swarm — a point mass with a position and velocity
 * that steers itself each frame using Reynolds "boids" rules plus hazard avoidance,
 * with an optional Q-learning nudge blended in.
 *
 * <p><b>Two layers meet here.</b> {@link #calculateWeightedAcc} produces the reactive
 * boid acceleration (cohesion + separation + alignment + obstacle / black-hole
 * avoidance + target seek). {@link #move} then blends in the learned steering bias
 * supplied by {@link SwarmQLearning#getQLearningBias} at 35% strength. This class
 * never references {@link SwarmQLearning}; it merely receives the bias vector as a
 * parameter, so the swarm logic stays fully decoupled from the reinforcement learning.</p>
 *
 * <p><b>Key fields.</b> {@code pos}/{@code vel} are the 2-D state; {@code max_acc} and
 * {@code max_vel} cap acceleration and speed; {@code rad_sep}/{@code rad_zus} are the
 * separation and cohesion/alignment neighbour radii; {@code id} is the 1-based id the
 * RL layer uses to index its per-vehicle cache. The {@code static} weight fields are
 * shared by every vehicle and tuned live by {@link ControlPanelBuilder}.</p>
 */
public class Vehicle {
	static int allId = 0; // class-wide counter; assigns each vehicle a unique id
	// Tunable avoidance constants (adjustable at runtime via UI)
	public static double BASE_AVOIDANCE_RADIUS = 8.0; // world units
	public static double AVOIDANCE_MULTIPLIER = 1.2;   // intensity scaling
	public static double OBS_WEIGHT = 1.0;            // weight used when combining forces
	public static double F_ZUS_WEIGHT = 0.6;
	public static double F_SEP_WEIGHT = 1.2;
	public static double F_AUS_WEIGHT = 0.4;
	int id;
	double rad_sep;
	double rad_zus;
	int type;


	final double FZL;
	final double FZB;

	double[] pos;
	double[] vel;

	final double max_acc;
	final double max_vel;

	Vehicle() {
		allId++;
		this.id = allId;
		this.FZL = 3;
		this.FZB = 1.5;
		this.rad_sep = 5;
		this.rad_zus = 25;
		this.max_acc = 0.2;
		this.max_vel = 1;

		pos = new double[2];
		vel = new double[2];

		pos[0] = Simulation.pix * 800 * Math.random();
		pos[1] = Simulation.pix * 800 * Math.random();
		double angle = 2 * Math.PI * Math.random();
		vel[0] = Math.cos(angle) * max_vel;
		vel[1] = Math.sin(angle) * max_vel;
	}

	/**
	 * Returns every other vehicle whose distance from this one falls in the ring
	 * {@code [radius1, radius2)}. The flocking rules call this with different radii to
	 * perceive different neighbourhoods (e.g. close-in for separation, wider for
	 * cohesion/alignment). Excludes this vehicle itself.
	 */
	ArrayList<Vehicle> neighbours(ArrayList<Vehicle> all, double radius1, double radius2) {
		ArrayList<Vehicle> neighbours = new ArrayList<Vehicle>();
		for (int i = 0; i < all.size(); i++) {
			Vehicle v = all.get(i);
			if (v.id != this.id) {
				double dist = Math.sqrt(Math.pow(v.pos[0] - this.pos[0], 2) + Math.pow(v.pos[1] - this.pos[1], 2));
				if (dist >= radius1 && dist < radius2) {
					neighbours.add(v);
				}
			}
		}
		return neighbours;
	}

	/**
	 * Converts a desired velocity into a steering acceleration, the standard Reynolds
	 * way: normalise the desired direction, scale to {@code max_vel}, then subtract the
	 * current velocity ({@code desired - current}). Shared by every flocking rule.
	 */
	double[] calculateAcc(double[] vel_dest) {
		double[] acc_dest = new double[2];

		if (VectorCalculation.length(vel_dest) > 1e-8) {
			vel_dest = VectorCalculation.normalize(vel_dest);
		}

		vel_dest[0] = vel_dest[0] * max_vel;
		vel_dest[1] = vel_dest[1] * max_vel;

		acc_dest[0] = vel_dest[0] - vel[0];
		acc_dest[1] = vel_dest[1] - vel[1];

		return acc_dest;
	}

	/**
	 * COHESION rule: steer toward the average position (centre of mass) of neighbours
	 * in the ring {@code [rad_sep, rad_zus)} — keeps the group together.
	 */
	double[] cohesion(ArrayList<Vehicle> all) {
		ArrayList<Vehicle> neighbours;

		double[] pos_dest = new double[2];
		double[] vel_dest = new double[2];
		double[] acc_dest = new double[2];

		acc_dest[0] = 0;
		acc_dest[1] = 0;
		neighbours = neighbours(all, rad_sep, rad_zus);

		if (neighbours.size() > 0) {
			pos_dest[0] = 0;
			pos_dest[1] = 0;
			for (int i = 0; i < neighbours.size(); i++) {
				Vehicle v = neighbours.get(i);
				pos_dest[0] = pos_dest[0] + v.pos[0];
				pos_dest[1] = pos_dest[1] + v.pos[1];
			}
			pos_dest[0] = pos_dest[0] / neighbours.size();
			pos_dest[1] = pos_dest[1] / neighbours.size();

			vel_dest[0] = pos_dest[0] - pos[0];
			vel_dest[1] = pos_dest[1] - pos[1];

			acc_dest = calculateAcc(vel_dest);
			acc_dest = VectorCalculation.truncate(acc_dest, max_acc);

		}
		return acc_dest;
	}

	/**
	 * SEPARATION rule: push away from very close neighbours (within {@code rad_sep}),
	 * weighting closer neighbours more strongly — prevents crowding and collisions.
	 */
	double[] separation(ArrayList<Vehicle> all) {
		ArrayList<Vehicle> neighbours;
		double[] vel_dest = new double[2];
		double[] acc_dest = new double[2];

		acc_dest[0] = 0;
		acc_dest[1] = 0;
		neighbours  = neighbours(all, 0, rad_sep);

		if (neighbours.size() > 0) {
			vel_dest[0] = 0;
			vel_dest[1] = 0;

			for (int i = 0; i < neighbours.size(); i++) {
				Vehicle v    = neighbours.get(i);
				double[] vel = new double[2];
				double dist;

				vel[0] = v.pos[0] - pos[0];
				vel[1] = v.pos[1] - pos[1];

				dist   = rad_sep  - VectorCalculation.length(vel);
				if (dist < 0)System.out.println("mistake in rad");

				if (VectorCalculation.length(vel) > 1e-8) {
					vel = VectorCalculation.normalize(vel);
				}

				vel[0] = -vel[0] * dist;
				vel[1] = -vel[1] * dist;

				vel_dest[0] = vel_dest[0] + vel[0];
				vel_dest[1] = vel_dest[1] + vel[1];
			}

			acc_dest = calculateAcc(vel_dest);
			acc_dest = VectorCalculation.truncate(acc_dest, max_acc);

		}

		return acc_dest;
	}

	/**
	 * ALIGNMENT rule: steer to match the average heading (velocity) of neighbours
	 * within {@code rad_zus} — produces coordinated group flow.
	 */
	double[] alignment(ArrayList<Vehicle> all) {
		ArrayList<Vehicle> neighbours = new ArrayList<Vehicle>();
		double[] vel_dest = new double[2];
		double[] acc_dest = new double[2];
		acc_dest[0] = 0;
		acc_dest[1] = 0;

		neighbours = neighbours(all, 0, rad_zus);


		if (neighbours.size() > 0) {
			vel_dest[0] = 0;
			vel_dest[1] = 0;

			for (int i = 0; i < neighbours.size(); i++) {
				Vehicle v = neighbours.get(i);
				vel_dest[0] = vel_dest[0] + v.vel[0];
				vel_dest[1] = vel_dest[1] + v.vel[1];
			}
			vel_dest[0] = vel_dest[0] / neighbours.size();
			vel_dest[1] = vel_dest[1] / neighbours.size();


			acc_dest = calculateAcc(vel_dest);
			acc_dest = VectorCalculation.truncate(acc_dest, max_acc);

		}

		return acc_dest;
	}

	/**
	 * LEGACY / unused: an earlier 4-force integrator (cohesion + separation + alignment
	 * + obstacle) without black holes or consume-mode switching. Superseded by
	 * {@link #calculateWeightedAcc}; kept for reference.
	 */
	public double[] calculateWeightedAcc1(ArrayList<Vehicle> allVehicles, ArrayList<Obstacle> obstacles, double[] target) {
		double[] acc_dest;
		double[] acc_swarm = new double[2]; // sum of cohesion, separation, alignment
		double f_zus = F_ZUS_WEIGHT;
		double f_sep = F_SEP_WEIGHT;
		double f_aus = F_AUS_WEIGHT;
		double f_obs = OBS_WEIGHT; // High priority to avoid hitting boxes
		double f_target = 0.3;

		double[] acc_cohesion = cohesion(allVehicles);
		double[] acc_sep = separation(allVehicles);
		double[] acc_align = alignment(allVehicles);
		double[] acc_obs = obstacleAvoidance(obstacles);
		double[] acc_seek = seekTarget(target);


		double x = (f_zus * acc_cohesion[0]) + (f_sep * acc_sep[0]) +
				(f_aus * acc_align[0]) + (f_obs * acc_obs[0]);
		double y = (f_zus * acc_cohesion[1]) + (f_sep * acc_sep[1]) +
				(f_aus * acc_align[1]) + (f_obs * acc_obs[1]);

		acc_dest = new double[]{x, y};
		acc_dest = VectorCalculation.truncate(acc_dest, max_acc);
		return acc_dest;
	}

	/**
	 * The boid integrator: computes all six steering forces (cohesion, separation,
	 * alignment, obstacle avoidance, black-hole avoidance, target seek) and returns
	 * their weighted sum, truncated to {@code max_acc}.
	 *
	 * <p>It is also a small state machine via {@code isConsuming}: while consuming a
	 * target it switches OFF separation/alignment and switches ON target seeking
	 * ({@code f_target = 1.2}) so the swarm converges onto the food, and raises the
	 * obstacle weight for safety. Black-hole avoidance ({@code f_bh = 2.0}) is always
	 * high — a hole can never be "consumed". The result is the base acceleration that
	 * {@link #move} blends the Q-learning bias into.</p>
	 */
	public double[] calculateWeightedAcc(ArrayList<Vehicle> allVehicles, ArrayList<Obstacle> obstacles, ArrayList<BlackHole> blackHoles, double[] target, boolean isConsuming) {
		// 1. Define all weights
		double f_zus = F_ZUS_WEIGHT;
		double f_sep = isConsuming ? 0.0 : F_SEP_WEIGHT;
		double f_obs = isConsuming ? Math.max(OBS_WEIGHT, 2) : OBS_WEIGHT;
		double f_bh  = 2.0; // black hole avoidance — always high priority, cannot be consumed
		double f_aus = isConsuming ? 0.0 : F_AUS_WEIGHT;
		double f_target = isConsuming ? 1.2 : 0.00;

		// 2. Calculate individual force vectors
		double[] acc_cohesion = cohesion(allVehicles);
		double[] acc_sep      = separation(allVehicles);
		double[] acc_align    = alignment(allVehicles);
		double[] acc_obs      = obstacleAvoidance(obstacles);
		double[] acc_bh       = isConsuming?blackHoleAvoidance(blackHoles):new double[]{0,0};
		double[] acc_seek     = seekTarget(target);

		// 3. Combine all forces into a single X and Y sum
		double x = (f_zus * acc_cohesion[0]) +
				(f_sep    * acc_sep[0]) +
				(f_aus    * acc_align[0]) +
				(f_obs    * acc_obs[0]) +
				(f_bh     * acc_bh[0]) +
				(f_target * acc_seek[0]);

		double y = (f_zus * acc_cohesion[1]) +
				(f_sep    * acc_sep[1]) +
				(f_aus    * acc_align[1]) +
				(f_obs    * acc_obs[1]) +
				(f_bh     * acc_bh[1]) +
				(f_target * acc_seek[1]);

		// 4. Create the final acceleration vector and limit it to max_acc
		double[] acc_dest = new double[]{x, y};
		return VectorCalculation.truncate(acc_dest, max_acc);
	}

	/**
	 * Advances the vehicle one step and is the single point where the swarm and RL
	 * layers fuse. It takes the boid acceleration from {@link #calculateWeightedAcc},
	 * adds the Q-learning {@code qBias} at 35% of {@code max_acc} (so the learned
	 * policy advises but never overrides the safety forces), caps the result to
	 * {@code max_acc}, integrates velocity (with light braking near the target) and
	 * position, then bounces off the arena edges via {@link #position_Box}.
	 *
	 * @param qBias learned steering direction from {@link SwarmQLearning#getQLearningBias};
	 *              its components are already in {@code [-1, 1]}.
	 */
	void move(ArrayList<Vehicle> allVehicles, ArrayList<Obstacle> obs, ArrayList<BlackHole> blackHoles, double[] target, boolean isConsuming, boolean isDispersing, double[] qBias) {

		// 1. Calculate Acceleration
		double[] acc;

		// Get base acceleration from flocking behaviors
		double[] baseAcc = calculateWeightedAcc(allVehicles, obs, blackHoles, target, isConsuming);

		// Blend Q-learning bias.
		// Weight of 0.35 lets a well-trained Q-table meaningfully steer
		// the swarm without overriding the flocking safety behaviours.
		// (bias magnitude is already [0,1]-scaled by confidence in getQLearningBias)
		double qBiasWeight = 0.35;
		acc = new double[]{
				baseAcc[0] + (qBias[0] * qBiasWeight * max_acc),
				baseAcc[1] + (qBias[1] * qBiasWeight * max_acc)
		};
		acc = VectorCalculation.truncate(acc, max_acc);

		// 2. Speed Update with "Braking" logic
		vel[0] = vel[0] + acc[0];
		vel[1] = vel[1] + acc[1];

		if (isConsuming && target != null) {
			double dist = Math.sqrt(Math.pow(target[0] - pos[0], 2) + Math.pow(target[1] - pos[1], 2));
			if (dist < 10) {
				vel[0] *= 0.5;
				vel[1] *= 0.5;
			}
		}

		// 3. Ensure we don't exceed max_vel
		double currentSpeed = VectorCalculation.length(vel);
		if (currentSpeed > max_vel) {
			vel = VectorCalculation.normalize(vel);
			vel[0] *= max_vel;
			vel[1] *= max_vel;
		}

		// 4. Update Position
		pos[0] = pos[0] + vel[0];
		pos[1] = pos[1] + vel[1];

		position_Box();
	}

	/**
	 * Steering force straight toward the target position. Only weighted non-zero while
	 * consuming (during the search phase {@code f_target} is 0, so the swarm relies on
	 * flocking and the learned bias to find the target).
	 */
	double[] seekTarget(double[] target) {
		double[] acc_dest = new double[2];
		if (target == null) return acc_dest;

		double[] vel_dest = new double[2];
		vel_dest[0] = target[0] - pos[0];
		vel_dest[1] = target[1] - pos[1];

		acc_dest = calculateAcc(vel_dest); // Reuse existing utility
		return VectorCalculation.truncate(acc_dest, max_acc);
	}

	/**
	 * Repulsive steering away from rectangular obstacles. It projects a speed-scaled
	 * "look-ahead" point in front of the vehicle, finds the closest point on each
	 * rectangle to it, and pushes away with an intensity that grows quadratically as
	 * the look-ahead nears the box — so the vehicle veers off <i>before</i> contact.
	 */
	double[] obstacleAvoidance(ArrayList<Obstacle> obstacles) {
		double[] acc_total = new double[2];
		// Use a short look-ahead so vehicles begin steering before the body reaches the box.
		double speed = VectorCalculation.length(vel);
		double lookAheadDistance = Math.max(12.0, speed * 18.0);
		double[] lookAhead = new double[]{pos[0], pos[1]};
		if (speed > 1e-8) {
			double[] forward = VectorCalculation.normalize(vel);
			lookAhead[0] += forward[0] * lookAheadDistance;
			lookAhead[1] += forward[1] * lookAheadDistance;
		}

		for (Obstacle obs : obstacles) {
			double ox = obs.position[0];
			double oy = obs.position[1];
			double ow = obs.getObstacle_width();
			double oh = obs.getObstacle_height();

			// Closest point on the rectangle to the look-ahead point.
			double closestX = Math.max(ox, Math.min(lookAhead[0], ox + ow));
			double closestY = Math.max(oy, Math.min(lookAhead[1], oy + oh));

			double dx = lookAhead[0] - closestX;
			double dy = lookAhead[1] - closestY;
			double dist = Math.sqrt(dx * dx + dy * dy);

			// Sensing radius grows with obstacle size so larger boxes are avoided earlier.
			double avoidanceRadius = BASE_AVOIDANCE_RADIUS + Math.max(ow, oh) * 0.5;

			if (dist < avoidanceRadius) {
				double[] pushAway = new double[]{dx, dy};

				if (VectorCalculation.length(pushAway) > 1e-8) {
					pushAway = VectorCalculation.normalize(pushAway);
				} else {
					// If the look-ahead lands inside the obstacle, push away from the obstacle center.
					pushAway[0] = lookAhead[0] - (ox + ow / 2.0);
					pushAway[1] = lookAhead[1] - (oy + oh / 2.0);
					if (VectorCalculation.length(pushAway) > 1e-8) {
						pushAway = VectorCalculation.normalize(pushAway);
					} else {
						// Final fallback: steer sideways relative to current heading.
						pushAway[0] = -vel[1];
						pushAway[1] = vel[0];
						if (VectorCalculation.length(pushAway) > 1e-8) {
							pushAway = VectorCalculation.normalize(pushAway);
						}
					}
				}

				// Stronger response the closer the look-ahead point is to the obstacle.
				double gap = Math.max(avoidanceRadius - dist, 0.0);
				double intensity = Math.pow(gap / Math.max(avoidanceRadius, 1.0), 2) * AVOIDANCE_MULTIPLIER;
				acc_total[0] += pushAway[0] * intensity;
				acc_total[1] += pushAway[1] * intensity;
			}
		}

		// If there is an avoidance force, truncate it to max_acc
		if (VectorCalculation.length(acc_total) > 1e-8) {
			return VectorCalculation.truncate(acc_total, max_acc);
		}
		return new double[]{0, 0};
	}

	/**
	 * Black hole avoidance — mirrors obstacleAvoidance but for circular danger zones.
	 *
	 * Uses a look-ahead point and pushes the vehicle radially outward when that
	 * point enters the black hole's influence radius.  The force grows quadratically
	 * as the look-ahead gets closer to the centre so vehicles feel a strong push
	 * well before they actually enter the hole.
	 *
	 * Influence radius is 2× the hole's actual radius so avoidance starts early
	 * enough for the vehicle's inertia to be overcome in time.
	 */
	double[] blackHoleAvoidance(ArrayList<BlackHole> blackHoles) {
		double[] acc_total = new double[]{0, 0};
		if (blackHoles == null || blackHoles.isEmpty()) return acc_total;

		double speed = VectorCalculation.length(vel);
		double lookAheadDist = Math.max(15.0, speed * 20.0);
		double[] lookAhead = new double[]{pos[0], pos[1]};
		if (speed > 1e-8) {
			double[] fwd = VectorCalculation.normalize(vel);
			lookAhead[0] += fwd[0] * lookAheadDist;
			lookAhead[1] += fwd[1] * lookAheadDist;
		}

		for (BlackHole bh : blackHoles) {
			double cx = bh.position[0];
			double cy = bh.position[1];
			double influenceRadius = bh.getHole_radius() * 2.5;

			double dx   = lookAhead[0] - cx;
			double dy   = lookAhead[1] - cy;
			double dist = Math.sqrt(dx * dx + dy * dy);

			if (dist < influenceRadius) {
				double[] pushAway = new double[]{dx, dy};

				if (VectorCalculation.length(pushAway) > 1e-8) {
					pushAway = VectorCalculation.normalize(pushAway);
				} else {
					// Look-ahead is exactly on the centre — push sideways
					pushAway[0] = -vel[1];
					pushAway[1] =  vel[0];
					if (VectorCalculation.length(pushAway) > 1e-8)
						pushAway = VectorCalculation.normalize(pushAway);
				}

				// Quadratic intensity: strongest near the centre, zero at influence edge
				double gap       = influenceRadius - dist;
				double intensity = Math.pow(gap / Math.max(influenceRadius, 1.0), 2)
						* AVOIDANCE_MULTIPLIER * 1.5;

				acc_total[0] += pushAway[0] * intensity;
				acc_total[1] += pushAway[1] * intensity;
			}
		}

		if (VectorCalculation.length(acc_total) > 1e-8)
			return VectorCalculation.truncate(acc_total, max_acc);
		return new double[]{0, 0};
	}

	/**
	 * Keeps the vehicle inside the arena: if it crosses an edge of the active world
	 * (10 .. {@code WIDTH*pix} by 10 .. {@code HEIGHT*pix}), the relevant velocity
	 * component is flipped so it bounces back inward — a hard reflective boundary.
	 */
	public void position_Box() {

		//   If the position is close to the left-edge then
		if (pos[0] < 10) {
			vel[0] = Math.abs(vel[0]);
			pos[0] = pos[0] + vel[0];
		}

		//   If the position is close to the right-edge then velocity is set to negative to move back to the left edge
		if (pos[0] > Simulation.WIDTH * Simulation.pix) {
			vel[0] = -Math.abs(vel[0]);
			pos[0] = pos[0] + vel[0];
		}

		//   If the position is close to the top-edge then
		if (pos[1] < 10) {
			vel[1] = Math.abs(vel[1]);
			pos[1] = pos[1] + vel[1];
		}

		//   If the position is close to the bottom-edge then
		if (pos[1] > Simulation.HEIGHT * Simulation.pix) {
			vel[1] = -Math.abs(vel[1]);
			pos[1] = pos[1] + vel[1];
		}
	}
}