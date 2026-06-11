package com.swarm_reinforcement_learning;

/**
 * Immutable-ish data model for a circular "black hole" — a lethal region of the arena.
 *
 * <p>Pure data, no behaviour: it stores the circle's {@code position} (its
 * <b>centre</b>, in world units), its {@code hole_radius}, and a display
 * {@code name}. Avoidance steering lives in {@link Vehicle#blackHoleAvoidance} and the
 * learning penalty in {@link SwarmQLearning}; both treat the hole as far more
 * dangerous than an obstacle (a vehicle is meant to die inside one).</p>
 *
 * <p>Unlike {@link Obstacle}, {@code position} is the <b>centre</b>, so distance tests
 * are a simple {@code sqrt((x-cx)^2 + (y-cy)^2)} against {@code hole_radius}.</p>
 */
public class BlackHole {
    private double hole_radius;
    private String hole_name;

    double[] position;

    // constructor with blackhole positions, radius and name
    BlackHole(double[] position, double hole_radius, String hole_name) {
        this.position = position;
        this.hole_radius = hole_radius;
        this.hole_name = hole_name;
    }

    // constructor with only position of the blackhole and default radius
    public String getHole_name() {
        return hole_name;
    }

    public void setHole_name(String hole_name) {
        this.hole_name = hole_name;
    }

    public double getHole_radius() {
        return hole_radius;
    }
    public void setHole_radius(double hole_radius) {
        this.hole_radius = hole_radius;
    }
}
