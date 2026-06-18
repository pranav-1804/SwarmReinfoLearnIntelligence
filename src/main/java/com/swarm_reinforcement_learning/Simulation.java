package com.swarm_reinforcement_learning;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;

import javax.swing.*;
import java.awt.BorderLayout;

/**
 * Application entry point and central orchestrator (a Swing {@link JFrame}).
 *
 * <p>{@code Simulation} owns the world ({@code allVehicles}, {@code allObstacles},
 * {@code allBlackHoles}), the {@link SwarmQLearning} brain, the {@link Canvas} view
 * and the {@link ControlPanelBuilder} UI, and drives everything from a single
 * {@link Timer} firing every {@code sleep} ms. It is also the <b>mediator</b> between
 * the swarm and RL layers: {@link Vehicle} and {@link SwarmQLearning} never call each
 * other — this class reads each vehicle's position, feeds it to the brain, and passes
 * the brain's bias back into the vehicle.</p>
 *
 * <p><b>Coordinate space.</b> Physics runs in a scaled world of {@code WIDTH*pix} by
 * {@code HEIGHT*pix} (590 × 320); the on-screen window is the full {@code WIDTH ×
 * HEIGHT} (1475 × 800), and {@link Canvas} converts world → screen by dividing by
 * {@code pix}.</p>
 *
 * <p><b>Each tick</b> the timer runs {@link #checkTargetStatus} (the episode state
 * machine), then for every vehicle: {@code stepUpdate} (learn) → {@code getQLearningBias}
 * (act) → {@code move} (blend), and finally repaints.</p>
 */

public class Simulation extends JFrame {

    static final double SPAWN_POINT_RADIUS = 40.0;

    int anzFz = 100; // number of cars (Anzahl Fahrzeuge)
    boolean isConsuming = false; // state when the vehicles are consuming the target
    boolean isDispersing = false; // state when the vehicles are done consuming the target
    long consumptionStartTime = 0; // timer after when the vehicles start consume the target
    long dispersalStartTime = 0; // timer for the dispersion of the vehicle
    double targetDetectionRadius = 15.0; // radius used to detect the target
    long targetSearchStartTime = 0; // time when the current target spawned
    long targetSearchElapsedMillis = 0; // live search timer for the current target
    long lastCaptureMillis = -1; // last time the swarm took to find a target

    Canvas myCanvas; // The Canvas (die Leinwand)
    static final int WIDTH = 1475;
    static final int HEIGHT = 800;
    final int WORLD_MARGIN = 10;
    final int WORLD_BORDER_WIDTH = 2;
    static int sleep = 1; // delay in frame
    static double pix = 0.4; // the scaling factor
    double[] currentTarget = null;

    int numObstacles = 0;// position of the current target

    ArrayList<Vehicle> allVehicles = new ArrayList<>(); // Array of vehicles
    ArrayList<Obstacle> allObstacles = new ArrayList<>(); // Array of Obstacles
    ArrayList<BlackHole> allBlackHoles = new ArrayList<>(); // Array of Black Holes

    private SwarmQLearning qLearning;

    /**
     * Builds the whole simulation and starts it: loads obstacles and black holes from
     * the resource files, generates the swarm, constructs the {@link SwarmQLearning}
     * brain (sized to the scaled world and vehicle count) and seeds each vehicle's
     * cell cache, wires up the {@link Canvas} and control panel, spawns the first
     * target, and launches the {@link Timer} loop that learns, acts and moves every tick.
     */
    Simulation() {

        setTitle("Die Schwarm intelligenz und Reinforcement Learning");
        System.out.println("\"Die Schwarmintelligenz\"");

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        System.out.println("Extracting Obstacles Positions");

        createObstacles();
        createBlackHoles();
        generateVehicles();

        // Pass anzFz so SwarmQLearning can allocate the per-vehicle cell cache
        qLearning = new SwarmQLearning(WIDTH * pix, HEIGHT * pix, anzFz);

        // Initialise each vehicle's previous-cell entry
        for (Vehicle v : allVehicles) {
            qLearning.initVehicle(v.id, v.pos);
        }

        myCanvas = new Canvas(allVehicles, pix, allObstacles, allBlackHoles, WIDTH, HEIGHT);
        myCanvas.setQLearning(qLearning); // read-only access for the grid / Q-value overlay

        myCanvas.setWorld_margin(WORLD_MARGIN);
        myCanvas.setWorld_border_thickness(WORLD_BORDER_WIDTH);

        // Layout: canvas center, controls at bottom
        getContentPane().setLayout(new BorderLayout());
        add(myCanvas, BorderLayout.CENTER);

        ControlPanelBuilder controlPanelBuilder = new ControlPanelBuilder(myCanvas, this);
        JPanel controlPanel = controlPanelBuilder.buildControlPanel();

        add(controlPanel, BorderLayout.SOUTH);
        setSize(WIDTH, HEIGHT);
        setVisible(true);

        spawnNextTarget();

        myCanvas.updateTarget(currentTarget, isConsuming, targetDetectionRadius, targetSearchElapsedMillis, lastCaptureMillis);

        new Timer(sleep, e -> {
            checkTargetStatus();
            myCanvas.updateTarget(currentTarget, isConsuming, targetDetectionRadius, targetSearchElapsedMillis, lastCaptureMillis);

            for (Vehicle v : allVehicles) {
                // Per-step Q-update: vehicle observes state, gets shaped reward
                // Black holes are passed so the Q-table learns to penalise entering them
                if (!isConsuming && !isDispersing && currentTarget != null) {
                    qLearning.stepUpdate(v, v.pos, currentTarget, allBlackHoles, allObstacles, this);
                }
                double[] qBias = qLearning.getQLearningBias(v.pos, currentTarget);
                v.move(allVehicles, allObstacles, allBlackHoles, currentTarget, isConsuming, isDispersing, qBias);
            }

            repaint();
        }).start();
    }

    /** Creates {@code anzFz} vehicles and places each one inside the spawn circle. */
    private void generateVehicles() {
        System.out.println("Generating Vehicles");

        for (int k = 0; k < anzFz; k++) {
            Vehicle car = new Vehicle();
            car.type = 1; // type 1 has visible boundary
            placeVehicleInSpawnCircle(car);
            allVehicles.add(car);
        }

        System.out.println("Vehicles Generated");

    }

    /**
     * Loads circular black holes from {@code resources/blackholes.txt} (first line =
     * count, then {@code x y radius name} per line), clamping each into the window.
     */
    private void createBlackHoles() {

        // Extract Black Holes
        System.out.println("Extracting Black Holes Positions");
        try{
            InputStream bhInput = getClass().getClassLoader().getResourceAsStream("blackholes.txt");
            if (bhInput != null) {
                BufferedReader reader = new BufferedReader(new InputStreamReader(bhInput));
                int numBH = Integer.parseInt(reader.readLine());
                System.out.println("Number of BlackHoles: "+numBH);
                for (int i = 0; i < numBH; i++) {
                    double[] bh_pos = new double[2];
                    String line = reader.readLine();
                    String[] parts = line.split(" ");
                    double parsedX = Integer.parseInt(parts[0]);
                    double parsedY = Integer.parseInt(parts[1]);
                    double bh_radius = Double.parseDouble(parts[2]);
                    String bh_name = parts.length > 3 ? parts[3] : "";

                    double maxX = WIDTH;
                    double maxY = HEIGHT;

                    bh_pos[0] = Math.max(WORLD_MARGIN, Math.min(parsedX, maxX));
                    bh_pos[1] = Math.max(WORLD_MARGIN, Math.min(parsedY, maxY));

                    allBlackHoles.add(new BlackHole(bh_pos, bh_radius, bh_name));
                }
                reader.close();
                System.out.println("Black Holes Position Extracted");
            } else {
                System.out.println("No text file for blackholes found!");
            }
        }
        catch (IOException ex) {
            ex.printStackTrace();
        }

    }

    /**
     * Loads rectangular obstacles from {@code resources/obstacles.txt} (first line =
     * count, then {@code x y width height name} per line), clamping each into the window.
     */
    private void createObstacles() {

        try{
            InputStream input = getClass().getClassLoader().getResourceAsStream("obstacles.txt");

            if (input != null) {

                BufferedReader reader = new BufferedReader(new InputStreamReader(input));

                numObstacles = Integer.parseInt(reader.readLine());

                System.out.println("Number of Obstacles: "+numObstacles);

                for (int i = 0; i < numObstacles; i++) {
                    double[] obs_pos = new double[2];
                    String line = reader.readLine();
                    String[] parts = line.split(" ");

                    // 1. Parse raw positions and dimensions from the file
                    double parsedX = Integer.parseInt(parts[0]);
                    double parsedY = Integer.parseInt(parts[1]);
                    double obs_width = Double.parseDouble(parts[2]);
                    double obs_height = Double.parseDouble(parts[3]);
                    String obs_name = parts[4];

                    // 2. Define your new centered world boundary thresholds for 1500x1500px window
                    double maxX = WIDTH - obs_width;
                    double maxY = HEIGHT - obs_height;

                    obs_pos[0] = Math.max(WORLD_MARGIN, Math.min(parsedX, maxX));
                    obs_pos[1] = Math.max(WORLD_MARGIN, Math.min(parsedY, maxY));

                    allObstacles.add(new Obstacle(obs_pos, obs_width, obs_height, obs_name));
                }

                reader.close();

                System.out.println("Obstacles Position Extracted");

                allObstacles.forEach(arr -> System.out.print(Arrays.toString(arr.position)+"\t"));

            } else {
                System.out.println("No text file for obstacles found!");
            }
        }
        catch (IOException ex) {
            ex.printStackTrace();
        }
        System.out.println("\nObstacles Generated");
    }

    /**
     * Rejection-samples a new target position inside the arena, re-rolling (up to 100
     * attempts) until it is clear of every obstacle (with a small buffer) and outside
     * every black hole's danger zone. Resets the search timer and clears
     * {@code isConsuming}, returning the machine to the SEARCHING state.
     */
    void spawnNextTarget() {
        currentTarget = new double[2];
        targetSearchStartTime = System.currentTimeMillis();
        targetSearchElapsedMillis = 0;
        boolean invalidLocation;
        int attempts = 0;

        // Define the same boundaries the vehicles use in position_Box()
        double minX = WORLD_MARGIN + 5;
        double maxX = (WIDTH * pix) - (WORLD_MARGIN + 5);
        double minY = WORLD_MARGIN + 5;
        double maxY = (HEIGHT * pix) - (WORLD_MARGIN + 5);

        do {
            invalidLocation = false;
            attempts++;

            // 1. Generate position strictly within the vehicle's "position_Box" limits
            currentTarget[0] = Math.random() * (maxX - minX) + minX;
            currentTarget[1] = Math.random() * (maxY - minY) + minY;

            // 2. Check if it's inside or too close to any obstacle rectangle
            for (Obstacle obs : allObstacles) {
                double ox = obs.position[0];
                double oy = obs.position[1];
                double ow = obs.getObstacle_width();
                double oh = obs.getObstacle_height();

                // Buffer must clear the target's drawn size AND its detection ring so
                // the target never visually overlaps (or grazes) an obstacle.
                double buffer = 20.0;

                boolean insideX = currentTarget[0] >= (ox - buffer) && currentTarget[0] <= (ox + ow + buffer);
                boolean insideY = currentTarget[1] >= (oy - buffer) && currentTarget[1] <= (oy + oh + buffer);

                if (insideX && insideY) {
                    invalidLocation = true;
                    break;
                }
            }

            // 3. Keep the target clear of black hole gravity zones plus a safety buffer
            //    large enough that the target and its detection ring never touch a hole.
            for (BlackHole bh : allBlackHoles) {
                if (isInsideBlackHoleBuffer(currentTarget, bh, 25.0)) {
                    invalidLocation = true;
                    break;
                }
            }

            if (attempts > 100) break;

        } while (invalidLocation);

        System.out.println("New Target Position:\t"+currentTarget[0]+","+currentTarget[1]);

        isConsuming = false;
    }

    /** True if {@code target} lies within {@code blackHole.radius + extraBuffer} of the hole's centre. */
    private boolean isInsideBlackHoleBuffer(double[] target, BlackHole blackHole, double extraBuffer) {
        double dx = target[0] - blackHole.position[0];
        double dy = target[1] - blackHole.position[1];
        double radius = blackHole.getHole_radius() + extraBuffer;
        return (dx * dx) + (dy * dy) <= radius * radius;
    }

    /**
     * Places one vehicle at a uniformly-random point inside the spawn disc in the
     * top-right corner ({@code distance = radius * sqrt(rand)} gives an even area
     * distribution), re-rolling if it lands in an obstacle, then gives it a fresh
     * random heading. Called at start-up and after every capture so each episode
     * begins as a coherent cluster.
     */
    public void placeVehicleInSpawnCircle(Vehicle car) {
        final double spawnRadius = SPAWN_POINT_RADIUS;
        final double spawnCenterX = (WIDTH * pix) - WORLD_MARGIN - spawnRadius;
        final double spawnCenterY = WORLD_MARGIN + spawnRadius;

        boolean invalidLocation;
        int attempts = 0;

        do {
            invalidLocation = false;
            attempts++;

            double angle = 2.0 * Math.PI * Math.random();
            double distance = spawnRadius * Math.sqrt(Math.random());

            car.pos[0] = spawnCenterX + Math.cos(angle) * distance;
            car.pos[1] = spawnCenterY + Math.sin(angle) * distance;

            for (Obstacle obs : allObstacles) {
                double ox = obs.position[0];
                double oy = obs.position[1];
                double ow = obs.getObstacle_width();
                double oh = obs.getObstacle_height();

                boolean insideX = car.pos[0] >= ox && car.pos[0] <= (ox + ow);
                boolean insideY = car.pos[1] >= oy && car.pos[1] <= (oy + oh);

                if (insideX && insideY) {
                    invalidLocation = true;
                    break;
                }
            }

            if (attempts > 100) {
                break;
            }
        } while (invalidLocation);

        double angle = 2.0 * Math.PI * Math.random();
        car.vel[0] = Math.cos(angle) * car.max_vel;
        car.vel[1] = Math.sin(angle) * car.max_vel;
    }

    /**
     * The episode state machine, run first on every tick. One boolean drives it:
     * {@code isConsuming} ({@code false} = SEARCHING, {@code true} = CONSUMING).
     *
     * <ul>
     *   <li>SEARCHING: if the nearest vehicle is within {@code targetDetectionRadius}
     *       of the target, that counts as a capture → flip to CONSUMING and record the
     *       search time. This is the RL <b>terminal-state detector</b> — the only thing
     *       that ends an episode.</li>
     *   <li>CONSUMING: after a 3-second dwell, run the episode close-out —
     *       {@code recordTargetCapture} for all vehicles, then {@code endEpisode} once,
     *       print progress, respawn the swarm and re-seed the caches, and
     *       {@link #spawnNextTarget} (which returns the machine to SEARCHING).</li>
     * </ul>
     */
    void checkTargetStatus() {
        if (currentTarget == null) return;

        if (!isConsuming) {
            targetSearchElapsedMillis = System.currentTimeMillis() - targetSearchStartTime;
        } else {
            targetSearchElapsedMillis = 0;
        }

        if (!isConsuming) {
            double nearestDistance = Double.MAX_VALUE;
            for (Vehicle v : allVehicles) {
                double d = Math.sqrt(Math.pow(v.pos[0] - currentTarget[0], 2) +
                        Math.pow(v.pos[1] - currentTarget[1], 2));
                if (d < nearestDistance) nearestDistance = d;
            }

            if (nearestDistance < targetDetectionRadius) {
                isConsuming = true;
                lastCaptureMillis = targetSearchElapsedMillis;
                targetSearchElapsedMillis = 0;
                consumptionStartTime = System.currentTimeMillis();
            }
        } else {
            if (System.currentTimeMillis() - consumptionStartTime > 3000) {
                // Record learning for every vehicle before resetting
                for (Vehicle v : allVehicles) {
                    qLearning.recordTargetCapture(v.pos, currentTarget, allBlackHoles, allObstacles);
                }
                // Close out the episode ONCE: capture-time stats + epsilon decay
                qLearning.endEpisode(lastCaptureMillis);
                // Print learning progress to console
                System.out.printf("Episode %3d | capture: %5dms | avg(last200): %6.0fms | ε=%.3f%n",
                        qLearning.getEpisodeCount(),
                        lastCaptureMillis,
                        qLearning.getAverageCaptureMs(),
                        qLearning.getEpsilon());

                // Respawn the whole swarm back at the spawn circle so they
                // start each episode as a coherent group rather than scattered
                // across the world from the previous dispersal.
                for (Vehicle v : allVehicles) {
                    placeVehicleInSpawnCircle(v);
                    qLearning.initVehicle(v.id, v.pos);
                }

                isDispersing = false; // no random scatter phase needed
                spawnNextTarget();
            }
        }
    }

    /** Launches the simulation. */
    public static void main(String[] args) {
        new Simulation();
    }
}