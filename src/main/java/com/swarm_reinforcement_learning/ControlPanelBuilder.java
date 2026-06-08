package com.swarm_reinforcement_learning;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.BorderLayout;
import java.awt.GridLayout;

/**
 * Builds the bottom control panel and wires its widgets to live state.
 *
 * <p>Keeps UI concerns out of {@link Simulation}. Each slider registers a listener
 * that writes <b>directly</b> into the shared {@code static} weights on {@link Vehicle}
 * (e.g. {@code F_ZUS_WEIGHT}, {@code OBS_WEIGHT}, {@code BASE_AVOIDANCE_RADIUS}) or
 * into {@link Simulation} fields (e.g. {@code targetDetectionRadius}), so tuning takes
 * effect on the very next tick with no restart. Toggles flip {@code show…} flags on
 * {@link Canvas}; the "New Target" button calls {@link Simulation#spawnNextTarget}.</p>
 *
 * <p>Note: the RL hyperparameters, the black-hole weight ({@code f_bh = 2.0}) and the
 * Q-bias blend weight (0.35) are intentionally <i>not</i> exposed here — they live as
 * constants in their own classes.</p>
 */
public class ControlPanelBuilder {

    private JPanel controlPanel;
    private Canvas canvas;
    private Simulation simulation;

    public ControlPanelBuilder(Canvas canvas, Simulation simulation) {
        this.canvas = canvas;
        this.simulation = simulation;
    }

    /**
     * Build the complete control panel with all sliders and toggles
     */
    public JPanel buildControlPanel() {
        controlPanel = new JPanel(new GridLayout(2, 5, 0, 4));
        controlPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Control Panel"),
                BorderFactory.createEmptyBorder(8, 0, 10, 0)));

        // Add all control tiles
        controlPanel.add(createAvoidanceRadiusSlider());
        controlPanel.add(createAvoidanceMultiplierSlider());
        controlPanel.add(createObstacleWeightSlider());
        controlPanel.add(createCohesionWeightSlider());
        controlPanel.add(createSeparationWeightSlider());
        controlPanel.add(createAlignmentWeightSlider());
        controlPanel.add(createTargetDetectionRadiusSlider());
        controlPanel.add(createObstacleRadiusToggle());
        controlPanel.add(createBlackHoleRadiusToggle());
        controlPanel.add(createTargetRadiusToggle());
        controlPanel.add(createShowGridToggle());
        controlPanel.add(createNewTargetButton());

        // Fill remaining cells to maintain 2x5 grid shape
        while (controlPanel.getComponentCount() < 10) {
            controlPanel.add(new JPanel());
        }

        return controlPanel;
    }

    /**
     * Reusable helper to create a control tile (label + slider)
     */
    private JPanel createControlTile(JLabel label, JSlider slider) {
        JPanel tile = new JPanel(new BorderLayout(4, 4));
        tile.add(label, BorderLayout.NORTH);
        tile.add(slider, BorderLayout.CENTER);
        return tile;
    }

    /**
     * Avoidance radius slider (0 - 200)
     */
    private JPanel createAvoidanceRadiusSlider() {
        JLabel lblRadius = new JLabel("AvoidRadius: " + (int) Vehicle.BASE_AVOIDANCE_RADIUS);
        JSlider sliderRadius = new JSlider(0, 200, (int) Math.round(Vehicle.BASE_AVOIDANCE_RADIUS));
        sliderRadius.setMajorTickSpacing(50);
        sliderRadius.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                Vehicle.BASE_AVOIDANCE_RADIUS = sliderRadius.getValue();
                lblRadius.setText("AvoidRadius: " + sliderRadius.getValue());
                canvas.repaint();
            }
        });
        return createControlTile(lblRadius, sliderRadius);
    }

    /**
     * Avoidance multiplier slider (0.0 - 10.0 mapped to 0 - 100)
     */
    private JPanel createAvoidanceMultiplierSlider() {
        JLabel lblMult = new JLabel("AvoidMult: " + Vehicle.AVOIDANCE_MULTIPLIER);
        JSlider sliderMult = new JSlider(0, 100, (int) Math.round(Vehicle.AVOIDANCE_MULTIPLIER * 10));
        sliderMult.setMajorTickSpacing(25);
        sliderMult.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                Vehicle.AVOIDANCE_MULTIPLIER = sliderMult.getValue() / 10.0;
                lblMult.setText("AvoidMult: " + String.format("%.1f", Vehicle.AVOIDANCE_MULTIPLIER));
                canvas.repaint();
            }
        });
        return createControlTile(lblMult, sliderMult);
    }

    /**
     * Obstacle weight slider (0.0 - 2.0 mapped to 0 - 200)
     */
    private JPanel createObstacleWeightSlider() {
        JLabel lblWeight = new JLabel("ObsWeight: " + Vehicle.OBS_WEIGHT);
        JSlider sliderWeight = new JSlider(0, 200, (int) Math.round(Vehicle.OBS_WEIGHT * 100));
        sliderWeight.setMajorTickSpacing(50);
        sliderWeight.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                Vehicle.OBS_WEIGHT = sliderWeight.getValue() / 100.0;
                lblWeight.setText("ObsWeight: " + String.format("%.2f", Vehicle.OBS_WEIGHT));
                canvas.repaint();
            }
        });
        return createControlTile(lblWeight, sliderWeight);
    }

    /**
     * Cohesion weight slider (0.0 - 2.0 mapped to 0 - 200)
     */
    private JPanel createCohesionWeightSlider() {
        JLabel lblZus = new JLabel("F_zus: " + String.format("%.2f", Vehicle.F_ZUS_WEIGHT));
        JSlider sliderZus = new JSlider(0, 200, (int) Math.round(Vehicle.F_ZUS_WEIGHT * 100));
        sliderZus.setMajorTickSpacing(50);
        sliderZus.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                Vehicle.F_ZUS_WEIGHT = sliderZus.getValue() / 100.0;
                lblZus.setText("F_zus: " + String.format("%.2f", Vehicle.F_ZUS_WEIGHT));
                canvas.repaint();
            }
        });
        return createControlTile(lblZus, sliderZus);
    }

    /**
     * Separation weight slider (0.0 - 2.0 mapped to 0 - 200)
     */
    private JPanel createSeparationWeightSlider() {
        JLabel lblSep = new JLabel("F_sep: " + String.format("%.2f", Vehicle.F_SEP_WEIGHT));
        JSlider sliderSep = new JSlider(0, 200, (int) Math.round(Vehicle.F_SEP_WEIGHT * 100));
        sliderSep.setMajorTickSpacing(50);
        sliderSep.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                Vehicle.F_SEP_WEIGHT = sliderSep.getValue() / 100.0;
                lblSep.setText("F_sep: " + String.format("%.2f", Vehicle.F_SEP_WEIGHT));
                canvas.repaint();
            }
        });
        return createControlTile(lblSep, sliderSep);
    }

    /**
     * Alignment weight slider (0.0 - 2.0 mapped to 0 - 200)
     */
    private JPanel createAlignmentWeightSlider() {
        JLabel lblAus = new JLabel("F_aus: " + String.format("%.2f", Vehicle.F_AUS_WEIGHT));
        JSlider sliderAus = new JSlider(0, 200, (int) Math.round(Vehicle.F_AUS_WEIGHT * 100));
        sliderAus.setMajorTickSpacing(50);
        sliderAus.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                Vehicle.F_AUS_WEIGHT = sliderAus.getValue() / 100.0;
                lblAus.setText("F_aus: " + String.format("%.2f", Vehicle.F_AUS_WEIGHT));
                canvas.repaint();
            }
        });
        return createControlTile(lblAus, sliderAus);
    }

    /**
     * Target detection radius slider (1 - 50)
     */
    private JPanel createTargetDetectionRadiusSlider() {
        JLabel lblDetect = new JLabel("Target DetectRadius: " + String.format("%.1f", simulation.targetDetectionRadius));
        JSlider sliderDetect = new JSlider(1, 50, (int) Math.round(simulation.targetDetectionRadius));
        sliderDetect.setMajorTickSpacing(10);
        sliderDetect.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                simulation.targetDetectionRadius = sliderDetect.getValue();
                lblDetect.setText("Target DetectRadius: " + String.format("%.1f", simulation.targetDetectionRadius));
                canvas.repaint();
            }
        });
        return createControlTile(lblDetect, sliderDetect);
    }

    /**
     * Toggle obstacle avoidance radius visualization
     */
    private JPanel createObstacleRadiusToggle() {
        JCheckBox chkRadius = new JCheckBox("Show obstacle radius", false);
        chkRadius.addActionListener(e -> {
            canvas.setShowObstacleRadius(chkRadius.isSelected());
            canvas.repaint();
        });
        JPanel tile = new JPanel(new BorderLayout(4, 4));
        tile.add(new JLabel("Toggle Obstacle Radius"), BorderLayout.NORTH);
        tile.add(chkRadius, BorderLayout.CENTER);
        return tile;
    }

    /**
     * Toggle black hole radius visualization
     */
    private JPanel createBlackHoleRadiusToggle() {
        JCheckBox chkBlackHoleRadius = new JCheckBox("Show black hole radius", false);
        chkBlackHoleRadius.addActionListener(e -> {
            canvas.setShowBlackHoleRadius(chkBlackHoleRadius.isSelected());
            canvas.repaint();
        });
        JPanel tile = new JPanel(new BorderLayout(4, 4));
        tile.add(new JLabel("Toggle Black Hole Radius"), BorderLayout.NORTH);
        tile.add(chkBlackHoleRadius, BorderLayout.CENTER);
        return tile;
    }

    /**
     * Toggle target detection radius visualization + show type1 circles
     */
    private JPanel createTargetRadiusToggle() {
        JCheckBox chkTargetRadius = new JCheckBox("Show target radius", true);
        chkTargetRadius.addActionListener(e -> {
            canvas.setShowTargetDetectionRadius(chkTargetRadius.isSelected());
            canvas.repaint();
        });

        JCheckBox chkType1Circle = new JCheckBox("Show type1 circle", false);
        chkType1Circle.addActionListener(e -> {
            canvas.setShowType1Circle(chkType1Circle.isSelected());
            canvas.repaint();
        });

        JPanel combinedToggleCenter = new JPanel(new GridLayout(1, 2, 12, 4));
        combinedToggleCenter.add(chkTargetRadius);
        combinedToggleCenter.add(chkType1Circle);

        JPanel tile = new JPanel(new BorderLayout(4, 4));
        tile.add(new JLabel("Toggle Target/Type1"), BorderLayout.NORTH);
        tile.add(combinedToggleCenter, BorderLayout.CENTER);

        // Initialize canvas toggles to match checkbox defaults
        canvas.setShowTargetDetectionRadius(chkTargetRadius.isSelected());
        canvas.setShowType1Circle(chkType1Circle.isSelected());

        return tile;
    }


    /**
     * Toggle the reinforcement-learning grid overlay: draws the state grid and prints
     * each cell's Q-value for the current target (numbers only — no colours or arrows).
     */
    private JPanel createShowGridToggle() {
        JCheckBox chkGrid = new JCheckBox("Show grid + Q-values", false);
        chkGrid.addActionListener(e -> {
            canvas.setShowGrid(chkGrid.isSelected());
            canvas.repaint();
        });
        JPanel tile = new JPanel(new BorderLayout(4, 4));
        tile.add(new JLabel("Toggle Q-Grid"), BorderLayout.NORTH);
        tile.add(chkGrid, BorderLayout.CENTER);
        return tile;
    }

    /**
     * Manual target regeneration button
     */
    private JPanel createNewTargetButton() {
        JButton btnNewTarget = new JButton("New Target");
        btnNewTarget.addActionListener(e -> {
            simulation.isDispersing = false;
            simulation.spawnNextTarget();
            canvas.updateTarget(simulation.currentTarget, simulation.isConsuming,
                    simulation.targetDetectionRadius, simulation.targetSearchElapsedMillis,
                    simulation.lastCaptureMillis);
            canvas.repaint();
        });
        JPanel tile = new JPanel(new BorderLayout(4, 4));
        tile.add(btnNewTarget, BorderLayout.CENTER);
        return tile;
    }
}