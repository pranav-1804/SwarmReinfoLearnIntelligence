package com.swarm_reinforcement_learning;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.util.ArrayList;
import javax.swing.JPanel;

/**
 * The render surface (a Swing {@link JPanel}). Read-only: {@link Simulation} pushes
 * state in via {@link #updateTarget} and {@code repaint()}, and {@link #paintComponent}
 * redraws the scene each frame — it never mutates the simulation.
 *
 * <p>All drawing converts world coordinates to screen pixels by dividing by
 * {@code pix}. Draw order (back to front): world frame, spawn circle + timer,
 * spawnability overlay, target, vehicles, obstacles, black holes — so obstacles and
 * holes paint on top of the agents. Debug overlays are gated by the {@code show…}
 * toggles set from {@link ControlPanelBuilder}.</p>
 */
public class Canvas extends JPanel {

    ArrayList<Vehicle> allVehicles;
    double pix;
    ArrayList<Obstacle> allObstacles;
    ArrayList<BlackHole> allBlackHoles;

    // New fields to track the target state
    double[] currentTarget;
    boolean isConsuming;
    boolean showObstacleRadius;
    boolean showBlackHoleRadius;
    boolean showTargetDetectionRadius;
    double targetDetectionRadius;
    boolean showType1Circle;
    boolean showTimer;
    long targetSearchElapsedMillis;
    long lastCaptureMillis;

    boolean showGrid; // toggles the RL state-grid + Q-value overlay
    private SwarmQLearning qLearning; // read-only reference, used only by the grid/Q-value overlay

    private int world_margin;
    private int world_border_thickness;

    double[] canvas_dimensions = new double[2];

    Canvas(ArrayList<Vehicle> allVehicles, double pix, ArrayList<Obstacle> obstacles, ArrayList<BlackHole> blackHoles, int width, int height) {
        this.allVehicles = allVehicles;
        this.pix = pix;
        this.allObstacles = obstacles;
        this.allBlackHoles = blackHoles;
        this.setBackground(Color.lightGray);
        this.canvas_dimensions[0] = width;
        this.canvas_dimensions[1] = height;
        setSize(width, height);
    }

    /**
     * Receives the current target and episode state from {@link Simulation} each tick
     * and stores it for the next {@link #paintComponent}. This is the one channel by
     * which the simulation feeds live data to the view.
     */
    public void updateTarget(double[] target, boolean consuming, double targetDetectionRadius, long targetSearchElapsedMillis, long lastCaptureMillis) {
        this.currentTarget = target;
        this.isConsuming = consuming;
        this.targetDetectionRadius = targetDetectionRadius;
        this.targetSearchElapsedMillis = targetSearchElapsedMillis;
        this.lastCaptureMillis = lastCaptureMillis;
    }

    public void setShowObstacleRadius(boolean showObstacleRadius) {
        this.showObstacleRadius = showObstacleRadius;
    }

    public void setShowBlackHoleRadius(boolean showBlackHoleRadius) {
        this.showBlackHoleRadius = showBlackHoleRadius;
    }

    public void setShowTargetDetectionRadius(boolean showTargetDetectionRadius) {
        this.showTargetDetectionRadius = showTargetDetectionRadius;
    }

    public void setShowType1Circle(boolean showType1Circle) {
        this.showType1Circle = showType1Circle;
    }

    public void setShowTimer(boolean showTimer) {
        this.showTimer = showTimer;
    }

    public void setShowGrid(boolean showGrid) {
        this.showGrid = showGrid;
    }

    /** Gives the view read-only access to the Q-learner for the grid/Q-value overlay. */
    public void setQLearning(SwarmQLearning qLearning) {
        this.qLearning = qLearning;
    }

    public Polygon kfzInPolygon(Vehicle fz) {
        Polygon q = new Polygon();
        int l = (int)(fz.FZL / pix);
        int b = (int)(fz.FZB / pix);
        int x = (int)(fz.pos[0] / pix);
        int y = (int)(fz.pos[1] / pix);
        int dia = (int)(Math.sqrt(Math.pow(l / 2, 2) + Math.pow(b / 2, 2)));
        double t = VectorCalculation.angle(fz.vel);
        double phi1 = Math.atan(fz.FZB / fz.FZL);
        double phi2 = Math.PI - phi1;
        double phi3 = Math.PI + phi1;
        double phi4 = 2 * Math.PI - phi1;

        q.addPoint((int)(x + (dia * Math.cos(t + phi1))), (int)(y + (dia * Math.sin(t + phi1))));
        q.addPoint((int)(x + (dia * Math.cos(t + phi2))), (int)(y + (dia * Math.sin(t + phi2))));
        q.addPoint((int)(x + (dia * Math.cos(t + phi3))), (int)(y + (dia * Math.sin(t + phi3))));
        q.addPoint((int)(x + (dia * Math.cos(t + phi4))), (int)(y + (dia * Math.sin(t + phi4))));
        return q;
    }

    /**
     * Renders the whole scene once per frame in back-to-front order (see the class
     * documentation). Each decorative element is drawn on a disposable
     * {@code g2d.create()} copy so stroke/font changes don't leak between elements.
     */
    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        // 1. Draw the World Border (Matches vehicle physical boundaries)
        int minPixelX = (int)(world_margin / pix);
        int minPixelY = (int)(world_margin / pix);
        int maxPixelX = (int)(canvas_dimensions[0] * Simulation.pix / pix);
        int maxPixelY = (int)(canvas_dimensions[1] * Simulation.pix / pix);

        int borderWidth = maxPixelX - minPixelX;
        int borderHeight = maxPixelY - minPixelY;

        // Choose how round you want the outer arena to look (e.g., 30 pixels)
        double worldCornerRounding = 30.0;

        // Create a rounded rectangle for the global boundary area
        java.awt.geom.RoundRectangle2D roundedWorld = new java.awt.geom.RoundRectangle2D.Double(
                minPixelX, minPixelY, borderWidth, borderHeight, worldCornerRounding, worldCornerRounding
        );

        // Draw a soft light-gray filled background inside the active simulation zone
        g2d.setColor(new Color(245, 245, 245));
        g2d.fill(roundedWorld);

        // Draw the thick dark frame line
        g2d.setColor(Color.DARK_GRAY);
        g2d.setStroke(new java.awt.BasicStroke(world_border_thickness));
        g2d.draw(roundedWorld);

        // 1b. Paint the initial vehicle spawn circle in the top-right corner.
        double spawnRadiusWorld = Simulation.SPAWN_POINT_RADIUS;
        double spawnCenterWorldX = (canvas_dimensions[0] * Simulation.pix) - world_margin - spawnRadiusWorld;
        double spawnCenterWorldY = world_margin + spawnRadiusWorld;
        double spawnRadiusPx = spawnRadiusWorld / pix;
        double spawnCenterPxX = spawnCenterWorldX / pix;
        double spawnCenterPxY = spawnCenterWorldY / pix;

        Graphics2D spawnGraphics = (Graphics2D) g2d.create();
        float[] spawnDashPattern = {4.0f, 6.0f};
        spawnGraphics.setColor(new Color(220, 0, 0));
        spawnGraphics.setStroke(new java.awt.BasicStroke(
                1.6f,
                java.awt.BasicStroke.CAP_BUTT,
                java.awt.BasicStroke.JOIN_MITER,
                10.0f,
                spawnDashPattern,
                0.0f
        ));
        spawnGraphics.draw(new java.awt.geom.Ellipse2D.Double(
                spawnCenterPxX - spawnRadiusPx,
                spawnCenterPxY - spawnRadiusPx,
                spawnRadiusPx * 2.0,
                spawnRadiusPx * 2.0
        ));

        spawnGraphics.setFont(spawnGraphics.getFont().deriveFont(java.awt.Font.BOLD, 12.0f));
        java.awt.FontMetrics spawnMetrics = spawnGraphics.getFontMetrics();
        String spawnLabelTop = "SPAWN POINT";
        String spawnLabelBottom = "(Spawnpunkt)";
        int topLabelWidth = spawnMetrics.stringWidth(spawnLabelTop);
        int bottomLabelWidth = spawnMetrics.stringWidth(spawnLabelBottom);
        int labelAscent = spawnMetrics.getAscent();
        int lineGap = spawnMetrics.getHeight() - labelAscent;
        spawnGraphics.setColor(new Color(160, 0, 0));
        spawnGraphics.drawString(
                spawnLabelTop,
                (float)(spawnCenterPxX - topLabelWidth / 2.0),
                (float)(spawnCenterPxY - 2.0)
        );
        spawnGraphics.drawString(
                spawnLabelBottom,
                (float)(spawnCenterPxX - bottomLabelWidth / 2.0),
                (float)(spawnCenterPxY + labelAscent + lineGap)
        );

        if (showTimer) {
            spawnGraphics.setFont(spawnGraphics.getFont().deriveFont(java.awt.Font.BOLD, 13.0f));
            java.awt.FontMetrics timerMetrics = spawnGraphics.getFontMetrics();
            String searchText = String.format("Search time: %.1f s", targetSearchElapsedMillis / 1000.0);
            String captureText = lastCaptureMillis >= 0
                    ? String.format("Last capture: %.1f s", lastCaptureMillis / 1000.0)
                    : "Last capture: --";
            int searchTextWidth = timerMetrics.stringWidth(searchText);
            int captureTextWidth = timerMetrics.stringWidth(captureText);
            int timerWidth = Math.max(searchTextWidth, captureTextWidth) + 18;
            int timerHeight = timerMetrics.getHeight() * 2 + 10;
            int timerX = (int)(spawnCenterPxX - timerWidth / 2.0);
            int timerY = (int)(spawnCenterPxY + spawnRadiusPx + 8.0);

            spawnGraphics.setColor(new Color(255, 255, 255, 210));
            spawnGraphics.fillRoundRect(timerX, timerY, timerWidth, timerHeight, 16, 16);
            spawnGraphics.setColor(new Color(70, 70, 70));
            spawnGraphics.drawRoundRect(timerX, timerY, timerWidth, timerHeight, 16, 16);
            spawnGraphics.setColor(Color.BLACK);
            spawnGraphics.drawString(searchText, timerX + 9, timerY + timerMetrics.getAscent() + 2);
            spawnGraphics.drawString(captureText, timerX + 9, timerY + timerMetrics.getAscent() + timerMetrics.getHeight() + 2);
        }
        spawnGraphics.dispose();

        // 2. Paint spawnability overlay (green = allowed, red = blocked)
        Graphics2D overlay = (Graphics2D) g2d.create();
        int stepPx = 12; // grid cell size in pixels (tune for speed/clarity)
        double buffer = 5.0; // buffer used when checking obstacle containment (world units)
        int panelW = getWidth();
        int panelH = getHeight();
        for (int px = 0; px < panelW; px += stepPx) {
            for (int py = 0; py < panelH; py += stepPx) {
                double worldX = px * pix;
                double worldY = py * pix;
                boolean blocked = false;
                for (Obstacle obs : allObstacles) {
                    double ox = obs.position[0];
                    double oy = obs.position[1];
                    double ow = obs.getObstacle_width();
                    double oh = obs.getObstacle_height();
                    boolean insideX = worldX >= (ox - buffer) && worldX <= (ox + ow + buffer);
                    boolean insideY = worldY >= (oy - buffer) && worldY <= (oy + oh + buffer);
                    if (insideX && insideY) { blocked = true; break; }
                }
                if (blocked) overlay.setColor(new java.awt.Color(255, 0, 0, 40));
                else overlay.setColor(new java.awt.Color(0, 255, 0, 20));
                int w = Math.min(stepPx, panelW - px);
                int h = Math.min(stepPx, panelH - py);
                overlay.fillRect(px, py, w, h);
            }
        }
        overlay.dispose();

        // 3. Paint Target
        if (currentTarget != null) {
            int tx = (int)(currentTarget[0] / pix);
            int ty = (int)(currentTarget[1] / pix);
            int size = 20;

            g2d.setColor(isConsuming ? Color.GREEN : Color.RED);
            g2d.fillOval(tx - size / 2, ty - size / 2, size, size);

            if (showTargetDetectionRadius) {
                double radiusPx = targetDetectionRadius / pix;
                double diameterPx = radiusPx * 2.0;

                Graphics2D targetRadiusGraphics = (Graphics2D) g2d.create();
                float[] dashPattern = {6.0f, 6.0f};
                targetRadiusGraphics.setColor(new Color(0, 0, 0, 180));
                targetRadiusGraphics.setStroke(new java.awt.BasicStroke(
                        1.5f,
                        java.awt.BasicStroke.CAP_BUTT,
                        java.awt.BasicStroke.JOIN_MITER,
                        10.0f,
                        dashPattern,
                        0.0f
                ));
                targetRadiusGraphics.draw(new java.awt.geom.Ellipse2D.Double(
                        tx - radiusPx,
                        ty - radiusPx,
                        diameterPx,
                        diameterPx
                ));
                targetRadiusGraphics.dispose();
            }
        }

        // 3. Paint Vehicles
        for (Vehicle fz : allVehicles) {
            Polygon q = kfzInPolygon(fz);
            g2d.setColor(Color.BLACK);
            g2d.draw(q);

            if (fz.type == 1 && showType1Circle) {
                int seite = (int)(fz.rad_zus / pix);
                g2d.drawOval((int)(fz.pos[0] / pix) - seite, (int)(fz.pos[1] / pix) - seite, 2 * seite, 2 * seite);
                seite = (int)(fz.rad_sep / pix);
                g2d.drawOval((int)(fz.pos[0] / pix) - seite, (int)(fz.pos[1] / pix) - seite, 2 * seite, 2 * seite);
            }
        }

        // 4. Paint Obstacles
        for (Obstacle obs : allObstacles) {
            // Scale everything to pixel positions
            double x = obs.position[0] / pix;
            double y = obs.position[1] / pix;
            double w = obs.getObstacle_width() / pix;
            double h = obs.getObstacle_height() / pix;

            double cornerRounding = 15.0;

            java.awt.geom.RoundRectangle2D roundedBox = new java.awt.geom.RoundRectangle2D.Double(
                    x, y, w, h, cornerRounding, cornerRounding
            );

            g2d.setColor(new Color(255, 236, 153));
            g2d.fill(roundedBox);
            g2d.draw(roundedBox);

            String obstacleName = obs.getObstacle_name();
            if (obstacleName != null && !obstacleName.isBlank()) {
                Graphics2D labelGraphics = (Graphics2D) g2d.create();
                labelGraphics.setColor(new Color(90, 60, 0));

                double maxFontSize = Math.max(10.0, Math.min(18.0, h * 0.35));
                labelGraphics.setFont(labelGraphics.getFont().deriveFont(java.awt.Font.BOLD, (float) maxFontSize));
                java.awt.FontMetrics metrics = labelGraphics.getFontMetrics();

                String labelText = obstacleName;
                while (metrics.stringWidth(labelText) > (w - 8.0) && labelText.length() > 1) {
                    labelText = labelText.substring(0, labelText.length() - 1);
                }

                int textWidth = metrics.stringWidth(labelText);
                int textHeight = metrics.getAscent() - metrics.getDescent();
                float textX = (float) (x + (w - textWidth) / 2.0);
                float textY = (float) (y + (h + textHeight) / 2.0);
                labelGraphics.drawString(labelText, textX, textY);
                labelGraphics.dispose();
            }

            if (showObstacleRadius) {
                double centerX = x + (w / 2.0);
                double centerY = y + (h / 2.0);
                double radiusWorld = Vehicle.BASE_AVOIDANCE_RADIUS + Math.max(obs.getObstacle_width() / 2.0, obs.getObstacle_height() / 2.0);
                double radiusPx = radiusWorld / pix;
                double diameterPx = radiusPx * 2.0;

                Graphics2D radiusGraphics = (Graphics2D) g2d.create();
                float[] dashPattern = {8.0f, 8.0f};
                radiusGraphics.setColor(Color.BLACK);
                radiusGraphics.setStroke(new java.awt.BasicStroke(
                        1.5f,
                        java.awt.BasicStroke.CAP_BUTT,
                        java.awt.BasicStroke.JOIN_MITER,
                        10.0f,
                        dashPattern,
                        0.0f
                ));
                radiusGraphics.draw(new java.awt.geom.Ellipse2D.Double(
                        centerX - radiusPx,
                        centerY - radiusPx,
                        diameterPx,
                        diameterPx
                ));
                radiusGraphics.dispose();
            }
        }

        // 5. Paint Black Holes
        if (allBlackHoles != null) {
            for (BlackHole bh : allBlackHoles) {
                double cx = bh.position[0] / pix;
                double cy = bh.position[1] / pix;
                double r = bh.getHole_radius() / pix;

                Graphics2D bhG = (Graphics2D) g2d.create();
                java.awt.geom.Ellipse2D holeShape = new java.awt.geom.Ellipse2D.Double(cx - r, cy - r, r * 2.0, r * 2.0);
                bhG.setPaint(new java.awt.RadialGradientPaint(
                        new java.awt.geom.Point2D.Double(cx, cy),
                        (float) r,
                        new float[]{0.0f, 0.45f, 1.0f},
                        new Color[]{Color.BLACK, new Color(55, 55, 55), new Color(130, 130, 130)}
                ));
                bhG.fill(holeShape);

                bhG.setColor(new Color(170, 170, 170, 180));
                bhG.draw(holeShape);

                if (showBlackHoleRadius) {
                    double radiusWorld = bh.getHole_radius() + 20.0;
                    double radiusPx = radiusWorld / pix;
                    double diameterPx = radiusPx * 2.0;

                    Graphics2D radiusGraphics = (Graphics2D) bhG.create();
                    float[] dashPattern = {8.0f, 8.0f};
                    radiusGraphics.setColor(new Color(0, 0, 0, 180));
                    radiusGraphics.setStroke(new java.awt.BasicStroke(
                            1.5f,
                            java.awt.BasicStroke.CAP_BUTT,
                            java.awt.BasicStroke.JOIN_MITER,
                            10.0f,
                            dashPattern,
                            0.0f
                    ));
                    radiusGraphics.draw(new java.awt.geom.Ellipse2D.Double(
                            cx - radiusPx,
                            cy - radiusPx,
                            diameterPx,
                            diameterPx
                    ));
                    radiusGraphics.dispose();
                }

                // Draw name centered inside the black hole (reduced size)
                String name = bh.getHole_name();
                if (name != null && !name.isBlank()) {
                    float fontSize = Math.max(8f, (float)(r * 0.25));
                    bhG.setFont(bhG.getFont().deriveFont(java.awt.Font.BOLD, fontSize));
                    java.awt.FontMetrics fm = bhG.getFontMetrics();
                    int w = fm.stringWidth(name);
                    bhG.setColor(Color.WHITE);
                    float textX = (float)(cx - w / 2.0);
                    float textY = (float)(cy + (fm.getAscent() - fm.getDescent()) / 2.0);
                    bhG.drawString(name, textX, textY);
                }

                bhG.dispose();
            }
        }

        // Q-table grid overlay (debug): the state grid with each cell's Q-value inside.
        if (showGrid) {
            paintGrid(g2d);
        }
    }

    /**
     * Draws the reinforcement-learning state grid and, inside each cell, that cell's
     * Q-value for the CURRENT target (the best/maximum Q over the four actions).
     * Just the grid lines and the numbers — no colour shading and no arrows.
     */
    private void paintGrid(Graphics2D g2dOuter) {
        if (qLearning == null) return;
        Graphics2D g2d = (Graphics2D) g2dOuter.create();

        int gw = qLearning.getGridW();
        int gh = qLearning.getGridH();
        double screenW = (canvas_dimensions[0] * Simulation.pix) / pix;
        double screenH = (canvas_dimensions[1] * Simulation.pix) / pix;
        double cellW = screenW / gw;
        double cellH = screenH / gh;

        g2d.setFont(g2d.getFont().deriveFont(java.awt.Font.PLAIN, 9.0f));
        java.awt.FontMetrics fm = g2d.getFontMetrics();

        for (int gx = 0; gx < gw; gx++) {
            for (int gy = 0; gy < gh; gy++) {
                int px = (int) Math.round(gx * cellW);
                int py = (int) Math.round(gy * cellH);

                // Grid cell outline.
                g2d.setColor(new Color(110, 110, 110, 130));
                g2d.drawRect(px, py, (int) cellW, (int) cellH);

                // The cell's Q-value for the current target, centred inside the cell.
                double value = qLearning.getBestValueForCell(gx, gy, currentTarget);
                String txt = String.format("%.1f", value);
                g2d.setColor(Color.BLACK);
                int tw = fm.stringWidth(txt);
                g2d.drawString(txt,
                        (float) (px + (cellW - tw) / 2.0),
                        (float) (py + cellH / 2.0 + 3));
            }
        }
        g2d.dispose();
    }

    public int getWorld_border_thickness() {
        return world_border_thickness;
    }

    public void setWorld_border_thickness(int world_border_thickness) {
        this.world_border_thickness = world_border_thickness;
    }

    public int getWorld_margin() {
        return world_margin;
    }

    public void setWorld_margin(int world_margin) {
        this.world_margin = world_margin;
    }
}