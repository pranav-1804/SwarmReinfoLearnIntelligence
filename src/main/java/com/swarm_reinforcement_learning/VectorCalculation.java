package com.swarm_reinforcement_learning;

/**
 * Stateless helper library for 2-D vector maths used throughout the simulation.
 *
 * <p>Every method is {@code static} and operates on {@code double[2]} arrays of the
 * form {@code {x, y}} — there is no reason to instantiate this class. These helpers
 * underpin every steering force in {@link Vehicle}: normalising directions, capping a
 * force's magnitude to {@code max_acc}, and measuring lengths and angles.</p>
 *
 * <p>Several names are German in origin, reflecting the project's roots:
 * {@code winkel} = angle, {@code betrag} = magnitude, {@code richtung} = direction,
 * {@code lot} = perpendicular foot, {@code skalPro} = scalar (dot) product.</p>
 */
public class VectorCalculation {

	/**
	 * Clamps a signed scalar {@code x} into the range {@code [-y, y]}, preserving its
	 * sign. Used by the vector overload to cap a force's magnitude. Prints a warning
	 * if the bound {@code y} is negative (a misuse).
	 */
	static double truncate(double x, double y) {
		if (y < 0)
			System.out.println("Mistake truncate");
		if (x > 0)
			return Math.min(x, y);
		else
			return Math.max(x, -y);
	}

	/**
	 * Returns a unit-length copy of {@code x} (same direction). If {@code x} is
	 * effectively zero (magnitude &lt; 1e-8) it is returned unchanged, guarding
	 * against division by zero.
	 */
	static double[] normalize(double[] x) {
		double[] res = new double[2];
		double norm = Math.sqrt(Math.pow(x[0], 2) + Math.pow(x[1], 2));
		res[0] = x[0];
		res[1] = x[1];
		if (norm > 1e-8) {
			res[0] = x[0] / norm;
			res[1] = x[1] / norm;
		}

		return res;
	}

	/**
	 * Caps the magnitude of vector {@code x} to {@code y} <b>without changing its
	 * direction</b>: if it is already shorter than {@code y} it is returned as-is.
	 * This is how every steering force is limited to {@code max_acc}.
	 */
	static double[] truncate(double[] x, double y) {
		if (y < 0)
			System.out.println("Mistake truncate");
		double[] res = normalize(x);
		res[0] = res[0] * truncate(length(x), y);
		res[1] = res[1] * truncate(length(x), y);
		return res;
	}

	/** Euclidean length of the vector: {@code sqrt(x0^2 + x1^2)}. */
	static double length(double[] x) {
		double res = Math.sqrt(Math.pow(x[0], 2) + Math.pow(x[1], 2));
		return res;
	}

	/**
	 * Absolute heading of {@code v1} measured from the +x axis, in {@code [0, 2*PI)}.
	 * Used by {@link Canvas} to rotate a vehicle's drawn shape along its velocity.
	 */
	static double angle(double[] v1) {

		double[] k = new double[2];
		double w;

		k[0] = 1;
		k[1] = 0;
		w = angle(k, v1);
		if (v1[1] < 0)
			w = 2 * Math.PI - w;
		return w;
	}

	/**
	 * Unsigned angle between {@code v1} and {@code v2} in {@code [0, PI]}, via the
	 * dot-product / {@code acos} formula. The cosine is clamped to {@code [-1, 1]}
	 * first so floating-point drift can never make {@code acos} return NaN.
	 */
	static double angle(double[] v1, double[] v2) {
		double betrag_v1 = Math.sqrt(Math.pow(v1[0], 2) + Math.pow(v1[1], 2));
		double betrag_v2 = Math.sqrt(Math.pow(v2[0], 2) + Math.pow(v2[1], 2));
		double winkelRad;//angleRadiant
		double skalPro;

		if (betrag_v1 == 0 || betrag_v2 == 0) {
			winkelRad = 0;
		} else {
			skalPro = (v1[0] * v2[0]) + (v1[1] * v2[1]);
			winkelRad = skalPro / (betrag_v1 * betrag_v2);
			if (winkelRad > 1)
				winkelRad = 1;
			if (winkelRad < -1)
				winkelRad = -1;
			winkelRad = Math.acos(winkelRad);
		}
		return winkelRad;
	}

	/**
	 * Closest point on the line <b>segment</b> {@code ort1}–{@code ort2} to the point
	 * {@code pkt}: if the perpendicular projection falls beyond an endpoint it snaps to
	 * that endpoint, otherwise it returns the perpendicular foot ({@code lot}).
	 *
	 * <p>NOTE: currently unused — {@link Vehicle#obstacleAvoidance} computes its
	 * closest point inline with axis clamps — kept for reference / potential reuse.</p>
	 */
	static double[] dist(double[] pkt, double[] ort1, double[] ort2) {
		double[] abstandsPkt = new double[2];
		abstandsPkt[0] = 0;
		abstandsPkt[1] = 0;

		double dist;
		double winkel1;
		double winkel2;

		double[] richtung1 = new double[2];
		double[] richtung2 = new double[2];

		richtung1[0] = ort2[0] - ort1[0];
		richtung1[1] = ort2[1] - ort1[1];
		richtung2[0] = pkt[0] - ort1[0];
		richtung2[1] = pkt[1] - ort1[1];
		winkel1 = angle(richtung1, richtung2);
		richtung1[0] = ort1[0] - ort2[0];
		richtung1[1] = ort1[1] - ort2[1];
		richtung2[0] = pkt[0] - ort2[0];
		richtung2[1] = pkt[1] - ort2[1];
		winkel2 = angle(richtung1, richtung2);

		if (winkel1 >= Math.PI / 2) {
			abstandsPkt[0] = ort1[0];
			abstandsPkt[1] = ort1[1];
		} else if (winkel2 >= Math.PI / 2) {
			abstandsPkt[0] = ort2[0];
			abstandsPkt[1] = ort2[1];
		} else {
			richtung1[0] = ort2[0] - ort1[0];
			richtung1[1] = ort2[1] - ort1[1];
			richtung2[0] = pkt[0] - ort1[0];
			richtung2[1] = pkt[1] - ort1[1];
			winkel1 = angle(richtung1, richtung2);
			dist = length(richtung2);
			double lot = dist * Math.cos(winkel1);
			double[] lotPkt = normalize(richtung1);
			abstandsPkt[0] = ort1[0] + lot * lotPkt[0];
			abstandsPkt[1] = ort1[1] + lot * lotPkt[1];
		}

		return abstandsPkt;
	}

}
