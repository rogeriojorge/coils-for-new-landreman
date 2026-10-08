"""Coils for the sheared-iota analytic MHD equilibria of Landreman (arXiv:2609.26742, section 3): weighted penalties."""
import numpy as np
from landreman_equilibria import case, sheared_iota_axis, CASES
from coil_optimization import targets, initial_coils, optimize, report, save_results

""" Equilibrium: A and B from the paper's figure 2, D new (see README); 1 m mean axis radius, 1 T on axis """
CASE = "D"
eq = case(CASE, major_radius=1.0, B_axis=1.0, inner_fraction=0.25)  # interior field matched at half radius
print(f"Case {CASE}: (eps, S, k_b, lambda) = {CASES[CASE]}, iota(0) = {sheared_iota_axis(*CASES[CASE][:2]):.4f}")

""" Coils and penalties """
N_COILS, ORDER, N_SEGMENTS, COIL_MINOR_RADIUS = 5, 4, 100, 0.55
LIMITS = dict(length=4.0, curvature=4.0, msc=9.0, total_curvature=3 * np.pi, arclength=0.05,
              coil_distance=0.12, surface_distance=0.2)  # m, 1/m, 1/m^2, rad, -, m, m
WEIGHTS = dict(field=1e4, boundary=1e4, length=1e2, curvature=1e1, msc=1e1, total_curvature=1e1, arclength=1e-2,
               coil_distance=1e3, surface_distance=1e3, linking=1e1)
MAXITER = 1000

""" Setting up the exact targets and the initial coils (centred on the elliptical axis) """
t = targets(eq["surface"], eq["B"], eq["inner"])
coils0 = initial_coils(t, N_COILS, ORDER, N_SEGMENTS, 1.0, COIL_MINOR_RADIUS, axis=eq["axis"])

""" Running the optimization """
report("Initial coils", coils0, t, N_COILS)
coils, history, snapshots = optimize(coils0, t, LIMITS, WEIGHTS, N_COILS, MAXITER)

""" Results """
report("Optimized coils", coils, t, N_COILS)
save_results(f"sheared_{CASE}", coils, t, history, snapshots, eq["title"] + " (penalty)")
