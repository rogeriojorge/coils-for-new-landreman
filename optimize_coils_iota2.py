"""Coils for the iota = 2 analytic MHD equilibrium of Landreman (arXiv:2609.26742, section 2): weighted penalties."""
import sys
import numpy as np
from landreman_equilibria import case
from coil_optimization import targets, initial_coils, optimize, report, save_results

""" Equilibrium: figure 1 of the paper (eps = 1/2, edge psi = 1/64), 1 m axis radius, 1 T on axis;
    "iota2_tau" or "iota2_tau_mirror" as argument for Issan et al.'s tau = +-0.5 (no stellarator symmetry) """
CASE = sys.argv[1] if len(sys.argv) > 1 else "iota2"
eq = case(CASE, major_radius=1.0, B_axis=1.0, inner_fraction=0.25)  # interior field matched at half radius

""" Coils and penalties """
N_COILS, ORDER, N_SEGMENTS, COIL_MINOR_RADIUS = 6 * (2 - eq["stellsym"]), 6, 140, 0.55  # per half period or period
LIMITS = dict(length=4.5, curvature=5.0, msc=14.0, total_curvature=5 * np.pi, arclength=0.05,
              coil_distance=0.12, surface_distance=0.2)  # m, 1/m, 1/m^2, rad, -, m, m
WEIGHTS = dict(field=1e4, boundary=1e4, length=1e2, curvature=1e1, msc=1e1, total_curvature=1e1, arclength=1e-2,
               coil_distance=1e3, surface_distance=1e3, linking=1e1)
MAXITER = 3000

""" Setting up the exact targets and the initial coils """
t = targets(eq["surface"], eq["B"], eq["inner"], stellsym=eq["stellsym"])
coils0 = initial_coils(t, N_COILS, ORDER, N_SEGMENTS, 1.0, COIL_MINOR_RADIUS)

""" Running the optimization """
report("Initial coils", coils0, t, N_COILS)
coils, history, snapshots = optimize(coils0, t, LIMITS, WEIGHTS, N_COILS, MAXITER)

""" Results """
report("Optimized coils", coils, t, N_COILS)
save_results(CASE, coils, t, history, snapshots, eq["title"] + " (penalty)")
