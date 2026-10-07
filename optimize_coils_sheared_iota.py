"""Coils for the sheared-iota analytic MHD equilibria of Landreman (arXiv:2609.26742, section 3): weighted penalties."""
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from landreman_equilibria import sheared_B, sheared_surface, sheared_iota_axis
from coil_optimization import targets, initial_coils, terms, residuals, report, save_results

""" Equilibrium: (eps, S, k_b, lambda). A and B are from the paper's figure 2; D keeps the singular foci
    (x, y) = (0, +-sqrt(eps)) far from the plasma, so |B| varies little on the boundary """
CASE = "D"
EPS, S, K_EDGE, LAM = dict(A=(1.08, 3.0, 0.7, 3.5), B=(4.0, 3.5, 0.7, 3.5), D=(1.0, 2.0, 0.5, 3.5))[CASE]
MAJOR_RADIUS, B_AXIS = 1.0, 1.0             # mean axis radius (m) and |B| on axis at phi = 0 (T)
TARGET_FLUX_FRACTION = 0.25                 # interior field matched on psi = 0.25 psi_edge (half radius)

""" Coils and penalties """
N_COILS, ORDER, N_SEGMENTS, COIL_MINOR_RADIUS = 6, 4, 100, 0.55
LIMITS = dict(length=4.0, curvature=4.0, msc=9.0, total_curvature=3 * np.pi, arclength=0.05,
              coil_distance=0.12, surface_distance=0.2)  # m, 1/m, 1/m^2, rad, -, m, m
WEIGHTS = dict(field=1e4, normal=1e4, length=1e2, curvature=1e1, msc=1e1, total_curvature=1e1, arclength=1e-2,
               coil_distance=1e3, surface_distance=1e3, linking=1e1)
MAXITER, SNAPSHOT_EVERY = 1000, 10

""" Setting up the equilibrium, the exact targets and the initial coils """
h = np.sqrt(4 * S**2 + EPS**2)
L = 2 * MAJOR_RADIUS / (np.sqrt((h - EPS) / 2) + np.sqrt((h + EPS) / 2))  # axis semi-axes, eq. (3.27)
b = B_AXIS / jnp.linalg.norm(sheared_B(sheared_surface(0., 0., EPS, S, LAM, 0.), EPS, S, LAM))
B = lambda x: b * sheared_B(x / L, EPS, S, LAM)
surface = lambda theta, zeta: L * sheared_surface(theta, zeta, EPS, S, LAM, K_EDGE)
print(f"Case {CASE}: eps = {EPS}, S = {S}, k_b = {K_EDGE}, lambda = {LAM}, iota(0) = {sheared_iota_axis(EPS, S):.4f}")
t = targets(surface, B, lambda theta, zeta: L * sheared_surface(theta, zeta, EPS, S, LAM,
                                                                np.sqrt(TARGET_FLUX_FRACTION) * K_EDGE))
coils0 = initial_coils(t, N_COILS, ORDER, N_SEGMENTS, MAJOR_RADIUS, COIL_MINOR_RADIUS,
                       axis=(L * np.sqrt((h - EPS) / 2), L * np.sqrt((h + EPS) / 2)))

objective = lambda dofs: (lambda d: (sum(WEIGHTS[k] * v for k, v in d.items()), d))(
    terms(coils0.with_dofs(dofs), t, LIMITS, N_COILS))
value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
history, snapshots = [], []

def fun(x):
    (value, d), grad = value_and_grad(jnp.asarray(x))
    history.append({k: float(v) for k, v in d.items()})
    if len(history) % SNAPSHOT_EVERY == 1:
        coils = coils0.with_dofs(jnp.asarray(x))
        snapshots.append((len(history), np.asarray(coils.gamma),
                          float(jnp.mean(jnp.linalg.norm(residuals(coils, t)[0], axis=1)))))
    if len(history) % 100 == 0:
        print(f"  eval {len(history)}: total {value:.3e}, " + ", ".join(f"{k} {v:.2e}" for k, v in history[-1].items()))
    return float(value), np.asarray(grad)

""" Running the optimization """
report("Initial coils", coils0, t, N_COILS)
t0 = time.time()
result = minimize(fun, np.asarray(coils0.dofs), jac=True, method="L-BFGS-B",
                  options=dict(maxiter=MAXITER, maxcor=50, ftol=1e-15, gtol=1e-12))
print(f"{result.message} after {result.nit} iterations in {time.time() - t0:.1f} s")
coils = coils0.with_dofs(jnp.asarray(result.x))

""" Results """
report("Optimized coils", coils, t, N_COILS)
save_results(f"sheared_{CASE}", coils, t, history, snapshots, f"sheared ι, case {CASE} (penalty)")
