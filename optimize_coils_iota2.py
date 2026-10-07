"""Coils for the iota = 2 analytic MHD equilibrium of Landreman (arXiv:2609.26742, section 2): weighted penalties."""
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from landreman_equilibria import iota2_B, iota2_surface
from coil_optimization import targets, initial_coils, terms, residuals, report, save_results

""" Equilibrium: figure 1 of the paper (eps = 1/2, edge psi = 1/64), in physical units """
EPS, PSI_EDGE = 0.5, 1 / 64
MAJOR_RADIUS, B_AXIS = 1.0, 1.0             # axis radius (m) and |B| on axis at phi = 0 (T)
TARGET_FLUX_FRACTION = 0.25                 # interior field matched on psi = 0.25 psi_edge (half radius)

""" Coils and penalties """
N_COILS, ORDER, N_SEGMENTS, COIL_MINOR_RADIUS = 6, 4, 100, 0.55
LIMITS = dict(length=4.0, curvature=4.0, msc=9.0, total_curvature=3 * np.pi, arclength=0.05,
              coil_distance=0.12, surface_distance=0.2)  # m, 1/m, 1/m^2, rad, -, m, m
WEIGHTS = dict(field=1e4, normal=1e4, length=1e2, curvature=1e1, msc=1e1, total_curvature=1e1, arclength=1e-2,
               coil_distance=1e3, surface_distance=1e3, linking=1e1)
MAXITER, SNAPSHOT_EVERY = 1000, 10

""" Setting up the equilibrium, the exact targets and the initial coils """
L = MAJOR_RADIUS / np.sqrt(1 - EPS**2)
b = B_AXIS / jnp.linalg.norm(iota2_B(iota2_surface(0., 0., EPS, 0.), EPS))
B = lambda x: b * iota2_B(x / L, EPS)
surface = lambda theta, zeta: L * iota2_surface(theta, zeta, EPS, PSI_EDGE)
t = targets(surface, B, lambda theta, zeta: L * iota2_surface(theta, zeta, EPS, TARGET_FLUX_FRACTION * PSI_EDGE))
coils0 = initial_coils(t, N_COILS, ORDER, N_SEGMENTS, MAJOR_RADIUS, COIL_MINOR_RADIUS)

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
save_results("iota2", coils, t, history, snapshots, "ι = 2, ε = 1/2 (penalty)")
