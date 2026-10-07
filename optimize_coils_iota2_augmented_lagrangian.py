"""Coils for the iota = 2 analytic MHD equilibrium of Landreman (arXiv:2609.26742): augmented Lagrangian.

The field mismatch is the objective; every engineering limit is a constraint with its own multiplier
(essos.augmented_lagrangian), so no penalty weights need tuning."""
import time
import jax
import jax.numpy as jnp
import numpy as np
import essos.augmented_lagrangian as alm
from landreman_equilibria import iota2_B, iota2_surface
from coil_optimization import targets, initial_coils, terms, constraints, residuals, report, save_results

""" Equilibrium: figure 1 of the paper (eps = 1/2, edge psi = 1/64), in physical units """
EPS, PSI_EDGE = 0.5, 1 / 64
MAJOR_RADIUS, B_AXIS = 1.0, 1.0             # axis radius (m) and |B| on axis at phi = 0 (T)
TARGET_FLUX_FRACTION = 0.25                 # interior field matched on psi = 0.25 psi_edge (half radius)

""" Coils and constraints """
N_COILS, ORDER, N_SEGMENTS, COIL_MINOR_RADIUS = 6, 4, 100, 0.55
LIMITS = dict(length=4.0, curvature=4.0, msc=9.0, total_curvature=3 * np.pi, arclength=0.05,
              coil_distance=0.12, surface_distance=0.2)  # m, 1/m, 1/m^2, rad, -, m, m
FIELD_SCALE = 1e2                           # objective = FIELD_SCALE^2 (field + normal mismatch)
OUTER_ITERATIONS = 40                       # multiplier updates, each an inner L-BFGS-B solve
AL = dict(model_lagrangian="Squared", beta=2.0, mu_max=1e4, eta_tol=1e-6, omega_tol=1e-8)

""" Setting up the equilibrium, the exact targets and the initial coils """
L = MAJOR_RADIUS / np.sqrt(1 - EPS**2)
b = B_AXIS / jnp.linalg.norm(iota2_B(iota2_surface(0., 0., EPS, 0.), EPS))
B = lambda x: b * iota2_B(x / L, EPS)
surface = lambda theta, zeta: L * iota2_surface(theta, zeta, EPS, PSI_EDGE)
t = targets(surface, B, lambda theta, zeta: L * iota2_surface(theta, zeta, EPS, TARGET_FLUX_FRACTION * PSI_EDGE))
coils0 = initial_coils(t, N_COILS, ORDER, N_SEGMENTS, MAJOR_RADIUS, COIL_MINOR_RADIUS)

""" Objective and constraints """
def mismatch(dofs):
    dB, Bn = residuals(coils0.with_dofs(dofs), t)
    return FIELD_SCALE * jnp.concatenate([dB.ravel() / np.sqrt(len(dB)), Bn.ravel() / np.sqrt(Bn.size)])

pick = lambda key: lambda dofs: constraints(coils0.with_dofs(dofs), t, LIMITS, N_COILS)[key]
C = alm.combine(*[alm.eq(pick(k), model_lagrangian=AL["model_lagrangian"])
                  for k in constraints(coils0, t, LIMITS, N_COILS)])
model = alm.ALM_model_jaxopt_lbfgsb(constraints=C, loss=mismatch, **AL)
params = coils0.dofs, C.init(coils0.dofs)
state, grad, info = model.init(params)

""" Running the optimization """
report("Initial coils", coils0, t, N_COILS)
history, snapshots, t0 = [], [], time.time()
for i in range(OUTER_ITERATIONS):
    params, state, grad, info = model.update(params, state, grad, info)
    coils = coils0.with_dofs(params[0])
    history.append({k: float(v) for k, v in terms(coils, t, LIMITS, N_COILS).items()})
    snapshots.append((i + 1, np.asarray(coils.gamma), float(jnp.mean(jnp.linalg.norm(residuals(coils, t)[0], axis=1)))))
    print(f"  outer {i + 1}: field {history[-1]['field']:.2e}, normal {history[-1]['normal']:.2e}, "
          f"infeasibility {alm.total_infeasibility(info[2]):.2e}, |grad| {jnp.linalg.norm(grad[0]):.2e}")
    if jnp.linalg.norm(grad[0]) < AL["omega_tol"] and alm.norm_constraints(info[2]) < AL["eta_tol"]:
        break
print(f"Augmented Lagrangian: {i + 1} outer iterations in {time.time() - t0:.1f} s")

""" Results """
report("Optimized coils", coils, t, N_COILS)
save_results("iota2_al", coils, t, history, snapshots, "ι = 2, ε = 1/2 (augmented Lagrangian)")
