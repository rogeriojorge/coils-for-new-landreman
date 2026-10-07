"""Coils for the iota = 2 analytic MHD equilibrium of Landreman (arXiv:2609.26742, section 2)."""
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.objective_functions import (loss_coil_separation, loss_coil_surface_distance,
                                       loss_linkingnumber, loss_lorentz_force_coils)
from landreman_equilibria import NFP, boundary_normal_target, iota2_B, iota2_surface, coil_field, fit_surface

""" Equilibrium: figure 1 of the paper (eps = 1/2, edge psi = 1/64), in physical units """
EPS, PSI_EDGE = 0.5, 1 / 64
MAJOR_RADIUS, B_AXIS = 1.0, 1.0            # axis radius (m) and |B| on axis at phi = 0 (T)
TARGET_FLUX_FRACTION = 0.25                 # coil field matched on psi = 0.25 psi_edge (half radius)
NTHETA_TARGET, NZETA_TARGET = 24, 16        # target points over half a field period
NTHETA_BN, NPHI_BN, VC_DIGITS = 32, 32, 6        # boundary B.n target by on-surface virtual casing
NTHETA_VC, NZETA_VC = 192, 4608             # boundary quadrature for the exact coil-field target

""" Coils: circular start, then shape and current optimization """
N_COILS, ORDER, N_SEGMENTS, COIL_MINOR_RADIUS = 6, 4, 100, 0.55
LENGTH_MAX, CURVATURE_MAX, MSC_MAX = 4.0, 4.0, 9.0  # m, 1/m, 1/m^2 (mean squared curvature)
TOTAL_CURVATURE_MAX = 3 * jnp.pi             # integral of curvature; a planar loop adds 2 pi
MIN_COIL_DISTANCE, MIN_SURFACE_DISTANCE = 0.12, 0.2  # m
CONDUCTOR_RADIUS = 0.05                     # m, for the reported self-force
WEIGHTS = dict(field=1e4, normal=1e4, length=1e2, curvature=1e1, msc=1e1, total_curvature=1e1, arclength=1e-2,
               coil_distance=1e3, surface_distance=1e3, linking=1e1)
MAXITER = 1000

""" Setting up the equilibrium, the exact target field, and the initial coils """
L = MAJOR_RADIUS / np.sqrt(1 - EPS**2)
b = B_AXIS / jnp.linalg.norm(iota2_B(iota2_surface(0., 0., EPS, 0.), EPS))
B = lambda x: b * iota2_B(x / L, EPS)
surface = lambda theta, zeta, psi=PSI_EDGE: L * iota2_surface(theta, zeta, EPS, psi)

theta, zeta = (a.ravel() for a in jnp.meshgrid(
    jnp.linspace(0, 2 * jnp.pi, NTHETA_TARGET, endpoint=False), jnp.linspace(0, jnp.pi / NFP, NZETA_TARGET)))
points = jax.vmap(lambda t, z: surface(t, z, TARGET_FLUX_FRACTION * PSI_EDGE))(theta, zeta)
B_points = jnp.linalg.norm(jax.vmap(B)(points), axis=1)
t0 = time.time()
B_target = coil_field(surface, B, points, NTHETA_VC, NZETA_VC)
B_check = coil_field(surface, B, points, 2 * NTHETA_VC // 3, 2 * NZETA_VC // 3)  # bounds the error
print(f"Exact coil-field target on {len(points)} points in {time.time() - t0:.1f} s; "
      f"error bound from 2/3 resolution max|dB|/B = {jnp.max(jnp.linalg.norm(B_target - B_check, axis=1) / B_points):.2e}")
print(f"Plasma-current share of B at the target points: "
      f"{jnp.mean(jnp.linalg.norm(jax.vmap(B)(points) - B_target, axis=1) / B_points):.3f}")
gamma_b, normal_b, Bn_target, B_b = (a[:NPHI_BN // 2] for a in  # half field period suffices (stellsym)
                                     boundary_normal_target(surface, B, NTHETA_BN, NPHI_BN, VC_DIGITS))
print(f"Boundary B.n target: max|B_coils.n|/B = {jnp.max(jnp.abs(Bn_target) / B_b):.3e} (plasma-current part)")
plasma = fit_surface(lambda t, z: surface(t, z), 12, 12, range_torus="full torus")  # distances, plots

curves = CreateEquallySpacedCurves(N_COILS, ORDER, MAJOR_RADIUS, COIL_MINOR_RADIUS,
                                   n_segments=N_SEGMENTS, nfp=NFP, stellsym=True)
B_unit = jax.vmap(BiotSavart(Coils(curves, jnp.ones(N_COILS))).B)(points)
current = jnp.vdot(B_unit, B_target) / jnp.vdot(B_unit, B_unit)  # least-squares common current
coils0 = Coils(curves, current * jnp.ones(N_COILS))
print(f"Initial coil current {current:.4e} A, {N_COILS} coils per half period, {coils0.dofs.size} dofs")

""" Objective: exact-field mismatch plus engineering penalties """
def objective(dofs):
    coils = coils0.with_dofs(dofs)
    dB = (jax.vmap(BiotSavart(coils).B)(points) - B_target) / B_points[:, None]
    Bn = (jnp.sum(jax.vmap(jax.vmap(BiotSavart(coils).B))(gamma_b) * normal_b, -1) - Bn_target) / B_b
    kappa, speed = coils.curvature[:N_COILS], jnp.linalg.norm(coils.gamma_dash[:N_COILS], axis=-1)
    terms = dict(
        field=jnp.mean(jnp.sum(dB**2, axis=1)),
        normal=jnp.mean(Bn**2),
        length=jnp.sum(jnp.maximum(coils.length[:N_COILS] - LENGTH_MAX, 0)**2),
        curvature=jnp.mean(jnp.maximum(coils.curvature - CURVATURE_MAX, 0)**2),
        msc=jnp.sum(jnp.maximum(jnp.mean(kappa**2 * speed, 1) / coils.length[:N_COILS] - MSC_MAX, 0)**2),
        total_curvature=jnp.sum(jnp.maximum(jnp.mean(kappa * speed, 1) - TOTAL_CURVATURE_MAX, 0)**2),
        arclength=jnp.sum(jnp.var(speed, 1) / jnp.mean(speed, 1)**2),
        coil_distance=loss_coil_separation(coils, MIN_COIL_DISTANCE),
        surface_distance=loss_coil_surface_distance(coils, plasma, MIN_SURFACE_DISTANCE),
        linking=jnp.sum(jnp.abs(loss_linkingnumber(coils))))
    return sum(WEIGHTS[k] * v for k, v in terms.items()), terms

value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
history = []

def fun(x):
    (value, terms), grad = value_and_grad(jnp.asarray(x))
    history.append({k: float(v) for k, v in terms.items()})
    return float(value), np.asarray(grad)

def report(label, coils):
    dB = jnp.linalg.norm(jax.vmap(BiotSavart(coils).B)(points) - B_target, axis=1) / B_points
    Bn = jnp.abs(jnp.sum(jax.vmap(jax.vmap(BiotSavart(coils).B))(gamma_b) * normal_b, -1) - Bn_target) / B_b
    print(f"{label}: interior max|dB|/B = {jnp.max(dB):.3e}, mean = {jnp.mean(dB):.3e}; "
          f"boundary max|dBn|/B = {jnp.max(Bn):.3e}, mean = {jnp.mean(Bn):.3e}, "
          f"lengths = {np.round(np.asarray(coils.length[:N_COILS]), 3)} m, "
          f"max curvature = {jnp.max(coils.curvature):.2f} 1/m, total curvature / 2pi = "
          f"{np.round(np.asarray(jnp.mean(coils.curvature[:N_COILS] * jnp.linalg.norm(coils.gamma_dash[:N_COILS], axis=-1), 1)) / (2 * np.pi), 2)}, mean force = "
          f"{loss_lorentz_force_coils(coils, threshold=0., conductor_radius=CONDUCTOR_RADIUS) / len(coils):.3e} N/m, currents = {np.asarray(coils.currents[:N_COILS])} A")
    print("  terms:", {k: f"{float(v):.3e}" for k, v in objective(coils.dofs)[1].items()})

""" Running the optimization """
report("Initial circular coils", coils0)
t0 = time.time()
result = minimize(fun, np.asarray(coils0.dofs), jac=True, method="L-BFGS-B",
                  callback=lambda x: len(history) % 50 or print(f"  eval {len(history)}: {history[-1]}"),
                  options=dict(maxiter=MAXITER, maxcor=50, ftol=1e-15, gtol=1e-12))
print(f"{result.message} after {result.nit} iterations in {time.time() - t0:.1f} s")
coils = coils0.with_dofs(jnp.asarray(result.x))
report("Optimized coils", coils)

""" Saving and plotting results """
coils.to_json("coils_iota2.json")
fig = plt.figure(figsize=(11, 4.5))
ax = fig.add_subplot(121, projection="3d")
plasma.plot(ax=ax, show=False)
coils.plot(ax=ax, show=False)
ax = fig.add_subplot(122)
ax.semilogy([h["field"] for h in history], label="field mismatch")
ax.semilogy([sum(WEIGHTS[k] * v for k, v in h.items()) for h in history], label="total objective")
ax.set_xlabel("function evaluation"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("coils_iota2.png", dpi=150)
print("Wrote coils_iota2.json and coils_iota2.png")
