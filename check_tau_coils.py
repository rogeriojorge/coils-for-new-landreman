"""Checks of the coils for Issan et al.'s non-stellarator-symmetric cases (tau = +-0.5, +-0.4; arXiv:2610.07304).

1. Boundary field error against a finer target than the optimization used (64 x 64 per field period, 6 digits,
   instead of 32 x 32, 4 digits), so the numbers are not limited by the target.
2. Mirror check. tau -> -tau maps the equilibrium by the rotation R: (x, y, z) -> (x, -y, -z) with B reversed, so the
   coils for -tau, optimized independently, should be the tau coils rotated by R with reversed currents. Compared:
   coil shapes (distance from each rotated tau coil to the nearest -tau coil), the two coil fields on the boundary,
   and the B.n error of the rotated tau coils against the -tau target (equal to the tau error if all is symmetric)."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from essos.coils import Coils
from essos.fields import BiotSavart
from landreman_equilibria import case, boundary_target

CASES = dict(iota2_tau="coils_iota2_tau.json", iota2_tau_mirror="coils_iota2_tau_mirror.json",
             issan_tau="coils_sheared_issan_tau.json", issan_tau_mirror="coils_sheared_issan_tau_mirror.json")
R = jnp.array([1., -1., -1.])
summary, gammas = {}, {}
for name, coil_file in CASES.items():
    eq, coils = case(name), Coils.from_json(coil_file)
    gamma, normal, B_ext, B_total = boundary_target(eq["surface"], eq["B"], 64, 64, digits=6)
    B, Bc = jnp.linalg.norm(B_total, axis=-1), jax.vmap(jax.vmap(BiotSavart(coils).B))(gamma)
    dBn, dB = jnp.abs(jnp.sum((Bc - B_ext) * normal, -1)) / B, jnp.linalg.norm(Bc - B_ext, axis=-1) / B
    summary[name] = dict(Bn_mean=float(dBn.mean()), Bn_max=float(dBn.max()), B_mean=float(dB.mean()), B_max=float(dB.max()))
    if name.endswith("_mirror"):  # field of the rotated tau coils (reversed currents) on the -tau boundary
        plus = BiotSavart(Coils.from_json(CASES[name[:-7]]))
        Bp = -R * jax.vmap(jax.vmap(plus.B))(gamma * R)
        gp, gm = np.asarray(Coils.from_json(CASES[name[:-7]]).gamma) * np.asarray(R), np.asarray(coils.gamma).reshape(-1, 3)
        summary[name].update(mirror_coil_distance_max_mm=1e3 * max(
            float(np.max(np.min(np.linalg.norm(c[:, None] - gm[None], axis=-1), 1))) for c in gp),
            mirror_field_difference_max=float(jnp.max(jnp.linalg.norm(Bc - Bp, axis=-1) / B)),
            rotated_tau_coils_Bn_max=float(jnp.max(jnp.abs(jnp.sum((Bp - B_ext) * normal, -1)) / B)))
    print(name, {k: f"{v:.3e}" for k, v in summary[name].items()})
json.dump(summary, open("tau_results.json", "w"), indent=1)
