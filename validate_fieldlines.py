"""Field lines with the real coils: do the optimized coils keep the analytic flux surfaces?

Inside the plasma, the field with real coils is B_analytic + dB, with dB = B_coils(ESSOS) - B_coils(exact). The
exact coil field comes from the interior virtual-casing integral. dB is tabulated, fitted with a smooth model, and
field lines of B_analytic + dB are traced (plasma currents held at their analytic values) and compared with the
analytic surfaces."""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from essos.coils import Coils
from essos.fields import BiotSavart
from landreman_equilibria import case, coil_field, section_zeta

""" Parameters """
CASE, COILS = "D", "coils_sheared_D.json"
RHOS = (0.2, 0.4, 0.6, 0.8)                     # traced surfaces, rho^2 = psi / psi_edge
NTHETA_FIT, NPHI_FIT = 16, 12                   # dB samples per surface over one field period
QUADRATURE = {0.2: (128, 1024), 0.4: (128, 1024), 0.6: (128, 1024), 0.8: (256, 4096)}  # exact field, per rho
FOURIER, DEGREE = 4, 6                          # fit: cos/sin(2 k phi) x polynomial in (R - 1, Z)
TRANSITS, STEPS = 300, 60                       # field periods traced, RK4 steps per field period

""" Tabulating the coil-field error inside the plasma """
eq, field = case(CASE), BiotSavart(Coils.from_json(COILS))
point = lambda theta, rho, phi: (lambda s: s(theta, section_zeta(s, theta, phi)))(lambda t, z: eq["family"](t, z, rho))
theta, phi = np.linspace(0, 2 * np.pi, NTHETA_FIT, endpoint=False), np.linspace(0, np.pi, NPHI_FIT, endpoint=False)
P, dB = [], []
for rho in RHOS:
    p = jnp.array([point(t, rho, f) for f in phi for t in theta])
    dB.append(np.asarray(jax.vmap(field.B)(p) - coil_field(eq["surface"], eq["B"], p, *QUADRATURE[rho])))
    P.append(np.asarray(p))
    print(f"rho = {rho}: |dB|/|B| = {np.mean(np.linalg.norm(dB[-1], axis=1) / np.linalg.norm(jax.vmap(eq['B'])(p), axis=1)):.2e}")
P, dB = np.concatenate(P), np.concatenate(dB)

""" Fitting dB in cylindrical components """
def cylindrical(P, V):
    c, s = np.cos(np.arctan2(P[:, 1], P[:, 0])), np.sin(np.arctan2(P[:, 1], P[:, 0]))
    return np.hypot(P[:, 0], P[:, 1]), np.arctan2(P[:, 1], P[:, 0]), P[:, 2], np.stack([c * V[:, 0] + s * V[:, 1], -s * V[:, 0] + c * V[:, 1], V[:, 2]], 1)

def basis(R, ph, Z, xp=np):
    angular = [1.0 + 0 * ph] + [f(2 * k * ph) for k in range(1, FOURIER) for f in (xp.cos, xp.sin)]
    return xp.stack([a * (R - 1)**i * Z**j for a in angular for i in range(DEGREE + 1) for j in range(DEGREE + 1 - i)], -1)

R, ph, Z, dBcyl = cylindrical(P, dB)
coef = np.linalg.lstsq(basis(R, ph, Z), dBcyl, rcond=1e-12)[0]
print(f"fit residual / |dB| = {np.linalg.norm(basis(R, ph, Z) @ coef - dBcyl) / np.linalg.norm(dBcyl):.2f}")

""" Tracing field lines (R, Z) versus phi, with and without dB """
def B_cyl(R, phi, Z, real):
    B = eq["B"](jnp.array([R * jnp.cos(phi), R * jnp.sin(phi), Z]))
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([c * B[0] + s * B[1], -s * B[0] + c * B[1], B[2]]) + real * basis(R, phi, Z, jnp) @ coef

@jax.jit
def orbit(u0, real):
    h = np.pi / STEPS
    rhs = lambda p, u: (lambda B: jnp.array([u[0] * B[0] / B[1], u[0] * B[2] / B[1]]))(B_cyl(u[0], p, u[1], real))

    def step(c, _):
        u, p = c
        k1 = rhs(p, u); k2 = rhs(p + h / 2, u + h / 2 * k1); k3 = rhs(p + h / 2, u + h / 2 * k2)
        return (u + h / 6 * (k1 + 2 * k2 + 2 * k3 + rhs(p + h, u + h * k3)), p + h), u

    def period(u, _):  # one field period, phi = 0 -> pi; sections at phi = 0 and pi / 2
        (u1, _), us = jax.lax.scan(step, (u, 0.), None, STEPS)
        return u1, (us[0], us[STEPS // 2])
    return jax.lax.scan(period, u0, None, TRANSITS)[1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
t = np.linspace(0, 2 * np.pi, 200)
for rho in RHOS:
    sections = orbit(jnp.array(np.asarray(point(0., rho, 0.))[[0, 2]]), 1.0)
    for ax, sec, phi0 in zip(axes, sections, (0., np.pi / 2)):
        exact = np.array([np.asarray(point(a, rho, phi0)) for a in t])
        exact = np.stack([np.hypot(exact[:, 0], exact[:, 1]), exact[:, 2]], 1)
        dist = np.min(np.linalg.norm(np.asarray(sec)[:, None] - exact[None], axis=-1), 1)
        ax.plot(*exact.T, "k-", lw=0.8); ax.plot(*np.asarray(sec).T, ".", color="firebrick", ms=1.5)
        print(f"rho = {rho}, phi = {phi0 / np.pi:.1f} pi: field lines {1e3 * dist.mean():.2f} mm mean, {1e3 * dist.max():.2f} mm max from the analytic surface")
for ax, phi0 in zip(axes, (0., 0.5)):
    ax.set(title=f"φ = {phi0}π", xlabel="R [m]", ylabel="Z [m]", aspect="equal")
fig.suptitle(f"{eq['title']}: analytic surfaces (black), field lines with the optimized coils (red)")
plt.tight_layout(); plt.savefig(f"fieldlines_{CASE}.png", dpi=130)
print(f"Wrote fieldlines_{CASE}.png")
