"""Poincare sections with the optimized coils against the analytic surfaces and free-boundary VMEX.

Inside the plasma, the field with real coils is B_analytic + dB, with dB = B_coils(ESSOS) - B_coils(exact). The exact
coil field comes from the interior virtual-casing integral. dB is tabulated, fitted with a smooth model, and field
lines of B_analytic + dB are traced, with the plasma currents held at their analytic values. Each figure overlays the
analytic surfaces (black), the traced field lines (red) and, when benchmark_vmex.py has written wout_free_<case>.nc,
the free-boundary VMEX surfaces (blue) at the same s = Phi / Phi_edge."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from essos.coils import Coils
from essos.fields import BiotSavart
from landreman_equilibria import case, coil_field, point, flux_profiles

""" Parameters """
CASES = dict(iota2="coils_iota2.json", A="coils_sheared_A.json", D="coils_sheared_D.json", E="coils_sheared_E.json")
S_TRACE = (0.1, 0.4, 0.7)                       # traced surfaces, s = Phi / Phi_edge (on VMEX's ns = 11 grid)
N_START = 4                                     # field lines per surface (theta = 0, pi/2, pi, 3pi/2)
NTHETA_FIT, NPHI_FIT = 16, 12                   # dB samples per surface over one field period
QUADRATURE = lambda rho: (128, 2048) if rho < 0.7 else (320, 8192)  # exact coil field, converged to ~1e-12
FOURIER, DEGREE = 4, 6                          # fit: cos/sin(2 k phi) x polynomial in (R - 1, Z)
TRANSITS, STEPS = 200, 60                       # field periods traced, RK4 steps per field period
PHIS = (0.0, np.pi / 4, np.pi / 2)              # Poincare sections


def cylindrical(P, V):
    c, s = np.cos(np.arctan2(P[:, 1], P[:, 0])), np.sin(np.arctan2(P[:, 1], P[:, 0]))
    return (np.hypot(P[:, 0], P[:, 1]), np.arctan2(P[:, 1], P[:, 0]), P[:, 2],
            np.stack([c * V[:, 0] + s * V[:, 1], -s * V[:, 0] + c * V[:, 1], V[:, 2]], 1))


def basis(R, ph, Z, xp=np):
    angular = [1.0 + 0 * ph] + [f(2 * k * ph) for k in range(1, FOURIER) for f in (xp.cos, xp.sin)]
    return xp.stack([a * (R - 1)**i * Z**j for a in angular for i in range(DEGREE + 1) for j in range(DEGREE + 1 - i)], -1)


summary = {}
for name, coil_file in CASES.items():
    """ Surfaces in s, and the coil-field error on them """
    eq, field = case(name), BiotSavart(Coils.from_json(coil_file))
    rho_grid = np.linspace(0, 1, 17)[1:]
    s_grid = flux_profiles(eq, rho_grid)[0]
    rhos = np.interp(S_TRACE, np.r_[0, s_grid / s_grid[-1]], np.r_[0, rho_grid])
    theta = np.linspace(0, 2 * np.pi, NTHETA_FIT, endpoint=False)
    phi = np.linspace(0, np.pi, NPHI_FIT, endpoint=False)
    P, dB = [], []
    for rho in rhos:
        p = jnp.array([point(eq, t, rho, f) for f in phi for t in theta])
        exact = coil_field(eq["surface"], eq["B"], p, *QUADRATURE(rho))
        check = coil_field(eq["surface"], eq["B"], p[::16], *(2 * n // 3 for n in QUADRATURE(rho)))
        Bp = jnp.linalg.norm(jax.vmap(eq["B"])(p), axis=1)
        dB.append(np.asarray(jax.vmap(field.B)(p) - exact)); P.append(np.asarray(p))
        print(f"{name}, rho = {rho:.2f}: coil |dB|/|B| = {np.mean(np.linalg.norm(dB[-1], axis=1) / Bp):.2e}, "
              f"exact-field error bound {np.max(np.linalg.norm(exact[::16] - check, axis=1) / Bp[::16]):.1e}")
    R, ph, Z, dBcyl = cylindrical(np.concatenate(P), np.concatenate(dB))
    coef = np.linalg.lstsq(basis(R, ph, Z), dBcyl, rcond=1e-12)[0]

    """ Tracing field lines (R, Z) versus phi """
    def B_cyl(R, phi, Z):
        B = eq["B"](jnp.array([R * jnp.cos(phi), R * jnp.sin(phi), Z]))
        c, s = jnp.cos(phi), jnp.sin(phi)
        return jnp.array([c * B[0] + s * B[1], -s * B[0] + c * B[1], B[2]]) + basis(R, phi, Z, jnp) @ coef

    @jax.jit
    def orbit(u0):
        h = np.pi / STEPS
        rhs = lambda p, u: (lambda B: jnp.array([u[0] * B[0] / B[1], u[0] * B[2] / B[1]]))(B_cyl(u[0], p, u[1]))

        def step(c, _):
            u, p = c
            k1 = rhs(p, u); k2 = rhs(p + h / 2, u + h / 2 * k1); k3 = rhs(p + h / 2, u + h / 2 * k2)
            return (u + h / 6 * (k1 + 2 * k2 + 2 * k3 + rhs(p + h, u + h * k3)), p + h), u

        def period(u, _):  # one field period, phi = 0 -> pi
            (u1, _), us = jax.lax.scan(step, (u, 0.), None, STEPS)
            return u1, us[np.array([0, STEPS // 4, STEPS // 2])]  # sections at phi = 0, pi/4, pi/2
        return jax.lax.scan(period, u0, None, TRANSITS)[1]

    """ Figure: analytic (black), field lines (red), free-boundary VMEX (blue) """
    wout_file = f"wout_free_{name}.nc"
    wout = None
    if os.path.exists(wout_file):
        from vmex.core.wout import read_wout
        from vmex.core.plotting import surface_rz
        wout = read_wout(wout_file)
    fig, axes = plt.subplots(1, len(PHIS), figsize=(4.6 * len(PHIS), 4.4))
    t = np.linspace(0, 2 * np.pi, 240)
    deviation = []
    for rho, s in zip(rhos, S_TRACE):
        starts = [np.asarray(point(eq, t0, rho, 0.))[[0, 2]] for t0 in np.linspace(0, 2 * np.pi, N_START, endpoint=False)]
        sections = np.concatenate([np.asarray(orbit(jnp.array(u0))) for u0 in starts])  # (orbits*transits, 3, 2)
        for k, (ax, phi0) in enumerate(zip(axes, PHIS)):
            exact = np.array([np.asarray(point(eq, a, rho, phi0)) for a in t])
            exact = np.stack([np.hypot(exact[:, 0], exact[:, 1]), exact[:, 2]], 1)
            deviation += list(np.min(np.linalg.norm(sections[:, k][:, None] - exact[None], axis=-1), 1))
            ax.plot(*exact.T, "k-", lw=1.0)
            ax.plot(*sections[:, k].T, ".", color="firebrick", ms=1.2)
            if wout is not None:
                R_v, Z_v = surface_rz(wout, s_index=int(round(s * (len(np.asarray(wout.phi)) - 1))), theta=t, phi=np.array([phi0]))
                ax.plot(R_v[:, 0], Z_v[:, 0], "--", color="tab:blue", lw=1.2)
    for ax, phi0 in zip(axes, PHIS):
        boundary = np.array([np.asarray(point(eq, a, 1.0, phi0)) for a in t])
        ax.plot(np.hypot(boundary[:, 0], boundary[:, 1]), boundary[:, 2], "k-", lw=1.6)
        if wout is not None:
            R_v, Z_v = surface_rz(wout, s_index=len(np.asarray(wout.phi)) - 1, theta=t, phi=np.array([phi0]))
            ax.plot(R_v[:, 0], Z_v[:, 0], "--", color="tab:blue", lw=1.6)
        ax.set(title=f"φ = {phi0 / np.pi:.2g}π", xlabel="R [m]", ylabel="Z [m]", aspect="equal")
    handles = [plt.Line2D([], [], color="k", label="analytic surfaces"),
               plt.Line2D([], [], color="firebrick", marker=".", ls="", label="field lines, optimized coils")]
    if wout is not None:
        handles.append(plt.Line2D([], [], color="tab:blue", ls="--", label="free-boundary VMEX, optimized coils"))
    axes[0].legend(handles=handles, fontsize=7, loc="upper left")
    fig.suptitle(f"{eq['title']}: Poincaré sections, s = {', '.join(map(str, S_TRACE))} and the boundary")
    plt.tight_layout(); plt.savefig(f"fieldlines_{name}.png", dpi=130); plt.close(fig)
    summary[name] = (1e3 * np.mean(deviation), 1e3 * np.max(deviation))
    print(f"{name}: field lines {summary[name][0]:.2f} mm mean, {summary[name][1]:.2f} mm max from the analytic surfaces; "
          f"wrote fieldlines_{name}.png")
