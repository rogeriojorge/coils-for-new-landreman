"""VMEX benchmark of the analytic equilibrium and the optimized coils.

VMEX gets the analytic profiles (toroidal flux, pressure p(s), enclosed toroidal current I(s)) and either the exact
boundary (fixed boundary) or only the coils (free boundary). Flux surfaces and iota(s) are compared with the exact
solution. These current-carrying equilibria are radially unstable (the vertical-field decay index at the axis, printed
below, exceeds 3/2), so VMEX's descent cannot settle in free boundary; the free-boundary solve uses Newton
(vmex_newton.py), which converges and counts the unstable modes. The same coils are also solved at zero pressure
with the analytic current ("vacuum"): no member of either family has p = 0, so the coils target the exterior
(virtual-casing) field. With I = 0 as well, iota is ~0.05 (iota is current-driven) and there are no surfaces to compare. Cases as arguments (default A D E)."""
import json
import os
import sys
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import vmex as vj
from vmex.core.wout import write_wout
from vmex.core.plotting import surface_rz
from essos.coils import Coils
from essos.fields import BiotSavart
from landreman_equilibria import case, point, flux_profiles, boundary_fourier, NFP
from vmex_newton import free_boundary_newton

""" Benchmark parameters """
CASES = sys.argv[1:] or ["A", "D", "E"]       # also iota2_tau(_mirror), issan_tau(_mirror): LASYM = T, coils_<case>.json
COIL_FILE = lambda name: f"coils_{name if name.startswith('iota2') else 'sheared_' + name}.json"
NRHO, NTHETA, PROFILE_DEGREE = 16, 64, 8      # analytic profile samples and polynomial degree in s
MPOL, NTOR, NZETA, NS = 5, 5, 16, 11          # VMEX resolution (NZETA >= 2 NTOR + 4 and divides the mgrid planes)
NS_ARRAY, NITER, FTOL, DELT = (NS,), (30000,), (1e-10,), 0.1  # best free-boundary settings found
GRID = dict(rmin=0.3, rmax=1.7, zmin=-0.95, zmax=0.95, ir=128, jz=176, kp=32)  # tau shifts the plasma in Z
PHIS = (0.0, np.pi / 4, np.pi / 2)            # cross-sections (one field period is pi)
FIXED_FTOL = dict(issan_tau=1e-6, issan_tau_mirror=1e-6)  # Newton's fixed-boundary start (default 3e-9); slow at MPOL 5
S_PLOT = (0.2, 0.5, 1.0)                      # flux surfaces drawn and compared (on the ns grid)


def vmex_input(eq, rho, Phi, I):
    """Free-boundary VMEX input from the analytic boundary and profiles, as polynomials in s = Phi / Phi_edge."""
    s = Phi / Phi[-1]
    am = np.polynomial.polynomial.polyfit(s, eq["pressure"](rho), PROFILE_DEGREE)
    basis = np.stack([s**(i + 1) for i in range(PROFILE_DEGREE)], 1)  # power_series_i: I(s) = sum c_i s^(i+1)
    ac = np.linalg.lstsq(basis, I / I[-1], rcond=None)[0]
    rbc, zbs, rbs, zbc = (np.asarray(a) for a in boundary_fourier(eq["surface"], MPOL - 1, NTOR))
    asym = {} if eq["stellsym"] else dict(lasym=True, rbs=rbs, zbc=zbc, raxis_s=-rbs[NTOR:, 0], zaxis_c=zbc[NTOR:, 0])
    pad = lambda c: list(c) + [0.0] * (21 - len(c))
    return vj.VmecInput(nfp=NFP, mpol=MPOL, ntor=NTOR, nzeta=NZETA, lfreeb=True, mgrid_file="essos_coils(direct)",
                        ns_array=list(NS_ARRAY), niter_array=list(NITER), ftol_array=list(FTOL), phiedge=float(Phi[-1]),  # B along +phi; VMEC2000 rejects the opposite sign
                        pmass_type="power_series", am=pad(am), pres_scale=1.0,
                        ncurr=1, pcurr_type="power_series_i", ac=pad(ac), curtor=float(I[-1]),
                        rbc=rbc, zbs=zbs, raxis_c=rbc[NTOR:, 0], zaxis_s=-zbs[NTOR:, 0], delt=DELT, **asym)


def wout(inp, res):
    return vj.wout_from_state(inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz), fsql=float(res.fsql),
                              niter=int(res.iterations), converged=bool(res.converged), vacuum_output=getattr(res, "vacuum", None))


""" Running the benchmark """
summary = json.load(open("benchmark_results.json")) if os.path.exists("benchmark_results.json") else {}
for name in CASES:
    eq, coils = case(name), Coils.from_json(COIL_FILE(name))
    rho = np.linspace(0, 1, NRHO + 1)[1:]
    Phi, I = flux_profiles(eq, rho, NTHETA)
    print(f"\n=== {eq['title']}: Phi_edge = {Phi[-1]:.4e} Wb, I_edge = {I[-1]:.4e} A, "
          f"p_axis = {eq['pressure'](0.):.4e} Pa")
    inp = vmex_input(eq, rho, Phi, I)
    for phi in PHIS:  # vertical-field decay index n = -(R / B_Z) dB_Z/dR of the coils at the magnetic axis
        axis = np.asarray(point(eq, 0., 0., phi)); R0 = float(np.hypot(*axis[:2]))
        BZ = lambda R: float(BiotSavart(coils).B(jnp.array([R * np.cos(phi), R * np.sin(phi), axis[2]]))[2])
        print(f"decay index at the axis, phi = {phi / np.pi:.2g} pi: n = {-R0 / BZ(R0) * (BZ(R0 + 1e-3) - BZ(R0 - 1e-3)) / 2e-3:.2f}")
    fixed = vj.solve_multigrid(replace(inp, lfreeb=False, mgrid_file="NONE"), raise_on_max_iterations=False)
    newton = {key: free_boundary_newton(v, coils, GRID, fixed_ftol=FIXED_FTOL.get(name, 3e-9))  # descent cannot settle on these unstable equilibria
              for key, v in dict(free=inp, vacuum=replace(inp, pres_scale=0.0)).items()}
    wouts = dict(fixed=wout(replace(inp, lfreeb=False, mgrid_file="NONE"), fixed),
                 **{key: n["wout"] for key, n in newton.items()})
    print(f"VMEX fixed boundary: converged {bool(fixed.converged)}, {int(fixed.iterations)} iterations, "
          f"fsq = {float(fixed.fsqr) + float(fixed.fsqz) + float(fixed.fsql):.1e}")
    for key, n in newton.items():
        print(f"VMEX free boundary (Newton, {key}): |F| = {n['residual']:.1e}, converged {n['converged']}, "
              f"{n['n_unstable']} unstable modes; most unstable (eigenvalue, [(fraction, m, n)]): {n['modes']}")

    """ Comparison: flux surfaces at three planes and iota(s), fixed- and free-boundary """
    s_of_rho = Phi / Phi[-1]
    rho_of_s = lambda s: np.interp(s, np.r_[0, s_of_rho], np.r_[0, rho])
    theta = np.linspace(0, 2 * np.pi, 181)
    styles = dict(fixed=dict(color="tab:blue", ls=":", label="fixed-boundary VMEX"),
                  free=dict(color="tab:red", ls="--", label="free-boundary VMEX (Newton), optimized coils"),
                  vacuum=dict(color="tab:green", ls="-.", label="free boundary, p = 0 (vacuum), same coils"))
    fig, axes = plt.subplots(1, len(PHIS) + 1, figsize=(4.0 * (len(PHIS) + 1), 4.0))
    deviation = {key: [] for key in wouts}
    for ax, phi in zip(axes, PHIS):
        for s in S_PLOT:
            exact = np.asarray(jax.vmap(lambda t: point(eq, t, rho_of_s(s), phi))(jnp.asarray(theta)))
            Re, Ze = np.hypot(exact[:, 0], exact[:, 1]), exact[:, 2]
            ax.plot(Re, Ze, "k-", lw=1.2, label="analytic" if s == S_PLOT[0] else None)
            for key, w in wouts.items():
                R, Z = surface_rz(w, s_index=int(round(s * (NS - 1))), theta=theta, phi=np.array([phi]))
                ax.plot(R[:, 0], Z[:, 0], lw=1.4, **{**styles[key], "label": styles[key]["label"] if s == S_PLOT[0] else None})
                if s == 1.0:  # distance of each VMEX boundary point to the analytic boundary
                    deviation[key] += list(np.min(np.hypot(R[:, 0, None] - Re[None], Z[:, 0, None] - Ze[None]), 1))
        ax.set(title=f"φ = {phi / np.pi:.2g}π", xlabel="R [m]", ylabel="Z [m]", aspect="equal")
    axes[0].legend(fontsize=7)
    s_full = np.linspace(0, 1, NS)
    iota_exact = np.array([float(eq["iota"](rho_of_s(s))) for s in s_full[1:]])
    axes[-1].plot(s_full[1:], iota_exact, "k-", label="analytic")
    for key, w in wouts.items():
        axes[-1].plot(s_full[1:], np.abs(np.asarray(w.iotaf))[1:], lw=1.4, **styles[key])
    axes[-1].set(title="rotational transform", xlabel="s = Φ/Φ_edge", ylabel="ι"); axes[-1].legend(fontsize=7)
    fig.suptitle(f"{eq['title']}: analytic equilibrium vs VMEX")
    plt.tight_layout(); plt.savefig(f"benchmark_{name}.png", dpi=120); plt.close(fig)

    minor = float(np.sqrt(abs(Phi[-1]) / (np.pi * jnp.linalg.norm(eq["B"](eq["family"](0., 0., 0.))))))  # rough minor radius
    write_wout(f"wout_free_{name}.nc", wouts["free"])  # read by validate_fieldlines.py
    summary[name] = dict(rho=rho.tolist(), s=s_of_rho.tolist())
    status = dict(fixed=dict(converged=bool(fixed.converged), residual=float(fixed.fsqr + fixed.fsqz + fixed.fsql)),
                  **{key: dict(converged=n["converged"], residual=n["residual"], unstable_modes=n["n_unstable"])
                     for key, n in newton.items()})
    for key in wouts:
        iota_vmex = np.abs(np.asarray(wouts[key].iotaf))[1:]
        summary[name][key] = dict(**status[key], beta=float(wouts[key].betatotal),
                                  boundary_deviation_mean_mm=1e3 * float(np.mean(deviation[key])),
                                  boundary_deviation_max_mm=1e3 * float(np.max(deviation[key])),
                                  boundary_deviation_over_minor_radius=float(np.max(deviation[key])) / minor,
                                  iota_max_relative_error=float(np.max(np.abs(iota_vmex / iota_exact - 1))))
        print(f"Benchmark {key}:", {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in summary[name][key].items()})

json.dump(summary, open("benchmark_results.json", "w"), indent=1)
print("Wrote benchmark_<case>.png and benchmark_results.json")
