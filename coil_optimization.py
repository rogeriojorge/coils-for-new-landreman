"""Shared pieces of the coil optimizations: exact targets, objective terms, AL constraints, report, plots, movie."""
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.objective_functions import (loss_coil_separation, loss_coil_surface_distance,
                                       loss_linkingnumber, loss_lorentz_force_coils)
from landreman_equilibria import NFP, boundary_normal_target, coil_field, fit_surface


def targets(surface, B, inner_surface, ntheta=24, nzeta=16, quadrature=(192, 4608), nbn=32, digits=6):
    """Exact coil field at interior points (on inner_surface) and the coil B.n on the exact boundary."""
    theta, zeta = (a.ravel() for a in jnp.meshgrid(jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False),
                                                    jnp.linspace(0, jnp.pi / NFP, nzeta)))
    points = jax.vmap(inner_surface)(theta, zeta)
    B_points = jnp.linalg.norm(jax.vmap(B)(points), axis=1)
    t0 = time.time()
    B_target = coil_field(surface, B, points, *quadrature)
    B_check = coil_field(surface, B, points, *(2 * n // 3 for n in quadrature))  # bounds the error
    print(f"Exact coil-field target on {len(points)} points in {time.time() - t0:.1f} s; error bound "
          f"(2/3 resolution) max|dB|/B = {jnp.max(jnp.linalg.norm(B_target - B_check, axis=1) / B_points):.2e}; "
          f"plasma-current share {jnp.mean(jnp.linalg.norm(jax.vmap(B)(points) - B_target, axis=1) / B_points):.3f}")
    gamma_b, normal_b, Bn_target, B_b = boundary_normal_target(surface, B, nbn, nbn, digits)  # one field period
    print(f"Boundary target by virtual casing ({digits} digits): max|B_coils.n|/B = {jnp.max(jnp.abs(Bn_target) / B_b):.3e}")
    return dict(points=points, B_target=B_target, B_points=B_points, gamma_b=gamma_b, normal_b=normal_b,
                Bn_target=Bn_target, B_b=B_b, half=nbn // 2,  # stellarator symmetry: half a period suffices
                plasma=fit_surface(surface, 12, 12, range_torus="full torus"))


def initial_coils(t, n_coils, order, n_segments, major_radius, minor_radius, axis=None):
    """Circular coils (centred on an elliptical axis with semi-axes `axis`) with the least-squares common current."""
    curves = CreateEquallySpacedCurves(n_coils, order, major_radius, minor_radius,
                                       n_segments=n_segments, nfp=NFP, stellsym=True)
    if axis is not None:
        angle = jnp.arctan2(curves.dofs[:, 1, 0], curves.dofs[:, 0, 0])
        curves = curves.with_dofs(curves.dofs.at[:, 0, 0].set(axis[0] * jnp.cos(angle))
                                             .at[:, 1, 0].set(axis[1] * jnp.sin(angle)))
    B_unit = jax.vmap(BiotSavart(Coils(curves, jnp.ones(n_coils))).B)(t["points"])
    current = jnp.vdot(B_unit, t["B_target"]) / jnp.vdot(B_unit, B_unit)
    print(f"Initial coils: {n_coils} per half period, order {order}, common current {current:.4e} A")
    return Coils(curves, current * jnp.ones(n_coils))


def residuals(coils, t):
    """Relative interior field mismatch (3 per point) and boundary normal-field mismatch."""
    field = BiotSavart(coils)
    dB = (jax.vmap(field.B)(t["points"]) - t["B_target"]) / t["B_points"][:, None]
    g, n = t["gamma_b"][:t["half"]], t["normal_b"][:t["half"]]
    Bn = (jnp.sum(jax.vmap(jax.vmap(field.B))(g) * n, -1) - t["Bn_target"][:t["half"]]) / t["B_b"][:t["half"]]
    return dB, Bn


def geometry(coils, n_coils):
    """Per-coil length, mean squared curvature and total curvature (int kappa dl), plus pointwise curvature."""
    kappa, speed = coils.curvature[:n_coils], jnp.linalg.norm(coils.gamma_dash[:n_coils], axis=-1)
    length = coils.length[:n_coils]
    return dict(length=length, kappa=coils.curvature, msc=jnp.mean(kappa**2 * speed, 1) / length,
                total_curvature=jnp.mean(kappa * speed, 1), arclength=jnp.var(speed, 1) / jnp.mean(speed, 1)**2)


def terms(coils, t, limits, n_coils):
    """Penalty terms; each is zero when its engineering limit is met."""
    dB, Bn = residuals(coils, t)
    g = geometry(coils, n_coils)
    over = lambda x, limit: jnp.maximum(x - limit, 0)**2
    return dict(field=jnp.mean(jnp.sum(dB**2, 1)), normal=jnp.mean(Bn**2),
                length=jnp.sum(over(g["length"], limits["length"])),
                curvature=jnp.mean(over(g["kappa"], limits["curvature"])),
                msc=jnp.sum(over(g["msc"], limits["msc"])),
                total_curvature=jnp.sum(over(g["total_curvature"], limits["total_curvature"])),
                arclength=jnp.sum(g["arclength"]),
                coil_distance=loss_coil_separation(coils, limits["coil_distance"]),
                surface_distance=loss_coil_surface_distance(coils, t["plasma"], limits["surface_distance"]),
                linking=jnp.sum(jnp.abs(loss_linkingnumber(coils))))


def constraints(coils, t, limits, n_coils):
    """Augmented-Lagrangian form: violations that must vanish, max(value - limit, 0) = 0 for every limit."""
    g = geometry(coils, n_coils)
    over = lambda x, limit: jnp.maximum(x - limit, 0)
    return dict(length=over(g["length"], limits["length"]), curvature=over(g["kappa"].ravel(), limits["curvature"]),
                msc=over(g["msc"], limits["msc"]), total_curvature=over(g["total_curvature"], limits["total_curvature"]),
                arclength=over(g["arclength"], limits["arclength"]),
                coil_distance=loss_coil_separation(coils, limits["coil_distance"]),
                surface_distance=loss_coil_surface_distance(coils, t["plasma"], limits["surface_distance"]),
                linking=jnp.sum(jnp.abs(loss_linkingnumber(coils))))


def report(label, coils, t, n_coils, conductor_radius=0.05):
    """One-line summary: interior and boundary errors, geometry, force."""
    dB, Bn = residuals(coils, t)
    dB, Bn, g = jnp.linalg.norm(dB, axis=1), jnp.abs(Bn), geometry(coils, n_coils)
    force = loss_lorentz_force_coils(coils, threshold=0., conductor_radius=conductor_radius) / len(coils)
    summary = dict(interior_mean=float(jnp.mean(dB)), interior_max=float(jnp.max(dB)),
                   boundary_mean=float(jnp.mean(Bn)), boundary_max=float(jnp.max(Bn)),
                   max_length=float(jnp.max(g["length"])), max_curvature=float(jnp.max(g["kappa"])),
                   max_total_curvature_over_2pi=float(jnp.max(g["total_curvature"]) / (2 * np.pi)),
                   mean_force=float(force))
    print(f"{label}: interior |dB|/B mean {summary['interior_mean']:.2e} max {summary['interior_max']:.2e}; "
          f"boundary |dB.n|/B mean {summary['boundary_mean']:.2e} max {summary['boundary_max']:.2e}; "
          f"length {np.round(np.asarray(g['length']), 2)} m; max curvature {summary['max_curvature']:.2f} 1/m; "
          f"total curvature/2pi {np.round(np.asarray(g['total_curvature']) / (2 * np.pi), 2)}; "
          f"mean force {summary['mean_force']:.2e} N/m")
    return summary


def optimize(coils0, t, limits, weights, n_coils, maxiter=1000, snapshot_every=10, verbose=True):
    """Weighted-penalty optimization with SciPy L-BFGS-B and exact JAX gradients."""
    from scipy.optimize import minimize
    objective = lambda dofs: (lambda d: (sum(weights[k] * v for k, v in d.items()), d))(
        terms(coils0.with_dofs(dofs), t, limits, n_coils))
    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    history, snapshots = [], []

    def fun(x):
        (value, d), grad = value_and_grad(jnp.asarray(x))
        history.append({k: float(v) for k, v in d.items()})
        if len(history) % snapshot_every == 1:
            coils = coils0.with_dofs(jnp.asarray(x))
            snapshots.append((len(history), np.asarray(coils.gamma),
                              float(jnp.mean(jnp.linalg.norm(residuals(coils, t)[0], axis=1)))))
        if verbose and len(history) % 100 == 0:
            print(f"  eval {len(history)}: total {value:.3e}, " + ", ".join(f"{k} {v:.2e}" for k, v in history[-1].items()))
        return float(value), np.asarray(grad)

    t0 = time.time()
    result = minimize(fun, np.asarray(coils0.dofs), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=maxiter, maxcor=50, ftol=1e-15, gtol=1e-12))
    print(f"{result.message} after {result.nit} iterations in {time.time() - t0:.1f} s")
    return coils0.with_dofs(jnp.asarray(result.x)), history, snapshots


def save_results(name, coils, t, history, snapshots, title):
    """coils_<name>.json, a figure (coils on the boundary coloured by |dB.n|/B, convergence) and a GIF."""
    coils.to_json(f"coils_{name}.json")
    # boundary error on one field period, rotated to the full torus
    g, n = t["gamma_b"], t["normal_b"]
    Bn = np.asarray(jnp.abs(jnp.sum(jax.vmap(jax.vmap(BiotSavart(coils).B))(g) * n, -1) - t["Bn_target"]) / t["B_b"])
    rot = lambda k, v: np.stack([np.cos(2 * np.pi * k / NFP) * v[..., 0] - np.sin(2 * np.pi * k / NFP) * v[..., 1],
                                 np.sin(2 * np.pi * k / NFP) * v[..., 0] + np.cos(2 * np.pi * k / NFP) * v[..., 1],
                                 v[..., 2]], -1)
    close = lambda a: np.concatenate([np.concatenate([a, a[:1]], 0), np.concatenate([a, a[:1]], 0)[:, :1]], 1)
    g = close(np.concatenate([rot(k, np.asarray(g)) for k in range(NFP)], 0))  # full torus, closed in both angles
    Bn = close(np.tile(Bn, (NFP, 1)))
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d", position=[0.0, 0.0, 0.55, 1.0])
    norm = plt.Normalize(0, Bn.max())
    ax.plot_surface(g[..., 0], g[..., 1], g[..., 2], facecolors=plt.cm.viridis(norm(Bn)),
                    rstride=1, cstride=1, linewidth=0, shade=False)
    for curve in np.asarray(coils.gamma):
        ax.plot(*np.vstack([curve, curve[:1]]).T, color="firebrick", lw=1.5)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.5, pad=0.0,
                 label="|ΔB·n|/|B| on boundary")
    r = np.abs(np.asarray(coils.gamma)[..., :2]).max()
    ax.set(xlim=(-r, r), ylim=(-r, r), zlim=(-0.4 * r, 0.4 * r)); ax.set_box_aspect((1, 1, 0.4))
    ax.set_axis_off(); ax.set_title(title, y=0.92)
    ax = fig.add_axes([0.62, 0.14, 0.35, 0.74])
    for key, label in (("field", "interior field"), ("normal", "boundary B·n")):
        ax.semilogy([h[key] for h in history], label=label)
    ax.set_xlabel("iteration"); ax.set_ylabel("mean squared relative mismatch"); ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(f"coils_{name}.png", dpi=130); plt.close(fig)
    movie(f"coils_{name}.gif", snapshots, t, title)
    print(f"Wrote coils_{name}.json, coils_{name}.png and coils_{name}.gif")


def movie(path, snapshots, t, title, frames=40):
    """GIF of the coils evolving around the plasma boundary."""
    idx = np.unique(np.linspace(0, len(snapshots) - 1, frames).astype(int))
    p = np.asarray(t["plasma"].gamma)
    fig = plt.figure(figsize=(5, 4.2))
    ax = fig.add_subplot(projection="3d")

    def draw(i):
        ax.clear()
        ax.plot_surface(p[..., 0], p[..., 1], p[..., 2], color="steelblue", alpha=0.35, linewidth=0)
        step, gammas, err = snapshots[idx[i]]
        for curve in gammas:
            ax.plot(*np.vstack([curve, curve[:1]]).T, color="firebrick", lw=1.2)
        ax.set_box_aspect((1, 1, 0.4)); ax.set_axis_off(); ax.view_init(30, 30 + 2 * i)
        ax.set_title(f"{title}\nevaluation {step}, interior |ΔB|/|B| = {err:.1e}", fontsize=9)
    FuncAnimation(fig, draw, frames=len(idx)).save(path, writer=PillowWriter(fps=8), dpi=80)
    plt.close(fig)
