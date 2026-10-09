"""Free-boundary VMEX equilibria by Newton, with an ideal-MHD stability count.

VMEX's default solver is steepest descent on the MHD energy W, which cannot settle on an unstable equilibrium (a
saddle of W). Newton on the same force balance converges to it. At the converged state the force Jacobian
J = dF/dz = K d2W (K > 0, VMEC's force metric) has the inertia of the energy Hessian, so each negative eigenvalue is
a direction with delta W < 0: an ideal-MHD instability (magnitudes are in VMEC's force metric, not growth rates).
Built on VMEX internals (vmex.core.implicit / freeboundary_implicit); a candidate for an upstream VMEX feature."""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import vmex as vj
from vmex.core import implicit as im, freeboundary_implicit as fbi
from vmex.core.wout import wout_from_state


def _basis(cfg, mask):
    """Orthonormal basis (n_full x n_dof) of the range of VMEX's dof projector, and the state unraveler."""
    P = im._dof_projector(cfg, mask)
    flat, unravel = ravel_pytree(jax.tree.map(jnp.zeros_like, mask))
    active = np.nonzero(np.asarray(ravel_pytree(mask)[0]) > 0)[0]
    E = np.zeros((len(active), flat.size)); E[np.arange(len(active)), active] = 1.0
    cols = np.asarray(jax.jit(jax.vmap(lambda v: ravel_pytree(P(unravel(v)))[0]))(jnp.asarray(E)))
    _, keep = np.unique((np.abs(cols) > 1e-14), axis=0, return_index=True)  # one column per support pattern
    return np.linalg.qr(cols[np.sort(keep)].T)[0], unravel


def _jacobian(f, z, Q, chunk=32):
    """Reduced Jacobian Q^T (dF/dz) Q by forward-mode products."""
    jvp = jax.jit(jax.vmap(lambda t: jax.jvp(f, (z,), (t,))[1]))
    cols = [np.asarray(jvp(jnp.asarray(np.pad(Q[:, i:i + chunk], ((0, 0), (0, max(0, i + chunk - Q.shape[1])))).T)))
            [:min(chunk, Q.shape[1] - i)] for i in range(0, Q.shape[1], chunk)]
    return Q.T @ np.concatenate(cols).T


def _modes(vec, unravel, cfg, top=2):
    """Dominant (m, n) of an eigenvector by energy fraction, summed over surfaces and R, Z, lambda."""
    state, table = unravel(jnp.asarray(vec)), im._static_tables(cfg.resolution)[0]
    m, n = np.asarray(table.m), np.asarray(table.n)
    energy = sum((np.asarray(getattr(state, f))**2).sum(0) for f in im._STATE_FIELDS)
    order = np.argsort(energy)[::-1][:top]
    return [(round(float(energy[k] / energy.sum()), 2), int(m[k]), int(n[k])) for k in order]


def free_boundary_newton(inp, coils, grid, *, fixed_ftol=3e-9, max_newton=15, nev=4):
    """Free-boundary equilibrium of `inp` in the field of `coils` by Newton, starting from the fixed-boundary
    solution. Returns dict(wout, residual, converged, n_unstable, modes)."""
    jax.config.update("jax_enable_x64", True)
    fixed = im.run(replace(inp, lfreeb=False, mgrid_file="NONE", ftol_array=[fixed_ftol], niter_array=[40000]))
    rcon0, zcon0 = fixed.runtime.rcon0, fixed.runtime.zcon0
    field = vj.MgridField.from_coils(coils, nfp=inp.nfp, **grid)
    free = replace(inp, lfreeb=True, ftol_array=[1e-10])
    cfg = fbi.make_free_boundary_config(free, field); icfg = cfg.implicit
    params = im.params_from_input(free)
    rt = replace(im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0, lfreeb=True,
                 jmax=int(icfg.resolution.ns), presf_ns_scale=jnp.asarray(fbi._presf_ns_scale(free, int(icfg.resolution.ns))))
    rt = replace(rt, bsqvac_edge=jax.lax.stop_gradient(cfg.vacuum_program.bsq(fixed.state, rt, field)))
    mask = im._dof_mask(fixed.state, rt, icfg, evaluator=lambda x: im.evaluate_forces(x, rt)[0], fixed_edge=False)
    mask = jax.tree.map(lambda a: jnp.asarray(np.asarray(a)), mask)
    state, _ = fbi._anchor_root(cfg, params, field, fixed.state, mask, rcon0, zcon0)  # Krylov Newton
    frozen, P = jax.lax.stop_gradient(state), im._dof_projector(icfg, mask)
    Q, unravel = _basis(icfg, mask)
    residual = fbi._projected_residual(cfg, mask, formulation="raw")
    f = lambda z: ravel_pytree(residual(unravel(z), params, field, frozen, rcon0, zcon0))[0]
    fj, z = jax.jit(f), ravel_pytree(P(state))[0]
    for _ in range(max_newton):  # damped dense Newton when the Krylov anchor stalls
        r0 = float(jnp.linalg.norm(fj(z)))
        if r0 < 1e-11:
            break
        dz = jnp.asarray(Q @ -np.linalg.solve(_jacobian(f, z, Q), Q.T @ np.asarray(fj(z))))
        a = 1.0
        while a > 1e-3 and float(jnp.linalg.norm(fj(z + a * dz))) >= r0:
            a /= 2
        z = z + a * dz
    state = jax.tree.map(jnp.add, frozen, P(jax.tree.map(jnp.subtract, unravel(z), frozen)))
    r = float(jnp.linalg.norm(fj(z)))
    w, V = np.linalg.eig(_jacobian(f, z, Q))
    order = np.argsort(w.real)
    return dict(wout=wout_from_state(inp=free, state=state, fsqr=r, fsqz=0.0, fsql=0.0, converged=r < 1e-9),
                residual=r, converged=r < 1e-9, n_unstable=int((w.real < 0).sum()),
                modes=[(float(w[k].real), _modes(Q @ V[:, k].real, unravel, icfg)) for k in order[:nev]])
