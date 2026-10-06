"""Landreman's explicit non-axisymmetric MHD equilibria (arXiv:2609.26742) in JAX.

Two families with exact nested flux surfaces, two field periods and stellarator symmetry:
``iota2_*`` (uniform transform iota = 2) and ``sheared_*`` (sheared transform). Every
function is pure ``jax.numpy``, so fields, currents ``curl B``, pressure gradients and
boundary shapes are differentiable in position and in the family parameters.
Lengths and fields are in the paper's normalized units; ``coil_target`` takes the
physical scale factors.
"""
import jax
import jax.numpy as jnp
from essos.surfaces import surfacerzfourier_from_boundary

jax.config.update("jax_enable_x64", True)
NFP = 2


# ---------------- family 1: iota = 2 (section 2 of the paper) ----------------
def iota2_B(x, eps):
    """Eq. (2.2): B(x) = A B0(A^-1 x) with A = diag(sqrt(1+eps), sqrt(1-eps), 1)."""
    a, b = jnp.sqrt(1 + eps), jnp.sqrt(1 - eps)
    s = x[0]**2 / a**2 + x[1]**2 / b**2
    F = jnp.sqrt(1 - (1 - s)**2 - 4 * x[2]**2)
    return jnp.array([(2 * x[2] * x[0] - a / b * F * x[1]) / s,
                      (2 * x[2] * x[1] + b / a * F * x[0]) / s, 1 - s])


def iota2_psi(x, eps):
    """Eq. (2.4): flux label, zero on the axis; pressure p = p_a - 2 psi."""
    return (x[0]**2 + x[1]**2 + 4 * x[2]**2 + jnp.sum(iota2_B(x, eps)**2) - 2 + eps**2) / 4


def iota2_surface(theta, zeta, eps, psi):
    """Eqs. (2.9)-(2.10), (2.14): point on the surface psi. The field-line label alpha = 2 zeta - theta
    untwists the iota = 2 winding, so theta is a poloidal angle (smooth, near-orthogonal grid)."""
    alpha = 2 * zeta - theta
    u, v = -eps / 2 + jnp.sqrt(psi) * jnp.cos(alpha), jnp.sqrt(psi) * jnp.sin(alpha)
    L = jnp.sqrt((1 + jnp.sqrt(1 - 4 * (u**2 + v**2))) / 2)
    c, s = jnp.cos(zeta), jnp.sin(zeta)
    return jnp.array([jnp.sqrt(1 + eps) * (L * c + (u * c + v * s) / L),
                      jnp.sqrt(1 - eps) * (L * s + (v * c - u * s) / L),
                      v * jnp.cos(2 * zeta) - u * jnp.sin(2 * zeta)])


# ---------------- family 2: sheared iota (section 3 of the paper) ----------------
def sheared_B(x, eps, S, lam):
    """Eqs. (3.1)-(3.2), with the principal square root (positive real part)."""
    w = jnp.conj(x[0] + 1j * x[1])
    K = w * jnp.sqrt(1 + eps / w**2)
    Xi = jnp.conj(w) * K + jnp.pi / 2 - S
    phase = jnp.exp(-1j * lam * x[2])
    Bxy = phase * 1j * jnp.sin(Xi) / (2 * K)
    return jnp.array([Bxy.real, Bxy.imag, jnp.real(phase * jnp.cos(Xi)) / lam])


def sheared_psi(x, eps, S, lam):
    """Eq. (3.3): psi = (X^2 + Y^2)/2 with X = lam B_z, Y = -sin(lam z)."""
    return ((lam * sheared_B(x, eps, S, lam)[2])**2 + jnp.sin(lam * x[2])**2) / 2


def sheared_surface(theta, zeta, eps, S, lam, k):
    """Eqs. (3.15)-(3.21): point on the surface psi = k^2/2 at angles (chi = theta, zeta)."""
    X, Y = -k * jnp.cos(theta), k * jnp.sin(theta)
    nu = eps / 2 * jnp.sin(2 * zeta)
    sigma = (S + jnp.arctan(jnp.tanh(nu) * Y / jnp.sqrt(1 - Y**2))
             - jnp.arcsin(X / jnp.sqrt(jnp.cosh(nu)**2 - Y**2)))
    h = jnp.sqrt(4 * sigma**2 + eps**2)
    return jnp.array([jnp.sqrt((h - eps) / 2) * jnp.cos(zeta),
                      jnp.sqrt((h + eps) / 2) * jnp.sin(zeta), -jnp.arcsin(Y) / lam])


def sheared_iota_axis(eps, S, n=512):
    """Eq. (3.29): on-axis transform iota(0) = h(S) <sech nu>."""
    zeta = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
    return jnp.sqrt(4 * S**2 + eps**2) * jnp.mean(1 / jnp.cosh(eps / 2 * jnp.sin(2 * zeta)))


# ---------------- exact boundary and coil target ----------------
def boundary(surface, ntheta, nphi, newton_steps=30):
    """Exact points and unit normals of surface(theta, zeta) on a grid uniform in theta and in
    the cylindrical angle phi over one field period, shape (nphi, ntheta, 3). zeta(phi) is a
    Newton root (phi is monotonic in zeta); normals use the exact tangents d/dtheta x d/dzeta."""
    theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0, 2 * jnp.pi / NFP, nphi, endpoint=False)

    def point(t, p):
        angle = lambda z: jnp.arctan2(surface(t, z)[1], surface(t, z)[0])
        step = lambda z, _: (z - jnp.angle(jnp.exp(1j * (angle(z) - p))) / jax.grad(angle)(z), None)
        z = jax.lax.scan(step, p, None, newton_steps)[0]
        normal = jnp.cross(jax.jacfwd(surface, 0)(t, z), jax.jacfwd(surface, 1)(t, z))
        return surface(t, z), normal / jnp.linalg.norm(normal)
    return jax.vmap(lambda p: jax.vmap(lambda t: point(t, p))(theta))(phi)


def coil_field(surface, B, points, ntheta=256, nzeta=512):
    """Exact field of all currents outside the plasma (the coils) at points inside it.

    Virtual casing with the surface current K = n x B on the plasma boundary gives
    B_coils(x) = -(1/4 pi) \oint (n' x B') x (x - x') / |x - x'|^3 dA' for x inside. The integrand is
    smooth and periodic in (theta, zeta), so the trapezoidal rule converges exponentially; away
    from the boundary it reaches machine precision (see README for the resolution study)."""
    t, z = (a.ravel() for a in jnp.meshgrid(jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False),
                                             jnp.linspace(0, 2 * jnp.pi, nzeta, endpoint=False)))
    r = jax.vmap(surface)(t, z)
    N = jnp.cross(jax.vmap(jax.jacfwd(surface, 0))(t, z), jax.vmap(jax.jacfwd(surface, 1))(t, z))
    KdA = jnp.cross(N * jnp.sign(jnp.sum(r * N)), jax.vmap(B)(r)) * (2 * jnp.pi)**2 / (ntheta * nzeta)

    def one(x):
        d = x - r
        return -jnp.sum(jnp.cross(KdA, d) / jnp.linalg.norm(d, axis=1)[:, None]**3, axis=0) / (4 * jnp.pi)
    return jax.lax.map(one, points, batch_size=32)


def fit_surface(surface, mpol, ntor, ntheta=64, nphi=64, **kwargs):
    """Stellarator-symmetric Fourier fit of surface(theta, zeta) -> ESSOS SurfaceRZFourier (plots)."""
    gamma = boundary(surface, ntheta, nphi)[0]
    R, Z = jnp.hypot(gamma[..., 0], gamma[..., 1]), gamma[..., 2]
    theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0, 2 * jnp.pi / NFP, nphi, endpoint=False)
    m, n = jnp.arange(mpol + 1), jnp.arange(-ntor, ntor + 1)
    angle = m[None, :, None, None] * theta - NFP * n[:, None, None, None] * phi[:, None]
    weight = jnp.where((m[None] == 0) & (n[:, None] == 0), 1.0, 2.0) / (ntheta * nphi)
    keep = (m[None] > 0) | (n[:, None] >= 0)
    rbc = jnp.where(keep, weight * jnp.sum(R * jnp.cos(angle), axis=(-2, -1)), 0)
    zbs = jnp.where(keep, weight * jnp.sum(Z * jnp.sin(angle), axis=(-2, -1)), 0)
    return surfacerzfourier_from_boundary(rbc, zbs, NFP, **kwargs)
