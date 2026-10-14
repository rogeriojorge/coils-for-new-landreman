"""Machine-precision checks of the analytic equilibria and of the exact coil-field target."""
import jax
import jax.numpy as jnp
import pytest
from landreman_equilibria import (iota2_B, iota2_psi, iota2_surface, sheared_B, sheared_psi,
                                  sheared_surface, sheared_iota_axis, boundary, coil_field)

FAMILIES = {  # B(x), psi(x), surface(theta, zeta, edge fraction), dp/dpsi, edge psi
    "iota2": (lambda x: iota2_B(x, .5), lambda x: iota2_psi(x, .5),
              lambda t, z, f=1.: iota2_surface(t, z, .5, f / 64), -2., 1 / 64),
    "sheared": (lambda x: sheared_B(x, 4., 3.5, 3.5), lambda x: sheared_psi(x, 4., 3.5, 3.5),
                lambda t, z, f=1.: sheared_surface(t, z, 4., 3.5, 3.5, .7 * f**.5), -1 / 3.5**2, .245),
    "iota2_tau": (lambda x: iota2_B(x, .5, .5), lambda x: iota2_psi(x, .5, .5),
                  lambda t, z, f=1.: iota2_surface(t, z, .5, f / 64, .5), -2. / .75, 1 / 64),
    "sheared_tau": (lambda x: sheared_B(x, 1.2, 1.4, 1.6, .4), lambda x: sheared_psi(x, 1.2, 1.4, 1.6, .4),
                    lambda t, z, f=1.: sheared_surface(t, z, 1.2, 1.4, 1.6, (.28 * f)**.5, .4), -1 / 1.6**2, .14)}
curl = lambda J: jnp.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


@pytest.mark.parametrize("name", FAMILIES)
def test_mhd_equilibrium(name):
    B, psi, surface, dp, edge = FAMILIES[name]
    points = jax.vmap(surface)(jnp.linspace(0, 6, 9), jnp.linspace(0, 3, 9), jnp.linspace(.1, 1, 9))
    for x in points:
        J = jax.jacfwd(B)(x)
        assert abs(jnp.trace(J)) < 1e-13                                    # div B = 0
        assert jnp.allclose(jnp.cross(curl(J), B(x)), dp * jax.grad(psi)(x), atol=1e-13)  # J x B = grad p
    assert jnp.allclose(jax.vmap(psi)(jax.vmap(surface)(jnp.arange(5.), jnp.arange(5.))), edge, atol=1e-14)
    gamma, normal = boundary(surface, 8, 8)
    assert jnp.max(jnp.abs(jnp.sum(jax.vmap(jax.vmap(B))(gamma) * normal, -1))) < 1e-13  # B . n = 0


def test_sheared_iota_axis():
    assert abs(sheared_iota_axis(2., 1.) - 2.28690) < 1e-5  # value quoted in the paper


def test_coil_field_is_vacuum():
    B, _, surface, _, _ = FAMILIES["iota2"]
    x = surface(1., .5, .25)
    J = jax.jacfwd(lambda y: coil_field(surface, B, y[None], 128, 2048)[0])(x)
    assert abs(jnp.trace(J)) < 1e-12 and jnp.max(jnp.abs(curl(J))) < 1e-9


def test_tau_mirror():  # tau -> -tau is the rotation by pi about x: B_-tau(Rx) = -R B_tau(x), R = diag(1, -1, -1)
    R, x = jnp.array([1., -1., -1.]), jnp.array([.9, .3, .05])
    assert jnp.allclose(iota2_B(R * x, .5, -.5), -R * iota2_B(x, .5, .5), atol=1e-14)
    assert jnp.allclose(sheared_B(R * x, 1.2, 1.4, 1.6, -.4), -R * sheared_B(x, 1.2, 1.4, 1.6, .4), atol=1e-14)
