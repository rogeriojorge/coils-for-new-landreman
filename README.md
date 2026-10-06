# coils-for-new-landreman

Coils for the explicit non-axisymmetric MHD equilibria of
[Landreman, arXiv:2609.26742](https://arxiv.org/abs/2609.26742), optimized with
[ESSOS](https://github.com/uwplasma/ESSOS).

The paper gives two families of smooth equilibria with exact nested flux surfaces in
closed form: one with uniform rotational transform ι = 2, and one with sheared ι. Both have two field
periods, stellarator symmetry and finite pressure. Here both families are written in JAX,
so the field, current, pressure and surfaces are differentiable in position and in the
family parameters. Coils are then found that reproduce the exact vacuum part of each field.

## Usage

```bash
pip install -r requirements.txt
python optimize_coils_iota2.py          # iota = 2 family, paper figure 1 (eps = 1/2)
python optimize_coils_sheared_iota.py   # sheared family, CASE = "A", "B" or "C" (paper figure 2)
pytest -q                               # machine-precision checks of the equilibria
```

Each script runs on its own: imports, input parameters, setup, a verbose optimization,
then results. It writes `coils_*.json` (ESSOS format) and `coils_*.png` (coils on the
plasma boundary, plus the convergence history).

## Files

| file | content |
| --- | --- |
| `landreman_equilibria.py` | both families in JAX (`iota2_B/psi/surface`, `sheared_B/psi/surface/iota_axis`), the exact boundary, the exact coil-field target `coil_field`, and a Fourier fit `fit_surface` for plotting |
| `optimize_coils_iota2.py` | coils for the ι = 2 family |
| `optimize_coils_sheared_iota.py` | coils for the sheared-ι family |
| `tests/test_equilibria.py` | div B = 0, J × B = ∇p, B · ∇ψ = 0, B · n = 0 on the boundary, ι(0) from the paper, and a check that the coil-field target is curl- and divergence-free |

## Method

**Exact coil-field target.** The plasma carries current, so the coils must not reproduce the
total field B. They must reproduce the field B_coils of the currents outside the plasma.
Virtual casing turns the boundary current K = n × B into an exact expression for B_coils at
any interior point x:

    B_coils(x) = -(1/4π) ∮ (n' × B') × (x - x') / |x - x'|³ dA'

The analytic solution supplies B and the boundary parameterization r(θ, ζ) exactly. The
integrand is smooth and periodic in (θ, ζ), so the trapezoidal rule converges exponentially
and reaches machine precision away from the boundary. The scripts match the coil field to
B_coils at half the minor radius (ψ = ψ_edge/4). They print an error bound for the target,
taken from the same quadrature at 2/3 resolution. For ι = 2 the boundary label is untwisted
(α = 2ζ − θ), which keeps the quadrature grid near-orthogonal. That reduces the cost by
more than an order of magnitude.

**Coil optimization.** The coils start as circles and use stellarator symmetry and two
field periods. The objective is the mean squared mismatch |B_coils,ESSOS − B_coils|²/|B|²
on the target points, plus penalties on length, curvature, coil–coil and coil–plasma
distance, and linking number (no winding). Gradients are exact (JAX). L-BFGS-B does the
minimization, and the coil currents are free. Coil forces are reported but not penalized.

## Results

Default settings: major radius 1 m, |B| = 1 T on the axis, 6 coils per half period (24 in
total), Fourier order 4, length ≤ 3.4 m, curvature ≤ 3 m⁻¹, 1000 L-BFGS-B iterations
(about 70 s on a laptop CPU). |ΔB|/|B| is the mismatch with the exact coil field at the
384 target points. The target error bound comes from the 2/3-resolution quadrature and
bounds the error of the target itself from above.

| case | ι on axis | target error bound | mean / max \|ΔB\|/\|B\| | max curvature (m⁻¹) | mean force (N/m) |
| --- | --- | --- | --- | --- | --- |
| ι = 2, ε = 1/2 (`optimize_coils_iota2.py`) | 2 | 1.4e-8 | 2.2e-4 / 5.9e-4 | 3.00 | 9.4e4 |
| sheared, case A (`CASE = "A"`) | 5.69 | 1.9e-15 | 3.3e-3 / 8.2e-3 | 3.10 | 8.6e4 |
| sheared, case B (`CASE = "B"`) | 4.36 | 2.7e-10 | 1.6e-2 / 4.6e-2 | 4.63 | 8.8e5 |

![iota = 2 coils](coils_iota2.png)
![sheared case A coils](coils_sheared_A.png)
![sheared case B coils](coils_sheared_B.png)

All three equilibria have large rotational transform, so modular coils need strong
three-dimensional shaping. The ι = 2 coils reach a sub-0.1% match but still form wide loops.
Case B (strongly elliptical axis, ι ≈ 4.4) is not yet resolved at these engineering
limits. Ways to improve it include more coils, a higher Fourier order, and a coil
geometry that starts closer to the elliptical axis.

## Reference

M. Landreman, *Analytic toroidal 3D MHD equilibria and steady Euler flows with invariant
surfaces*, [arXiv:2609.26742](https://arxiv.org/abs/2609.26742);
scripts at [landreman/analytic_3d_equilibria](https://github.com/landreman/analytic_3d_equilibria).
