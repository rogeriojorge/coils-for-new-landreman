# coils-for-new-landreman

Coils, optimized with [ESSOS](https://github.com/uwplasma/ESSOS), for the explicit
non-axisymmetric MHD equilibria of [Landreman, arXiv:2609.26742](https://arxiv.org/abs/2609.26742).

The paper gives two families of smooth equilibria with exact nested flux surfaces in closed form:
one with rotational transform ι = 2, and one with sheared ι. Both have two field periods,
stellarator symmetry and finite pressure. Here both families are written in JAX, so the
equilibrium is known exactly, with all derivatives. That exact solution is used to compute,
to machine precision, the field the coils must produce. Coils are then optimized for it with
two methods: weighted penalties and an augmented Lagrangian.

![iota = 2 coils, penalty method](coils_iota2.gif) ![sheared iota case D coils, penalty method](coils_sheared_D.gif)

## Usage

```bash
pip install -r requirements.txt
python optimize_coils_iota2.py                              # ι = 2, weighted penalties
python optimize_coils_iota2_augmented_lagrangian.py         # ι = 2, augmented Lagrangian
python optimize_coils_sheared_iota.py                       # sheared ι, CASE = "A", "B" or "D"
python optimize_coils_sheared_iota_augmented_lagrangian.py  # sheared ι, augmented Lagrangian
python scan_coils.py                                        # coil count / curvature / length scan (108 runs)
pytest -q                                                   # machine-precision checks of the equilibria
```

Each script runs on its own and prints its progress. It writes `coils_<case>.json` (ESSOS coils),
`coils_<case>.png` (coils on the boundary coloured by the B·n error, plus convergence) and
`coils_<case>.gif` (the optimization). Penalty runs take about 2 minutes on a laptop CPU, augmented-Lagrangian runs 9–15 minutes, the scan about 1.5 hours.

| file | content |
| --- | --- |
| `landreman_equilibria.py` | both families in JAX, the exact boundary, the exact coil-field target, the boundary B·n target |
| `coil_optimization.py` | targets, objective terms, constraints, penalty optimizer, report, figure and movie |
| `optimize_coils_*.py` | one script per family and method |
| `scan_coils.py` | scan of coils per half period, curvature limit and length limit (`scan_results.json`, `scan.png`) |
| `tests/test_equilibria.py` | div B = 0, J × B = ∇p, B·∇ψ = 0, B·n = 0 on the boundary, ι(0) from the paper, vacuum coil field |

## Equilibria

| case | parameters | ι on axis | \|B\|max/\|B\|min on boundary | note |
| --- | --- | --- | --- | --- |
| ι = 2 | ε = 1/2, ψ_edge = 1/64 (paper fig. 1) | 2 | 1.6 | non-planar axis, closed field lines |
| sheared A | ε, S, k_b, λ = 1.08, 3, 0.7, 3.5 (paper fig. 2) | 5.69 | 1.6 | nearly circular axis |
| sheared B | 4, 3.5, 0.7, 3.5 (paper fig. 2) | 4.36 | 5.4 | not reproducible by coils (below) |
| sheared D | 1, 2, 0.5, 3.5 (new) | 3.88 | 1.6 | elliptical axis, b/a = 1.28 |

All cases are scaled to a 1 m axis radius and 1 T on the axis.

The sheared family is built on confocal ellipses whose foci, (x, y) = (0, ±√ε), are field
singularities. In paper cases B and C the foci lie about 0.2 m inside the plasma's outer edge,
and |B| on the boundary varies 5–12×. Coils at a practical distance cannot reproduce that
structure. Every weight, limit, coil count (6 or 8), Fourier order (4 or 6) and initial shape
tried for B left 5–8% mean B·n error. Case D was found by scanning (ε, S, k_b, λ) for
parameters that keep the foci away from the plasma while keeping an elliptical axis and
sheared ι.

## Method

**Exact coil field.** The plasma carries current, so the coils must reproduce only the field of
the currents outside it. For any point x inside the plasma, virtual casing with the surface
current n × B gives that field exactly:

    B_coils(x) = -(1/4π) ∮ (n' × B') × (x - x') / |x - x'|³ dA'

B and the boundary r(θ, ζ) are analytic, and the integrand is smooth and periodic, so the
trapezoidal rule converges exponentially. At half the minor radius, where the coils are
matched, the target is accurate to 2e-15 (sheared A, D) and 1.4e-8 (ι = 2), measured against
the same quadrature at 2/3 resolution. On the boundary itself the integral is singular. There,
the on-surface singular quadrature of
[virtual_casing_jax](https://github.com/uwplasma/virtual_casing_jax) gives the B·n the coils
must supply.

**Objective.** Two field terms are needed:
- the interior mismatch |B_coils − target|²/|B|², which fixes the total coil current and the
  toroidal circulation;
- the boundary B·n mismatch.

Matching B·n alone leaves the total current free, giving 19–370% interior errors in the scan.
The coil limits are:
- length ≤ 4 m;
- curvature ≤ 4 m⁻¹;
- mean squared curvature ≤ 9 m⁻²;
- total curvature ∫κ dl ≤ 3π (each loop adds 2π, so this prevents loops);
- arclength variation;
- coil–coil distance ≥ 0.12 m and coil–plasma distance ≥ 0.2 m;
- zero linking number.

All gradients are exact (JAX). The coils start as circles, 5 per half period (20 in total, chosen by the scan),
Fourier order 4, centred on the elliptical axis for the sheared cases.

**Penalty vs augmented Lagrangian.**
- *Penalty*: the limits are weighted penalties, minimized with SciPy L-BFGS-B (1000 iterations).
- *Augmented Lagrangian*: the field mismatch is the objective and every limit is an equality
  constraint, max(value − limit, 0) = 0, each with its own multiplier
  (`essos.augmented_lagrangian`). No weights need tuning. It runs 10 outer iterations, each an
  L-BFGS-B solve of up to 400 iterations with tolerance 1e-6.

## Results

| case | method | interior \|ΔB\|/\|B\| mean / max | boundary \|ΔB·n\|/\|B\| mean / max | max κ (m⁻¹) | max ∫κdl / 2π | time |
| --- | --- | --- | --- | --- | --- | --- |
| ι = 2 | penalty | 2.0e-4 / 6.5e-4 | 4.3e-4 / 2.0e-3 | 4.01 | 1.50 | 101 s |
| ι = 2 | augmented Lagrangian | 2.6e-4 / 8.4e-4 | 5.0e-4 / 2.5e-3 | 4.00 | 1.50 | 527 s |
| sheared A | penalty | 3.6e-3 / 9.5e-3 | 9.2e-3 / 4.6e-2 | 4.34 | 1.50 | 93 s |
| sheared A | augmented Lagrangian | 4.6e-3 / 9.7e-3 | 1.3e-2 / 5.3e-2 | 4.00 | 1.50 | 891 s |
| sheared D | penalty | 1.1e-3 / 2.4e-3 | 3.2e-3 / 1.1e-2 | 4.07 | 1.50 | 98 s |
| sheared D | augmented Lagrangian | 1.1e-3 / 2.7e-3 | 3.3e-3 / 1.1e-2 | 4.00 | 1.50 | 870 s |

- The penalty method reaches the lowest field error, by letting the limits overshoot slightly
  (curvature 4.01–4.34 m⁻¹ for a 4 m⁻¹ limit).
- The augmented Lagrangian meets every limit exactly with no weights to tune. It matches the
  penalty field for case D and is within 1.3× for ι = 2 and 1.4× for A. It takes 5–10× longer.
- Case D gives about 1% maximum boundary B·n with a strongly elliptical axis: 4× better than A,
  with lower ι.
- Mean coil forces are 1.0–1.6 × 10⁵ N/m. They are reported, not penalized.

| | penalty | augmented Lagrangian |
| --- | --- | --- |
| ι = 2 | ![](coils_iota2.png) | ![](coils_iota2_al.png) |
| sheared A | ![](coils_sheared_A.png) | ![](coils_sheared_A_al.png) |
| sheared D | ![](coils_sheared_D.png) | ![](coils_sheared_D_al.png) |

Movies of every run: `coils_<case>.gif` and `coils_<case>_al.gif`.

## Parameter scan

`scan_coils.py` runs the penalty optimization (500 iterations) for every combination of:
- 3, 4, 5 or 6 coils per half period;
- curvature limit 3, 4 or 5 m⁻¹ (mean squared curvature limit (0.75 κ)²);
- length limit 3, 4 or 5 m;

for each of the three cases, 108 runs in total. Each case's targets are computed once.

![parameter scan](scan.png)

| case | 3 coils | 4 coils | 5 coils | 6 coils |
| --- | --- | --- | --- | --- |
| ι = 2 | 2.7e-2 | 6.7e-3 | 2.2e-3 | 1.7e-3 |
| sheared A | 6.0e-2 | 5.1e-2 | 4.6e-2 | 4.2e-2 |
| sheared D | 1.9e-2 | 1.3e-2 | 9.8e-3 | 9.7e-3 |

*Maximum boundary \|ΔB·n\|/\|B\| at curvature ≤ 4 m⁻¹ and length ≤ 4 m.*

- **Coil length matters most for ι = 2.** Going from 3 to 4 m cuts the error 2.4–6× for ι = 2,
  and 1.3–2.6× for A and D. 5 m helps only ι = 2 with 3–4 coils.
- **Curvature: 4 m⁻¹ is enough.** 5 m⁻¹ changes the error by at most 15%. 3 m⁻¹ costs 1.5–9×,
  most with short coils.
- **Five coils per half period is the compromise.** It is within 1.3× of six coils in every
  case, while 4 coils cost 3× for ι = 2. Case D still reaches 1.3% with 4.
- **The loop cap binds.** The total curvature sits at 1.5·2π whenever coils may be 4 m or
  longer. For case A this, not coil count or length, sets the 4–5% plateau.

Earlier objective variants (500 iterations each) fixed the rest of the defaults:
- Interior field plus boundary B·n is required; either term alone fails.
- A total-curvature cap of 1.25·2π instead of 1.5·2π costs about 25% (ι = 2) and 5% (A).
- A stronger coil–coil distance term changed nothing.

## Upstream fixes

- **virtual_casing_jax**: left-handed (φ, θ) grids inflated the singular quadrature about 2500×,
  which looked like a hang.
  [uwplasma/virtual_casing_jax#19](https://github.com/uwplasma/virtual_casing_jax/pull/19)
  fixes it; released in virtual-casing-jax 0.0.10.
- **ESSOS**: `ALM_model_jaxopt_lbfgsb` capped every inner solve at jaxopt's default 50
  iterations. [uwplasma/ESSOS#147](https://github.com/uwplasma/ESSOS/pull/147), now merged,
  exposes the cap. Raising it brought the augmented Lagrangian from 1.5e-3 to 3.4e-4 mean
  boundary error for ι = 2. Inequality
  constraints are also unsupported ([uwplasma/ESSOS#145](https://github.com/uwplasma/ESSOS/issues/145)),
  so the scripts write each limit as an equality on its violation.

## Reference

M. Landreman, *Analytic toroidal 3D MHD equilibria and steady Euler flows with invariant
surfaces*, [arXiv:2609.26742](https://arxiv.org/abs/2609.26742);
scripts at [landreman/analytic_3d_equilibria](https://github.com/landreman/analytic_3d_equilibria).
