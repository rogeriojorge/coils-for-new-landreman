# coils-for-new-landreman

Coils, optimized with [ESSOS](https://github.com/uwplasma/ESSOS), for the explicit
non-axisymmetric MHD equilibria of [Landreman, arXiv:2609.26742](https://arxiv.org/abs/2609.26742),
and a check that they hold those equilibria.

The paper gives two families of smooth equilibria with exact nested flux surfaces in closed form:
one with rotational transform ι = 2, and one with sheared ι. Both have two field periods,
stellarator symmetry, finite pressure and net toroidal current. Here both families are written in
JAX, so the equilibrium and all its derivatives are known exactly. Virtual casing turns that
solution into the field the coils must produce. Coils are optimized for it with weighted penalties
or an augmented Lagrangian, then checked against VMEX and by field-line tracing.

![iota = 2 coils, penalty method](coils_iota2.gif) ![sheared iota case D coils, penalty method](coils_sheared_D.gif)

## Usage

```bash
pip install -r requirements.txt
python optimize_coils_iota2.py                              # ι = 2, weighted penalties
python optimize_coils_iota2_augmented_lagrangian.py         # ι = 2, augmented Lagrangian
python optimize_coils_sheared_iota.py                       # sheared ι, case "A", "B", "D" or "E" as argument
python optimize_coils_sheared_iota_augmented_lagrangian.py  # sheared ι, augmented Lagrangian
python scan_coils.py                                        # coil count / curvature / length / loop-cap scan
python benchmark_vmex.py                                    # fixed- and free-boundary (Newton) VMEX vs the analytic solution
python validate_fieldlines.py                               # field lines with the coils vs the analytic surfaces
pytest -q                                                   # machine-precision checks of the equilibria
```

Every script runs on its own and prints its progress. The optimizations write
`coils_<case>.json` (ESSOS coils), `coils_<case>.png` (coils on the boundary coloured by the
B·n error, plus convergence) and `coils_<case>.gif` (the optimization).

| file | content |
| --- | --- |
| `landreman_equilibria.py` | both families in JAX, the cases, the exact boundary, the exact interior coil field, the boundary target |
| `coil_optimization.py` | targets, objective terms, constraints, penalty optimizer, report, figure and movie |
| `optimize_coils_*.py` | one script per family and method |
| `scan_coils.py` | parameter scans (`scan_results.json`, `scan.png`, `scan_loop_cap.png`) |
| `benchmark_vmex.py` | VMEX benchmark (`benchmark_<case>.png`, `benchmark_results.json`, `wout_free_<case>.nc`) |
| `vmex_newton.py` | free-boundary VMEX by Newton, with a count of unstable ideal-MHD modes |
| `validate_fieldlines.py` | Poincaré sections with the coils vs analytic and free-boundary surfaces (`fieldlines_<case>.png`) |
| `check_tau_coils.py` | non-symmetric cases: errors against a finer target, mirror check (`tau_results.json`) |
| `tests/test_equilibria.py` | div B = 0, J × B = ∇p, B·∇ψ = 0, B·n = 0 on the boundary, ι(0) from the paper, vacuum coil field |

## Equilibria

| case | parameters | ι axis → edge | \|B\|max/\|B\|min on boundary | note |
| --- | --- | --- | --- | --- |
| ι = 2 | ε = 1/2, ψ_edge = 1/64 (paper fig. 1) | 2 | 1.6 | non-planar axis, closed field lines |
| sheared A | ε, S, k_b, λ = 1.08, 3, 0.7, 3.5 (paper fig. 2) | 5.69 → | 1.6 | nearly circular axis |
| sheared B | 4, 3.5, 0.7, 3.5 (paper fig. 2) | 4.36 → | 5.4 | not reproducible by coils |
| sheared D | 1, 2, 0.5, 3.5 (new) | 3.88 → 3.98 | 1.6 | elliptical axis, b/a = 1.28 |
| sheared E | 1, 1.75, 0.5, 3.5 (new) | 3.43 → 3.49 | 1.7 | like D, ι away from the ι = 4 resonance |

All cases are scaled to a 1 m mean axis radius and 1 T on the axis.

The sheared family is built on confocal ellipses whose foci, (x, y) = (0, ±√ε), are singularities
of the field. In paper cases B and C the foci lie about 0.2 m inside the plasma's outer edge, and
|B| on the boundary varies 5–12×. No coil set tried for B gets below 5% mean B·n error. Cases D
and E were found by scanning (ε, S, k_b, λ) for parameters that keep the foci away from the plasma
while keeping an elliptical axis and sheared ι.

## Method

**Exact coil field.** The plasma carries current, so the coils must reproduce only the field of the
currents outside it, B_coils = B − B_plasma. For points inside the plasma, virtual casing gives it
exactly:

    B_coils(x) = -(1/4π) ∮ (n' × B') × (x - x') / |x - x'|³ dA'

The boundary r(θ, ζ) and B are analytic, so the trapezoidal rule converges exponentially. At half
the minor radius the target is accurate to 2e-15 (sheared cases) and 1.4e-8 (ι = 2). On the
boundary itself the integral is singular. There, the on-surface singular quadrature of
[virtual_casing_jax](https://github.com/uwplasma/virtual_casing_jax) gives all three components.
A 32 × 32 grid at 4 digits is converged to about 1e-6 of |B|.

**Objective.** The coils must match:
- the interior field, which fixes the total coil current;
- the full boundary field: the normal part for B·n = 0, and the tangential part for the pressure
  balance a free-boundary equilibrium needs.

The engineering limits are:
- length ≤ 4.5 m and curvature ≤ 5 m⁻¹;
- mean squared curvature ≤ 14 m⁻²;
- total curvature ∫κ dl ≤ 5π (each loop adds 2π);
- arclength variation;
- coil–coil distance ≥ 0.12 m and coil–plasma distance ≥ 0.2 m;
- zero linking number.

The coils start as circles, 6 per half period (24 in total), Fourier order 6. For the sheared
cases they are centred on the elliptical axis. All gradients are exact (JAX).

**Methods.**
- *Penalty*: the limits are weighted penalties, minimized with L-BFGS-B (3000 iterations).
- *Augmented Lagrangian*: the field mismatch is the objective and each limit is a constraint,
  max(value − limit, 0) = 0, with its own multiplier (`essos.augmented_lagrangian`). It runs
  10 outer iterations of up to 400 inner L-BFGS-B iterations each. The augmented-Lagrangian
  scripts keep the earlier limits (5 coils, order 4, length 4 m, curvature 4 m⁻¹, total curvature
  3π), so they are compared with penalty runs at those limits.

## Results

Penalty method, final limits (6 coils per half period, length 4.5 m, curvature 5 m⁻¹, total
curvature 5π, 3000 iterations, about 10 min each):

| case | interior \|ΔB\|/\|B\| mean / max | boundary \|ΔB·n\|/\|B\| mean / max | boundary \|ΔB\|/\|B\| max | max κ (m⁻¹) |
| --- | --- | --- | --- | --- |
| ι = 2 | 8.7e-5 / 2.7e-4 | 1.7e-4 / 4.8e-3 | 6.5e-2 | 5.00 |
| sheared A | 4.8e-4 / 1.5e-3 | 1.8e-3 / 6.1e-3 | 6.6e-3 | 5.13 |
| sheared D | 6.2e-5 / 1.5e-4 | 1.9e-4 / 7.0e-4 | 7.5e-4 | 5.01 |
| sheared E | 4.1e-5 / 8.5e-5 | 1.3e-4 / 4.8e-4 | 5.5e-4 | 5.00 |

| ι = 2 | sheared A |
| --- | --- |
| ![](coils_iota2.png) | ![](coils_sheared_A.png) |
| **sheared D** | **sheared E** |
| ![](coils_sheared_D.png) | ![](coils_sheared_E.png) |

Movies of every run: `coils_<case>.gif` and `coils_<case>_al.gif`.

- D and E hold the whole boundary field to better than 0.1%, A to 0.7%.
- Relaxing the limits, mainly the loop cap (3π → 5π total curvature) together with 6 coils of
  order 6, cuts the errors 4–20× from the earlier limits below. 8 coils per half period gain only
  25% more.
- For ι = 2, the boundary field error is 6.5% at a few points on the inboard midplane near φ = 0,
  where |B| is highest, for every coil set tried. The target there is converged to 1e-6, so it is
  a limit of the coils, not of the target. Elsewhere it is below 0.5%.
- Coil forces are reported, not penalized.

**Augmented Lagrangian vs penalty**, at the earlier limits (5 coils, order 4, length 4 m,
curvature 4 m⁻¹, total curvature 3π, penalty 1000 iterations):

| case | method | interior mean / max | boundary B·n mean / max | max κ (m⁻¹) | time |
| --- | --- | --- | --- | --- | --- |
| ι = 2 | penalty | 3.4e-4 / 9.3e-4 | 6.2e-4 / 4.7e-3 | 4.02 | 94 s |
| ι = 2 | augmented Lagrangian | 2.7e-4 / 8.6e-4 | 5.4e-4 / 4.6e-3 | 4.00 | 430 s |
| sheared A | penalty | 3.9e-3 / 9.8e-3 | 8.7e-3 / 4.8e-2 | 4.56 | 65 s |
| sheared A | augmented Lagrangian | 4.9e-3 / 1.0e-2 | 1.2e-2 / 5.2e-2 | 4.01 | 854 s |
| sheared D | penalty | 1.2e-3 / 2.5e-3 | 3.1e-3 / 9.2e-3 | 4.09 | 62 s |
| sheared D | augmented Lagrangian | 1.2e-3 / 3.0e-3 | 3.2e-3 / 9.7e-3 | 4.00 | 867 s |

The augmented Lagrangian meets every limit exactly and matches the penalty field error for ι = 2
and D, but takes 5–14× longer. Figures: `coils_<case>_al.png`.

## Parameter scan

`scan_coils.py` runs the penalty optimization (500 iterations, 5-coil order-4 limits) for:
- every combination of 3–6 coils per half period, curvature limit 3–5 m⁻¹ and length limit
  3–5 m (108 runs);
- case A with total-curvature caps of 1.25–3·2π.

![parameter scan](scan.png)

| case | 3 coils | 4 coils | 5 coils | 6 coils |
| --- | --- | --- | --- | --- |
| ι = 2 | 2.5e-2 | 6.3e-3 | 4.8e-3 | 4.8e-3 |
| sheared A | 5.8e-2 | 4.9e-2 | 4.8e-2 | 3.9e-2 |
| sheared D | 1.9e-2 | 1.3e-2 | 1.1e-2 | 9.4e-3 |

*Maximum boundary \|ΔB·n\|/\|B\|, curvature ≤ 4 m⁻¹, length ≤ 4 m.*

- **Coils:** 5 per half period is within 1.24× of 6 in every case. Going to 4 costs up to 1.3×,
  and 3 costs 1.2–5×.
- **Length:** going from 3 to 4 m cuts the error 1.1–5.4×. 5 m helps mainly with 3 coils
  (1.5–2.9×).
- **Curvature:** 4 m⁻¹ is enough; 5 m⁻¹ changes the error by at most 18%. 3 m⁻¹ costs up to
  7× with 3 m coils.

![loop cap](scan_loop_cap.png)

- **Loop cap (case A):** removing the cap lets the coils reach 1.87·2π, which improves the error
  by only 8% at 5 coils of order 4. Tightening it to 1.25·2π costs 1.4–2×. With 6 coils of order
  6 the extra freedom pays off: A drops from 4.8% to 0.6% maximum B·n error (Results above), so
  A's earlier floor was set by the coil limits, not by the equilibrium.

## Validation

**Field lines.** `validate_fieldlines.py` traces field lines of B + (B_coils,ESSOS − B_coils,exact).
The plasma currents are held at their analytic values. The coil-field error is tabulated against the
exact interior field (accurate to 1e-11 or better) and fitted smoothly. Each figure overlays three
things: the analytic surfaces (black), the traced field lines (red) and the free-boundary VMEX
surfaces with the same coils (blue dashed).

| case | coil \|ΔB\|/\|B\| inside | field lines from analytic surfaces, mean / max |
| --- | --- | --- |
| sheared A | 4e-4 – 1.4e-3 | 3.95 / 25.8 mm |
| sheared D | 4e-5 – 1.8e-4 | 0.63 / 2.29 mm |
| sheared E | 2.6e-5 – 1.2e-4 | 0.57 / 1.98 mm |

| | |
| --- | --- |
| ![](fieldlines_D.png) | ![](fieldlines_E.png) |
| ![](fieldlines_A.png) | ![](fieldlines_iota2.png) |

- For D and E the field lines lie on the analytic surfaces to within about 2 mm, with no islands.
- For A, the lines near s = 0.7 spread into a band a few cm wide. A has high ι (≈ 5.7) and
  β ≈ 20%, so the remaining 0.1% field error resonates more strongly there.
- ι = 2 has closed field lines on every surface, so any error breaks the surfaces into short arcs.
  This is expected.

**VMEX.** `benchmark_vmex.py` gives VMEX the analytic profiles: toroidal flux, pressure p(s) and
enclosed current I(s). It runs a fixed-boundary solve on the exact boundary and a free-boundary
solve with only the coils.

![VMEX benchmark](benchmark_E.png)

| case | fixed boundary: deviation mean / max, ι error | free boundary (Newton): deviation mean / max, ι error | unstable modes |
| --- | --- | --- | --- |
| sheared A | 0.47 / 0.87 mm, 0.28% | 3.9 / 11.4 mm, 0.50% | 8 |
| sheared D | 0.08 / 0.15 mm, 0.26% | 3.0 / 6.4 mm, 0.44% | 7 |
| sheared E | 0.09 / 0.16 mm, 0.31% | 2.6 / 4.9 mm, 0.49% | 6 |

Figures for the other cases: `benchmark_A.png`, `benchmark_D.png`.

- Fixed-boundary VMEX reproduces the analytic equilibria.
- Free-boundary VMEX converges (|F| ≈ 2e-13) within a few mm of the analytic boundary, and ι is
  within 0.5%. The coil field lines stay much closer than that, within about 2 mm for D and E. The
  difference is a coherent shift that the unstable equilibrium is sensitive to.
- These equilibria are ideal-MHD unstable. The coils' vertical field has decay index
  n = −(R/B_Z) ∂B_Z/∂R = 2.4–9.5 at the axis, above the 3/2 limit for radial stability of a
  ~300 kA plasma. The coils match the external field closely, so this index belongs to the
  equilibrium. The Newton solve finds 6–8 directions with δW < 0, including the radial shift.
- [uwplasma/vmex#570](https://github.com/uwplasma/vmex/pull/570) adds vertical-field position
  control. It removes the radial drift in case E, but a helical axis displacement then grows.
- ι = 2 is not benchmarked: with ι exactly 2 everywhere, every field line closes on itself, and
  VMEC cannot converge to that with pressure.

**Newton vs descent.** VMEX's default solver is steepest descent on the MHD energy, and it should
stay the default. It is robust from a cold start, compiles fast and uses little memory. It cannot
settle on an unstable equilibrium, though: these free-boundary runs drift instead of converging.
`vmex_newton.free_boundary_newton(inp, coils, grid)` takes over in that case:
1. It runs descent on the fixed-boundary problem, to about 3e-9.
2. It solves the free-boundary force balance by Newton: a Krylov anchor, then dense damped steps.
3. It reaches |F| ≈ 1e-13 and returns the wout.
4. From the eigenvalues of the force Jacobian it returns the number of unstable modes and their
   (m, n).

Use it when a free-boundary solve stalls, or when you need residuals far below 1e-10. Always start
it from a converged descent state; from cold, or from 1e-4, it is fragile. It costs 14–52 s of
compile time and 1.5–3× the memory. Its converged equilibrium may be a saddle of the MHD energy,
so check the returned unstable-mode count.

VMEC sign conventions matter here. PHIEDGE must be +Φ (B along +φ) and curtor +I. VMEC2000 stops
on the wrong sign; VMEX did not, which [uwplasma/vmex#571](https://github.com/uwplasma/vmex/pull/571)
fixes. NZETA must be at least 2·NTOR + 4.

## Non-stellarator-symmetric equilibria (Issan et al.)

[Issan et al., arXiv:2610.07304](https://arxiv.org/abs/2610.07304) add a parameter τ to both
families that breaks stellarator symmetry and keeps the solution exact. τ → −τ is the image under
the rotation by π about the x axis, (x, y, z) → (x, −y, −z), with B reversed. The cases follow
[vmex-benchmark-analytical](https://github.com/rogeriojorge/vmex-benchmark-analytical)
(`integer_3d_tau`, `sheared_issan_tau` and their mirrors), scaled to 1 m and 1 T as above:

| case | parameters | ι | β | plasma-current share of B inside |
| --- | --- | --- | --- | --- |
| `iota2_tau` (`_mirror`) | ε = 1/2, ψ_edge = 1/64, τ = 0.5 (−0.5) | 2 | 4.5% | 17% |
| `issan_tau` (`_mirror`) | ε, S, k_b, λ = 1.2, 1.4, √0.28, 1.6, τ = 0.4 (−0.4) | 2.65 → 2.64 | 22% | 35% |

Neither family has a zero-pressure member, so there is no separate vacuum target: the coils
match the exterior (virtual-casing) field of the finite-β equilibrium, and the same coil set is
used for the vacuum and finite-β free-boundary runs below.

```bash
python optimize_coils_iota2.py iota2_tau                 # also iota2_tau_mirror
python optimize_coils_sheared_iota.py issan_tau          # also issan_tau_mirror
python check_tau_coils.py                                # fine-target errors and mirror check (tau_results.json)
python benchmark_vmex.py iota2_tau issan_tau             # VMEX, LASYM = T
```

**What changes without symmetry.** The JAX fields, flux labels and surfaces take τ, and the tests
check div B = 0, J × B = ∇p, B·n = 0 and the mirror map at τ ≠ 0. The optimization covers a full
field period instead of half: 12 coils per period (24 in total, as before) with
`stellsym=False`. ESSOS needed no change, and neither did the virtual-casing call, which already
used full-period grids. VMEX gets LASYM = T with the rbs and zbc boundary tables. The penalties and
limits are the same as for the symmetric cases.

**Coils.** Errors against a finer target (64 × 64 per period, 6 digits) than the optimization used
(32 × 32, 4 digits):

| case | boundary \|ΔB·n\|/\|B\| mean / max | boundary \|ΔB\|/\|B\| max | interior \|ΔB\|/\|B\| mean / max | max κ (m⁻¹) |
| --- | --- | --- | --- | --- |
| `iota2_tau` | 1.4e-4 / 6.0e-4 | 7.5e-4 | 7.3e-5 / 2.5e-4 | 5.00 |
| `iota2_tau_mirror` | 1.1e-4 / 5.0e-4 | 8.4e-4 | 4.9e-5 / 1.4e-4 | 5.00 |
| `issan_tau` | 3.8e-4 / 3.2e-3 | 3.2e-3 | 1.0e-4 / 5.6e-4 | 5.01 |
| `issan_tau_mirror` | 3.7e-4 / 3.1e-3 | 3.2e-3 | 1.0e-4 / 4.7e-4 | 5.01 |

| ι = 2, τ = 0.5 | sheared, τ = 0.4 |
| --- | --- |
| ![](coils_iota2_tau.png) | ![](coils_sheared_issan_tau.png) |

- All coils stay within the limits: length ≤ 4.5 m, curvature ≤ 5.01 m⁻¹, total curvature
  ≤ 2.22·2π (cap 2.5·2π).
- For ι = 2 the coarse target is the limit. Against it, `iota2_tau` shows a 1.6% boundary-field
  error, but its 32 → 64 change is 1.5% of |B|. Against the fine target the error is 7.5e-4. For
  `issan_tau` the coarse target agrees with the fine one to 2e-4.
- **Mirror check.** Rotating the τ coils by π about x and reversing their currents gives exactly
  the τ error on the −τ target (max B·n 6.026e-4 and 3.157e-3, the same to all printed digits). The
  equilibria, targets and coil handling are mirror symmetric. The −τ coils optimized on their own
  reach the same error, and their field agrees with the rotated τ coils to 8e-4 (ι = 2) and 2e-3
  (sheared). Their shapes differ by up to 0.43 m and 0.21 m: the optimization is mirror
  equivariant only in exact arithmetic, and round-off sends the two runs to different minima of
  equal quality.

**VMEX, LASYM = T.** Fixed boundary uses the exact boundary. Free boundary uses only the coils,
with Newton (`vmex_newton.py`): at finite β with the analytic p(s) and I(s), and in "vacuum" with
p = 0 and the same I(s).

| case | fixed boundary: fsq, deviation mean / max | free, finite β: \|F\|, deviation mean / max | free, p = 0: \|F\|, deviation mean / max |
| --- | --- | --- | --- |
| `iota2_tau` | 1.7e-10, 0.58 / 0.94 mm | 8.5e-3, 3.4 / 15 mm | 1.9e-2, 13 / 28 mm |
| `issan_tau` | 5.5e-7 (not converged), 2.4 / 3.4 mm | 6.3e-3, 11 / 54 mm | 1.2e-1, 29 / 79 mm |

![VMEX, ι = 2, τ = 0.5](benchmark_iota2_tau.png)

- **No free-boundary run converges** (`benchmark_issan_tau.png` shows the sheared case). Neither does
  VMEX's descent (fsq ≈ 2e-6 after 30000 iterations, 13 / 44 mm for `iota2_tau`).
- **The cause is not LASYM.** With LASYM = T forced, the symmetric case D converges by Newton to
  |F| = 2.3e-13, as it does with LASYM = F. The symmetric ι = 2 case (τ = 0) fails the same way
  with LASYM = F and T (descent fsq 1.3e-6 / 1.4e-6, 5.9 / 6.3 mm mean deviation). Constant
  rational ι = 2 is what VMEX cannot settle, with or without symmetry.
- **`issan_tau` needs more resolution.** Its fixed-boundary solve stalls at 5.5e-7 at MPOL = 5,
  NTOR = 5, and at 3.4e-8 and 2.1e-8 with (6, 8) and (8, 12). The benchmark deck uses (17, 24).
  The dense Newton Jacobian is impractical at that size, so the free-boundary numbers are an upper
  bound set by the solver, not by the coils.
- **Without pressure the shape moves 1–3 cm** at fixed current. For ι = 2, removing the current as
  well leaves ι ≈ 0.05: ι is current-driven, so a p = I = 0 run has no surfaces to compare.
- The coil decay index at the axis is 0.3, 0.15 and −0.6 (ι = 2), and −2.4, −2.6 and −0.2
  (sheared). Unlike A, D and E (n = 2.4–9.5), neither case is radially unstable. n < 0
  points to a vertical instability instead.

## Reproducing

| result | command | time (laptop CPU) |
| --- | --- | --- |
| coils, penalty | `python optimize_coils_iota2.py`, `python optimize_coils_sheared_iota.py` (`CASE`) | about 10 min each |
| coils, augmented Lagrangian | the `*_augmented_lagrangian.py` scripts | 7–15 min each |
| parameter scans | `python scan_coils.py` | about 1.5 h |
| VMEX benchmark | `python benchmark_vmex.py` | about 30 min |
| field lines | `python validate_fieldlines.py` | about 1.5 h (4 cases) |

Peak memory is about 10 GB, mostly in the virtual-casing step. Run the scripts one at a time on a
24 GB machine.

## Upstream fixes

- **virtual_casing_jax**: left-handed (φ, θ) grids inflated the singular quadrature about 2500×.
  Fixed in [uwplasma/virtual_casing_jax#19](https://github.com/uwplasma/virtual_casing_jax/pull/19),
  released in 0.0.10.
- **ESSOS**: the augmented Lagrangian capped every inner solve at 50 iterations. Fixed in
  [uwplasma/ESSOS#147](https://github.com/uwplasma/ESSOS/pull/147), which brought ι = 2 from 1.5e-3
  to 3.4e-4 mean boundary error. Inequality constraints are unsupported
  ([uwplasma/ESSOS#145](https://github.com/uwplasma/ESSOS/issues/145)), so limits are written as
  equalities on their violation. [uwplasma/ESSOS#158](https://github.com/uwplasma/ESSOS/pull/158)
  adds a virtual-casing coil example.
- **VMEX**: wrong-sign PHIEDGE check
  ([uwplasma/vmex#571](https://github.com/uwplasma/vmex/pull/571)) and free-boundary position
  control ([uwplasma/vmex#570](https://github.com/uwplasma/vmex/pull/570)).

## Reference

M. Landreman, *Analytic toroidal 3D MHD equilibria and steady Euler flows with invariant
surfaces*, [arXiv:2609.26742](https://arxiv.org/abs/2609.26742);
scripts at [landreman/analytic_3d_equilibria](https://github.com/landreman/analytic_3d_equilibria).
