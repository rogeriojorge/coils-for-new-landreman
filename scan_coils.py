"""Parameter scans (weighted penalties): coils per half period x curvature limit x length limit for each case,
then the loop cap (total curvature limit) for case A. Results are appended to scan_results.json (re-running skips
finished points) and plotted in scan.png and scan_loop_cap.png."""
import itertools
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from landreman_equilibria import case
from coil_optimization import targets, initial_coils, optimize, report

""" Scan parameters """
CASES = ("iota2", "A", "D")
N_COILS = (3, 4, 5, 6)                      # coils per half field period
CURVATURE = (3.0, 4.0, 5.0)                 # 1/m; mean squared curvature limit = (0.75 kappa)^2
LENGTH = (3.0, 4.0, 5.0)                    # m
ORDER, N_SEGMENTS, COIL_MINOR_RADIUS, MAXITER = 4, 100, 0.55, 500
WEIGHTS = dict(field=1e4, boundary=1e4, length=1e2, curvature=1e1, msc=1e1, total_curvature=1e1, arclength=1e-2,
               coil_distance=1e3, surface_distance=1e3, linking=1e1)
LOOP_CAPS = (1.25, 1.5, 2.0, 2.5, 3.0)       # total curvature / 2 pi, case A at 5 coils, kappa <= 4, L <= 4 m
OUTPUT = "scan_results.json"

""" Running the scans """
results = json.load(open(OUTPUT)) if os.path.exists(OUTPUT) else []
key = lambda r: (r["case"], r["n_coils"], r["curvature"], r["length"], r.get("loop_cap", 1.5))
done = {key(r) for r in results}
runs = [(name, n, kappa, length, 1.5) for name in CASES for n, kappa, length in itertools.product(N_COILS, CURVATURE, LENGTH)]
runs += [("A", 5, 4.0, 4.0, cap) for cap in LOOP_CAPS]
built = {}
for name, n, kappa, length, cap in runs:
    if (name, n, kappa, length, cap) in done:
        continue
    eq = case(name)
    t = built[name] = built.get(name) or targets(eq["surface"], eq["B"], eq["inner"])
    print(f"\n=== case {name}: {n} coils per half period, curvature <= {kappa} 1/m, length <= {length} m, "
          f"total curvature <= {cap} x 2 pi")
    limits = dict(length=length, curvature=kappa, msc=(0.75 * kappa)**2, total_curvature=cap * 2 * np.pi,
                  arclength=0.05, coil_distance=0.12, surface_distance=0.2)
    coils0 = initial_coils(t, n, ORDER, N_SEGMENTS, 1.0, COIL_MINOR_RADIUS, axis=eq["axis"])
    coils, _, _ = optimize(coils0, t, limits, WEIGHTS, n, MAXITER, snapshot_every=10**9, verbose=False)
    results.append(dict(case=name, n_coils=n, curvature=kappa, length=length, loop_cap=cap,
                        **report("Optimized", coils, t, n)))
    json.dump(results, open(OUTPUT, "w"), indent=1)

""" Plotting: boundary B.n error vs coil count, one panel per case """
fig, axes = plt.subplots(1, len(CASES), figsize=(4.2 * len(CASES), 3.8), sharey=True)
colors, styles = dict(zip(CURVATURE, ("tab:blue", "tab:orange", "tab:green"))), dict(zip(LENGTH, (":", "--", "-")))
for ax, name in zip(axes, CASES):
    for kappa, length in itertools.product(CURVATURE, LENGTH):
        rows = sorted((r for r in results if (r["case"], r["curvature"], r["length"], r.get("loop_cap", 1.5))
                       == (name, kappa, length, 1.5)),
                      key=lambda r: r["n_coils"])
        ax.semilogy([r["n_coils"] for r in rows], [r["boundary_max"] for r in rows], marker="o", ms=4,
                    color=colors[kappa], ls=styles[length], label=f"κ ≤ {kappa:g}, L ≤ {length:g} m")
    ax.set(title=case(name)["title"], xlabel="coils per half period", xticks=N_COILS)
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("max |ΔB·n|/|B| on boundary")
axes[-1].legend(fontsize=7, ncol=1, loc="upper right")
plt.tight_layout(); plt.savefig("scan.png", dpi=130)

""" Plotting: case A against the loop cap """
rows = sorted((r for r in results if key(r)[:4] == ("A", 5, 4.0, 4.0)), key=lambda r: r.get("loop_cap", 1.5))
fig, ax = plt.subplots(figsize=(4.6, 3.6))
caps = [r.get("loop_cap", 1.5) for r in rows]
ax.semilogy(caps, [r["boundary_max"] for r in rows], "o-", label="max |ΔB·n|/|B|")
ax.semilogy(caps, [r["boundary_mean"] for r in rows], "s--", label="mean |ΔB·n|/|B|")
ax.set(xlabel="total curvature limit / 2π", title="sheared ι, case A: loop cap"); ax.grid(alpha=0.3, which="both")
ax.legend(); plt.tight_layout(); plt.savefig("scan_loop_cap.png", dpi=130)
print("Wrote scan_results.json, scan.png and scan_loop_cap.png")
