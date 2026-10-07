"""Parameter scan: coils per half period, curvature limit and length limit, for each case (weighted penalties).
Results are appended to scan_results.json (re-running skips finished points) and plotted in scan.png."""
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
WEIGHTS = dict(field=1e4, normal=1e4, length=1e2, curvature=1e1, msc=1e1, total_curvature=1e1, arclength=1e-2,
               coil_distance=1e3, surface_distance=1e3, linking=1e1)
OUTPUT = "scan_results.json"

""" Running the scan """
results = json.load(open(OUTPUT)) if os.path.exists(OUTPUT) else []
done = {(r["case"], r["n_coils"], r["curvature"], r["length"]) for r in results}
for name in CASES:
    eq = case(name)
    t = None
    for n, kappa, length in itertools.product(N_COILS, CURVATURE, LENGTH):
        if (name, n, kappa, length) in done:
            continue
        t = t or targets(eq["surface"], eq["B"], eq["inner"])
        print(f"\n=== case {name}: {n} coils per half period, curvature <= {kappa} 1/m, length <= {length} m")
        limits = dict(length=length, curvature=kappa, msc=(0.75 * kappa)**2, total_curvature=3 * np.pi,
                      arclength=0.05, coil_distance=0.12, surface_distance=0.2)
        coils0 = initial_coils(t, n, ORDER, N_SEGMENTS, 1.0, COIL_MINOR_RADIUS, axis=eq["axis"])
        coils, _, _ = optimize(coils0, t, limits, WEIGHTS, n, MAXITER, snapshot_every=10**9, verbose=False)
        results.append(dict(case=name, n_coils=n, curvature=kappa, length=length, **report("Optimized", coils, t, n)))
        json.dump(results, open(OUTPUT, "w"), indent=1)

""" Plotting: boundary B.n error vs coil count, one panel per case """
fig, axes = plt.subplots(1, len(CASES), figsize=(4.2 * len(CASES), 3.8), sharey=True)
colors, styles = dict(zip(CURVATURE, ("tab:blue", "tab:orange", "tab:green"))), dict(zip(LENGTH, (":", "--", "-")))
for ax, name in zip(axes, CASES):
    for kappa, length in itertools.product(CURVATURE, LENGTH):
        rows = sorted((r for r in results if (r["case"], r["curvature"], r["length"]) == (name, kappa, length)),
                      key=lambda r: r["n_coils"])
        ax.semilogy([r["n_coils"] for r in rows], [r["boundary_max"] for r in rows], marker="o", ms=4,
                    color=colors[kappa], ls=styles[length], label=f"κ ≤ {kappa:g}, L ≤ {length:g} m")
    ax.set(title=case(name)["title"], xlabel="coils per half period", xticks=N_COILS)
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("max |ΔB·n|/|B| on boundary")
axes[-1].legend(fontsize=7, ncol=1, loc="upper right")
plt.tight_layout(); plt.savefig("scan.png", dpi=130)
print("Wrote scan_results.json and scan.png")
