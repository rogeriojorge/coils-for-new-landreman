"""Fewer coils per half period at the final limits (order 6, curvature 5 1/m, total curvature 5 pi), with the coil
length allowed to grow so fewer coils can still shape the field. Results go to scan_fewer_results.json (re-running
skips finished points) and scan_fewer.png."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from landreman_equilibria import case
from coil_optimization import targets, initial_coils, optimize, report

""" Scan parameters """
CASES = ("iota2", "A", "D", "E")
N_COILS = (2, 3, 4, 6)                      # coils per half field period (6 = the final coils, for reference)
LENGTH = (4.5, 6.0)                         # m
ORDER, N_SEGMENTS, COIL_MINOR_RADIUS, MAXITER = 6, 140, 0.55, 1500
LIMITS = dict(curvature=5.0, msc=14.0, total_curvature=5 * np.pi, arclength=0.05, coil_distance=0.12, surface_distance=0.2)
WEIGHTS = dict(field=1e4, boundary=1e4, length=1e2, curvature=1e1, msc=1e1, total_curvature=1e1, arclength=1e-2,
               coil_distance=1e3, surface_distance=1e3, linking=1e1)
OUTPUT = "scan_fewer_results.json"

""" Running the scan """
results = json.load(open(OUTPUT)) if os.path.exists(OUTPUT) else []
done = {(r["case"], r["n_coils"], r["length"]) for r in results}
for name in CASES:
    eq = case(name)
    t = targets(eq["surface"], eq["B"], eq["inner"])
    for n in N_COILS:
        for length in LENGTH:
            if (name, n, length) in done:
                continue
            print(f"\n=== case {name}: {n} coils per half period, length <= {length} m", flush=True)
            coils0 = initial_coils(t, n, ORDER, N_SEGMENTS, 1.0, COIL_MINOR_RADIUS,
                                   **({} if name == "iota2" else dict(axis=eq["axis"])))
            coils, _, _ = optimize(coils0, t, dict(LIMITS, length=length), WEIGHTS, n, MAXITER, snapshot_every=10**9, verbose=False)
            results.append(dict(case=name, n_coils=n, length=length, **report("Optimized", coils, t, n)))
            json.dump(results, open(OUTPUT, "w"), indent=1)
            coils.to_json(f"scan_fewer_{name}_{n}_{length:g}.json")

""" Plotting: boundary B.n error vs coil count """
fig, axes = plt.subplots(1, len(CASES), figsize=(3.6 * len(CASES), 3.6), sharey=True)
for ax, name in zip(axes, CASES):
    for length, ls in zip(LENGTH, ("--", "-")):
        rows = sorted((r for r in results if (r["case"], r["length"]) == (name, length)), key=lambda r: r["n_coils"])
        ax.semilogy([r["n_coils"] for r in rows], [r["boundary_max"] for r in rows], "o" + ls, label=f"L ≤ {length:g} m, max")
        ax.semilogy([r["n_coils"] for r in rows], [r["boundary_mean"] for r in rows], "s" + ls, alpha=0.5, label=f"L ≤ {length:g} m, mean")
    ax.set(title=case(name)["title"], xlabel="coils per half period", xticks=N_COILS); ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("|ΔB·n|/|B| on boundary"); axes[-1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("scan_fewer.png", dpi=130)
print("Wrote scan_fewer_results.json and scan_fewer.png")
