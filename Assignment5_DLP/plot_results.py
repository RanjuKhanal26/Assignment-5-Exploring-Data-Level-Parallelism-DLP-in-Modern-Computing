"""
Generates Figures 1, 3, 5, 6 from results/*.csv into figures/ and screenshots/.
Skips any figure whose CSV is missing.
Handles BOTH results_simd.csv layouts:
  legacy single-size : implementation,time_seconds
  multi-regime       : N,regime,path,time_seconds,instr_per_element
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures",     exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

def read_csv(path):
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))

def save(fig, name):
    for d in ("figures", "screenshots"):
        fig.savefig(f"{d}/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {name}")

# ---------- Figure 1: scalar loop vs NumPy vectorised ----------
rows = read_csv("results/results_vector.csv")
if rows:
    labels = [r["implementation"] for r in rows]
    times  = [float(r["time_seconds"]) for r in rows]
    sp = times[0] / times[1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, times, color=["#888", "#2b7a78"], edgecolor="k", width=0.55)
    for b, v in zip(bars, times):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                f"{v*1000:.2f} ms", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(times)*1.20)
    ax.set_ylabel("Execution time (s)")
    ax.set_title(f"Figure 1: Element-wise add — scalar vs NumPy  (speedup {sp:.0f}\u00d7)")
    ax.grid(axis="y", linestyle=":", alpha=0.6); ax.set_axisbelow(True)
    save(fig, "fig1_vector.png")

# ---------- Figure 3: SIMD dot product ----------
rows = read_csv("results/results_simd.csv")
if rows:
    first = rows[0]
    colors = {"scalar": "#888", "sse": "#2b7a78", "avx2": "#c0504d"}

    # ---- (A) multi-regime layout: N,regime,path,... ----
    if "regime" in first:
        regimes, paths = [], []
        for r in rows:
            if r["regime"] not in regimes: regimes.append(r["regime"])
            if r["path"]   not in paths:   paths.append(r["path"])
        data = {(r["regime"], r["path"]):
                  (float(r["time_seconds"]), float(r["instr_per_element"]))
                for r in rows}
        base = {reg: data[(reg, "scalar")][0] for reg in regimes}

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
        x = np.arange(len(regimes)); width = 0.8 / len(paths)

        for i, p in enumerate(paths):
            t = [data[(reg, p)][0]*1000 for reg in regimes]
            ax1.bar(x + i*width - (len(paths)-1)*width/2, t, width,
                    label=p.upper(), color=colors.get(p, "#666"),
                    edgecolor="k", linewidth=0.6)
        ax1.set_xticks(x); ax1.set_xticklabels(regimes)
        ax1.set_yscale("log"); ax1.set_ylabel("Time (ms)")
        ax1.set_title("(a) Execution time"); ax1.legend()
        ax1.grid(axis="y", linestyle=":", alpha=0.6); ax1.set_axisbelow(True)

        for i, p in enumerate(paths):
            if p == "scalar": continue
            sp = [base[reg] / data[(reg, p)][0] for reg in regimes]
            ax2.bar(x + i*width - (len(paths)-1)*width/2, sp, width,
                    label=p.upper(), color=colors.get(p, "#666"),
                    edgecolor="k", linewidth=0.6)
        for p, ideal in (("sse", 4.0), ("avx2", 8.0)):
            if p in paths:
                ax2.axhline(ideal, ls="--", color=colors[p], alpha=0.6,
                            label=f"{p.upper()} ideal ({ideal:.0f}\u00d7)")
        ax2.set_xticks(x); ax2.set_xticklabels(regimes)
        ax2.set_ylabel("Speedup vs scalar")
        ax2.set_title("(b) Speedup (narrows outside cache)")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", linestyle=":", alpha=0.6); ax2.set_axisbelow(True)

        ipe = [data[(regimes[0], p)][1] for p in paths]
        bars = ax3.bar([p.upper() for p in paths], ipe,
                       color=[colors.get(p, "#666") for p in paths],
                       edgecolor="k", linewidth=0.6)
        for b, v in zip(bars, ipe):
            ax3.text(b.get_x()+b.get_width()/2, b.get_height()*1.03,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax3.set_ylabel("Arithmetic instructions / element")
        ax3.set_title("(c) Instr. per element")
        ax3.grid(axis="y", linestyle=":", alpha=0.6); ax3.set_axisbelow(True)

        fig.suptitle("Figure 3: Dot product — scalar vs SIMD across cache regimes")
        fig.tight_layout()
        save(fig, "fig3_simd.png")

    # ---- (B) legacy layout: implementation,time_seconds ----
    else:
        labels = [r["implementation"] for r in rows]
        times  = [float(r["time_seconds"]) for r in rows]
        base   = times[0]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        bars = ax.bar(labels, times,
                      color=[colors.get(l, "#666") for l in labels],
                      edgecolor="k", width=0.55)
        for b, v in zip(bars, times):
            tag = "baseline" if v == base else f"{base/v:.2f}\u00d7"
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                    f"{v*1000:.2f} ms\n({tag})",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(times)*1.25)
        ax.set_ylabel("Execution time (s)")
        ax.set_title("Figure 3: Dot-product kernel — scalar vs available SIMD\n"
                     "(N = 10,000,000 float32; single DRAM-resident working set)")
        ax.grid(axis="y", linestyle=":", alpha=0.6); ax.set_axisbelow(True)
        fig.tight_layout()
        save(fig, "fig3_simd.png")

# ---------- Figure 5: naive vs tiled CUDA matmul ----------
rows = read_csv("results/results_matmul.csv")
if rows:
    Ns = sorted({int(r["N"]) for r in rows})
    naive = [next(float(r["time_ms"]) for r in rows
                  if int(r["N"])==N and r["kernel"]=="naive") for N in Ns]
    tiled = [next(float(r["time_ms"]) for r in rows
                  if int(r["N"])==N and r["kernel"]=="tiled") for N in Ns]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ns, naive, "o-", label="naive")
    ax.plot(Ns, tiled, "s-", label="tiled")
    ax.set_xlabel("Matrix size N"); ax.set_ylabel("Time (ms)")
    ax.set_title("Figure 5: GPU matrix multiply — naive vs tiled")
    ax.legend(); ax.grid(linestyle=":", alpha=0.6); ax.set_axisbelow(True)
    save(fig, "fig5_matmul.png")

# ---------- Figure 6: OpenMP scaling ----------
rows = read_csv("results/results_openmp.csv")
if rows:
    ths = [int(r["threads"]) for r in rows]
    ts  = [float(r["time_seconds"]) for r in rows]
    sp  = [ts[0] / t for t in ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(ths))
    bars = ax1.bar(x, ts, color="#2b7a78", edgecolor="k", linewidth=0.6, width=0.6)
    for b, t in zip(bars, ts):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                 f"{t:.3f} s", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels([f"{t}T" for t in ths])
    ax1.set_xlabel("Threads"); ax1.set_ylabel("Time (s)")
    ax1.set_title("(a) Execution time")
    ax1.grid(axis="y", linestyle=":", alpha=0.6); ax1.set_axisbelow(True)

    bars = ax2.bar(x, sp, color="#c0504d", edgecolor="k", linewidth=0.6,
                   width=0.6, label="Measured")
    ax2.plot(x, ths, "k--o", label="Ideal linear", markersize=8)
    for b, s in zip(bars, sp):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                 f"{s:.2f}\u00d7", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels([f"{t}T" for t in ths])
    ax2.set_xlabel("Threads"); ax2.set_ylabel("Speedup vs 1 thread")
    ax2.set_title("(b) Speedup vs ideal (gap = memory-bandwidth limit)")
    ax2.legend()
    ax2.grid(axis="y", linestyle=":", alpha=0.6); ax2.set_axisbelow(True)

    fig.suptitle("Figure 6: OpenMP scaling — memory-bound sum (N = 100M floats)")
    fig.tight_layout()
    save(fig, "fig6_openmp.png")

print("\nDone.")
