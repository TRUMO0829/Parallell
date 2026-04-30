#!/usr/bin/env python3
"""
Generate report graphs from the radix-sort benchmark CSVs.

  Graph 3 — CUDA GPU time breakdown (stacked bar)
  Graph 5 — Best execution time per N (grouped bar)
  Graph A — GPU speedup over sequential (with-transfer vs compute-only)
  Graph B — Throughput scaling (MOPS vs N, CUDA vs sequential)
  Graph D — Crossover plot (all implementations, log-log)

Inputs  (project root):
  radix_sort_stats.csv          — sequential
  radix_sort_pthread_stats.csv  — pthread (5 thread counts)
  radix_sort_omp_stats.csv      — openmp  (5 thread counts)
  results_cuda.csv              — cuda

Outputs:
  stats/graph3_gpu_breakdown.png
  stats/graph5_best_per_n.png
  stats/graphA_gpu_speedup.png
  stats/graphB_throughput_scaling.png
  stats/graphD_crossover.png

Run:
  python3 plot_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "stats"


def load_data():
    seq = pd.read_csv(ROOT / "radix_sort_stats.csv")
    pth = pd.read_csv(ROOT / "radix_sort_pthread_stats.csv")
    omp = pd.read_csv(ROOT / "radix_sort_omp_stats.csv")
    gpu = pd.read_csv(ROOT / "results_cuda.csv")
    return seq, pth, omp, gpu


def fmt_n(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# ─────────────────────────────────────────────────────────────────────────────
# Graph 3 — CUDA time breakdown
# ─────────────────────────────────────────────────────────────────────────────
def graph3_gpu_breakdown(gpu, out_path):
    gpu = gpu.sort_values("N").reset_index(drop=True)
    n_labels = [fmt_n(n) for n in gpu["N"]]

    compute = gpu["computation_ms"].to_numpy(dtype=float)
    transfer = gpu["transfer_ms"].to_numpy(dtype=float)
    overhead = (gpu["execution_ms"].to_numpy(dtype=float)
                - compute - transfer).clip(min=0)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    x = np.arange(len(n_labels))

    ax.bar(x, compute, label="Compute (kernels)", color="#3b82f6")
    ax.bar(x, transfer, bottom=compute,
           label="Transfer (H2D + D2H)", color="#f97316")
    ax.bar(x, overhead, bottom=compute + transfer,
           label="Overhead", color="#94a3b8")

    totals = compute + transfer + overhead
    for i, t in enumerate(totals):
        ax.text(i, t * 1.02, f"{t:.2f} ms",
                ha="center", va="bottom", fontsize=9)

    for i in range(len(x)):
        if totals[i] > 0 and transfer[i] > 0:
            pct = 100 * transfer[i] / totals[i]
            ax.text(i, compute[i] + transfer[i] / 2,
                    f"{pct:.0f}%",
                    ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(n_labels)
    ax.set_xlabel("Problem size (N)")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title("CUDA GPU Time Breakdown — Compute vs PCIe Transfer")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    ymax = float(totals.max())
    ax.set_ylim(0, ymax * 1.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 5 — Best time per N (grouped bar, log-Y)
# ─────────────────────────────────────────────────────────────────────────────
def graph5_best_per_n(seq, pth, omp, gpu, out_path):
    sizes = sorted(seq["N"].unique())

    rows = []
    for n in sizes:
        s_row = seq[seq["N"] == n].iloc[0]
        p_best = pth[pth["N"] == n].sort_values("execution_ms").iloc[0]
        o_best = omp[omp["N"] == n].sort_values("execution_ms").iloc[0]
        g_row = gpu[gpu["N"] == n].iloc[0] if (gpu["N"] == n).any() else None

        rows.append({
            "N": n,
            "Sequential": float(s_row["execution_ms"]),
            "Pthread (best T)": float(p_best["execution_ms"]),
            "Pthread T": int(p_best["threads"]),
            "OpenMP (best T)": float(o_best["execution_ms"]),
            "OpenMP T": int(o_best["threads"]),
            "CUDA total": float(g_row["execution_ms"]) if g_row is not None else np.nan,
            "CUDA compute-only": float(g_row["computation_ms"]) if g_row is not None else np.nan,
        })
    df = pd.DataFrame(rows)

    impls = [
        ("Sequential",         "#64748b", None),
        ("Pthread (best T)",   "#10b981", "Pthread T"),
        ("OpenMP (best T)",    "#3b82f6", "OpenMP T"),
        ("CUDA total",         "#ef4444", None),
        ("CUDA compute-only",  "#f59e0b", None),
    ]

    n_labels = [fmt_n(n) for n in df["N"]]
    x = np.arange(len(n_labels))
    width = 0.16

    fig, ax = plt.subplots(figsize=(11, 6.2))

    for i, (impl, color, t_col) in enumerate(impls):
        offset = (i - 2) * width
        vals = df[impl].to_numpy(dtype=float)
        ax.bar(x + offset, vals, width, label=impl, color=color)

        for j, v in enumerate(vals):
            if not np.isnan(v):
                if t_col is not None:
                    label = f"{v:.2f}\nT={int(df[t_col].iloc[j])}"
                else:
                    label = f"{v:.2f}"
                ax.text(x[j] + offset, v * 1.08, label,
                        ha="center", va="bottom", fontsize=7)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels)
    ax.set_xlabel("Problem size (N)")
    ax.set_ylabel("Execution time (ms, log scale)")
    ax.set_title("Best Execution Time per N — Implementation Comparison")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph A — GPU speedup over sequential (with-transfer vs compute-only)
# ─────────────────────────────────────────────────────────────────────────────
def graphA_gpu_speedup(seq, gpu, out_path):
    sizes = sorted(set(seq["N"]) & set(gpu["N"]))

    seq_t = np.array([float(seq[seq["N"] == n]["execution_ms"].iloc[0]) for n in sizes])
    cuda_total = np.array([float(gpu[gpu["N"] == n]["execution_ms"].iloc[0]) for n in sizes])
    cuda_comp = np.array([float(gpu[gpu["N"] == n]["computation_ms"].iloc[0]) for n in sizes])

    sp_total = seq_t / cuda_total
    sp_comp = seq_t / cuda_comp

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(sizes, sp_comp, "o-", color="#f59e0b",
            linewidth=2.2, markersize=8, label="CUDA compute-only")
    ax.plot(sizes, sp_total, "s-", color="#ef4444",
            linewidth=2.2, markersize=8, label="CUDA total (incl. H2D + D2H)")

    ax.axhline(1.0, color="#64748b", linestyle="--",
               linewidth=1.2, label="Break-even with sequential CPU")

    for x, y in zip(sizes, sp_total):
        ax.annotate(f"{y:.2f}×", (x, y),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="#ef4444")
    for x, y in zip(sizes, sp_comp):
        ax.annotate(f"{y:.2f}×", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#b45309")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem size (N)")
    ax.set_ylabel("Speedup vs sequential CPU (×)")
    ax.set_title("CUDA Speedup over Sequential — PCIe Transfer Penalty Visible")
    ax.set_xticks(sizes)
    ax.set_xticklabels([fmt_n(n) for n in sizes])
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph B — Throughput scaling (MOPS vs N)
# ─────────────────────────────────────────────────────────────────────────────
def graphB_throughput_scaling(seq, gpu, out_path):
    sizes = sorted(set(seq["N"]) & set(gpu["N"]))

    seq_mops = [float(seq[seq["N"] == n]["performance_mops"].iloc[0]) for n in sizes]
    cuda_mops = [float(gpu[gpu["N"] == n]["performance_mops"].iloc[0]) for n in sizes]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(sizes, cuda_mops, "o-", color="#ef4444",
            linewidth=2.2, markersize=9, label="CUDA GPU")
    ax.plot(sizes, seq_mops, "s--", color="#64748b",
            linewidth=1.8, markersize=7, label="Sequential CPU (reference)")

    for x, y in zip(sizes, cuda_mops):
        ax.annotate(f"{y:.0f}", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#b91c1c")
    for x, y in zip(sizes, seq_mops):
        ax.annotate(f"{y:.0f}", (x, y),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="#475569")

    ax.set_xscale("log")
    ax.set_xlabel("Problem size (N)")
    ax.set_ylabel("Throughput (MOPS)")
    ax.set_title("GPU Throughput Scaling — Amortization Curve")
    ax.set_xticks(sizes)
    ax.set_xticklabels([fmt_n(n) for n in sizes])
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    growth = cuda_mops[-1] / cuda_mops[0]
    ax.text(0.98, 0.04,
            f"GPU throughput grows {growth:.0f}× from N=10K to N=10M\n"
            f"(same hardware, same code — purely amortization)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7",
                      edgecolor="#f59e0b", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph D — Crossover plot (log-log line chart)
# ─────────────────────────────────────────────────────────────────────────────
def graphD_crossover(seq, omp, gpu, out_path):
    sizes = sorted(set(seq["N"]) & set(gpu["N"]))

    seq_t = [float(seq[seq["N"] == n]["execution_ms"].iloc[0]) for n in sizes]
    omp_best = []
    omp_t = []
    for n in sizes:
        row = omp[omp["N"] == n].sort_values("execution_ms").iloc[0]
        omp_best.append(float(row["execution_ms"]))
        omp_t.append(int(row["threads"]))
    cuda_total = [float(gpu[gpu["N"] == n]["execution_ms"].iloc[0]) for n in sizes]
    cuda_comp = [float(gpu[gpu["N"] == n]["computation_ms"].iloc[0]) for n in sizes]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sizes, seq_t, "s-", color="#64748b", linewidth=2,
            markersize=8, label="Sequential CPU")
    ax.plot(sizes, omp_best, "D-", color="#3b82f6", linewidth=2,
            markersize=8, label="OpenMP (best T per N)")
    ax.plot(sizes, cuda_total, "o-", color="#ef4444", linewidth=2,
            markersize=8, label="CUDA total (incl. transfer)")
    ax.plot(sizes, cuda_comp, "^--", color="#f59e0b", linewidth=2,
            markersize=8, label="CUDA compute-only")

    for x, y, t in zip(sizes, omp_best, omp_t):
        ax.annotate(f"T={t}", (x, y),
                    textcoords="offset points", xytext=(8, -10),
                    fontsize=7, color="#1e40af")

    seq_arr = np.array(seq_t)
    cuda_arr = np.array(cuda_total)
    cross_idx = None
    for i in range(len(sizes) - 1):
        if seq_arr[i] < cuda_arr[i] and seq_arr[i + 1] > cuda_arr[i + 1]:
            cross_idx = i
            break
    if cross_idx is not None:
        # log-log linear interpolation for crossover N
        x1, x2 = np.log10(sizes[cross_idx]), np.log10(sizes[cross_idx + 1])
        d1 = seq_arr[cross_idx] - cuda_arr[cross_idx]
        d2 = seq_arr[cross_idx + 1] - cuda_arr[cross_idx + 1]
        frac = d1 / (d1 - d2)
        x_cross = 10 ** (x1 + frac * (x2 - x1))
        y_cross = 10 ** (np.log10(seq_arr[cross_idx]) +
                         frac * (np.log10(seq_arr[cross_idx + 1]) -
                                 np.log10(seq_arr[cross_idx])))
        ax.axvline(x_cross, color="#dc2626", linestyle=":", linewidth=1.2, alpha=0.6)
        ax.annotate(f"CUDA total beats\nsequential here\nN ≈ {x_cross/1000:.0f}K",
                    xy=(x_cross, y_cross),
                    xytext=(x_cross * 1.4, y_cross * 0.15),
                    fontsize=9, color="#dc2626",
                    arrowprops=dict(arrowstyle="->", color="#dc2626", alpha=0.7))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem size (N)")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title("CPU vs GPU Crossover — When Does the GPU Become Worthwhile?")
    ax.set_xticks(sizes)
    ax.set_xticklabels([fmt_n(n) for n in sizes])
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    seq, pth, omp, gpu = load_data()
    graph3_gpu_breakdown(gpu, OUT_DIR / "graph3_gpu_breakdown.png")
    graph5_best_per_n(seq, pth, omp, gpu, OUT_DIR / "graph5_best_per_n.png")
    graphA_gpu_speedup(seq, gpu, OUT_DIR / "graphA_gpu_speedup.png")
    graphB_throughput_scaling(seq, gpu, OUT_DIR / "graphB_throughput_scaling.png")
    graphD_crossover(seq, omp, gpu, OUT_DIR / "graphD_crossover.png")


if __name__ == "__main__":
    main()
