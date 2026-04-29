"""
plot_all.py — generate SpeedUp graphs from the result CSVs.

Inputs (read from <project root>):
    radix_sort_stats.csv             (1_sequential.cpp -> sequential baseline)
    radix_sort_pthread_stats.csv     (2_pthread.cpp)
                                     NOTE: 2_pthread.cpp writes to a hardcoded
                                     Colab path. If running locally, copy the
                                     CSV into <project root> first, or change
                                     the path in 2_pthread.cpp.
    radix_sort_omp_stats.csv         (3_openmp.cpp)
    results_cuda.csv                 (4_radix_gpu.cpp — old schema, unchanged)

Outputs (saved into <project root>/stats/):
    scaling_pthread.png   SpeedUp vs threads, line per N (pthread)
    scaling_openmp.png    SpeedUp vs threads, line per N (OpenMP)
    speedup_gpu.png       CUDA end-to-end vs kernel-only SpeedUp (bar)

Run from anywhere:
    python3 scripts/plot_all.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Project layout --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR      = PROJECT_ROOT
STATS_DIR    = PROJECT_ROOT / "stats"
STATS_DIR.mkdir(exist_ok=True)

HW_CONCURRENCY = os.cpu_count()


# ---- Loaders ---------------------------------------------------------------
# CPU CSVs (new schema): elements,[threads,]avg_time_ms,throughput_meps,correct
def load_cpu(name: str) -> Optional[pd.DataFrame]:
    path = CSV_DIR / name
    if not path.exists():
        print(f"Note: {path.name} not found — chart that needs it will be skipped.")
        return None
    df = pd.read_csv(path)
    df["elements"] = df["elements"].astype(np.int64)
    if "threads" in df.columns:
        df["threads"] = df["threads"].astype(int)
    return df


# CUDA CSV (old schema): N,execution_ms,computation_ms,transfer_ms,...
def load_cuda() -> Optional[pd.DataFrame]:
    path = CSV_DIR / "results_cuda.csv"
    if not path.exists():
        print(f"Note: {path.name} not found — CUDA chart will be skipped.")
        return None
    df = pd.read_csv(path)
    df["N"] = df["N"].astype(np.int64)
    return df


seq_df  = load_cpu("radix_sort_stats.csv")
thr_df  = load_cpu("radix_sort_pthread_stats.csv")
omp_df  = load_cpu("radix_sort_omp_stats.csv")
cuda_df = load_cuda()

assert seq_df is not None, \
    "radix_sort_stats.csv (sequential) is required — it provides the SpeedUp baseline"

# Sequential baseline: elements -> avg_time_ms (each row is mean of 5 runs)
seq_baseline    = seq_df.set_index("elements")["avg_time_ms"].sort_index()
seq_baseline_ns = set(seq_baseline.index.astype(int))


# ===========================================================================
# Strong-scaling SpeedUp plot (per CPU implementation)
# X = thread count (log_2)   Y = SpeedUp = t_seq / t_impl   line per N
# ===========================================================================
def plot_scaling(df: Optional[pd.DataFrame], png_name: str, title: str) -> None:
    if df is None:
        return
    if "threads" not in df.columns:
        print(f"Skip {png_name}: CSV has no `threads` column")
        return

    sizes   = sorted(set(df["elements"].unique()) & seq_baseline_ns)
    plot_df = df[df["elements"].isin(sizes)]
    threads = sorted(plot_df["threads"].unique())
    if not sizes or not threads:
        print(f"Skip {png_name}: no overlap with sequential baseline")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")

    max_speedup = 0.0
    for i, N in enumerate(sizes):
        sub = plot_df[plot_df["elements"] == N].sort_values("threads")
        speedup = float(seq_baseline.loc[N]) / sub["avg_time_ms"].values
        max_speedup = max(max_speedup, float(speedup.max()))
        line, = ax.plot(sub["threads"], speedup, marker="o",
                        color=cmap(i), label=f"N = {int(N):,}")
        for x, y in zip(sub["threads"], speedup):
            ax.annotate(f"{y:.2f}x", xy=(x, y), xytext=(7, 0),
                        textcoords="offset points", fontsize=8,
                        ha="left", va="center", color=line.get_color())

    # Sequential baseline reference at y = 1.0
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1, alpha=0.6)
    ax.text(threads[0], 1.0, "  Sequential baseline (1.0x)",
            color="black", fontsize=8, va="bottom", ha="left")

    # Hardware-concurrency vertical reference
    if HW_CONCURRENCY and threads[0] <= HW_CONCURRENCY <= threads[-1]:
        ax.axvline(HW_CONCURRENCY, linestyle=":", color="grey", alpha=0.7)
        ax.text(HW_CONCURRENCY, max_speedup * 1.12,
                f" hw cores = {HW_CONCURRENCY}",
                color="grey", fontsize=8, va="top", ha="left")

    ax.set_ylim(0, max(max_speedup * 1.20, 1.05))
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("Урсгалын тоо")
    ax.set_ylabel("SpeedUp = t_sequential / t_implementation")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = STATS_DIR / png_name
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# CUDA SpeedUp — end-to-end vs kernel-only (bar chart per N)
# ===========================================================================
def plot_speedup_gpu() -> None:
    if cuda_df is None:
        return

    # Bridge the schema mismatch: cuda uses N, sequential uses elements.
    baseline = pd.DataFrame({
        "N":     list(seq_baseline.index),
        "t_seq": list(seq_baseline.values),
    })
    df = cuda_df.merge(baseline, on="N", how="inner").sort_values("N")
    if df.empty:
        print("Skip speedup_gpu.png: no N overlap between cuda and sequential CSVs")
        return

    df["se"] = df["t_seq"] / df["execution_ms"]
    df["sk"] = df["t_seq"] / df["computation_ms"]

    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, df["se"], w, label="CUDA end-to-end (incl. H2D + D2H)")
    b2 = ax.bar(x + w/2, df["sk"], w, label="CUDA kernel only")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
               label="Sequential baseline (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n):,}" for n in df["N"]])
    ax.set_xlabel("N (input size)")
    ax.set_ylabel("SpeedUp = t_sequential / t_cuda")
    ax.set_title("CUDA SpeedUp vs Sequential Baseline\n(end-to-end vs kernel-only)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.2f}x",
                        xy=(r.get_x() + r.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = STATS_DIR / "speedup_gpu.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Console: SpeedUp pivot tables (drop straight into the report)
# ===========================================================================
def print_speedup_pivot(df: Optional[pd.DataFrame], label: str) -> None:
    if df is None or "threads" not in df.columns:
        return
    sub = df[df["elements"].isin(seq_baseline_ns)].copy()
    if sub.empty:
        return
    sub["speedup"] = sub["elements"].map(seq_baseline) / sub["avg_time_ms"]
    print(f"{label}:")
    print(sub.pivot(index="threads", columns="elements", values="speedup")
              .round(2).to_string())
    print()


def main() -> None:
    print_speedup_pivot(thr_df, "SpeedUp summary (pthread)")
    print_speedup_pivot(omp_df, "SpeedUp summary (OpenMP)")

    plot_scaling(thr_df, "scaling_pthread.png",
                 "pthread Strong Scaling (SpeedUp vs Урсгалын тоо)")
    plot_scaling(omp_df, "scaling_openmp.png",
                 "OpenMP Strong Scaling (SpeedUp vs Урсгалын тоо)")
    plot_speedup_gpu()


if __name__ == "__main__":
    main()
