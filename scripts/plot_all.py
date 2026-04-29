"""
plot_all.py — generate every chart used in the report from the 4 result CSVs.

Inputs (read from <project root>):
    results_sequential.csv
    results_stdthread.csv
    results_openmp.csv
    results_cuda.csv

Outputs (saved into <project root>/stats/):
    scaling_stdthread.png       SpeedUp vs num_threads, line per N (std::thread)
    scaling_openmp.png          SpeedUp vs num_threads, line per N (OpenMP)
    speedup_gpu.png             CUDA end-to-end vs kernel-only SpeedUp (bar)
    exec_time_comparison.png    Execution time vs N, all 4 implementations
    performance_mops.png        Achievable performance (MOPS), grouped by N
    cuda_breakdown.png          CUDA time breakdown (kernel / transfer / overhead)

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


# ---- Load CSVs (missing files are skipped, not fatal) ----------------------
def load_csv(name: str) -> Optional[pd.DataFrame]:
    path = CSV_DIR / name
    if not path.exists():
        print(f"Note: {path.name} not found — charts that need it will be skipped.")
        return None
    df = pd.read_csv(path)
    df["N"] = df["N"].astype(np.int64)
    return df


seq  = load_csv("results_sequential.csv")
thr  = load_csv("results_stdthread.csv")
omp  = load_csv("results_openmp.csv")
cuda = load_csv("results_cuda.csv")

assert seq is not None, "results_sequential.csv is required (provides SpeedUp baseline)"

seq_baseline      = seq.set_index("N")["execution_ms"].sort_index()
seq_baseline_ns   = set(seq_baseline.index.astype(int))


# ---- Helpers: best (min exec_ms / max MOPS) per N across thread counts ----
def best_exec_per_N(df: pd.DataFrame) -> pd.Series:
    if "num_threads" in df.columns:
        return df.groupby("N")["execution_ms"].min().sort_index()
    return df.set_index("N")["execution_ms"].sort_index()


def best_mops_per_N(df: pd.DataFrame) -> pd.Series:
    if "num_threads" in df.columns:
        return df.groupby("N")["performance_mops"].max().sort_index()
    return df.set_index("N")["performance_mops"].sort_index()


# ===========================================================================
# Chart 1 & 2: Strong-scaling SpeedUp (one PNG per CPU implementation)
# ===========================================================================
def plot_scaling(df: Optional[pd.DataFrame], png_name: str, title: str) -> None:
    if df is None:
        return

    sizes   = sorted(set(df["N"].unique()) & seq_baseline_ns)
    plot_df = df[df["N"].isin(sizes)]
    threads = sorted(plot_df["num_threads"].unique())
    if not sizes or not threads:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")

    max_speedup = 0.0
    for i, N in enumerate(sizes):
        sub = plot_df[plot_df["N"] == N].sort_values("num_threads")
        speedup = float(seq_baseline.loc[N]) / sub["execution_ms"].values
        max_speedup = max(max_speedup, float(speedup.max()))
        line, = ax.plot(sub["num_threads"], speedup, marker="o",
                        color=cmap(i), label=f"N = {int(N):,}")
        for x, y in zip(sub["num_threads"], speedup):
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
# Chart 3: CUDA SpeedUp — end-to-end vs kernel-only
# ===========================================================================
def plot_speedup_gpu() -> None:
    if cuda is None:
        return

    df = cuda.merge(seq[["N", "execution_ms"]].rename(
        columns={"execution_ms": "t_seq"}), on="N")
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
# Chart 4: Execution time vs N — all 4 implementations (line, log-log)
# ===========================================================================
def plot_exec_time_comparison() -> None:
    impls = [("Sequential", best_exec_per_N(seq))]
    if thr  is not None: impls.append(("std::thread (best T)", best_exec_per_N(thr)))
    if omp  is not None: impls.append(("OpenMP (best T)",      best_exec_per_N(omp)))
    if cuda is not None: impls.append(("CUDA (end-to-end)",    best_exec_per_N(cuda)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, (label, s) in enumerate(impls):
        ax.plot(s.index, s.values, marker="o", color=cmap(i), label=label)
        for x, y in zip(s.index, s.values):
            ax.annotate(f"{y:.2f} ms", xy=(x, y), xytext=(6, 4),
                        textcoords="offset points", fontsize=8,
                        color=cmap(i))

    ax.set_xscale("log")
    ax.set_yscale("log")
    sizes = sorted(seq["N"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{int(N):,}" for N in sizes])
    ax.set_xlabel("Input size N")
    ax.set_ylabel("Execution time (ms, log scale)")
    ax.set_title("Execution Time vs Input Size (all implementations)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = STATS_DIR / "exec_time_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Chart 5: Achievable Performance (MOPS) — grouped bar by N
# ===========================================================================
def plot_mops() -> None:
    impls = [("Sequential", best_mops_per_N(seq))]
    if thr  is not None: impls.append(("std::thread (best T)", best_mops_per_N(thr)))
    if omp  is not None: impls.append(("OpenMP (best T)",      best_mops_per_N(omp)))
    if cuda is not None: impls.append(("CUDA",                 best_mops_per_N(cuda)))

    sizes    = sorted(seq["N"].unique())
    n_impls  = len(impls)
    x        = np.arange(len(sizes))
    w        = 0.8 / n_impls
    cmap     = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (label, s) in enumerate(impls):
        offsets = x + w * (i - (n_impls - 1) / 2)
        values  = [float(s.get(N, 0.0)) for N in sizes]
        bars = ax.bar(offsets, values, w, color=cmap(i), label=label)
        for r, v in zip(bars, values):
            ax.annotate(f"{v:,.0f}",
                        xy=(r.get_x() + r.get_width()/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(N):,}" for N in sizes])
    ax.set_xlabel("Input size N")
    ax.set_ylabel("Achievable performance (MOPS)")
    ax.set_title("Achievable Performance — best MOPS per implementation per N")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = STATS_DIR / "performance_mops.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Chart 6: CUDA time breakdown — stacked bar (kernel / transfer / overhead)
# ===========================================================================
def plot_cuda_breakdown() -> None:
    if cuda is None:
        return

    df = cuda.sort_values("N").copy()
    kernel   = df["computation_ms"].values
    transfer = df["transfer_ms"].values
    overhead = (df["execution_ms"] - df["computation_ms"] - df["transfer_ms"]) \
                  .clip(lower=0).values

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, kernel,   color="tab:blue",   label="Kernel (computation)")
    ax.bar(x, transfer, bottom=kernel,
           color="tab:orange", label="Transfer (H2D + D2H)")
    ax.bar(x, overhead, bottom=kernel + transfer,
           color="tab:grey",   label="Overhead (malloc / free / sync)")

    totals = kernel + transfer + overhead
    for xi, total in zip(x, totals):
        ax.annotate(f"{total:.2f} ms", xy=(xi, total), xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n):,}" for n in df["N"]])
    ax.set_xlabel("N (input size)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("CUDA Execution Time Breakdown")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = STATS_DIR / "cuda_breakdown.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Console: SpeedUp pivot tables for the report
# ===========================================================================
def print_speedup_pivot(df: Optional[pd.DataFrame], label: str) -> None:
    if df is None or "num_threads" not in df.columns:
        return
    sub = df[df["N"].isin(seq_baseline.index)].copy()
    if sub.empty:
        return
    sub["speedup"] = sub["N"].map(seq_baseline) / sub["execution_ms"]
    print(f"{label}:")
    print(sub.pivot(index="num_threads", columns="N", values="speedup")
              .round(2).to_string())
    print()


def main() -> None:
    print_speedup_pivot(thr, "SpeedUp summary (std::thread)")
    print_speedup_pivot(omp, "SpeedUp summary (OpenMP)")

    plot_scaling(thr, "scaling_stdthread.png",
                 "std::thread Strong Scaling (SpeedUp vs Урсгалын тоо)")
    plot_scaling(omp, "scaling_openmp.png",
                 "OpenMP Strong Scaling (SpeedUp vs Урсгалын тоо)")
    plot_speedup_gpu()
    plot_exec_time_comparison()
    plot_mops()
    plot_cuda_breakdown()


if __name__ == "__main__":
    main()
