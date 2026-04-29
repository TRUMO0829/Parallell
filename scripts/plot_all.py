#!/usr/bin/env python3
"""
Plot scaling/performance graphs for the radix-sort implementations.

Inputs (CSV in repo root):
  - radix_sort_stats.csv          (sequential, 1 thread)
  - radix_sort_omp_stats.csv      (openmp, 1..16 threads)
  - radix_sort_pthread_stats.csv  (pthread, 1..16 threads)

Outputs (PNGs in stats/):
  1. exec_time_vs_N.png         exec time vs N, one line per impl (log-log)
  2. speedup_vs_threads.png     speedup vs threads, one line per N (OMP | pthread panels)
  3. efficiency_vs_threads.png  efficiency vs threads, one line per N (OMP | pthread panels)
  4. throughput_vs_threads.png  MOPS vs threads, one line per N (OMP | pthread panels)

Speedup baseline = each implementation's own 1-thread execution_ms.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
STATS_DIR = REPO / "stats"
STATS_DIR.mkdir(exist_ok=True)

SEQ_CSV = REPO / "radix_sort_stats.csv"
OMP_CSV = REPO / "radix_sort_omp_stats.csv"
PTH_CSV = REPO / "radix_sort_pthread_stats.csv"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seq = pd.read_csv(SEQ_CSV)
    omp = pd.read_csv(OMP_CSV)
    pth = pd.read_csv(PTH_CSV)
    return seq, omp, pth


def plot_exec_time_vs_N(seq: pd.DataFrame, omp: pd.DataFrame, pth: pd.DataFrame) -> None:
    """Exec time vs N (log-log). 1-thread point per impl — sanity check for O(N)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    series = [
        ("sequential", seq[seq["threads"] == 1], "o-"),
        ("openmp (1 thread)", omp[omp["threads"] == 1], "s-"),
        ("pthread (1 thread)", pth[pth["threads"] == 1], "^-"),
    ]
    for label, df, style in series:
        df = df.sort_values("N")
        ax.loglog(df["N"], df["execution_ms"], style, label=label, linewidth=1.8, markersize=7)

    # Reference O(N) guideline anchored at the smallest sequential point.
    ref = seq[seq["threads"] == 1].sort_values("N")
    if not ref.empty:
        n0, t0 = ref.iloc[0]["N"], ref.iloc[0]["execution_ms"]
        ns = ref["N"].values
        ax.loglog(ns, t0 * (ns / n0), "k--", alpha=0.4, linewidth=1, label="O(N) reference")

    ax.set_xlabel("N (input size)")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title("Execution time vs N — single-thread, all implementations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(STATS_DIR / "exec_time_vs_N.png", dpi=150)
    plt.close(fig)


def _compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'speedup' column using each (implementation, N)'s own 1-thread time."""
    base = (
        df[df["threads"] == 1]
        .set_index("N")["execution_ms"]
        .rename("t1")
    )
    out = df.join(base, on="N")
    out["speedup"] = out["t1"] / out["execution_ms"]
    out["efficiency"] = out["speedup"] / out["threads"]
    return out


def _panel_per_N(ax, df: pd.DataFrame, ycol: str, title: str, ylabel: str, *, ideal: bool) -> None:
    threads_axis = sorted(df["threads"].unique())
    for n in sorted(df["N"].unique()):
        sub = df[df["N"] == n].sort_values("threads")
        ax.plot(sub["threads"], sub[ycol], "o-", label=f"N={n:,}", linewidth=1.6, markersize=6)
    if ideal:
        ax.plot(threads_axis, threads_axis, "k--", alpha=0.5, linewidth=1, label="ideal (y=x)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads_axis)
    ax.set_xticklabels([str(t) for t in threads_axis])
    ax.set_xlabel("Threads")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def plot_speedup(omp: pd.DataFrame, pth: pd.DataFrame) -> None:
    omp_s = _compute_speedup(omp)
    pth_s = _compute_speedup(pth)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _panel_per_N(axes[0], omp_s, "speedup", "OpenMP — speedup vs threads", "Speedup (T1 / Tp)", ideal=False)
    _panel_per_N(axes[1], pth_s, "speedup", "pthread — speedup vs threads", "Speedup (T1 / Tp)", ideal=False)
    ymax = max(omp_s["speedup"].max(), pth_s["speedup"].max()) * 1.6
    for ax in axes:
        ax.set_ylim(0, ymax)
    fig.suptitle("Speedup vs thread count (baseline = each impl's own 1-thread time)")
    fig.tight_layout()
    fig.savefig(STATS_DIR / "speedup_vs_threads.png", dpi=150)
    plt.close(fig)


def plot_efficiency(omp: pd.DataFrame, pth: pd.DataFrame) -> None:
    omp_s = _compute_speedup(omp)
    pth_s = _compute_speedup(pth)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _panel_per_N(axes[0], omp_s, "efficiency", "OpenMP — parallel efficiency", "Efficiency (speedup / threads)", ideal=False)
    _panel_per_N(axes[1], pth_s, "efficiency", "pthread — parallel efficiency", "Efficiency (speedup / threads)", ideal=False)
    for ax in axes:
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.4, linewidth=1)
    fig.suptitle("Parallel efficiency vs thread count")
    fig.tight_layout()
    fig.savefig(STATS_DIR / "efficiency_vs_threads.png", dpi=150)
    plt.close(fig)


def plot_throughput(omp: pd.DataFrame, pth: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _panel_per_N(axes[0], omp, "performance_mops", "OpenMP — throughput", "Throughput (MOPS)", ideal=False)
    _panel_per_N(axes[1], pth, "performance_mops", "pthread — throughput", "Throughput (MOPS)", ideal=False)
    fig.suptitle("Throughput vs thread count")
    fig.tight_layout()
    fig.savefig(STATS_DIR / "throughput_vs_threads.png", dpi=150)
    plt.close(fig)


def main() -> None:
    seq, omp, pth = load()
    plot_exec_time_vs_N(seq, omp, pth)
    plot_speedup(omp, pth)
    plot_efficiency(omp, pth)
    plot_throughput(omp, pth)
    print(f"Wrote 4 plots to {STATS_DIR}")


if __name__ == "__main__":
    main()
