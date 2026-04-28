"""
plot_scaling.py
Strong-scaling SpeedUp study for the CPU parallel implementations.

For each parallel implementation (std::thread, OpenMP), produces a plot:
  X = num_threads (log scale base 2; whatever appears in the CSV)
  Y = SpeedUp = t_sequential[N] / t_implementation[N, T]
  One line per problem size N that has a matching row in the sequential CSV.

Works with any N / thread grid you benchmark — change sizes in the C++ driver,
re-run benchmarks, update CSVs; plots follow the data automatically.

Rules:
  - Sequential baseline times must exist in the sequential CSV for every N you
    want plotted on the parallel side (same integer N after normalizing).

Inputs / outputs are listed in SCALING_JOBS below (paths relative to CWD).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Tune these if you like ------------------------------------------------
SEQUENTIAL_CSV = "results_sequential.csv"
# Add/remove jobs when you have more CSVs (e.g. pthread vs std::thread).
SCALING_JOBS = [
    {
        "csv": "results_stdthread.csv",
        "png": "scaling_stdthread.png",
        "title": "std::thread Strong Scaling (SpeedUp vs Thread Count)",
        "summary_label": "std::thread",
    },
    {
        "csv": "results_openmp.csv",
        "png": "scaling_openmp.png",
        "title": "OpenMP Strong Scaling (SpeedUp vs Thread Count)",
        "summary_label": "OpenMP",
    },
]
# If total (N × thread) points exceeds this, point labels are skipped (legend only).
MAX_ANNOTATED_POINTS = 48

HW_CONCURRENCY = os.cpu_count()


def load_sequential_baseline(path: str) -> pd.Series:
    """Unique N → execution_ms; N coerced to int for reliable CSV matching."""
    seq = pd.read_csv(path)
    if seq.empty:
        raise FileNotFoundError(f"No rows in {path}")
    if seq["N"].duplicated().any():
        dups = seq["N"][seq["N"].duplicated()].unique().tolist()
        print(
            f"Note ({path}): duplicate N {dups}; using last row per N.",
            flush=True,
        )
        seq = seq.drop_duplicates(subset=["N"], keep="last")
    seq = seq.assign(N=seq["N"].astype(np.int64))
    return seq.set_index("N")["execution_ms"].sort_index()


def normalize_parallel_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["N"] = out["N"].astype(np.int64)
    return out


seq_baseline = load_sequential_baseline(SEQUENTIAL_CSV)
_BASELINE_NS: set[int] = set(seq_baseline.index.astype(int))


def warn_missing_baseline(csv_path: str, df: pd.DataFrame) -> None:
    need = {int(x) for x in df["N"].unique()}
    missing = sorted(need - _BASELINE_NS)
    if missing:
        print(
            f"Note ({csv_path}): no sequential baseline for N = {missing}; "
            "those rows are omitted from speedup plots/summary. "
            f"Add matching rows to {SEQUENTIAL_CSV} or drop them from the "
            "parallel CSV.",
            flush=True,
        )


def _palette(n_curves: int) -> np.ndarray:
    if n_curves <= 10:
        cmap = plt.get_cmap("tab10")
        return cmap(np.linspace(0, 1, n_curves, endpoint=False))
    if n_curves <= 20:
        cmap = plt.get_cmap("tab20")
        return cmap(np.linspace(0, 1, n_curves, endpoint=False))
    cmap = plt.get_cmap("viridis")
    return cmap(np.linspace(0, 1, n_curves, endpoint=False))


@dataclass(frozen=True)
class ScalingJob:
    csv_path: str
    png_path: str
    title: str


def plot_scaling(job: ScalingJob) -> None:
    df = normalize_parallel_df(pd.read_csv(job.csv_path))
    warn_missing_baseline(job.csv_path, df)

    sizes = sorted(set(df["N"].unique()) & _BASELINE_NS)
    plot_df = df[df["N"].isin(sizes)]
    threads = sorted(plot_df["num_threads"].unique())

    if not sizes:
        print(f"Skipping plot {job.png_path}: no N overlap with sequential baseline.")
        return
    if not threads:
        print(f"Skipping plot {job.png_path}: no thread counts in data.")
        return

    total_pts = sum(len(plot_df[plot_df["N"] == N]) for N in sizes)
    annotate_points = total_pts <= MAX_ANNOTATED_POINTS

    palette = _palette(len(sizes))
    fig_w = min(14, 8 + 0.35 * len(sizes))
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))

    max_speedup = 0.0
    for i, N in enumerate(sizes):
        color = palette[i]
        sub = plot_df[plot_df["N"] == N].sort_values("num_threads")
        t_seq = float(seq_baseline.loc[N])
        speedup = t_seq / sub["execution_ms"].values
        max_speedup = max(max_speedup, float(np.max(speedup)))
        (line,) = ax.plot(
            sub["num_threads"],
            speedup,
            marker="o",
            color=color,
            label=f"N = {int(N):,}",
        )
        if annotate_points:
            for x, y in zip(sub["num_threads"], speedup):
                ax.annotate(
                    f"{y:.2f}x",
                    xy=(x, y),
                    xytext=(7, 0),
                    textcoords="offset points",
                    fontsize=8,
                    ha="left",
                    va="center",
                    color=line.get_color(),
                )

    ax.axhline(1.0, linestyle="--", color="black", linewidth=1, alpha=0.6)
    ax.text(
        threads[0],
        1.0,
        "  Sequential baseline (1.0x)",
        color="black",
        fontsize=8,
        va="bottom",
        ha="left",
    )

    if HW_CONCURRENCY and threads[0] <= HW_CONCURRENCY <= threads[-1]:
        ax.axvline(HW_CONCURRENCY, linestyle=":", color="grey", alpha=0.7)
        ax.text(
            HW_CONCURRENCY,
            max_speedup * 1.12,
            f" hw cores = {HW_CONCURRENCY}",
            color="grey",
            fontsize=8,
            va="top",
            ha="left",
        )

    y_top = max(max_speedup * 1.20, 1.05)
    ax.set_ylim(0, y_top)
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("Thread count")
    ax.set_ylabel("SpeedUp = t_sequential / t_implementation")
    ax.set_title(job.title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(job.png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {job.png_path}")


def print_speedup_table(csv_path: str, label: str) -> None:
    df = normalize_parallel_df(pd.read_csv(csv_path))
    warn_missing_baseline(csv_path, df)
    sub = df[df["N"].isin(seq_baseline.index)].copy()
    if sub.empty:
        print(f"{label}: (no rows with matching sequential N)\n")
        return
    sub["speedup"] = sub["N"].map(seq_baseline) / sub["execution_ms"]
    print(f"{label}:")
    print(
        sub.pivot(index="num_threads", columns="N", values="speedup")
        .round(2)
        .to_string()
    )
    print()


def main() -> None:
    for spec in SCALING_JOBS:
        lbl = spec.get("summary_label", spec["csv"])
        print_speedup_table(spec["csv"], f"SpeedUp summary ({lbl})")
    print()
    for spec in SCALING_JOBS:
        plot_scaling(
            ScalingJob(
                csv_path=spec["csv"],
                png_path=spec["png"],
                title=spec["title"],
            )
        )


if __name__ == "__main__":
    main()
