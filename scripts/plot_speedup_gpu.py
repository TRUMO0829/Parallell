"""
plot_speedup_gpu.py
SpeedUp graph for the CUDA implementation vs sequential baseline.

Two SpeedUp variants are plotted side-by-side per N:
  - End-to-end : t_sequential / cuda.execution_ms
                 (includes H2D + kernel + D2H — the realistic cost of using GPU)
  - Kernel only: t_sequential / cuda.computation_ms
                 (pure GPU kernel time — useful when transfer is amortized
                 across many sorts on the same data)

Inputs (in current directory):
  results_sequential.csv
  results_cuda.csv

Output:
  speedup_gpu.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Load CSVs ----------------------------------------------------------
seq  = pd.read_csv("results_sequential.csv")
cuda = pd.read_csv("results_cuda.csv")

# ---- Join on N and compute the two SpeedUp variants --------------------
df = seq[["N", "execution_ms"]].rename(columns={"execution_ms": "t_seq"})
df = df.merge(
    cuda[["N", "execution_ms", "computation_ms"]].rename(
        columns={"execution_ms": "cuda_exec",
                 "computation_ms": "cuda_kernel"}),
    on="N",
)

df["speedup_endtoend"]    = df["t_seq"] / df["cuda_exec"]
df["speedup_kernel_only"] = df["t_seq"] / df["cuda_kernel"]

print("SpeedUp table (CUDA):")
print(df[["N", "speedup_endtoend", "speedup_kernel_only"]].to_string(index=False))

# ---- Grouped bar chart -------------------------------------------------
x = np.arange(len(df))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w / 2, df["speedup_endtoend"],    w,
            label="CUDA end-to-end (incl. H2D + D2H)")
b2 = ax.bar(x + w / 2, df["speedup_kernel_only"], w,
            label="CUDA kernel only")

ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
           label="Sequential baseline (1.0)")

ax.set_xticks(x)
ax.set_xticklabels([f"{int(n):,}" for n in df["N"]])
ax.set_xlabel("N (input size)")
ax.set_ylabel("SpeedUp = t_sequential / t_cuda")
ax.set_title("CUDA SpeedUp vs Sequential Baseline\n(end-to-end vs kernel-only)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

# Value labels on top of each bar
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}x",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("speedup_gpu.png", dpi=150)
print("Saved speedup_gpu.png")
