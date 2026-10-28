"""
Simple Standalone Plotting Script for IEEE PerCom 2027.

Generates each subplot as a completely separate, individual single plot image.
No functions, no complex classes — simple, sequential, easy to edit directly.

Input CSV data files:
- logs/baselines/data_churn.csv
- logs/baselines/data_qoe.csv
- logs/baselines/data_qos_dwell.csv
- logs/baselines/data_weight_dwell.csv
- logs/baselines/data_weight_tradeoff.csv

Output plot files:
- plots/plot1_churn_noise_vs_shift.png & .pdf
- plots/plot2_average_video_qoe.png & .pdf
- plots/plot3_qos_vs_dwell_stability.png & .pdf
- plots/plot4_weight_server_dwell.png & .pdf
- plots/plot5_weight_latency_vs_bandwidth.png & .pdf
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS (Change font, style, output folder here)
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Apply IEEE styling (Okabe-Ito colors, clean grid, vector fonts)
try:
    plt.style.use("configs/plot_style.mplstyle")
except Exception:
    pass

# Distinct colors for the 8 policies
POLICY_COLORS = {
    "Nearest": "#888888",
    "Min-Latency": "#E69F00",
    "Max-Total-BW": "#56B4E9",
    "Max-Effective-BW": "#009E73",
    "AHP-TOPSIS": "#F0E442",
    "Static-Hysteresis": "#0072B2",
    "QoE-Greedy": "#D55E00",
    "Proposed-Margin": "#CC79A7",
}


# ===========================================================================
# PLOT 1: Churn under Noise vs Regime Shift (Separated Single Plot)
# ===========================================================================
df1 = pd.read_csv("logs/baselines/data_churn.csv")

plt.figure(figsize=(5.5, 3.8))
x = np.arange(len(df1))
width = 0.36

plt.bar(x - width/2, df1["noise_churn_switches_per_min"], width,
        label="Scenario A (Noise Churn)", color="#D55E00", edgecolor="black", linewidth=0.5)
plt.bar(x + width/2, df1["regime_shift_switches_per_min"], width,
        label="Scenario B (Regime Shift)", color="#0072B2", edgecolor="black", linewidth=0.5)

plt.title("Churn under Noise vs Regime Shift", fontsize=11, fontweight="bold")
plt.xlabel("Selection Policy", fontsize=9)
plt.ylabel("Switch Rate (switches / min)", fontsize=9)
plt.xticks(x, df1["policy"], rotation=40, ha="right", fontsize=8)
plt.legend(loc="upper left", fontsize=8)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "plot1_churn_noise_vs_shift.png", dpi=300)
plt.savefig(OUTPUT_DIR / "plot1_churn_noise_vs_shift.pdf")
plt.close()
print("Saved -> plot1_churn_noise_vs_shift.png / .pdf")


# ===========================================================================
# PLOT 2: Average Video QoE Across All Scenarios (Separated Single Plot)
# ===========================================================================
df2 = pd.read_csv("logs/baselines/data_qoe.csv")

plt.figure(figsize=(5.5, 3.8))
x = np.arange(len(df2))
colors2 = [POLICY_COLORS.get(p, "#333333") for p in df2["policy"]]

plt.bar(x, df2["mean_qoe"], yerr=df2["ci95_qoe"], capsize=3,
        color=colors2, edgecolor="black", linewidth=0.5, alpha=0.9)

plt.title("Average Video QoE (All Scenarios, 95% CI)", fontsize=11, fontweight="bold")
plt.xlabel("Selection Policy", fontsize=9)
plt.ylabel("Cumulative Video QoE (ITU-T P.1203)", fontsize=9)
plt.xticks(x, df2["policy"], rotation=40, ha="right", fontsize=8)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "plot2_average_video_qoe.png", dpi=300)
plt.savefig(OUTPUT_DIR / "plot2_average_video_qoe.pdf")
plt.close()
print("Saved -> plot2_average_video_qoe.png / .pdf")


# ===========================================================================
# PLOT 3: Connection Instability (Ping-Pong) vs Dwell Time (Separated Single Plot)
# ===========================================================================
df3 = pd.read_csv("logs/baselines/data_qos_dwell.csv")

plt.figure(figsize=(5.5, 3.8))

for _, row in df3.iterrows():
    p = row["policy"]
    plt.scatter(row["mean_dwell_time_s"], row["ping_pong_rate_pct"],
                color=POLICY_COLORS.get(p, "#333333"), s=90, edgecolors="black",
                linewidth=0.8, label=p, zorder=5)

plt.title("Ping-Pong Thrashing vs Stability (Dwell)", fontsize=11, fontweight="bold")
plt.xlabel("Mean Dwell Time (seconds)", fontsize=9)
plt.ylabel("Ping-Pong Switching Rate (%)", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right", fontsize=7.5, ncol=2)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "plot3_qos_vs_dwell_stability.png", dpi=300)
plt.savefig(OUTPUT_DIR / "plot3_qos_vs_dwell_stability.pdf")
plt.close()
print("Saved -> plot3_qos_vs_dwell_stability.png / .pdf")


# ===========================================================================
# PLOT 4: Server Dwell Allocation by Weight Profile (Separated Single Plot)
# ===========================================================================
df4 = pd.read_csv("logs/baselines/data_weight_dwell.csv")

plt.figure(figsize=(5.5, 3.8))
x = np.arange(len(df4))
width = 0.50

plt.bar(x, df4["s1_dwell_pct"], width,
        label="s1 (Low-Latency Edge)", color="#56B4E9", edgecolor="black", linewidth=0.5)
plt.bar(x, df4["s2_dwell_pct"], width, bottom=df4["s1_dwell_pct"],
        label="s2 (Balanced Edge)", color="#E69F00", edgecolor="black", linewidth=0.5)
bottom_s3 = df4["s1_dwell_pct"] + df4["s2_dwell_pct"]
plt.bar(x, df4["s3_dwell_pct"], width, bottom=bottom_s3,
        label="s3 (High-Capacity Cloud)", color="#009E73", edgecolor="black", linewidth=0.5)

plt.title("Server Preference by Weight Profile", fontsize=11, fontweight="bold")
plt.xlabel("Subjective Weight Profile", fontsize=9)
plt.ylabel("Server Dwell Allocation (%)", fontsize=9)
plt.xticks(x, df4["profile"], rotation=30, ha="right", fontsize=8)
plt.ylim(0, 105)
plt.legend(loc="upper right", fontsize=7.5)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "plot4_weight_server_dwell.png", dpi=300)
plt.savefig(OUTPUT_DIR / "plot4_weight_server_dwell.pdf")
plt.close()
print("Saved -> plot4_weight_server_dwell.png / .pdf")


# ===========================================================================
# PLOT 5: Latency vs Bandwidth Trade-off by Profile (Separated Single Plot)
# ===========================================================================
df5 = pd.read_csv("logs/baselines/data_weight_tradeoff.csv")

plt.figure(figsize=(5.5, 3.8))
prof_colors = ["#888888", "#E69F00", "#56B4E9", "#CC79A7", "#009E73"]

for i, row in df5.iterrows():
    prof = row["profile"]
    plt.scatter(row["mean_latency_ms"], row["mean_bandwidth_mbps"],
                s=110, color=prof_colors[i % len(prof_colors)], edgecolors="black",
                linewidth=0.8, label=prof, zorder=5)

plt.title("QoS Trade-off Across Weight Profiles", fontsize=11, fontweight="bold")
plt.xlabel("Average Observed Latency (ms)", fontsize=9)
plt.ylabel("Average Effective Bandwidth (Mbps)", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "plot5_weight_latency_vs_bandwidth.png", dpi=300)
plt.savefig(OUTPUT_DIR / "plot5_weight_latency_vs_bandwidth.pdf")
plt.close()
print("Saved -> plot5_weight_latency_vs_bandwidth.png / .pdf")

print("\nAll 5 individual single plots generated successfully.")
