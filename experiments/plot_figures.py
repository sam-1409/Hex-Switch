from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    plt.style.use("configs/plot_style.mplstyle")
except Exception:
    pass

POLICY_COLORS = {"Nearest": "#888888", "Min-Latency": "#E69F00", "Max-Total-BW": "#56B4E9",
    "Max-Effective-BW": "#009E73", "AHP-TOPSIS": "#F0E442", "Static-Hysteresis": "#0072B2",
    "QoE-Greedy": "#D55E00", "Proposed-Margin": "#CC79A7"
}

# PLOT 2: Average Video QoE Across All Scenarios (Separated Single Plot)
df2 = pd.read_csv("logs/baselines/data_qoe.csv")
plt.figure(figsize=(5.5, 3.8))
x = np.arange(len(df2))
colors2 = [POLICY_COLORS.get(p, "#333333") for p in df2["policy"]]
plt.bar(x, df2["mean_qoe"], yerr=df2["ci95_qoe"], capsize=3,
        color=colors2, edgecolor="black", linewidth=0.5, alpha=0.9)
plt.xlabel("Selection Policy", fontsize=18, weight="semibold")
plt.ylabel("Cumulative QoE", fontsize=18, weight="semibold")
plt.xticks(x, df2["policy"], rotation=30, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot2_average_video_qoe.png", dpi=800)
plt.close()
print("Saved -> plot2_average_video_qoe.png")

# PLOT 3: Connection Instability (Ping-Pong) vs Dwell Time (Separated Single Plot)
df3 = pd.read_csv("logs/baselines/data_qos_dwell.csv")
plt.figure(figsize=(5.5, 3.8))
for _, row in df3.iterrows():
    p = row["policy"]
    plt.scatter(row["mean_dwell_time_s"], row["ping_pong_rate_pct"],
                color=POLICY_COLORS.get(p, "#333333"), s=90, edgecolors="black",
                linewidth=0.8, label=p, zorder=5)
plt.xlabel("Mean Dwell Time (seconds)", fontsize=16, weight="semibold")
plt.ylabel("Switching Rate (%)", fontsize=16, weight="semibold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="center", fontsize=12, ncol=2, frameon=True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot3_qos_vs_dwell_stability.png", dpi=800)
plt.close()
print("Saved -> plot3_qos_vs_dwell_stability.png")

# PLOT 6: Criteria Weights Comparison (AHP Profiles vs Objective Methods)
df6 = pd.read_csv("logs/xai/data_weights_comparison.csv")
plt.figure(figsize=(6.2, 4.0))
criteria = ["latency", "effective_bw", "server_load", "packet_loss", "availability"]
labels = ["Latency", "Effective BW", "Server Load", "Packet Loss", "Availability"]
colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
x = np.arange(len(df6))
width = 0.15
for i, crit in enumerate(criteria):
    offset = (i - 2) * width
    plt.bar(x + offset, df6[crit], width, label=labels[i], color=colors[i], edgecolor="black", linewidth=0.5)
plt.xlabel("Weighting Methodology", fontsize=12, weight="semibold")
plt.ylabel("Assigned Weight ($w_k$)", fontsize=12, weight="semibold")
plt.xticks(x, df6["method"], rotation=25, ha="right", fontsize=10)
plt.ylim(0, 0.85)
plt.yticks(fontsize=10)
plt.legend(loc="upper left", fontsize=10, ncol=2)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot6_criteria_weights_comparison.png", dpi=800)
plt.close()
print("Saved -> plot6_criteria_weights_comparison.png")

# PLOT 8: Counterfactual Decision Margin & Noise Boundary Trajectory (Scenario B)
df8 = pd.read_csv("logs/xai/data_counterfactual_margins.csv")
plt.figure(figsize=(6.5, 4.5))
plt.plot(df8["time_s"], df8["decision_margin"], label="Decision Margin $M(t)$", color="#0072B2", linewidth=1.8)
plt.plot(df8["time_s"], df8["switching_hurdle"], label=r"Switching Hurdle $\lambda_{\text{switch}}$", color="#D55E00", linestyle="--", linewidth=1.4)
plt.plot(df8["time_s"], df8["switching_hurdle"] + df8["noise_threshold"], label="Hurdle + Noise Band ($k\\hat{\\sigma}$)", color="#E69F00", linestyle=":", linewidth=1.4)
plt.fill_between(df8["time_s"], 0, df8["switching_hurdle"] + df8["noise_threshold"], color="#E69F00", alpha=0.12, label="Noise Suppression Zone")
switch_rows = df8[df8["action"] == "SWITCH"]
if not switch_rows.empty:
    sw_t = switch_rows["time_s"].iloc[0]
    sw_m = switch_rows["decision_margin"].iloc[0]
    plt.scatter([sw_t], [sw_m], color="#D55E00", s=100, zorder=6, label="Explainable SWITCH Event")
    plt.annotate("Regime Shift\nHandover", xy=(sw_t, sw_m), xytext=(sw_t+2, sw_m + 0.025),
                 arrowprops=dict(arrowstyle="->", color="black", lw=2), fontsize=14, fontweight="bold")
plt.xlabel("Time (s)", fontsize=18, weight="semibold")
plt.ylabel("Utility Margin/Hurdle", fontsize=18, weight="semibold")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="lower left", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot8_counterfactual_margins.png", dpi=800)
plt.close()
print("Saved -> plot8_counterfactual_margins.png")

# PLOT 9: Explanation Stability Under Stationary Telemetry Jitter (30 seeds)
df9 = pd.read_csv("logs/xai/data_explanation_stability.csv")
plt.figure(figsize=(5.5, 3.8))
methods = df9["method"].tolist()
methods = [m.replace(' ','\n') for m in methods]
x = np.arange(len(methods))
ci_col = "ci95_stability_pct" if "ci95_stability_pct" in df9.columns else None
errors = df9[ci_col].to_numpy() if ci_col else None
stab_col = "mean_explanation_stability_pct" if "mean_explanation_stability_pct" in df9.columns else "explanation_stability_pct"
plt.bar(x, df9[stab_col], yerr=errors, capsize=4,
        color=["#D55E00", "#009E73"], edgecolor="black", linewidth=0.5)
for i, row in df9.iterrows():
    plt.text(i, row[stab_col] + 1.0,
             f"{row[stab_col]:.1f}%", ha="center", fontsize=15, weight="bold")
plt.xlabel("Decision Framework", fontsize=18, weight="semibold")
plt.ylabel("Explanation Stability", fontsize=18, weight="semibold")
plt.xticks(x, methods, fontsize=14)
plt.yticks(fontsize=12)
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot9_explanation_stability_noise.png", dpi=800)
plt.close()
print("Saved -> plot9_explanation_stability_noise.png")

# PLOT 11: Explanation Faithfulness & Monotonic Decision Inversion
df11 = pd.read_csv("logs/auditor/data_counterfactual_faithfulness.csv")
plt.figure(figsize=(6.0, 3.8))
crit_colors = {
    "effective_bw": "#009E73",
    "latency": "#0072B2",
    "server_load": "#E69F00",
    "packet_loss": "#D55E00",
}
crit_labels = {
    "effective_bw": "Effective BW",
    "latency": "Latency",
    "server_load": "Server Load",
    "packet_loss": "Packet Loss",
}
for crit in df11["criterion"].unique():
    sub = df11[df11["criterion"] == crit]
    plt.plot(sub["perturbation_delta"], sub["resulting_margin"],
             marker="o", markersize=4, linewidth=1.6,
             color=crit_colors.get(crit, "black"), label=crit_labels.get(crit, crit))
plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, label="Decision Boundary")
plt.xlabel(r"Perturbation Delta ($\Delta u_k$)", fontsize=18, weight="semibold")
plt.ylabel("Net Decision Margin", fontsize=18, weight="semibold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="upper right", fontsize=12, frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot11_counterfactual_faithfulness.png", dpi=800)
plt.close()
print("Saved -> plot11_counterfactual_faithfulness.png")

# PLOT 14: Algorithmic Runtime Complexity Scaling (M=3 to 32 Servers)
df14 = pd.read_csv("logs/scalability/data_server_scaling_latency.csv")
plt.figure(figsize=(6.2, 4.0))
x = np.arange(len(df14))
width = 0.55
b1 = df14["matrix_build_us"]
b2 = df14["policy_select_us"]
b3 = df14["explainer_us"]
b4 = df14["auditor_us"]
plt.bar(x, b1, width, label="Matrix Build $O(MK)$", color="#56B4E9", edgecolor="black", linewidth=0.5)
plt.bar(x, b2, width, bottom=b1, label="Policy Selection $O(M)$", color="#0072B2", edgecolor="black", linewidth=0.5)
plt.bar(x, b3, width, bottom=b1+b2, label="Explainer $O(K)$", color="#E69F00", edgecolor="black", linewidth=0.5)
plt.bar(x, b4, width, bottom=b1+b2+b3, label="Auditor $O(1)$", color="#009E73", edgecolor="black", linewidth=0.5)
plt.xlabel("Candidate Server Count ($M$)", fontsize=18, weight="semibold")
plt.ylabel(r"Execution Time ($\mu$s)", fontsize=18, weight="semibold")
plt.xticks(x, df14["num_servers_M"], fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot14_server_scaling_complexity.png", dpi=800)
plt.close()
print("Saved -> plot14_server_scaling_complexity.png")

# PLOT 16: Statistical Significance & Effect Sizes (Cohen's d)
df16 = pd.read_csv("logs/ablations/data_statistical_tests.csv")
plt.figure(figsize=(6.5, 4.0))
effect_col = "paired_cohens_dz" if "paired_cohens_dz" in df16.columns else "cohens_d"
sig_col = "significant_holm_0_05" if "significant_holm_0_05" in df16.columns else "statistically_significant"
clean_df16 = df16.dropna(subset=[effect_col])
comparisons = [c.replace("Full-Proposed vs ", "") for c in clean_df16["comparison"]]
y_pos = np.arange(len(comparisons))
bar_colors = ["#009E73" if sig else "#888888" for sig in clean_df16[sig_col]]
plt.barh(y_pos, clean_df16[effect_col], color=bar_colors, edgecolor="black", linewidth=0.5)
plt.axvline(0.8, color="#D55E00", linestyle="--", linewidth=1.0, label="Large Effect ($d=0.8$)")
plt.axvline(0.0, color="black", linestyle="-", linewidth=0.8)
plt.xlabel("Paired Cohen's $d_z$", fontsize=12, weight="semibold")
plt.yticks(y_pos, comparisons, fontsize=10)
plt.legend(loc="lower right", fontsize=10)
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot16_statistical_significance_pvalues.png", dpi=800)
plt.close()
print("Saved -> plot16_statistical_significance_pvalues.png")
print("\nAll available individual plots generated; plot13 is produced after the 30-seed scaling run.")