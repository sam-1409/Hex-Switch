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

# ===========================================================================
# PLOT 1: Churn under Noise vs Regime Shift (Separated Single Plot)
# ===========================================================================
# df1 = pd.read_csv("logs/baselines/data_churn.csv")

# plt.figure(figsize=(5.5, 3.8))
# x = np.arange(len(df1))
# width = 0.36

# plt.bar(x - width/2, df1["noise_churn_switches_per_min"], width,
#         label="Scenario A (Noise Churn)", color="#D55E00", edgecolor="black", linewidth=0.5)
# plt.bar(x + width/2, df1["regime_shift_switches_per_min"], width,
#         label="Scenario B (Regime Shift)", color="#0072B2", edgecolor="black", linewidth=0.5)

# plt.title("Churn under Noise vs Regime Shift", fontsize=11, fontweight="bold")
# plt.xlabel("Selection Policy", fontsize=9)
# plt.ylabel("Switch Rate (switches / min)", fontsize=9)
# plt.xticks(x, df1["policy"], rotation=40, ha="right", fontsize=8)
# plt.legend(loc="upper left", fontsize=8)
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.tight_layout()

# plt.savefig(OUTPUT_DIR / "plot1_churn_noise_vs_shift.png", dpi=800)
# plt.savefig(OUTPUT_DIR / "plot1_churn_noise_vs_shift.pdf")
# plt.close()
# print("Saved -> plot1_churn_noise_vs_shift.png / .pdf")

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

# PLOT 4: Server Dwell Allocation by Weight Profile (Separated Single Plot)
# df4 = pd.read_csv("logs/baselines/data_weight_dwell.csv")

# plt.figure(figsize=(5.5, 3.8))
# x = np.arange(len(df4))
# width = 0.50

# plt.bar(x, df4["s1_dwell_pct"], width,
#         label="s1 (Low-Latency Edge)", color="#56B4E9", edgecolor="black", linewidth=0.5)
# plt.bar(x, df4["s2_dwell_pct"], width, bottom=df4["s1_dwell_pct"],
#         label="s2 (Balanced Edge)", color="#E69F00", edgecolor="black", linewidth=0.5)
# bottom_s3 = df4["s1_dwell_pct"] + df4["s2_dwell_pct"]
# plt.bar(x, df4["s3_dwell_pct"], width, bottom=bottom_s3,
#         label="s3 (High-Capacity Cloud)", color="#009E73", edgecolor="black", linewidth=0.5)

# plt.title("Server Preference by Weight Profile", fontsize=11, fontweight="bold")
# plt.xlabel("Subjective Weight Profile", fontsize=9)
# plt.ylabel("Server Dwell Allocation (%)", fontsize=9)
# plt.xticks(x, df4["profile"], rotation=30, ha="right", fontsize=8)
# plt.ylim(0, 105)
# plt.legend(loc="upper right", fontsize=7.5)
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.tight_layout()

# plt.savefig(OUTPUT_DIR / "plot4_weight_server_dwell.png", dpi=800)
# plt.savefig(OUTPUT_DIR / "plot4_weight_server_dwell.pdf")
# plt.close()
# print("Saved -> plot4_weight_server_dwell.png / .pdf")

# ===========================================================================
# PLOT 5: Latency vs Bandwidth Trade-off by Profile (Separated Single Plot)
# ===========================================================================
# df5 = pd.read_csv("logs/baselines/data_weight_tradeoff.csv")

# plt.figure(figsize=(5.5, 3.8))
# prof_colors = ["#888888", "#E69F00", "#56B4E9", "#CC79A7", "#009E73"]

# for i, row in df5.iterrows():
#     prof = row["profile"]
#     plt.scatter(row["mean_latency_ms"], row["mean_bandwidth_mbps"],
#                 s=110, color=prof_colors[i % len(prof_colors)], edgecolors="black",
#                 linewidth=0.8, label=prof, zorder=5)

# plt.title("QoS Trade-off Across Weight Profiles", fontsize=11, fontweight="bold")
# plt.xlabel("Average Observed Latency (ms)", fontsize=9)
# plt.ylabel("Average Effective Bandwidth (Mbps)", fontsize=9)
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend(loc="upper right", fontsize=8)
# plt.tight_layout()

# plt.savefig(OUTPUT_DIR / "plot5_weight_latency_vs_bandwidth.png", dpi=800)
# plt.savefig(OUTPUT_DIR / "plot5_weight_latency_vs_bandwidth.pdf")
# plt.close()
# print("Saved -> plot5_weight_latency_vs_bandwidth.png / .pdf")

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

# PLOT 7: Exact Additive Feature Attributions Across Scenarios
# df7 = pd.read_csv("logs/xai/data_feature_attributions.csv")
# plt.figure(figsize=(6.5, 4.2))
# scenarios = df7["scenario"].tolist()
# y_pos = np.arange(len(scenarios))
# bar_height = 0.15

# crit_cols = ["phi_latency", "phi_effective_bw", "phi_server_load", "phi_packet_loss", "phi_availability"]
# crit_names = ["Latency", "Effective BW", "Server Load", "Packet Loss", "Availability"]
# crit_colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]

# for i, col in enumerate(crit_cols):
#     offset = (i - 2) * bar_height
#     plt.barh(y_pos + offset, df7[col], bar_height, label=crit_names[i], color=crit_colors[i], edgecolor="black", linewidth=0.5)

# plt.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
# plt.title(r"Exact Additive Feature Attributions ($\phi_k$) Across Canonical Scenarios", fontsize=10.5, fontweight="bold")
# plt.xlabel(r"Shapley Attribution Value $\phi_k = w_k \cdot \Delta u_k$", fontsize=9)
# plt.yticks(y_pos, scenarios, fontsize=8)
# plt.legend(loc="lower right", fontsize=7.5, ncol=2)
# plt.grid(axis="x", linestyle="--", alpha=0.5)
# plt.tight_layout()

# plt.savefig(OUTPUT_DIR / "plot7_feature_attributions_scenarios.png", dpi=800)
# plt.savefig(OUTPUT_DIR / "plot7_feature_attributions_scenarios.pdf")
# plt.close()
# print("Saved -> plot7_feature_attributions_scenarios.png / .pdf")

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

# PLOT 10: Auditor Fault Detection & Confusion Metrics
# df10 = pd.read_csv("logs/auditor/data_fault_detection_confusion.csv")
# plt.figure(figsize=(6.5, 4.0))
# fault_labels = [
#     "Stale\nTelemetry",
#     "Infeasible\nAssignment",
#     "Spurious\nHandover",
#     "Buffer\nStarvation",
#     "Weight\nCorrupt"
# ]
# x = np.arange(len(df10))
# width = 0.26

# plt.bar(x - width, df10["recall"] * 100, width, label="Recall (Sensitivity %)", color="#0072B2", edgecolor="black", linewidth=0.5)
# plt.bar(x, df10["precision"] * 100, width, label="Precision (%)", color="#009E73", edgecolor="black", linewidth=0.5)
# plt.bar(x + width, df10["specificity"] * 100, width, label="Specificity (1 - FPR %)", color="#E69F00", edgecolor="black", linewidth=0.5)

# plt.title("Post-Decision Auditor Fault Detection Performance", fontsize=11, fontweight="bold")
# plt.xlabel("Injected Safety Fault Class", fontsize=9)
# plt.ylabel("Classification Metric (%)", fontsize=9)
# plt.xticks(x, fault_labels, fontsize=8)
# plt.ylim(0, 115)
# plt.legend(loc="upper right", fontsize=8)
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.tight_layout()

# plt.savefig(OUTPUT_DIR / "plot10_auditor_fault_confusion.png", dpi=800)
# plt.savefig(OUTPUT_DIR / "plot10_auditor_fault_confusion.pdf")
# plt.close()
# print("Saved -> plot10_auditor_fault_confusion.png / .pdf")

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

# PLOT 12: Auditor Sub-Millisecond Runtime Execution Overhead
# df12 = pd.read_csv("logs/auditor/data_auditor_latency_overhead.csv")

# plt.figure(figsize=(5.8, 3.8))
# latencies = df12["latency_us"]

# plt.hist(latencies, bins=35, color="#56B4E9", edgecolor="black", linewidth=0.5, density=True, alpha=0.7, label="Empirical PDF")
# mean_lat = float(np.mean(latencies))
# p99_lat = float(np.percentile(latencies, 99))

# plt.axvline(mean_lat, color="#D55E00", linestyle="-", linewidth=1.5, label=f"Mean: {mean_lat:.1f} $\\mu$s")
# plt.axvline(p99_lat, color="#CC79A7", linestyle="--", linewidth=1.5, label=f"99th Pct: {p99_lat:.1f} $\\mu$s")

# plt.title("Post-Decision Auditor Runtime Latency Distribution", fontsize=11, fontweight="bold")
# plt.xlabel("Execution Time ($\\mu$s)", fontsize=9)
# plt.ylabel("Probability Density", fontsize=9)
# plt.xlim(0, max(25.0, p99_lat * 1.5))
# plt.legend(loc="upper right", fontsize=8)
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.tight_layout()

# plt.savefig(OUTPUT_DIR / "plot12_auditor_runtime_overhead.png", dpi=800)
# plt.savefig(OUTPUT_DIR / "plot12_auditor_runtime_overhead.pdf")
# plt.close()
# print("Saved -> plot12_auditor_runtime_overhead.png / .pdf")

# ===========================================================================
# PLOT 13: Multi-User Herding Rate vs Client Density (N=1 to 100)
# scale_seed_path = Path("logs/scalability/data_client_scaling_herding_seed_level.csv")
# if scale_seed_path.exists():
#     scale_seed = pd.read_csv(scale_seed_path)
#     plt.figure(figsize=(6.2, 4.0))
#     pol_colors = {"Proposed-Margin": "#0072B2", "AHP-TOPSIS": "#D55E00",
#                   "Min-Latency": "#009E73", "Max-Total-BW": "#E69F00"}
#     from scipy import stats as scipy_stats
#     for pol, sub in scale_seed.groupby("policy"):
#         summary = sub.groupby("num_clients").ping_pong_rate_pct.agg(["mean", "std", "count"]).reset_index()
#         half = [scipy_stats.t.ppf(0.975, int(n)-1) * sd / np.sqrt(n) if n > 1 else 0.0
#                 for sd, n in zip(summary["std"], summary["count"])]
#         plt.errorbar(summary.num_clients, summary["mean"], yerr=half, marker="o", capsize=2,
#                      linewidth=1.4, color=pol_colors.get(pol, "black"), label=pol)
#     plt.title("Multi-User Herding Under Contention (30 seeds; 95% t CI)", fontsize=10.5, fontweight="bold")
#     plt.xlabel("Connected Client Density per Cell ($N$)", fontsize=9)
#     plt.ylabel("Ping-Pong Switching Rate (%)", fontsize=9)
#     plt.ylim(-2, 105)
#     plt.legend(loc="center right", fontsize=8)
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(OUTPUT_DIR / "plot13_client_scaling_herding.png", dpi=800)
#     plt.savefig(OUTPUT_DIR / "plot13_client_scaling_herding.pdf")
#     plt.close()
#     print("Saved -> plot13_client_scaling_herding.png / .pdf")
# else:
#     print("SKIPPED plot13: run experiments/run_scalability_benchmark.py to create 30-seed data first.")

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

# PLOT 15: Component Ablation Study (QoE vs Switching Churn)
# df15 = pd.read_csv("logs/ablations/data_ablation_study.csv")
# plt.figure(figsize=(6.5, 4.2))
# colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#888888"]

# for i, row in df15.iterrows():
#     cfg = row["configuration"]
#     c = colors[i % len(colors)]
#     plt.errorbar(row["mean_switch_rate_per_min"], row["mean_qoe_proxy"],
#                  yerr=row.get("ci95_qoe_proxy", 0.0), fmt="o", markersize=7,
#                  color=c, ecolor="black", capsize=3, markeredgecolor="black",
#                  linewidth=0.8, label=cfg, zorder=5)

# plt.title("Component Ablation: Video QoE vs Switching Churn", fontsize=11, fontweight="bold")
# plt.xlabel("Switching Churn Rate (switches / min)", fontsize=9)
# plt.ylabel("Mean cumulative linear QoE proxy (score units)", fontsize=9)
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend(loc="lower left", fontsize=7.5)
# plt.tight_layout()

# plt.savefig(OUTPUT_DIR / "plot15_ablation_qoe_vs_churn.png", dpi=800)
# plt.savefig(OUTPUT_DIR / "plot15_ablation_qoe_vs_churn.pdf")
# plt.close()
# print("Saved -> plot15_ablation_qoe_vs_churn.png / .pdf")

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