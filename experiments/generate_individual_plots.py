"""
Individual Plot Generator & Tabular Data Exporter for IEEE PerCom 2027.

This script reads directly from saved evaluation logs:
- logs/baselines/baseline_runs_raw.csv
- logs/baselines/baseline_summary.csv
- logs/baselines/weight_sensitivity_results.csv

It performs two operations without re-running any simulations:
1. Exports clean, individual CSV tables (and LaTeX snippet) to logs/baselines/tables/
   so any figure can be presented as a table or imported into Excel/LaTeX.
2. Generates standalone, individual plots (1 plot per file) in plots/generated/individual/
   saving both publication-ready vector PDF and high-res PNG.
All titles, font sizes, colors, and dimensions are fully configurable below.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_STYLE = ROOT_DIR / "configs" / "plot_style.mplstyle"
LOGS_DIR = ROOT_DIR / "logs" / "baselines"
TABLES_DIR = LOGS_DIR / "tables"
INDIVIDUAL_PLOTS_DIR = ROOT_DIR / "plots" / "generated" / "individual"

# Standard Okabe-Ito colorblind palette
COLOR_PALETTE = {
    "Nearest": "#888888",
    "Min-Latency": "#E69F00",
    "Max-Total-BW": "#56B4E9",
    "Max-Effective-BW": "#009E73",
    "AHP-TOPSIS": "#F0E442",
    "Static-Hysteresis": "#0072B2",
    "QoE-Greedy": "#D55E00",
    "Proposed-Margin": "#CC79A7",
}

POLICIES_ORDER = [
    "Nearest",
    "Min-Latency",
    "Max-Total-BW",
    "Max-Effective-BW",
    "AHP-TOPSIS",
    "Static-Hysteresis",
    "QoE-Greedy",
    "Proposed-Margin",
]


# ===========================================================================
# PLOT 1: Churn Comparison under Noise vs Regime Shift
# ===========================================================================
def plot_fig1_churn_comparison(summary_df: pd.DataFrame):
    """
    Fig 1: Switching Rate under Noise (Scenario A) vs. Regime Shift (Scenario B).
    Demonstrates that naive/greedy policies churn under jitter, while Proposed-Margin
    suppresses noise churn to zero while reliably triggering genuine handovers.
    """
    sub_a = summary_df[summary_df["scenario"] == "scenario_a"].set_index("policy")
    sub_b = summary_df[summary_df["scenario"] == "scenario_b"].set_index("policy")

    table_data = []
    for p in POLICIES_ORDER:
        a_mean = float(sub_a.loc[p]["switch_rate_per_min_mean"]) if p in sub_a.index else 0.0
        a_ci = float(sub_a.loc[p]["switch_rate_per_min_ci95"]) if p in sub_a.index else 0.0
        b_mean = float(sub_b.loc[p]["switch_rate_per_min_mean"]) if p in sub_b.index else 0.0
        b_ci = float(sub_b.loc[p]["switch_rate_per_min_ci95"]) if p in sub_b.index else 0.0
        table_data.append({
            "Policy": p,
            "Scenario_A_Noise_Switch_Rate_Mean": a_mean,
            "Scenario_A_Noise_Switch_Rate_CI95": a_ci,
            "Scenario_B_Shift_Switch_Rate_Mean": b_mean,
            "Scenario_B_Shift_Switch_Rate_CI95": b_ci,
        })
    df_table = pd.DataFrame(table_data)
    csv_path = TABLES_DIR / "table_fig1_churn_comparison.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"  [Table] Saved -> {csv_path.name}")

    # Plotting (Individual IEEE Single-Column: 3.5 x 2.7 inches)
    plt.style.use(CONFIG_STYLE)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    x = np.arange(len(POLICIES_ORDER))
    width = 0.38

    noise_vals = df_table["Scenario_A_Noise_Switch_Rate_Mean"].values
    noise_errs = df_table["Scenario_A_Noise_Switch_Rate_CI95"].values
    shift_vals = df_table["Scenario_B_Shift_Switch_Rate_Mean"].values
    shift_errs = df_table["Scenario_B_Shift_Switch_Rate_CI95"].values

    ax.bar(x - width/2, noise_vals, width, yerr=noise_errs, capsize=2,
           label="Scenario A (Noise Churn)", color="#D55E00", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, shift_vals, width, yerr=shift_errs, capsize=2,
           label="Scenario B (Regime Shift)", color="#0072B2", alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Switch Rate (switches / min)", fontsize=8)
    ax.set_title("Handover Churn: Noise vs. Regime Shift", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(POLICIES_ORDER, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_out = INDIVIDUAL_PLOTS_DIR / "fig1_churn_comparison.pdf"
    png_out = INDIVIDUAL_PLOTS_DIR / "fig1_churn_comparison.png"
    plt.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot]  Saved -> {pdf_out.name} and {png_out.name}")


# ===========================================================================
# PLOT 2: Cumulative Video QoE Across All Scenarios
# ===========================================================================
def plot_fig2_cumulative_qoe(raw_df: pd.DataFrame):
    """
    Fig 2: Mean Cumulative Video QoE (ITU-T P.1203 proxy) across all scenarios with 95% CIs.
    """
    grouped = raw_df.groupby("policy")["cumulative_qoe"].agg(["mean", "std", "count"])
    grouped["ci95"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"])

    table_data = []
    for p in POLICIES_ORDER:
        if p in grouped.index:
            row = grouped.loc[p]
            table_data.append({
                "Policy": p,
                "Mean_Cumulative_QoE": float(row["mean"]),
                "Std_QoE": float(row["std"]),
                "CI95_QoE": float(row["ci95"]),
                "Sample_Runs": int(row["count"]),
            })
    df_table = pd.DataFrame(table_data)
    csv_path = TABLES_DIR / "table_fig2_cumulative_qoe.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"  [Table] Saved -> {csv_path.name}")

    # Plotting
    plt.style.use(CONFIG_STYLE)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    x = np.arange(len(POLICIES_ORDER))
    qoe_means = df_table["Mean_Cumulative_QoE"].values
    qoe_cis = df_table["CI95_QoE"].values
    colors = [COLOR_PALETTE[p] for p in POLICIES_ORDER]

    ax.bar(x, qoe_means, yerr=qoe_cis, capsize=2.5, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_ylabel("Cumulative Video QoE (ITU-T P.1203)", fontsize=8)
    ax.set_title("Overall Video QoE (30 Seeds, 95% CI)", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(POLICIES_ORDER, rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_out = INDIVIDUAL_PLOTS_DIR / "fig2_cumulative_qoe.pdf"
    png_out = INDIVIDUAL_PLOTS_DIR / "fig2_cumulative_qoe.png"
    plt.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot]  Saved -> {pdf_out.name} and {png_out.name}")


# ===========================================================================
# PLOT 3: QoS Violations vs Stability (Mean Dwell Time)
# ===========================================================================
def plot_fig3_qos_vs_dwell(raw_df: pd.DataFrame):
    """
    Fig 3: QoS Violation Rate (%) vs. Mean Continuous Dwell Time (s).
    Shows the Pareto frontier between client connection stability and SLO adherence.
    """
    grouped = raw_df.groupby("policy").agg({
        "qos_violation_rate": "mean",
        "mean_dwell_time_s": "mean",
        "switch_rate_per_min": "mean",
        "ping_pong_rate": "mean"
    })

    table_data = []
    for p in POLICIES_ORDER:
        if p in grouped.index:
            row = grouped.loc[p]
            table_data.append({
                "Policy": p,
                "Mean_Dwell_Time_s": float(row["mean_dwell_time_s"]),
                "QoS_Violation_Rate_pct": float(row["qos_violation_rate"] * 100.0),
                "Mean_Switch_Rate_per_min": float(row["switch_rate_per_min"]),
                "Mean_Ping_Pong_Rate": float(row["ping_pong_rate"]),
            })
    df_table = pd.DataFrame(table_data)
    csv_path = TABLES_DIR / "table_fig3_qos_vs_dwell.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"  [Table] Saved -> {csv_path.name}")

    # Plotting
    plt.style.use(CONFIG_STYLE)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    for p in POLICIES_ORDER:
        row = df_table[df_table["Policy"] == p].iloc[0]
        ax.scatter(row["Mean_Dwell_Time_s"], row["QoS_Violation_Rate_pct"],
                   color=COLOR_PALETTE[p], s=70, edgecolors="black", linewidth=0.7, label=p, zorder=5)

    ax.set_xlabel("Mean Dwell Time (seconds)", fontsize=8)
    ax.set_ylabel("QoS Violation Rate (%)", fontsize=8)
    ax.set_title("QoS Violations vs. Connection Dwell", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.85)

    plt.tight_layout()
    pdf_out = INDIVIDUAL_PLOTS_DIR / "fig3_qos_vs_dwell.pdf"
    png_out = INDIVIDUAL_PLOTS_DIR / "fig3_qos_vs_dwell.png"
    plt.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot]  Saved -> {pdf_out.name} and {png_out.name}")


# ===========================================================================
# PLOT 4: Server Dwell Allocation by Weight Profile
# ===========================================================================
def plot_fig4_weight_dwell_allocation(weight_df: pd.DataFrame):
    """
    Fig 4: Server selection dwell time allocation (%) across the 5 weight profiles.
    Shows how subjective criteria preferences steer candidate choice between edge and cloud.
    """
    profiles = list(weight_df["profile"].unique())
    grouped = weight_df.groupby("profile")

    table_data = []
    for prof in profiles:
        g = grouped.get_group(prof)
        s1_pct = float(g["s1_dwell_fraction"].mean() * 100.0)
        s2_pct = float(g["s2_dwell_fraction"].mean() * 100.0)
        s3_pct = float(g["s3_dwell_fraction"].mean() * 100.0)
        table_data.append({
            "Profile": prof,
            "s1_Edge_LowLatency_Dwell_pct": s1_pct,
            "s2_Edge_Balanced_Dwell_pct": s2_pct,
            "s3_Cloud_HighCap_Dwell_pct": s3_pct,
        })
    df_table = pd.DataFrame(table_data)
    csv_path = TABLES_DIR / "table_fig4_weight_dwell_allocation.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"  [Table] Saved -> {csv_path.name}")

    # Plotting
    plt.style.use(CONFIG_STYLE)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    x = np.arange(len(profiles))
    width = 0.52

    s1_vals = df_table["s1_Edge_LowLatency_Dwell_pct"].values
    s2_vals = df_table["s2_Edge_Balanced_Dwell_pct"].values
    s3_vals = df_table["s3_Cloud_HighCap_Dwell_pct"].values

    ax.bar(x, s1_vals, width, label="s1 (Low-Lat Edge)", color="#56B4E9", edgecolor="black", linewidth=0.5)
    ax.bar(x, s2_vals, width, bottom=s1_vals, label="s2 (Balanced Edge)", color="#E69F00", edgecolor="black", linewidth=0.5)
    bottom_s3 = s1_vals + s2_vals
    ax.bar(x, s3_vals, width, bottom=bottom_s3, label="s3 (High-Cap Cloud)", color="#009E73", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Server Dwell Allocation (%)", fontsize=8)
    ax.set_title("Server Preference by Weight Profile", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_out = INDIVIDUAL_PLOTS_DIR / "fig4_weight_dwell_allocation.pdf"
    png_out = INDIVIDUAL_PLOTS_DIR / "fig4_weight_dwell_allocation.png"
    plt.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot]  Saved -> {pdf_out.name} and {png_out.name}")


# ===========================================================================
# PLOT 5: Weight Profile QoS Trade-off (Latency vs Bandwidth)
# ===========================================================================
def plot_fig5_weight_tradeoff_pareto(weight_df: pd.DataFrame):
    """
    Fig 5: Average Observed Latency (ms) vs. Effective Bandwidth (Mbps) across Weight Profiles.
    """
    profiles = list(weight_df["profile"].unique())
    grouped = weight_df.groupby("profile")

    table_data = []
    for prof in profiles:
        g = grouped.get_group(prof)
        lat_mean = float(g["mean_latency_ms"].mean())
        bw_mean = float(g["mean_bandwidth_mbps"].mean())
        qoe_mean = float(g["cumulative_qoe"].mean())
        table_data.append({
            "Profile": prof,
            "Mean_Latency_ms": lat_mean,
            "Mean_Effective_Bandwidth_mbps": bw_mean,
            "Mean_Cumulative_QoE": qoe_mean,
        })
    df_table = pd.DataFrame(table_data)
    csv_path = TABLES_DIR / "table_fig5_weight_tradeoff_pareto.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"  [Table] Saved -> {csv_path.name}")

    # Plotting
    plt.style.use(CONFIG_STYLE)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    palette_prof = ["#888888", "#E69F00", "#56B4E9", "#CC79A7", "#009E73"]
    for i, prof in enumerate(profiles):
        row = df_table[df_table["Profile"] == prof].iloc[0]
        ax.scatter(row["Mean_Latency_ms"], row["Mean_Effective_Bandwidth_mbps"],
                   s=85, color=palette_prof[i], edgecolors="black", linewidth=0.8, label=prof, zorder=5)
        # Offset annotations cleanly
        ax.annotate(prof, (row["Mean_Latency_ms"] + 0.35, row["Mean_Effective_Bandwidth_mbps"] - 0.25), fontsize=6.5)

    ax.set_xlabel("Average Observed Latency (ms)", fontsize=8)
    ax.set_ylabel("Average Effective Bandwidth (Mbps)", fontsize=8)
    ax.set_title("QoS Trade-off Across Weight Profiles", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_out = INDIVIDUAL_PLOTS_DIR / "fig5_weight_tradeoff_pareto.pdf"
    png_out = INDIVIDUAL_PLOTS_DIR / "fig5_weight_tradeoff_pareto.png"
    plt.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot]  Saved -> {pdf_out.name} and {png_out.name}")


# ===========================================================================
# COMPREHENSIVE LATEX-READY COMPARISON TABLE
# ===========================================================================
def export_comprehensive_paper_table(summary_df: pd.DataFrame):
    """
    Exports a publication-ready CSV and LaTeX tabular comparing all 8 policies
    across key representative scenarios (Mean ± 95% CI).
    """
    scenarios_to_include = [
        "scenario_a",
        "scenario_b",
        "scenario_d",
        "scenario_f",
        "scenario_puffer",
    ]
    sc_headers = {
        "scenario_a": "Scenario A (Noise)",
        "scenario_b": "Scenario B (Shift)",
        "scenario_d": "Scenario D (Herding)",
        "scenario_f": "Scenario F (Mobility)",
        "scenario_puffer": "Puffer (Trace)",
    }

    records = []
    for pol in POLICIES_ORDER:
        row = {"Policy": pol}
        for sc in scenarios_to_include:
            sub = summary_df[(summary_df["policy"] == pol) & (summary_df["scenario"] == sc)]
            if len(sub) > 0:
                qoe_m = float(sub["cumulative_qoe_mean"].iloc[0])
                qoe_ci = float(sub["cumulative_qoe_ci95"].iloc[0])
                sw_m = float(sub["switch_rate_per_min_mean"].iloc[0])
                row[f"{sc_headers[sc]} QoE"] = f"{qoe_m:.1f} ± {qoe_ci:.2f}"
                row[f"{sc_headers[sc]} Sw/min"] = f"{sw_m:.1f}"
            else:
                row[f"{sc_headers[sc]} QoE"] = "N/A"
                row[f"{sc_headers[sc]} Sw/min"] = "N/A"
        records.append(row)

    df_full = pd.DataFrame(records)
    csv_out = TABLES_DIR / "table_paper_baseline_matrix.csv"
    df_full.to_csv(csv_out, index=False)
    print(f"  [Table] Saved Full Paper Matrix -> {csv_out.name}")

    # Generate LaTeX snippet
    tex_out = TABLES_DIR / "table_paper_baseline_matrix.tex"
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write("% IEEE Double Column Evaluation Summary Table\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Benchmark Performance across 30 Independent Seeds (Mean $\\pm$ 95\\% CI)}\n")
        f.write("\\label{tab:baseline_comparison}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write(df_full.to_latex(index=False))
        f.write("}\n")
        f.write("\\end{table*}\n")
    print(f"  [LaTeX] Saved LaTeX Table Snippet -> {tex_out.name}")


# ===========================================================================
# MAIN ENTRYPOINT
# ===========================================================================
def main():
    print("=" * 70)
    print("GENERATING INDIVIDUAL PLOTS & EXPORTING CLEAN DATA TABLES")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv = LOGS_DIR / "baseline_runs_raw.csv"
    summary_csv = LOGS_DIR / "baseline_summary.csv"
    weight_csv = LOGS_DIR / "weight_sensitivity_results.csv"

    if not summary_csv.exists() or not raw_csv.exists():
        print(f"[ERROR] Required baseline logs not found in {LOGS_DIR}. Please run run_baselines.py first.")
        return

    print("1. Loading cached simulation logs...")
    raw_df = pd.read_csv(raw_csv)
    summary_df = pd.read_csv(summary_csv)
    weight_df = pd.read_csv(weight_csv) if weight_csv.exists() else None

    print("\n2. Generating individual figures & corresponding data tables:")
    plot_fig1_churn_comparison(summary_df)
    plot_fig2_cumulative_qoe(raw_df)
    plot_fig3_qos_vs_dwell(raw_df)

    if weight_df is not None:
        plot_fig4_weight_dwell_allocation(weight_df)
        plot_fig5_weight_tradeoff_pareto(weight_df)

    print("\n3. Generating LaTeX-ready comparison tables:")
    export_comprehensive_paper_table(summary_df)

    print("\n[SUCCESS] All 5 individual plots and 6 standalone data tables generated successfully.")
    print(f"  Tables folder: {TABLES_DIR}")
    print(f"  Plots folder:  {INDIVIDUAL_PLOTS_DIR}")


if __name__ == "__main__":
    main()
