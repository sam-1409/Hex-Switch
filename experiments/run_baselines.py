"""
Comprehensive Baseline Evaluation & Weight Sensitivity Benchmark.
Target Venue: IEEE PerCom 2027 (A4E Workshop)

Runs:
1. Full 8-policy baseline comparison across 8 scenarios with >= 30 seeds.
2. Formal metrics calculation: QoE, QoS Violation Rate, Switch Rate, Ping-Pong Rate, Dwell Time.
3. Weight sensitivity sweep evaluating subjective preference trade-offs.
4. Publication-ready vector PDF plots following configs/plot_style.mplstyle.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from edge_switching.qos_model import Criterion
from edge_switching.simulator import VideoClientState
from edge_switching.scenarios import run_scenario
from edge_switching.metrics import compute_scenario_metrics
from edge_switching.policies import (
    NearestServerPolicy,
    MinLatencyPolicy,
    MaxTotalBandwidthPolicy,
    MaxEffectiveBandwidthPolicy,
    AhpTopsisPolicy,
    HysteresisOnlyPolicy,
    QoEGreedyPolicy,
    ProposedMarginPolicy,
)

CONFIG_STYLE = ROOT_DIR / "configs" / "plot_style.mplstyle"
LOGS_DIR = ROOT_DIR / "logs" / "baselines"
PLOTS_DIR = ROOT_DIR / "plots" / "generated"


def get_fresh_policy(name: str):
    """Instantiate a fresh policy instance to reset internal dwell counters."""
    if name == "Nearest":
        return NearestServerPolicy(target_server_id="s1")
    elif name == "Min-Latency":
        return MinLatencyPolicy()
    elif name == "Max-Total-BW":
        return MaxTotalBandwidthPolicy()
    elif name == "Max-Effective-BW":
        return MaxEffectiveBandwidthPolicy()
    elif name == "AHP-TOPSIS":
        return AhpTopsisPolicy()
    elif name == "Static-Hysteresis":
        return HysteresisOnlyPolicy(hysteresis_threshold=0.08)
    elif name == "QoE-Greedy":
        return QoEGreedyPolicy()
    elif name == "Proposed-Margin":
        return ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65, min_dwell_steps=4)
    else:
        raise ValueError(f"Unknown policy name: {name}")


def run_all_baselines_benchmark(num_seeds: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all 8 policies across 8 scenarios with num_seeds independent seeds.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    policies = [
        "Nearest",
        "Min-Latency",
        "Max-Total-BW",
        "Max-Effective-BW",
        "AHP-TOPSIS",
        "Static-Hysteresis",
        "QoE-Greedy",
        "Proposed-Margin",
    ]

    scenarios = [
        "scenario_a",
        "scenario_b",
        "scenario_c",
        "scenario_d",
        "scenario_e",
        "scenario_f",
        "scenario_edge_infra",
        "scenario_puffer",
    ]

    seeds = list(range(42, 42 + num_seeds))
    total_runs = len(policies) * len(scenarios) * len(seeds)
    print(f"Starting Baseline Evaluation: {len(policies)} policies x {len(scenarios)} scenarios x {len(seeds)} seeds = {total_runs} runs...")

    raw_records = []
    run_idx = 0

    for pol_name in policies:
        for sc in scenarios:
            for s in seeds:
                policy_inst = get_fresh_policy(pol_name)
                df = run_scenario(
                    scenario_id=sc,
                    duration_s=30.0,
                    dt_s=0.5,
                    num_clients=3,
                    seed=s,
                    policy=policy_inst,
                )
                metrics = compute_scenario_metrics(df, dt_s=0.5)

                raw_records.append({
                    "policy": pol_name,
                    "scenario": sc,
                    "seed": s,
                    "cumulative_qoe": metrics["cumulative_qoe"],
                    "total_switches": metrics["total_switches"],
                    "switch_rate_per_min": metrics["switch_rate_per_min"],
                    "ping_pong_rate": metrics["ping_pong_rate"],
                    "mean_dwell_time_s": metrics["mean_dwell_time_s"],
                    "qos_violation_rate": metrics["qos_violation_rate"],
                })
                run_idx += 1
                if run_idx % 200 == 0 or run_idx == total_runs:
                    print(f"  Progress: {run_idx}/{total_runs} runs completed ({run_idx / total_runs * 100:.1f}%)")

    raw_df = pd.DataFrame(raw_records)
    raw_csv = LOGS_DIR / "baseline_runs_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"Saved raw execution logs to: {raw_csv}")

    # Compute summary statistics: mean, std, 95% CI
    summary_records = []
    grouped = raw_df.groupby(["policy", "scenario"])
    for (pol, sc), g in grouped:
        n = len(g)
        rec = {"policy": pol, "scenario": sc, "n_seeds": n}
        for metric in ["cumulative_qoe", "total_switches", "switch_rate_per_min", "ping_pong_rate", "mean_dwell_time_s", "qos_violation_rate"]:
            mean_val = float(g[metric].mean())
            std_val = float(g[metric].std()) if n > 1 else 0.0
            ci95 = float(1.96 * std_val / np.sqrt(n)) if n > 1 else 0.0
            rec[f"{metric}_mean"] = mean_val
            rec[f"{metric}_std"] = std_val
            rec[f"{metric}_ci95"] = ci95
        summary_records.append(rec)

    summary_df = pd.DataFrame(summary_records)
    summary_csv = LOGS_DIR / "baseline_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved statistical summary to: {summary_csv}")

    return raw_df, summary_df


def run_weight_sensitivity_benchmark(num_seeds: int = 15) -> pd.DataFrame:
    """
    Evaluate how different weight configurations affect candidate server selection
    and resulting QoE under Proposed-Margin policy.
    """
    weight_profiles = {
        "Equal": {
            Criterion.LATENCY: 0.20,
            Criterion.EFFECTIVE_BW: 0.20,
            Criterion.SERVER_LOAD: 0.20,
            Criterion.PACKET_LOSS: 0.20,
            Criterion.AVAILABILITY: 0.20,
        },
        "Latency-Dominant": {
            Criterion.LATENCY: 0.50,
            Criterion.EFFECTIVE_BW: 0.20,
            Criterion.SERVER_LOAD: 0.10,
            Criterion.PACKET_LOSS: 0.10,
            Criterion.AVAILABILITY: 0.10,
        },
        "Bandwidth-Dominant": {
            Criterion.LATENCY: 0.15,
            Criterion.EFFECTIVE_BW: 0.55,
            Criterion.SERVER_LOAD: 0.10,
            Criterion.PACKET_LOSS: 0.10,
            Criterion.AVAILABILITY: 0.10,
        },
        "Loss-Dominant": {
            Criterion.LATENCY: 0.15,
            Criterion.EFFECTIVE_BW: 0.20,
            Criterion.SERVER_LOAD: 0.10,
            Criterion.PACKET_LOSS: 0.50,
            Criterion.AVAILABILITY: 0.05,
        },
        "Load-Dominant": {
            Criterion.LATENCY: 0.15,
            Criterion.EFFECTIVE_BW: 0.15,
            Criterion.SERVER_LOAD: 0.55,
            Criterion.PACKET_LOSS: 0.10,
            Criterion.AVAILABILITY: 0.05,
        },
    }

    test_scenarios = ["scenario_a", "scenario_b", "scenario_f", "scenario_edge_infra"]
    seeds = list(range(100, 100 + num_seeds))
    records = []

    print(f"\nStarting Weight Sensitivity Sweep: {len(weight_profiles)} profiles across {len(test_scenarios)} scenarios...")

    for prof_name, weights in weight_profiles.items():
        for sc in test_scenarios:
            for s in seeds:
                policy = ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65, min_dwell_steps=4)
                df = run_scenario(
                    scenario_id=sc,
                    duration_s=30.0,
                    dt_s=0.5,
                    num_clients=3,
                    seed=s,
                    policy=policy,
                    weights=weights,
                )
                metrics = compute_scenario_metrics(df, dt_s=0.5)

                # Count time spent on each server
                srv_counts = df["client_server"].value_counts(normalize=True)
                s1_pct = float(srv_counts.get("s1", 0.0))
                s2_pct = float(srv_counts.get("s2", 0.0))
                s3_pct = float(srv_counts.get("s3", 0.0))

                records.append({
                    "profile": prof_name,
                    "scenario": sc,
                    "seed": s,
                    "s1_dwell_fraction": s1_pct,
                    "s2_dwell_fraction": s2_pct,
                    "s3_dwell_fraction": s3_pct,
                    "mean_bandwidth_mbps": float(df["effective_bw_mbps"].mean()),
                    "mean_latency_ms": float(df["client_latency_ms"].mean()),
                    "cumulative_qoe": metrics["cumulative_qoe"],
                    "total_switches": metrics["total_switches"],
                    "qos_violation_rate": metrics["qos_violation_rate"],
                })

    weight_df = pd.DataFrame(records)
    out_csv = LOGS_DIR / "weight_sensitivity_results.csv"
    weight_df.to_csv(out_csv, index=False)
    print(f"Saved weight sensitivity results to: {out_csv}")
    return weight_df


def plot_baseline_comparison(raw_df: pd.DataFrame, summary_df: pd.DataFrame):
    """
    Generate IEEE PerCom standard 3-panel comparative performance plot.
    """
    plt.style.use(CONFIG_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    # Color palette (Okabe-Ito / Colorblind safe)
    palette = {
        "Nearest": "#888888",
        "Min-Latency": "#E69F00",
        "Max-Total-BW": "#56B4E9",
        "Max-Effective-BW": "#009E73",
        "AHP-TOPSIS": "#F0E442",
        "Static-Hysteresis": "#0072B2",
        "QoE-Greedy": "#D55E00",
        "Proposed-Margin": "#CC79A7",
    }

    policies = list(palette.keys())

    # -------------------------------------------------------------
    # Panel 1: Churn under Noise (Scenario A) vs Real Events (Scenario B)
    # -------------------------------------------------------------
    ax1 = axes[0]
    sc_noise = "scenario_a"
    sc_shift = "scenario_b"

    sub_noise = summary_df[summary_df["scenario"] == sc_noise].set_index("policy")
    sub_shift = summary_df[summary_df["scenario"] == sc_shift].set_index("policy")

    x = np.arange(len(policies))
    width = 0.38

    noise_vals = [sub_noise.loc[p]["switch_rate_per_min_mean"] if p in sub_noise.index else 0.0 for p in policies]
    shift_vals = [sub_shift.loc[p]["switch_rate_per_min_mean"] if p in sub_shift.index else 0.0 for p in policies]

    rects1 = ax1.bar(x - width/2, noise_vals, width, label="Scenario A (Noise Churn)", color="#D55E00", alpha=0.85)
    rects2 = ax1.bar(x + width/2, shift_vals, width, label="Scenario B (Regime Shift)", color="#0072B2", alpha=0.85)

    ax1.set_ylabel("Switch Rate (switches / min)")
    ax1.set_title("(a) Churn under Noise vs Regime Shift")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=45, ha="right", fontsize=7)
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 2: Mean Cumulative Video QoE across All Scenarios (with 95% CI)
    # -------------------------------------------------------------
    ax2 = axes[1]
    overall_qoe = raw_df.groupby("policy")["cumulative_qoe"].agg(["mean", "std", "count"])
    overall_qoe["ci95"] = 1.96 * overall_qoe["std"] / np.sqrt(overall_qoe["count"])

    qoe_means = [overall_qoe.loc[p]["mean"] for p in policies]
    qoe_cis = [overall_qoe.loc[p]["ci95"] for p in policies]
    bar_colors = [palette[p] for p in policies]

    ax2.bar(x, qoe_means, yerr=qoe_cis, capsize=3, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax2.set_ylabel("Cumulative Video QoE (ITU-T P.1203)")
    ax2.set_title("(b) Average Video QoE (All Scenarios, 95% CI)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, rotation=45, ha="right", fontsize=7)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 3: QoS Violation Rate vs Mean Dwell Time Trade-off
    # -------------------------------------------------------------
    ax3 = axes[2]
    overall_qos = raw_df.groupby("policy")[["qos_violation_rate", "mean_dwell_time_s"]].mean()

    for p in policies:
        qos_v = overall_qos.loc[p]["qos_violation_rate"] * 100.0  # as percentage
        dwell = overall_qos.loc[p]["mean_dwell_time_s"]
        ax3.scatter(dwell, qos_v, color=palette[p], s=80, edgecolors="black", linewidth=0.8, label=p, zorder=5)

    ax3.set_xlabel("Mean Dwell Time (seconds)")
    ax3.set_ylabel("QoS Violation Rate (%)")
    ax3.set_title("(c) QoS Violations vs Stability (Dwell)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.8)

    plt.tight_layout()
    pdf_out = PLOTS_DIR / "baseline_comparison.pdf"
    png_out = PLOTS_DIR / "baseline_comparison.png"
    plt.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated comparison plots:\n  {pdf_out}\n  {png_out}")


def plot_weight_sensitivity(weight_df: pd.DataFrame):
    """
    Generate plot demonstrating how subjective preference vectors alter server allocation.
    """
    plt.style.use(CONFIG_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    profiles = list(weight_df["profile"].unique())
    grouped = weight_df.groupby("profile")

    # Panel 1: Dwell Fraction by Server
    s1_means = [grouped.get_group(p)["s1_dwell_fraction"].mean() * 100.0 for p in profiles]
    s2_means = [grouped.get_group(p)["s2_dwell_fraction"].mean() * 100.0 for p in profiles]
    s3_means = [grouped.get_group(p)["s3_dwell_fraction"].mean() * 100.0 for p in profiles]

    x = np.arange(len(profiles))
    width = 0.55

    ax1.bar(x, s1_means, width, label="s1 (Low Latency Edge)", color="#56B4E9")
    ax1.bar(x, s2_means, width, bottom=s1_means, label="s2 (Balanced Edge)", color="#E69F00")
    bottom_s3 = np.array(s1_means) + np.array(s2_means)
    ax1.bar(x, s3_means, width, bottom=bottom_s3, label="s3 (High-Cap Cloud)", color="#009E73")

    ax1.set_ylabel("Server Dwell Time Allocation (%)")
    ax1.set_title("(a) Server Selection Allocation by Weight Profile")
    ax1.set_xticks(x)
    ax1.set_xticklabels(profiles, rotation=35, ha="right", fontsize=8)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Latency vs Bitrate Trade-off Across Profiles
    lat_means = [grouped.get_group(p)["mean_latency_ms"].mean() for p in profiles]
    bw_means = [grouped.get_group(p)["mean_bandwidth_mbps"].mean() for p in profiles]

    colors = ["#888888", "#E69F00", "#56B4E9", "#CC79A7", "#009E73"]
    for i, p in enumerate(profiles):
        ax2.scatter(lat_means[i], bw_means[i], s=100, color=colors[i], edgecolors="black", label=p, zorder=5)
        ax2.annotate(p, (lat_means[i] + 0.5, bw_means[i] - 0.2), fontsize=7)

    ax2.set_xlabel("Average Observed Latency (ms)")
    ax2.set_ylabel("Average Effective Bandwidth (Mbps)")
    ax2.set_title("(b) QoS Trade-off Across Weight Profiles")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_out = PLOTS_DIR / "weight_sensitivity.pdf"
    png_out = PLOTS_DIR / "weight_sensitivity.png"
    plt.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated weight sensitivity plots:\n  {pdf_out}\n  {png_out}")


def main():
    print("=" * 70)
    print("STEP 3: BASELINE SUITE & WEIGHT SENSITIVITY BENCHMARK")
    print("=" * 70)

    # 1. Run 30-seed baseline evaluation
    raw_df, summary_df = run_all_baselines_benchmark(num_seeds=30)

    # 2. Run weight sensitivity sweep
    weight_df = run_weight_sensitivity_benchmark(num_seeds=15)

    # 3. Generate publication-ready figures
    print("\nGenerating IEEE publication-quality figures...")
    plot_baseline_comparison(raw_df, summary_df)
    plot_weight_sensitivity(weight_df)

    print("\n[SUCCESS] Baseline benchmark and weight sensitivity completed successfully.")


if __name__ == "__main__":
    main()
