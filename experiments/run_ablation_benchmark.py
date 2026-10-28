"""
Ablation Study and Statistical Significance Benchmark for Step 6.
Target Venue: IEEE PerCom 2027 (A4E Workshop)

Generates:
1. data_ablation_study.csv: Performance breakdown across 5 architectural ablations over 30 seeds.
2. data_statistical_tests.csv: Paired Wilcoxon signed-rank tests, t-tests, Cohen's d effect sizes,
   and 95% confidence intervals comparing Full-Proposed against ablations and baselines.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from edge_switching.qos_model import (
    Criterion,
    DecisionMatrix,
    QoSFormalizationEngine,
    ServerTelemetry,
    SystemConfig,
)
from edge_switching.simulator import (
    ClosedLoopSimulator,
    EdgeServerNode,
    NoiseDistribution,
    VideoClientState,
)
from edge_switching.scenarios import run_scenario, create_standard_servers
from edge_switching.policies import (
    AhpTopsisPolicy,
    BaseSelectionPolicy,
    MaxTotalBandwidthPolicy,
    MinLatencyPolicy,
    ProposedMarginPolicy,
)
from edge_switching.weighting import get_literature_ahp_profile

LOGS_DIR = ROOT_DIR / "logs" / "ablations"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class AblationNoNoiseHurdlePolicy(ProposedMarginPolicy):
    """Ablation 1: Uses fixed static hurdle lambda without noise adaptation (k_noise = 0)."""
    def __init__(self, base_hurdle: float = 0.05):
        super().__init__(base_hurdle=base_hurdle, k_noise=0.0, min_dwell_steps=4)


class AblationNoBufferMultiplierPolicy(ProposedMarginPolicy):
    """Ablation 2: Does not scale switching hurdle when buffer is critically depleted."""
    def decide(self, client, matrix, engine, telemetry_history=None):
        # Override to keep hurdle fixed regardless of buffer
        saved_buf = client.buffer_occupancy_s
        client.buffer_occupancy_s = 10.0  # Force buffer multiplier to 1.0
        dec = super().decide(client, matrix, engine, telemetry_history)
        client.buffer_occupancy_s = saved_buf
        return dec


class AblationNoPlusOnePolicy(BaseSelectionPolicy):
    """Ablation 3: Uses raw advertised server capacity instead of (+1)-user effective bandwidth."""
    def __init__(self, base_hurdle: float = 0.05, k_noise: float = 1.65):
        super().__init__("Ablation-NoPlusOneBandwidth")
        self.base_hurdle = base_hurdle
        self.k_noise = k_noise

    def decide(self, client, matrix, engine, telemetry_history=None):
        curr_id = client.current_server_id
        curr_idx = matrix.server_ids.index(curr_id) if curr_id in matrix.server_ids else -1

        # Modify utility matrix to use raw capacity rather than fair-share
        U_raw = matrix.utility_matrix_U.copy()
        if matrix.raw_server_capacity is not None:
            # Overwrite effective bandwidth column (col 1) with raw capacity utility
            for i in range(len(matrix.server_ids)):
                raw_cap = matrix.raw_server_capacity[i]
                U_raw[i, 1] = float(np.clip(raw_cap / 1000.0, 0.0, 1.0))

        scores = np.dot(U_raw, matrix.weight_vector_w)
        best_idx = int(np.argmax(scores))
        best_id = matrix.server_ids[best_idx]

        if curr_idx < 0 or best_idx == curr_idx:
            return {"action": "RETAIN", "target_server": curr_id, "decision_margin": 0.0, "switching_hurdle": 0.05}

        gain = float(scores[best_idx] - scores[curr_idx])
        if gain > self.base_hurdle:
            return {"action": "SWITCH", "target_server": best_id, "decision_margin": gain, "switching_hurdle": self.base_hurdle}
        return {"action": "RETAIN", "target_server": curr_id, "decision_margin": gain, "switching_hurdle": self.base_hurdle}


def run_ablation_experiments():
    """Run 30-seed benchmark across all component ablations."""
    print("1. Running Component Ablation Suite (30 seeds across canonical scenarios)...")
    w_video, _, _ = get_literature_ahp_profile("video_streaming")
    num_seeds = 30

    configurations = {
        "Full-Proposed": ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65, min_dwell_steps=4),
        "Ablation-NoNoiseHurdle": AblationNoNoiseHurdlePolicy(base_hurdle=0.05),
        "Ablation-NoBufferMultiplier": AblationNoBufferMultiplierPolicy(base_hurdle=0.05, k_noise=1.65),
        "Ablation-NoPlusOneBW": AblationNoPlusOnePolicy(base_hurdle=0.05, k_noise=1.65),
        "Baseline-AHP-TOPSIS": AhpTopsisPolicy(),
        "Baseline-MinLatency": MinLatencyPolicy(),
        "Baseline-MaxTotalBW": MaxTotalBandwidthPolicy(),
    }

    # Store seed-level QoE vectors for statistical testing
    seed_qoe_records = {name: [] for name in configurations}

    records = []
    scenarios_to_eval = ["scenario_a", "scenario_b", "scenario_c", "scenario_d", "scenario_f"]

    for name, policy in configurations.items():
        print(f"  Evaluating configuration: {name}...")
        all_qoes = []
        all_switches = []
        all_ping_pongs = []
        all_stalls = []
        all_viols = []

        for seed in range(num_seeds):
            seed_qoe_acc = []
            seed_switches = 0
            seed_stalls = 0.0
            total_steps = 0
            total_viols = 0

            for sc in scenarios_to_eval:
                df = run_scenario(sc, duration_s=25.0, dt_s=0.5, num_clients=1, seed=seed, policy=policy, weights=w_video)
                # Compute QoE
                for _, row in df.iterrows():
                    bw = float(row["effective_bw_mbps"])
                    lat = float(row["client_latency_ms"])
                    sw = 1.0 if row["action"] == "SWITCH" else 0.0
                    stall = float(row.get("stall_duration_s", 0.0))

                    qoe = 0.8 * (bw / 25.0) - (0.005 * lat) - (0.5 * sw) - (4.3 * stall)
                    seed_qoe_acc.append(qoe)
                    seed_switches += int(sw)
                    seed_stalls += stall
                    if lat > 120.0 or stall > 0.0:
                        total_viols += 1
                    total_steps += 1

            mean_seed_qoe = float(np.mean(seed_qoe_acc))
            all_qoes.append(mean_seed_qoe)
            seed_qoe_records[name].append(mean_seed_qoe)

            all_switches.append((seed_switches / total_steps) * 120.0) # switches / min
            all_stalls.append(seed_stalls)
            all_viols.append((total_viols / total_steps) * 100.0)

        mean_qoe = float(np.mean(all_qoes))
        std_qoe = float(np.std(all_qoes))
        ci95 = 1.96 * (std_qoe / np.sqrt(num_seeds))

        records.append({
            "configuration": name,
            "mean_qoe": round(mean_qoe, 4),
            "std_qoe": round(std_qoe, 4),
            "ci95_qoe": round(ci95, 4),
            "switch_rate_per_min": round(float(np.mean(all_switches)), 2),
            "stall_duration_s": round(float(np.mean(all_stalls)), 3),
            "qos_violation_pct": round(float(np.mean(all_viols)), 2),
        })

    df_abl = pd.DataFrame(records)
    out_csv = LOGS_DIR / "data_ablation_study.csv"
    df_abl.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")

    # -------------------------------------------------------------
    # Statistical Significance Testing (Paired vs Full-Proposed)
    # -------------------------------------------------------------
    print("2. Performing Statistical Hypothesis Tests & Effect Size Analysis...")
    stat_records = []
    base_qoe = np.array(seed_qoe_records["Full-Proposed"])

    for comp_name in configurations:
        if comp_name == "Full-Proposed":
            continue

        comp_qoe = np.array(seed_qoe_records[comp_name])
        diff = base_qoe - comp_qoe

        # Paired Wilcoxon signed-rank test
        w_stat, w_pval = stats.wilcoxon(diff, alternative="greater")

        # Welch's two-sample t-test
        t_stat, t_pval = stats.ttest_rel(base_qoe, comp_qoe)

        # Cohen's d effect size
        d_val = float(np.mean(diff) / (np.std(diff) + 1e-8))

        stat_records.append({
            "comparison": f"Full-Proposed vs {comp_name}",
            "mean_qoe_difference": round(float(np.mean(diff)), 4),
            "wilcoxon_stat_W": round(float(w_stat), 1),
            "wilcoxon_p_value": float(f"{w_pval:.2e}"),
            "t_statistic": round(float(t_stat), 3),
            "t_p_value": float(f"{t_pval:.2e}"),
            "cohens_d": round(d_val, 3),
            "statistically_significant": bool(w_pval < 0.001),
        })

    df_stats = pd.DataFrame(stat_records)
    out_stat_csv = LOGS_DIR / "data_statistical_tests.csv"
    df_stats.to_csv(out_stat_csv, index=False)
    print(f"  Saved -> {out_stat_csv.name}")


def main():
    print("=" * 70)
    print("STEP 6: RUNNING ABLATION & STATISTICAL SIGNIFICANCE BENCHMARK")
    print("=" * 70)

    run_ablation_experiments()

    print("\n[SUCCESS] Ablation benchmarks and statistical tests completed successfully.")


if __name__ == "__main__":
    main()
