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
from edge_switching.metrics import compute_scenario_metrics

LOGS_DIR = ROOT_DIR / "logs" / "ablations"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class AblationNoNoiseHurdlePolicy(ProposedMarginPolicy):
    def __init__(self, base_hurdle: float = 0.05):
        super().__init__(base_hurdle=base_hurdle, k_noise=0.0, min_dwell_steps=4)

class AblationNoBufferMultiplierPolicy(ProposedMarginPolicy):
    def decide(self, client, matrix, engine, telemetry_history=None):
        # Override to keep hurdle fixed regardless of buffer
        saved_buf = client.buffer_occupancy_s
        client.buffer_occupancy_s = 10.0  # Force buffer multiplier to 1.0
        dec = super().decide(client, matrix, engine, telemetry_history)
        client.buffer_occupancy_s = saved_buf
        return dec

class AblationNoPlusOnePolicy(BaseSelectionPolicy):
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

def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return mean, sd, half

def _holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted

def run_ablation_experiments():
    print("1. Running component ablations (30 paired seeds)...")
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
    scenarios = ["scenario_a", "scenario_b", "scenario_c", "scenario_d", "scenario_f", "scenario_low_buffer"]
    seed_rows = []
    summary_rows = []
    paired_qoe = {name: [] for name in configurations}

    for name in configurations:
        print(f"  Evaluating {name}...")
        for seed in range(num_seeds):
            run_metrics = []
            for scenario in scenarios:
                # Fresh policy per run prevents hidden state leaking between scenarios.
                if name == "Full-Proposed":
                    policy = ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65, min_dwell_steps=4)
                elif name == "Ablation-NoNoiseHurdle":
                    policy = AblationNoNoiseHurdlePolicy(base_hurdle=0.05)
                elif name == "Ablation-NoBufferMultiplier":
                    policy = AblationNoBufferMultiplierPolicy(base_hurdle=0.05, k_noise=1.65)
                elif name == "Ablation-NoPlusOneBW":
                    policy = AblationNoPlusOnePolicy(base_hurdle=0.05, k_noise=1.65)
                elif name == "Baseline-AHP-TOPSIS":
                    policy = AhpTopsisPolicy()
                elif name == "Baseline-MinLatency":
                    policy = MinLatencyPolicy()
                else:
                    policy = MaxTotalBandwidthPolicy()
                frame = run_scenario(scenario, duration_s=25.0, dt_s=0.5,
                                     num_clients=1, seed=seed, policy=policy, weights=w_video)
                run_metrics.append(compute_scenario_metrics(frame, dt_s=0.5))

            total_qoe = float(sum(m["cumulative_qoe_proxy"] for m in run_metrics))
            duration_total = len(scenarios) * 25.0
            switches = sum(m["total_switches"] for m in run_metrics)
            seed_record = {
                "configuration": name, "seed": seed,
                "cumulative_qoe_proxy": total_qoe,
                "switch_rate_per_min": switches / duration_total * 60.0,
                "qos_violation_rate": float(np.mean([m["qos_violation_rate"] for m in run_metrics])),
                "availability_violation_rate": float(np.nanmean([m["availability_violation_rate"] for m in run_metrics])),
                "latency_violation_rate": float(np.mean([m["latency_violation_rate"] for m in run_metrics])),
                "stall_epoch_rate": float(np.mean([m["stall_epoch_rate"] for m in run_metrics])),
                "ping_pong_rate": float(np.mean([m["ping_pong_rate"] for m in run_metrics])),
            }
            seed_rows.append(seed_record)
            paired_qoe[name].append(total_qoe)

        subset = [r for r in seed_rows if r["configuration"] == name]
        qmean, qsd, qci = _mean_ci([r["cumulative_qoe_proxy"] for r in subset])
        summary_rows.append({
            "configuration": name, "n_seeds": num_seeds,
            "mean_qoe_proxy": qmean, "std_qoe_proxy": qsd, "ci95_qoe_proxy": qci,
            "mean_switch_rate_per_min": float(np.mean([r["switch_rate_per_min"] for r in subset])),
            "mean_qos_violation_rate": float(np.mean([r["qos_violation_rate"] for r in subset])),
            "mean_availability_violation_rate": float(np.nanmean([r["availability_violation_rate"] for r in subset])),
            "mean_latency_violation_rate": float(np.mean([r["latency_violation_rate"] for r in subset])),
            "mean_stall_epoch_rate": float(np.mean([r["stall_epoch_rate"] for r in subset])),
        })

    pd.DataFrame(seed_rows).to_csv(LOGS_DIR / "data_ablation_seed_level.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(LOGS_DIR / "data_ablation_study.csv", index=False)

    print("2. Paired Wilcoxon tests with Holm correction and paired effect sizes...")
    base = np.asarray(paired_qoe["Full-Proposed"], dtype=float)
    tests = []
    raw_ps = []
    for name in configurations:
        if name == "Full-Proposed":
            continue
        comp = np.asarray(paired_qoe[name], dtype=float)
        diff = base - comp
        if np.allclose(diff, 0):
            w_stat, p_value = 0.0, 1.0
        else:
            result = stats.wilcoxon(base, comp, alternative="two-sided", zero_method="wilcox")
            w_stat, p_value = float(result.statistic), float(result.pvalue)
        sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        dz = float(np.mean(diff) / sd_diff) if sd_diff > 0 else float("nan")
        diff_ci = (float(stats.t.ppf(0.975, len(diff)-1) * sd_diff / np.sqrt(len(diff)))
                   if len(diff) > 1 else 0.0)
        # Matched-pairs rank-biserial correlation from positive/negative ranks.
        nz = diff[~np.isclose(diff, 0.0)]
        if len(nz):
            ranks = stats.rankdata(np.abs(nz))
            rbc = float((np.sum(ranks[nz > 0]) - np.sum(ranks[nz < 0])) / np.sum(ranks))
        else:
            rbc = 0.0
        tests.append({"comparison": f"Full-Proposed vs {name}",
                      "mean_paired_difference_qoe_proxy": float(np.mean(diff)),
                      "ci95_paired_difference_qoe_proxy": diff_ci,
                      "wilcoxon_statistic": w_stat, "p_value": p_value,
                      "paired_cohens_dz": dz, "rank_biserial_correlation": rbc,
                      "n_pairs": len(diff)})
        raw_ps.append(p_value)
    adjusted = _holm_adjust(raw_ps)
    for row, p_adj in zip(tests, adjusted):
        row["holm_adjusted_p_value"] = float(p_adj)
        row["significant_holm_0_05"] = bool(p_adj < 0.05)
    pd.DataFrame(tests).to_csv(LOGS_DIR / "data_statistical_tests.csv", index=False)
    print("Saved seed-level, summary, and corrected statistical outputs.")

def main():
    print("=" * 70)
    print("HEX-SWITCH: COMPONENT ABLATION & STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    run_ablation_experiments()

    print("\n[SUCCESS] Ablation benchmarks and statistical tests completed successfully.")


if __name__ == "__main__":
    main()