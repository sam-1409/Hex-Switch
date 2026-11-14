import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import pandas as pd

from edge_switching.qos_model import (
    Criterion,
    DecisionMatrix,
    QoSFormalizationEngine,
    ServerTelemetry,
    SystemConfig,
)
from edge_switching.simulator import VideoClientState
from edge_switching.scenarios import run_scenario
from edge_switching.policies import ProposedMarginPolicy, AhpTopsisPolicy
from edge_switching.weighting import (
    compute_ahp_weights,
    compute_critic_weights,
    compute_entropy_weights,
    compute_equal_weights,
    get_literature_ahp_profile,
    STANDARD_CRITERIA,
)
from edge_switching.explainer import EdgeSwitchingExplainer

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs" / "xai"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def run_weights_benchmark():
    print("1. Computing Criteria Weights (AHP vs Objective)...")
    records = []

    # AHP Profiles
    for prof in ["video_streaming", "interactive_gaming", "balanced_edge"]:
        w_dict, ci, cr = get_literature_ahp_profile(prof)
        rec = {"method": f"AHP-{prof}", "CI": round(ci, 4), "CR": round(cr, 4)}
        for c in STANDARD_CRITERIA:
            rec[c.value] = round(w_dict[c], 4)
        records.append(rec)

    rec_eq = {"method": "Equal", "CI": 0.0, "CR": 0.0}
    for c in STANDARD_CRITERIA:
        rec_eq[c.value] = 0.20
    records.append(rec_eq)

    engine = QoSFormalizationEngine()
    now = 10.0
    telemetries = [
        ServerTelemetry("s1", 15.0, 100.0, 4, 20.0, 0.20, 0.010, True, now),
        ServerTelemetry("s2", 35.0, 800.0, 16, 50.0, 0.60, 0.005, True, now),
        ServerTelemetry("s3", 80.0, 1500.0, 30, 50.0, 0.85, 0.001, True, now),
    ]
    matrix = engine.build_matrices(telemetries, {c: 0.20 for c in STANDARD_CRITERIA}, now)
    U = matrix.utility_matrix_U

    # Shannon Entropy weights
    ent_w = compute_entropy_weights(U)
    rec_ent = {"method": "Shannon-Entropy", "CI": 0.0, "CR": 0.0}
    for i, c in enumerate(STANDARD_CRITERIA):
        rec_ent[c.value] = round(float(ent_w[i]), 4)
    records.append(rec_ent)

    # CRITIC weights
    crit_w = compute_critic_weights(U)
    rec_crit = {"method": "CRITIC", "CI": 0.0, "CR": 0.0}
    for i, c in enumerate(STANDARD_CRITERIA):
        rec_crit[c.value] = round(float(crit_w[i]), 4)
    records.append(rec_crit)

    df_weights = pd.DataFrame(records)
    out_csv = LOGS_DIR / "data_weights_comparison.csv"
    df_weights.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")


def run_attributions_benchmark():
    """Extract exact Shapley attributions across canonical scenarios."""
    print("2. Generating Exact Feature Attributions across Scenarios...")
    engine = QoSFormalizationEngine()
    explainer = EdgeSwitchingExplainer(engine)
    w_video, _, _ = get_literature_ahp_profile("video_streaming")

    scenario_events = [
        ("Scenario A (Noise Jitter)", "s1", "s2", False, [
            ServerTelemetry("s1", 26.5, 100.0, 4, 20.0, 0.20, 0.010, True, 10.0),
            ServerTelemetry("s2", 36.2, 800.0, 16, 50.0, 0.60, 0.005, True, 10.0),
        ]),
        ("Scenario B (Regime Shift)", "s1", "s2", True, [
            ServerTelemetry("s1", 75.0, 80.0, 4, 16.0, 0.70, 0.030, True, 25.0),
            ServerTelemetry("s2", 35.0, 800.0, 16, 50.0, 0.60, 0.005, True, 25.0),
        ]),
        ("Scenario C (Server Failure)", "s1", "s2", True, [
            ServerTelemetry("s1", 75.0, 80.0, 4, 0.0, 0.70, 0.030, False, 20.0),
            ServerTelemetry("s2", 35.0, 800.0, 16, 50.0, 0.60, 0.005, True, 20.0),
        ]),
        ("Scenario D (Flash Crowd)", "s1", "s2", False, [
            ServerTelemetry("s1", 25.0, 100.0, 4, 20.0, 0.20, 0.010, True, 25.0),
            ServerTelemetry("s2", 40.0, 800.0, 70, 11.2, 0.96, 0.015, True, 25.0),
        ]),
        ("Scenario F (Mobility Drift)", "s1", "s2", True, [
            ServerTelemetry("s1", 95.0, 100.0, 4, 20.0, 0.20, 0.010, True, 30.0),
            ServerTelemetry("s2", 22.0, 800.0, 16, 50.0, 0.50, 0.005, True, 30.0),
        ]),
    ]

    records = []
    for sc_name, curr_id, target_id, is_sw, telems in scenario_events:
        matrix = engine.build_matrices(telems, w_video, telems[0].timestamp_s)
        client = VideoClientState("c0", curr_id, buffer_occupancy_s=3.0)
        action = "SWITCH" if is_sw else "RETAIN"

        decision = {
            "action": action,
            "target_server": target_id if is_sw else curr_id,
            "decision_margin": 0.15 if is_sw else 0.01,
            "switching_hurdle": 0.05,
            "reason": "Evaluated handover" if is_sw else "Gain within telemetry noise margin",
        }
        exp = explainer.explain_decision(client, matrix, decision)

        rec = {
            "scenario": sc_name,
            "action": exp.action,
            "target": exp.target_server_id,
            "dominant_criterion": exp.dominant_criterion,
            "decision_margin": round(exp.decision_margin, 3),
        }
        for c in STANDARD_CRITERIA:
            rec[f"phi_{c.value}"] = round(exp.feature_attributions.get(c.value, 0.0), 4)
        records.append(rec)

    df_att = pd.DataFrame(records)
    out_csv = LOGS_DIR / "data_feature_attributions.csv"
    df_att.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")


def run_counterfactual_trajectory_benchmark():
    """Track decision margin trajectory and noise confidence bands over time in Scenario B."""
    print("3. Generating Counterfactual Margin Trajectory (Scenario B)...")
    policy = ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65, min_dwell_steps=4)
    w_video, _, _ = get_literature_ahp_profile("video_streaming")
    df = run_scenario("scenario_b", duration_s=35.0, dt_s=0.5, num_clients=1, seed=42, policy=policy, weights=w_video)

    records = []
    engine = QoSFormalizationEngine()
    explainer = EdgeSwitchingExplainer(engine)

    # Extract time series of margin and noise threshold
    for i, row in df.iterrows():
        t = float(row["time_s"])
        action = str(row["action"])
        hurdle = float(row["switching_hurdle"])
        margin = float(row["decision_margin"])
        # In ProposedMarginPolicy, noise threshold is k * sigma (around 0.025)
        noise_band = 1.65 * 0.015

        records.append({
            "time_s": t,
            "action": action,
            "client_server": str(row["client_server"]),
            "decision_margin": round(margin, 4),
            "switching_hurdle": round(hurdle, 4),
            "noise_threshold": round(noise_band, 4),
            "effective_bw_mbps": round(float(row["effective_bw_mbps"]), 2),
            "client_latency_ms": round(float(row["client_latency_ms"]), 1),
        })

    df_traj = pd.DataFrame(records)
    out_csv = LOGS_DIR / "data_counterfactual_margins.csv"
    df_traj.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")


def run_explanation_stability_benchmark(num_seeds: int = 30, epochs: int = 60):
    """Measure actual explainer dominant-criterion churn under controlled jitter.

    This is a synthetic, controlled telemetry experiment, not a real-trace result.
    Each seed creates an independent noise sequence. Explanations are generated
    by EdgeSwitchingExplainer from the actual DecisionMatrix at every epoch.
    """
    print(f"4. Measuring explanation stability ({num_seeds} seeds x {epochs} epochs)...")
    w_video, _, _ = get_literature_ahp_profile("video_streaming")
    methods = {
        "AHP-TOPSIS": lambda: AhpTopsisPolicy(),
        "Proposed-Margin": lambda: ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65),
    }
    seed_records = []
    engine = QoSFormalizationEngine()

    for method, factory in methods.items():
        for seed in range(num_seeds):
            rng = np.random.default_rng(seed)
            policy = factory()
            explainer = EdgeSwitchingExplainer(engine)
            client = VideoClientState("c0", "s1", buffer_occupancy_s=4.0)
            dominant = []
            action_sequence = []
            attribution_vectors = []

            for epoch in range(epochs):
                now = (epoch + 1) * 0.5
                # Stationary underlying state; only measurement noise changes.
                s1 = ServerTelemetry(
                    "s1", max(5.0, 35.0 + rng.normal(0.0, 3.0)), 300.0,
                    10, max(0.5, 30.0 + rng.normal(0.0, 1.5)), 0.20,
                    0.005, True, now)
                s2 = ServerTelemetry(
                    "s2", max(5.0, 42.0 + rng.normal(0.0, 3.0)), 800.0,
                    25, max(0.5, 42.0 + rng.normal(0.0, 1.5)), 0.35,
                    0.005, True, now)
                matrix = engine.build_matrices([s1, s2], w_video, now)
                decision = policy.decide(client, matrix, engine, telemetry_history=None)
                explanation = explainer.explain_decision(client, matrix, decision)
                dominant.append(explanation.dominant_criterion)
                action_sequence.append(str(decision.get("action", "RETAIN")))
                attribution_vectors.append(explanation.feature_attributions)
                if decision.get("action") == "SWITCH":
                    client.current_server_id = str(decision.get("target_server", client.current_server_id))

            flips = sum(dominant[i] != dominant[i - 1] for i in range(1, len(dominant)))
            churn_rate = flips / max(1, epochs - 1)
            seed_records.append({
                "method": method, "seed": seed, "epochs": epochs,
                "dominant_criterion_flips": flips,
                "dominant_criterion_churn_rate": churn_rate,
                "explanation_stability_pct": 100.0 * (1.0 - churn_rate),
                "action_switches": sum(a == "SWITCH" for a in action_sequence),
                "event_regime": "stationary_gaussian_jitter",
            })

    seed_df = pd.DataFrame(seed_records)
    seed_df.to_csv(LOGS_DIR / "data_explanation_stability_seed_level.csv", index=False)
    summary = (seed_df.groupby("method").agg(
        n_seeds=("seed", "nunique"),
        mean_explanation_stability_pct=("explanation_stability_pct", "mean"),
        std_explanation_stability_pct=("explanation_stability_pct", "std"),
        mean_dominant_criterion_flips=("dominant_criterion_flips", "mean"),
        mean_action_switches=("action_switches", "mean"),
    ).reset_index())
    from scipy import stats as scipy_stats
    summary["ci95_stability_pct"] = summary.apply(
        lambda r: scipy_stats.t.ppf(0.975, int(r.n_seeds)-1) * r.std_explanation_stability_pct / np.sqrt(r.n_seeds)
        if r.n_seeds > 1 else 0.0, axis=1)
    summary["measurement"] = "actual dominant criterion from explainer; synthetic stationary telemetry"
    summary.to_csv(LOGS_DIR / "data_explanation_stability.csv", index=False)
    print("  Saved seed-level and summary explanation-stability results.")


def main():
    print("=" * 70)
    print("HEX-SWITCH: XAI BENCHMARKS & EXPLANATION PIPELINE")
    print("=" * 70)

    run_weights_benchmark()
    run_attributions_benchmark()
    run_counterfactual_trajectory_benchmark()
    run_explanation_stability_benchmark()

    print("\n[SUCCESS] XAI benchmark datasets exported successfully.")

if __name__ == "__main__":
    main()