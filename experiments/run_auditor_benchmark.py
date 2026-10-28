"""
Benchmark and Data Generator for Step 5: Post-Decision Auditor & Explanation Faithfulness.
Target Venue: IEEE PerCom 2027 (A4E Workshop)

Generates:
1. data_fault_detection_confusion.csv: Precision, Recall, Specificity, F1 per fault class.
2. data_counterfactual_faithfulness.csv: Margin monotonicity trajectory and decision inversion.
3. data_auditor_latency_overhead.csv: Microsecond-level runtime execution latency distribution.
"""

import sys
import time
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
from edge_switching.weighting import get_literature_ahp_profile, STANDARD_CRITERIA
from edge_switching.explainer import EdgeSwitchingExplainer
from edge_switching.auditor import (
    AuditSeverity,
    CounterfactualFaithfulnessVerifier,
    FaultType,
    PostDecisionAuditor,
)

LOGS_DIR = ROOT_DIR / "logs" / "auditor"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def run_fault_injection_benchmark():
    """Evaluate Auditor detection recall, false alarm rate, and precision across fault classes."""
    print("1. Running Fault Injection & Confusion Matrix Benchmark...")
    engine = QoSFormalizationEngine()
    auditor = PostDecisionAuditor()
    w_video, _, _ = get_literature_ahp_profile("video_streaming")

    fault_classes = [
        FaultType.STALE_TELEMETRY,
        FaultType.INFEASIBLE_ASSIGNMENT,
        FaultType.SPURIOUS_HANDOVER,
        FaultType.BUFFER_STARVATION_HAZARD,
        FaultType.WEIGHT_INCONSISTENCY,
    ]

    records = []
    num_trials = 60
    np.random.seed(42)

    for target_fault in fault_classes:
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for trial in range(num_trials):
            now = 20.0 + trial * 0.5
            t1 = ServerTelemetry("s1", 25.0, 100.0, 4, 15.0, 0.25, 0.005, True, now)
            t2 = ServerTelemetry("s2", 20.0, 800.0, 16, 50.0, 0.35, 0.005, True, now)
            telem_dict = {"s1": t1, "s2": t2}
            matrix = engine.build_matrices([t1, t2], w_video, now)
            client = VideoClientState("c0", "s1", buffer_occupancy_s=3.0)

            # 50% positive cases (fault injected), 50% negative cases (clean)
            inject = (trial % 2 == 0)

            decision = {
                "action": "SWITCH" if inject else "RETAIN",
                "target_server": "s2" if inject else "s1",
                "decision_margin": 0.12 if inject else 0.01,
                "switching_hurdle": 0.05,
            }

            if inject:
                if target_fault == FaultType.STALE_TELEMETRY:
                    telem_dict["s2"] = ServerTelemetry("s2", 20.0, 800.0, 16, 50.0, 0.35, 0.005, True, now - 4.5)
                elif target_fault == FaultType.INFEASIBLE_ASSIGNMENT:
                    matrix.feasible_mask[1] = False
                elif target_fault == FaultType.SPURIOUS_HANDOVER:
                    decision["decision_margin"] = 0.02  # below hurdle 0.05 + noise 0.0247
                elif target_fault == FaultType.BUFFER_STARVATION_HAZARD:
                    client.buffer_occupancy_s = 0.06   # below switch delay 0.15s
                elif target_fault == FaultType.WEIGHT_INCONSISTENCY:
                    matrix.weight_vector_w = np.array([0.15, 0.15, 0.15, 0.15, 0.15]) # sum 0.75

            report = auditor.audit_decision(f"d_{trial}", client, matrix, decision, telem_dict, now)
            detected = (target_fault in report.detected_faults)

            if inject:
                if detected:
                    tp += 1
                else:
                    fn += 1
            else:
                if detected:
                    fp += 1
                else:
                    tn += 1

        recall = tp / max(1, (tp + fn))
        precision = tp / max(1, (tp + fp))
        specificity = tn / max(1, (tn + fp))
        f1 = 2.0 * (precision * recall) / max(1e-6, (precision + recall))

        records.append({
            "fault_class": target_fault.value,
            "total_trials": num_trials,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
        })

    df_faults = pd.DataFrame(records)
    out_csv = LOGS_DIR / "data_fault_detection_confusion.csv"
    df_faults.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")


def run_faithfulness_benchmark():
    """Evaluate explanation faithfulness monotonicity and counterfactual inversion accuracy."""
    print("2. Running Explanation Faithfulness & Counterfactual Inversion Benchmark...")
    engine = QoSFormalizationEngine()
    verifier = CounterfactualFaithfulnessVerifier(engine)
    w_video, _, _ = get_literature_ahp_profile("video_streaming")
    now = 10.0

    # Test across all standard criteria
    records = []
    perturbation_steps = np.linspace(0.0, 0.35, 8)

    criteria_scenarios = [
        (Criterion.EFFECTIVE_BW, 4.0, 50.0, 25.0, 20.0),
        (Criterion.LATENCY, 15.0, 15.0, 95.0, 20.0),
        (Criterion.SERVER_LOAD, 20.0, 20.0, 20.0, 20.0),
        (Criterion.PACKET_LOSS, 20.0, 20.0, 20.0, 20.0),
    ]

    for crit, bw1, bw2, lat1, lat2 in criteria_scenarios:
        t1 = ServerTelemetry("s1", lat1, 100.0, 4, bw1, 0.20, 0.005, True, now)
        t2 = ServerTelemetry("s2", lat2, 800.0, 16, bw2, 0.30, 0.005, True, now)
        matrix = engine.build_matrices([t1, t2], w_video, now)
        client = VideoClientState("c0", "s1", buffer_occupancy_s=3.0)

        decision = {
            "action": "SWITCH",
            "target_server": "s2",
            "decision_margin": 0.08,
            "switching_hurdle": 0.05,
        }

        report = verifier.verify_faithfulness(client, matrix, decision, crit, num_perturbation_steps=8)
        for step_idx, p_delta in enumerate(report.perturbation_steps):
            records.append({
                "criterion": crit.value,
                "perturbation_delta": round(float(p_delta), 3),
                "resulting_margin": round(float(report.margin_trajectory[step_idx]), 4),
                "is_monotonically_faithful": report.is_monotonically_faithful,
                "counterfactual_inverts_decision": report.counterfactual_inverts_decision,
            })

    df_faith = pd.DataFrame(records)
    out_csv = LOGS_DIR / "data_counterfactual_faithfulness.csv"
    df_faith.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")


def run_latency_overhead_benchmark():
    """Measure microsecond execution latency distribution of the Post-Decision Auditor."""
    print("3. Measuring Auditor Microsecond Execution Overhead (1,000 iterations)...")
    engine = QoSFormalizationEngine()
    auditor = PostDecisionAuditor()
    w_video, _, _ = get_literature_ahp_profile("video_streaming")
    now = 10.0

    t1 = ServerTelemetry("s1", 25.0, 100.0, 4, 15.0, 0.25, 0.005, True, now)
    t2 = ServerTelemetry("s2", 20.0, 800.0, 16, 50.0, 0.35, 0.005, True, now)
    telem_dict = {"s1": t1, "s2": t2}
    matrix = engine.build_matrices([t1, t2], w_video, now)
    client = VideoClientState("c0", "s1", buffer_occupancy_s=3.0)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.08,
        "switching_hurdle": 0.05,
    }

    latencies_us = []
    # Warm-up
    for _ in range(50):
        auditor.audit_decision("d_warm", client, matrix, decision, telem_dict, now)

    # Benchmark 1,000 runs
    for i in range(1000):
        rep = auditor.audit_decision(f"d_bench_{i}", client, matrix, decision, telem_dict, now)
        latencies_us.append(rep.audit_duration_us)

    df_lat = pd.DataFrame({"run_id": range(1000), "latency_us": latencies_us})
    out_csv = LOGS_DIR / "data_auditor_latency_overhead.csv"
    df_lat.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")
    print(f"  Mean Latency: {np.mean(latencies_us):.2f} us | Median: {np.median(latencies_us):.2f} us | 99th Pct: {np.percentile(latencies_us, 99):.2f} us")


def main():
    print("=" * 70)
    print("STEP 5: RUNNING AUDITOR & FAITHFULNESS BENCHMARK")
    print("=" * 70)

    run_fault_injection_benchmark()
    run_faithfulness_benchmark()
    run_latency_overhead_benchmark()

    print("\n[SUCCESS] All Step 5 data files exported successfully.")


if __name__ == "__main__":
    main()
