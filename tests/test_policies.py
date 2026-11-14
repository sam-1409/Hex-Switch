import pytest
import numpy as np

from edge_switching import (
    Criterion,
    DecisionMatrix,
    QoSFormalizationEngine,
    ServerTelemetry,
    SystemConfig,
    VideoClientState,
    NearestServerPolicy,
    MinLatencyPolicy,
    MaxTotalBandwidthPolicy,
    MaxEffectiveBandwidthPolicy,
    AhpTopsisPolicy,
    HysteresisOnlyPolicy,
    QoEGreedyPolicy,
    ProposedMarginPolicy,
    run_scenario,
)

def make_test_matrix():
    engine = QoSFormalizationEngine()
    now = 100.0
    telemetries = [
        ServerTelemetry("s1", 15.0, 300.0, 14, 30.0, 0.3, 0.001, True, now),
        ServerTelemetry("s2", 35.0, 800.0, 15, 60.0, 0.2, 0.000, True, now),
        ServerTelemetry("s3", 80.0, 1500.0, 29, 100.0, 0.1, 0.000, True, now),
    ]
    weights = {c: 0.20 for c in Criterion}
    matrix = engine.build_matrices(telemetries, weights, now)
    return engine, matrix

def test_nearest_server_policy():
    engine, matrix = make_test_matrix()
    client = VideoClientState("c1", "s2")
    policy = NearestServerPolicy("s1")
    dec = policy.decide(client, matrix, engine)
    assert dec["target_server"] == "s1"
    assert dec["action"] == "SWITCH"

def test_min_latency_policy():
    engine, matrix = make_test_matrix()
    client = VideoClientState("c1", "s2")
    policy = MinLatencyPolicy()
    dec = policy.decide(client, matrix, engine)
    assert dec["target_server"] == "s1"

def test_max_total_bandwidth_policy():
    engine, matrix = make_test_matrix()
    client = VideoClientState("c1", "s1")
    policy = MaxTotalBandwidthPolicy()
    dec = policy.decide(client, matrix, engine)
    assert dec["target_server"] == "s3"

def test_max_effective_bandwidth_policy():
    engine, matrix = make_test_matrix()
    client = VideoClientState("c1", "s1")
    policy = MaxEffectiveBandwidthPolicy()
    dec = policy.decide(client, matrix, engine)
    assert dec["target_server"] in ["s2", "s3"]

def test_ahp_topsis_policy():
    engine, matrix = make_test_matrix()
    client = VideoClientState("c1", "s1")
    policy = AhpTopsisPolicy()
    dec = policy.decide(client, matrix, engine)
    assert "closeness_scores" in dec
    assert dec["target_server"] in ["s1", "s2", "s3"]

def test_hysteresis_only_policy():
    engine, matrix = make_test_matrix()
    client = VideoClientState("c1", "s1")
    policy = HysteresisOnlyPolicy(hysteresis_threshold=0.25)
    dec = policy.decide(client, matrix, engine)
    assert dec["action"] == "RETAIN"
    assert dec["target_server"] == "s1"

def test_qoe_greedy_policy():
    engine, matrix = make_test_matrix()
    client = VideoClientState("c1", "s1")
    policy = QoEGreedyPolicy()
    dec = policy.decide(client, matrix, engine)
    assert dec["target_server"] in ["s1", "s2"]

def test_proposed_margin_policy_noise_suppression():
    policy = ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65, min_dwell_steps=4)
    df = run_scenario("scenario_a", duration_s=15.0, dt_s=0.5, num_clients=1, seed=42, policy=policy)
    switches = df[df["action"] == "SWITCH"]
    assert len(switches) == 0
    assert all(df["client_server"] == "s1")

def test_proposed_margin_policy_emergency_override():
    engine = QoSFormalizationEngine()
    now = 50.0
    telemetries = [
        ServerTelemetry("s1", 15.0, 300.0, 10, 30.0, 0.3, 0.001, False, now),
        ServerTelemetry("s2", 35.0, 800.0, 10, 60.0, 0.2, 0.000, True, now),
    ]
    weights = {c: 0.20 for c in Criterion}
    matrix = engine.build_matrices(telemetries, weights, now)

    client = VideoClientState("c1", "s1")
    policy = ProposedMarginPolicy(min_dwell_steps=10)
    dec = policy.decide(client, matrix, engine)
    assert dec["action"] == "SWITCH"
    assert dec["target_server"] == "s2"

def test_baseline_suite_initialization():
    from edge_switching.policies import get_baseline_suite
    suite = get_baseline_suite()
    assert len(suite) == 8
    assert "Proposed-Margin" in suite
    assert "AHP-TOPSIS" in suite
    assert "Static-Hysteresis" in suite