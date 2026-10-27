"""
Unit tests for Step 1: Application and QoS Formalization.

Verifies:
- Effective bandwidth b_eff with the (+1)-user effect and path bottleneck.
- Feasibility pruning (hard constraints vs soft objectives).
- Normalization and matrix dimensions (X vs U vs w).
- Multi-criteria trade-off (best value on one metric does not make the best server).
- Switching cost hurdle lambda_switch and buffer starvation protection.
"""

import pytest
import numpy as np

from edge_switching import (
    Criterion,
    DecisionMatrix,
    QoSFormalizationEngine,
    ServerTelemetry,
    SystemConfig,
)


def test_effective_bandwidth_plus_one_user_and_bottleneck():
    """Verify b_eff = min(C_path, C_srv / (N + 1)) strictly models contention."""
    # Scenario 1: Path bottleneck dominates
    t1 = ServerTelemetry(
        server_id="s1",
        latency_ms=20.0,
        server_capacity_mbps=1000.0,
        active_users=9,  # Fair share = 1000 / 10 = 100 Mbps
        path_bottleneck_mbps=15.0,  # Access link bottleneck = 15 Mbps
        server_load=0.20,
        packet_loss=0.0,
        is_available=True,
        timestamp_s=100.0,
    )
    assert t1.compute_effective_bandwidth() == 15.0

    # Scenario 2: Server contention (+1 user effect) dominates
    t2 = ServerTelemetry(
        server_id="s2",
        latency_ms=20.0,
        server_capacity_mbps=100.0,
        active_users=19,  # Fair share = 100 / (19 + 1) = 5.0 Mbps
        path_bottleneck_mbps=50.0,
        server_load=0.85,
        packet_loss=0.0,
        is_available=True,
        timestamp_s=100.0,
    )
    assert t2.compute_effective_bandwidth() == 5.0


def test_hard_feasibility_constraints():
    """Verify feasibility filter rejects servers violating safety SLOs."""
    engine = QoSFormalizationEngine()
    now = 100.0

    # 1. Latency SLO violation (> 120 ms)
    t_high_lat = ServerTelemetry("s_lat", 140.0, 500.0, 5, 50.0, 0.4, 0.0, True, now)
    is_feas, reasons = engine.check_feasibility(t_high_lat, now)
    assert not is_feas
    assert any("Latency SLO violated" in r for r in reasons)

    # 2. Stale telemetry (> 2.0 s old)
    t_stale = ServerTelemetry("s_stale", 30.0, 500.0, 5, 50.0, 0.4, 0.0, True, now - 3.5)
    is_feas, reasons = engine.check_feasibility(t_stale, now)
    assert not is_feas
    assert any("Telemetry stale" in r for r in reasons)

    # 3. Availability down (A = 0)
    t_down = ServerTelemetry("s_down", 25.0, 500.0, 5, 50.0, 0.4, 0.0, False, now)
    is_feas, reasons = engine.check_feasibility(t_down, now)
    assert not is_feas
    assert any("unavailable" in r for r in reasons)

    # 4. Valid server satisfies all constraints
    t_valid = ServerTelemetry("s_ok", 30.0, 500.0, 5, 50.0, 0.3, 0.001, True, now)
    is_feas, _ = engine.check_feasibility(t_valid, now)
    assert is_feas


def test_matrix_dimensions_and_utility_polarities():
    """Verify X vs U vs w dimensions and [0, 1] utility bounds."""
    engine = QoSFormalizationEngine()
    now = 50.0

    telemetries = [
        ServerTelemetry("s1", 20.0, 500.0, 10, 50.0, 0.2, 0.0, True, now),
        ServerTelemetry("s2", 60.0, 200.0, 5, 20.0, 0.5, 0.02, True, now),
        ServerTelemetry("s3", 100.0, 100.0, 2, 10.0, 0.8, 0.05, True, now),
    ]
    weights = {
        Criterion.LATENCY: 0.35,
        Criterion.EFFECTIVE_BW: 0.30,
        Criterion.SERVER_LOAD: 0.15,
        Criterion.PACKET_LOSS: 0.15,
        Criterion.AVAILABILITY: 0.05,
    }

    matrix = engine.build_matrices(telemetries, weights, now)
    assert matrix.raw_matrix_X.shape == (3, 5)
    assert matrix.utility_matrix_U.shape == (3, 5)
    assert matrix.weight_vector_w.shape == (5,)
    assert np.isclose(np.sum(matrix.weight_vector_w), 1.0)

    # All normalized utilities must be strictly in [0, 1]
    assert np.all(matrix.utility_matrix_U >= 0.0)
    assert np.all(matrix.utility_matrix_U <= 1.0)

    # Polarity check: lower latency must yield higher utility
    # s1 (20ms) should have higher latency utility than s3 (100ms)
    lat_col = matrix.criteria_order.index(Criterion.LATENCY)
    assert matrix.utility_matrix_U[0, lat_col] > matrix.utility_matrix_U[2, lat_col]


def test_best_on_one_metric_does_not_make_best_server():
    """
    Advisor Requirement: A server best on a single metric (e.g. latency)
    must not win if it suffers on other key criteria (packet loss, load).
    """
    engine = QoSFormalizationEngine()
    now = 10.0

    # s1 has lowest latency (12 ms) but poor packet loss (6%) and high load (85%)
    s1 = ServerTelemetry("s1_low_lat", 12.0, 300.0, 20, 10.0, 0.85, 0.06, True, now)
    # s2 has higher latency (30 ms) but zero loss, low load (20%), and strong bandwidth
    s2 = ServerTelemetry("s2_balanced", 30.0, 500.0, 4, 40.0, 0.20, 0.00, True, now)

    weights = {
        Criterion.LATENCY: 0.30,
        Criterion.EFFECTIVE_BW: 0.30,
        Criterion.SERVER_LOAD: 0.15,
        Criterion.PACKET_LOSS: 0.20,
        Criterion.AVAILABILITY: 0.05,
    }
    matrix = engine.build_matrices([s1, s2], weights, now)
    decision = engine.evaluate_switching_decision(matrix, current_server_id="s1_low_lat")

    # Balanced server s2 must beat s1 in multi-criteria utility
    assert decision["static_utilities"]["s2_balanced"] > decision["static_utilities"]["s1_low_lat"]
    assert decision["action"] == "SWITCH"
    assert decision["target_server"] == "s2_balanced"


def test_switching_cost_hurdle_prevents_marginal_churn():
    """
    Verify switching hurdle lambda_switch prevents switching when candidate
    offers only marginal, jitter-level improvements (hysteresis principle).
    """
    engine = QoSFormalizationEngine(SystemConfig(base_switching_cost=0.08))
    now = 20.0

    # Current server s_curr has good utility (~0.75)
    s_curr = ServerTelemetry("s_curr", 35.0, 300.0, 10, 25.0, 0.4, 0.005, True, now)
    # Candidate s_cand is marginally better (e.g. 30ms latency vs 35ms)
    s_cand = ServerTelemetry("s_cand", 30.0, 300.0, 10, 25.0, 0.4, 0.005, True, now)

    weights = {c: 0.20 for c in Criterion}
    matrix = engine.build_matrices([s_curr, s_cand], weights, now)

    decision = engine.evaluate_switching_decision(matrix, current_server_id="s_curr", buffer_occupancy_s=2.5)

    # Static utility of s_cand is slightly higher:
    assert decision["static_utilities"]["s_cand"] > decision["static_utilities"]["s_curr"]
    # BUT net utility after subtracting hurdle 0.08 is lower:
    assert decision["net_utilities"]["s_cand"] < decision["net_utilities"]["s_curr"]
    # Therefore, action must be RETAIN!
    assert decision["action"] == "RETAIN"
    assert decision["target_server"] == "s_curr"
    assert decision["decision_margin"] > 0.0


def test_buffer_starvation_amplifies_switching_hurdle():
    """Verify low buffer (< 150 ms) penalizes switching to prevent stalls."""
    config = SystemConfig(base_switching_cost=0.05, switch_delay_s=0.20, starvation_penalty_mult=2.0)
    engine = QoSFormalizationEngine(config)
    now = 30.0

    s_curr = ServerTelemetry("s1", 40.0, 200.0, 10, 15.0, 0.5, 0.01, True, now)
    s_cand = ServerTelemetry("s2", 20.0, 400.0, 5, 30.0, 0.2, 0.00, True, now)

    weights = {c: 0.20 for c in Criterion}
    matrix = engine.build_matrices([s_curr, s_cand], weights, now)

    # With healthy buffer (1.5s), hurdle is base (0.05)
    healthy_dec = engine.evaluate_switching_decision(matrix, "s1", buffer_occupancy_s=1.5)
    assert healthy_dec["switching_hurdle"] == 0.05

    # With starvation buffer (0.10s < switch_delay_s 0.20s), hurdle doubles to 0.10
    starving_dec = engine.evaluate_switching_decision(matrix, "s1", buffer_occupancy_s=0.10)
    assert starving_dec["switching_hurdle"] == 0.10
