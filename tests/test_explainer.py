"""
Unit tests for XAI explanation pipeline:
Validates Shapley additive efficiency axiom, counterfactual margins, and contrastive narratives.
"""

import numpy as np
import pytest

from edge_switching.qos_model import (
    Criterion,
    DecisionMatrix,
    QoSFormalizationEngine,
    ServerTelemetry,
    SystemConfig,
)
from edge_switching.simulator import VideoClientState
from edge_switching.policies import ProposedMarginPolicy
from edge_switching.explainer import EdgeSwitchingExplainer, ExplanationResult


def make_test_setup():
    engine = QoSFormalizationEngine()
    now = 10.0
    telemetries = [
        # s1: low latency (15ms) but congested/low bandwidth (25 Mbps / 10 = 2.5 Mbps)
        ServerTelemetry("s1", 15.0, 25.0, 9, 20.0, 0.05, 0.000, True, now),
        # s2: higher latency (35ms) but high capacity (400 Mbps / 16 = 25.0 Mbps)
        ServerTelemetry("s2", 35.0, 400.0, 15, 60.0, 0.05, 0.000, True, now),
    ]
    weights = {
        Criterion.LATENCY: 0.25,
        Criterion.EFFECTIVE_BW: 0.40,
        Criterion.SERVER_LOAD: 0.15,
        Criterion.PACKET_LOSS: 0.10,
        Criterion.AVAILABILITY: 0.10,
    }
    matrix = engine.build_matrices(telemetries, weights, now)
    client = VideoClientState("c1", "s1", buffer_occupancy_s=4.0)
    return engine, matrix, client


def test_additive_efficiency_axiom():
    """Exact feature attributions must sum to Delta U (Shapley efficiency axiom)."""
    engine, matrix, client = make_test_setup()
    explainer = EdgeSwitchingExplainer(engine)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.12,
        "switching_hurdle": 0.05,
    }
    exp = explainer.explain_decision(client, matrix, decision)

    # Calculate actual Delta U between s2 and s1
    static_utils = matrix.utility_matrix_U @ matrix.weight_vector_w
    expected_delta_u = static_utils[1] - static_utils[0]

    sum_attributions = sum(exp.feature_attributions.values())
    assert np.isclose(sum_attributions, expected_delta_u), (
        f"Efficiency axiom violated: sum(phi)={sum_attributions:.4f} != Delta U={expected_delta_u:.4f}"
    )


def test_dominant_criterion_identification():
    """Explainer must correctly identify the dominant feature driving the switch."""
    engine, matrix, client = make_test_setup()
    explainer = EdgeSwitchingExplainer(engine)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.15,
        "switching_hurdle": 0.05,
    }
    exp = explainer.explain_decision(client, matrix, decision)

    # s2 has much higher bandwidth (+30 Mbps, weighted 0.40), which dominates the latency trade-off
    assert exp.dominant_criterion == Criterion.EFFECTIVE_BW.value
    assert exp.feature_attributions[Criterion.EFFECTIVE_BW.value] > 0
    # Latency attribution should be negative (s2 has worse latency than s1)
    assert exp.feature_attributions[Criterion.LATENCY.value] < 0


def test_counterfactual_distance_calculation():
    """Counterfactual distances must specify positive physical improvements required to flip decision."""
    engine, matrix, client = make_test_setup()
    explainer = EdgeSwitchingExplainer(engine)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.10,
        "switching_hurdle": 0.05,
    }
    exp = explainer.explain_decision(client, matrix, decision)

    cf_bw = exp.counterfactual_distances.get(Criterion.EFFECTIVE_BW.value)
    cf_lat = exp.counterfactual_distances.get(Criterion.LATENCY.value)

    assert cf_bw is not None and cf_bw > 0.0
    assert cf_lat is not None and cf_lat > 0.0


def test_emergency_handover_explanation():
    """Failure handover must clearly declare operational constraint violation."""
    engine = QoSFormalizationEngine()
    now = 20.0
    telemetries = [
        # s1 crashed
        ServerTelemetry("s1", 15.0, 100.0, 4, 20.0, 0.05, 0.000, False, now),
        ServerTelemetry("s2", 35.0, 800.0, 15, 60.0, 0.2, 0.000, True, now),
    ]
    weights = {c: 0.20 for c in Criterion}
    matrix = engine.build_matrices(telemetries, weights, now)
    client = VideoClientState("c1", "s1")

    explainer = EdgeSwitchingExplainer(engine)
    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 1.0,
        "switching_hurdle": 0.05,
        "reason": "Emergency migration: current server violates feasibility",
    }
    exp = explainer.explain_decision(client, matrix, decision)

    assert exp.is_emergency is True
    assert "EMERGENCY SWITCH" in exp.natural_language_summary
    assert "feasibility" in exp.natural_language_summary


def test_noise_suppression_explanation():
    """Noise-suppressed retention must explain that candidate gain falls within confidence margin."""
    engine, matrix, client = make_test_setup()
    explainer = EdgeSwitchingExplainer(engine)

    decision = {
        "action": "RETAIN",
        "target_server": "s1",
        "decision_margin": 0.02,
        "switching_hurdle": 0.05,
        "reason": "Gain 0.03 within telemetry noise margin (0.05)",
    }
    exp = explainer.explain_decision(client, matrix, decision)

    assert exp.is_noise_suppressed is True
    assert "RETAIN" in exp.natural_language_summary
    assert "noise" in exp.natural_language_summary.lower()
