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
        ServerTelemetry("s1", 15.0, 25.0, 9, 20.0, 0.05, 0.000, True, now),
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
    engine, matrix, client = make_test_setup()
    explainer = EdgeSwitchingExplainer(engine)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.12,
        "switching_hurdle": 0.05,
    }
    exp = explainer.explain_decision(client, matrix, decision)

    static_utils = matrix.utility_matrix_U @ matrix.weight_vector_w
    expected_delta_u = static_utils[1] - static_utils[0]

    sum_attributions = sum(exp.feature_attributions.values())
    assert np.isclose(sum_attributions, expected_delta_u), (
        f"Efficiency axiom violated: sum(phi)={sum_attributions:.4f} != Delta U={expected_delta_u:.4f}"
    )

def test_dominant_criterion_identification():
    engine, matrix, client = make_test_setup()
    explainer = EdgeSwitchingExplainer(engine)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.15,
        "switching_hurdle": 0.05,
    }
    exp = explainer.explain_decision(client, matrix, decision)

    assert exp.dominant_criterion == Criterion.EFFECTIVE_BW.value
    assert exp.feature_attributions[Criterion.EFFECTIVE_BW.value] > 0
    assert exp.feature_attributions[Criterion.LATENCY.value] < 0

def test_counterfactual_distance_calculation():
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
    engine = QoSFormalizationEngine()
    now = 20.0
    telemetries = [
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

def test_pre_decision_and_post_decision_and_feedback():
    engine, matrix, client = make_test_setup()
    explainer = EdgeSwitchingExplainer(engine)

    pre_exp = explainer.explain_pre_decision(matrix)
    assert len(pre_exp.server_rankings) == 2
    assert "s1" in pre_exp.composite_scores
    assert "s2" in pre_exp.composite_scores
    assert "effective_bw" in pre_exp.criterion_contributions["s1"]

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.12,
        "switching_hurdle": 0.05,
    }
    telem_dict = {
        "s1": ServerTelemetry("s1", 15.0, 25.0, 9, 20.0, 0.05, 0.000, True, 10.0),
        "s2": ServerTelemetry("s2", 35.0, 400.0, 15, 60.0, 0.05, 0.000, True, 10.0),
    }
    post_exp = explainer.explain_post_decision(client, matrix, decision, telem_dict)
    assert post_exp.selected_server_id == "s2"
    assert "s1" in post_exp.alternative_server_ids
    assert post_exp.qos_parameter_values["s2"]["effective_bw_mbps"] > 0
    assert post_exp.is_auditor_verified is True

    initial_weights = {c: 0.20 for c in Criterion}
    updated_weights = explainer.generate_weight_feedback(
        initial_weights,
        user_feedback_action="prioritize_quality"
    )
    assert np.isclose(sum(updated_weights.values()), 1.0)
    assert updated_weights[Criterion.EFFECTIVE_BW] > initial_weights[Criterion.EFFECTIVE_BW]