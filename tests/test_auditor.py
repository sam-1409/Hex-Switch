import pytest
import numpy as np

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
    AuditReport,
    AuditSeverity,
    CounterfactualFaithfulnessVerifier,
    FaultType,
    PostDecisionAuditor,
)

def make_clean_test_setup():
    engine = QoSFormalizationEngine()
    w_video, _, _ = get_literature_ahp_profile("video_streaming")
    now = 10.0
    t1 = ServerTelemetry("s1", 20.0, 100.0, 4, 20.0, 0.20, 0.005, True, now)
    t2 = ServerTelemetry("s2", 35.0, 800.0, 16, 50.0, 0.60, 0.005, True, now)
    matrix = engine.build_matrices([t1, t2], w_video, now)
    client = VideoClientState("c0", "s1", buffer_occupancy_s=3.0)
    telem_dict = {"s1": t1, "s2": t2}
    return engine, matrix, client, telem_dict, now


def test_auditor_validates_clean_decision():
    engine, matrix, client, telem_dict, now = make_clean_test_setup()
    auditor = PostDecisionAuditor()

    decision = {
        "action": "RETAIN",
        "target_server": "s1",
        "decision_margin": 0.01,
        "switching_hurdle": 0.05,
        "reason": "Within noise margin",
    }

    report = auditor.audit_decision("d_clean", client, matrix, decision, telem_dict, now)
    assert report.is_valid is True
    assert len(report.detected_faults) == 0
    assert report.highest_severity == AuditSeverity.INFO
    assert report.audit_duration_us < 1000.0  # Must execute in under 1 ms

def test_auditor_detects_stale_telemetry():
    engine, matrix, client, telem_dict, now = make_clean_test_setup()
    auditor = PostDecisionAuditor()
    telem_dict["s2"] = ServerTelemetry("s2", 35.0, 800.0, 16, 50.0, 0.60, 0.005, True, now - 5.0)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.15,
        "switching_hurdle": 0.05,
    }

    report = auditor.audit_decision("d_stale", client, matrix, decision, telem_dict, now)
    assert report.is_valid is False
    assert FaultType.STALE_TELEMETRY in report.detected_faults
    assert report.highest_severity == AuditSeverity.CRITICAL
    assert "QUARANTINE TELEMETRY" in report.remedial_recommendation

def test_auditor_detects_infeasible_assignment():
    engine, matrix, client, telem_dict, now = make_clean_test_setup()
    auditor = PostDecisionAuditor()
    matrix.feasible_mask[1] = False

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.20,
        "switching_hurdle": 0.05,
    }

    report = auditor.audit_decision("d_infeas", client, matrix, decision, telem_dict, now)
    assert report.is_valid is False
    assert FaultType.INFEASIBLE_ASSIGNMENT in report.detected_faults
    assert report.highest_severity == AuditSeverity.CRITICAL
    assert "VETO SWITCH" in report.remedial_recommendation

def test_auditor_detects_spurious_handover():
    engine, matrix, client, telem_dict, now = make_clean_test_setup()
    auditor = PostDecisionAuditor(k_noise=1.65)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.02,
        "switching_hurdle": 0.05,
    }

    report = auditor.audit_decision("d_spurious", client, matrix, decision, telem_dict, now)
    assert report.is_valid is False
    assert FaultType.SPURIOUS_HANDOVER in report.detected_faults
    assert "SUPPRESS SWITCH" in report.remedial_recommendation

def test_auditor_detects_buffer_starvation_hazard():
    engine, matrix, client, telem_dict, now = make_clean_test_setup()
    auditor = PostDecisionAuditor()
    client_low_buffer = VideoClientState("c0", "s1", buffer_occupancy_s=0.05)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.25,
        "switching_hurdle": 0.05,
    }

    report = auditor.audit_decision("d_starve", client_low_buffer, matrix, decision, telem_dict, now)
    assert report.is_valid is False
    assert FaultType.BUFFER_STARVATION_HAZARD in report.detected_faults
    assert report.highest_severity == AuditSeverity.CRITICAL
    assert "DEFER SWITCH" in report.remedial_recommendation

def test_auditor_detects_weight_inconsistency():
    engine, matrix, client, telem_dict, now = make_clean_test_setup()
    auditor = PostDecisionAuditor()
    matrix.weight_vector_w = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    decision = {
        "action": "RETAIN",
        "target_server": "s1",
        "decision_margin": 0.01,
        "switching_hurdle": 0.05,
    }

    report = auditor.audit_decision("d_corrupt_w", client, matrix, decision, telem_dict, now)
    assert report.is_valid is False
    assert FaultType.WEIGHT_INCONSISTENCY in report.detected_faults
    assert "RESET CONFIGURATION" in report.remedial_recommendation

def test_counterfactual_faithfulness_and_monotonicity():
    engine = QoSFormalizationEngine()
    w_video, _, _ = get_literature_ahp_profile("video_streaming")
    now = 10.0
    t1 = ServerTelemetry("s1", 25.0, 100.0, 4, 4.0, 0.25, 0.005, True, now)
    t2 = ServerTelemetry("s2", 20.0, 800.0, 16, 50.0, 0.30, 0.005, True, now)
    matrix = engine.build_matrices([t1, t2], w_video, now)
    client = VideoClientState("c0", "s1", buffer_occupancy_s=3.0)

    decision = {
        "action": "SWITCH",
        "target_server": "s2",
        "decision_margin": 0.06,
        "switching_hurdle": 0.05,
    }

    verifier = CounterfactualFaithfulnessVerifier(engine)
    report = verifier.verify_faithfulness(client, matrix, decision, Criterion.EFFECTIVE_BW)

    assert report.is_monotonically_faithful is True
    margins = report.margin_trajectory
    assert margins[0] >= margins[-1]
    assert report.counterfactual_inverts_decision is True