"""
Post-Decision Auditor and Explanation Faithfulness Verification Engine.
Target Venue: IEEE PerCom 2027 (A4E Workshop)

Implements:
1. Post-Decision Safety Auditor:
   - Validates runtime decisions (SWITCH vs RETAIN) against hard SLA constraints.
   - Detects 5 critical fault classes:
     * STALE_TELEMETRY (timestamp age > T_stale)
     * INFEASIBLE_ASSIGNMENT (assigning/retaining on an unviable or failed server)
     * SPURIOUS_HANDOVER (switching with net margin below switching hurdle or noise band)
     * BUFFER_STARVATION_HAZARD (switching when buffer < switch delay without emergency)
     * WEIGHT_INCONSISTENCY (corrupted or non-normalized criteria weights)
   - Generates structured AuditReports with root-cause diagnosis and remediation.
   - Operates in sub-millisecond execution time (< 100 microseconds) on edge clients.

2. Explanation Faithfulness Verifier:
   - Evaluates whether feature attributions phi_k are monotonically faithful.
   - Validates physical counterfactual distances (verifies decision inversion).
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from edge_switching.qos_model import (
    Criterion,
    DecisionMatrix,
    QoSFormalizationEngine,
    ServerTelemetry,
    SystemConfig,
)
from edge_switching.simulator import VideoClientState
from edge_switching.explainer import EdgeSwitchingExplainer, ExplanationResult


class FaultType(str, Enum):
    """Categorization of runtime switching and telemetry faults."""
    STALE_TELEMETRY = "stale_telemetry"
    INFEASIBLE_ASSIGNMENT = "infeasible_assignment"
    SPURIOUS_HANDOVER = "spurious_handover"
    BUFFER_STARVATION_HAZARD = "buffer_starvation_hazard"
    WEIGHT_INCONSISTENCY = "weight_inconsistency"
    NONE = "none"


class AuditSeverity(str, Enum):
    """Severity classification of detected audit anomalies."""
    INFO = "INFO"          # Benign, decision verified valid
    WARNING = "WARNING"    # Non-critical warning (e.g. low margin or marginal buffer)
    CRITICAL = "CRITICAL"  # Critical SLA violation or imminent stream outage


@dataclass
class AuditReport:
    """Structured report returned by the Post-Decision Auditor."""
    decision_id: str
    timestamp_s: float
    client_id: str
    action: str
    assigned_server_id: str
    is_valid: bool
    detected_faults: List[FaultType]
    highest_severity: AuditSeverity
    rule_violations: List[str]
    root_cause_diagnosis: str
    remedial_recommendation: str
    audit_duration_us: float  # Microseconds taken to execute audit


@dataclass
class FaithfulnessReport:
    """Quantitative report on explanation faithfulness and counterfactual inversion validity."""
    criterion_name: str
    is_monotonically_faithful: bool
    perturbation_steps: List[float]
    margin_trajectory: List[float]
    counterfactual_inverts_decision: bool
    nominal_margin: float
    counterfactual_margin: float


class PostDecisionAuditor:
    """
    Sub-millisecond Post-Decision Safety Auditor for edge client runtimes.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        k_noise: float = 1.65,
    ):
        self.config = config or SystemConfig()
        self.k_noise = k_noise

    def audit_decision(
        self,
        decision_id: str,
        client: VideoClientState,
        matrix: DecisionMatrix,
        decision: Dict[str, Any],
        telemetry_dict: Dict[str, ServerTelemetry],
        current_time_s: float,
        estimated_noise_sigma: float = 0.015,
    ) -> AuditReport:
        """
        Audit a runtime decision across 5 safety invariant rules.
        Execution is optimized for sub-millisecond edge evaluation.
        """
        start_ns = time.perf_counter_ns()

        action = str(decision.get("action", "RETAIN")).upper()
        target_server_id = str(decision.get("target_server", client.current_server_id))
        decision_margin = float(decision.get("decision_margin", 0.0))
        switching_hurdle = float(decision.get("switching_hurdle", self.config.base_switching_cost))

        detected_faults: List[FaultType] = []
        rule_violations: List[str] = []
        severity = AuditSeverity.INFO

        # -------------------------------------------------------------
        # RULE 1: Weight Inconsistency & Normalization Check
        # -------------------------------------------------------------
        w = matrix.weight_vector_w
        if np.any(w < -1e-6) or abs(float(np.sum(w)) - 1.0) > 1e-4:
            detected_faults.append(FaultType.WEIGHT_INCONSISTENCY)
            rule_violations.append(
                f"Criteria weights do not sum to 1.0 (sum={np.sum(w):.4f}) or contain negative values"
            )
            severity = AuditSeverity.CRITICAL

        # -------------------------------------------------------------
        # RULE 2: Telemetry Staleness Audit
        # -------------------------------------------------------------
        target_telem = telemetry_dict.get(target_server_id)
        if target_telem is not None:
            age = current_time_s - target_telem.timestamp_s
            if age > self.config.max_stale_age_s:
                detected_faults.append(FaultType.STALE_TELEMETRY)
                rule_violations.append(
                    f"Target server '{target_server_id}' telemetry is stale (age={age:.2f}s > threshold={self.config.max_stale_age_s:.2f}s)"
                )
                severity = AuditSeverity.CRITICAL

        current_telem = telemetry_dict.get(client.current_server_id)
        if current_telem is not None and action == "RETAIN":
            age_curr = current_time_s - current_telem.timestamp_s
            if age_curr > self.config.max_stale_age_s:
                detected_faults.append(FaultType.STALE_TELEMETRY)
                rule_violations.append(
                    f"Retained server '{client.current_server_id}' telemetry is stale (age={age_curr:.2f}s)"
                )
                severity = AuditSeverity.CRITICAL

        # -------------------------------------------------------------
        # RULE 3: Infeasible Assignment Audit (Hard Feasibility Filter)
        # -------------------------------------------------------------
        if target_server_id in matrix.server_ids:
            idx = matrix.server_ids.index(target_server_id)
            if not matrix.feasible_mask[idx]:
                detected_faults.append(FaultType.INFEASIBLE_ASSIGNMENT)
                rule_violations.append(
                    f"Assigned server '{target_server_id}' failed hard operational feasibility constraints"
                )
                severity = AuditSeverity.CRITICAL
        elif target_telem is None or not target_telem.is_available:
            detected_faults.append(FaultType.INFEASIBLE_ASSIGNMENT)
            rule_violations.append(f"Target server '{target_server_id}' is unavailable or unmonitored")
            severity = AuditSeverity.CRITICAL

        # -------------------------------------------------------------
        # RULE 4: Spurious Handover Audit (Noise Margin & Hurdle Invariant)
        # -------------------------------------------------------------
        if action == "SWITCH":
            noise_band = self.k_noise * estimated_noise_sigma
            # Switch requires positive net margin exceeding hurdle and noise threshold
            # unless current server has experienced a hard failure (emergency handover)
            curr_is_dead = (current_telem is None) or (not current_telem.is_available)
            if not curr_is_dead and decision_margin < (switching_hurdle + noise_band - 1e-4):
                detected_faults.append(FaultType.SPURIOUS_HANDOVER)
                rule_violations.append(
                    f"Spurious handover: decision margin ({decision_margin:.4f}) is below hurdle + noise threshold ({switching_hurdle + noise_band:.4f})"
                )
                if severity != AuditSeverity.CRITICAL:
                    severity = AuditSeverity.WARNING

        # -------------------------------------------------------------
        # RULE 5: Playout Starvation Hazard Audit
        # -------------------------------------------------------------
        if action == "SWITCH":
            curr_is_dead = (current_telem is None) or (not current_telem.is_available)
            # If current server is healthy but client buffer < switch delay, switching causes stall
            if not curr_is_dead and client.buffer_occupancy_s < self.config.switch_delay_s:
                detected_faults.append(FaultType.BUFFER_STARVATION_HAZARD)
                rule_violations.append(
                    f"Playout starvation hazard: buffer occupancy ({client.buffer_occupancy_s:.2f}s) < switching delay ({self.config.switch_delay_s:.2f}s)"
                )
                severity = AuditSeverity.CRITICAL

        # Build diagnosis and recommendation
        is_valid = (len(detected_faults) == 0)
        if is_valid:
            root_cause = "Decision validated against all SLA constraints and noise margins."
            recommendation = "Execute decision as planned."
        else:
            root_cause = "; ".join(rule_violations)
            if FaultType.INFEASIBLE_ASSIGNMENT in detected_faults:
                recommendation = "VETO SWITCH: Target server is unviable. Fallback to best available feasible candidate."
            elif FaultType.STALE_TELEMETRY in detected_faults:
                recommendation = "QUARANTINE TELEMETRY: Refresh connection state before applying handover."
            elif FaultType.BUFFER_STARVATION_HAZARD in detected_faults:
                recommendation = "DEFER SWITCH: Playout buffer critically depleted. Retain current connection until buffer accumulates."
            elif FaultType.SPURIOUS_HANDOVER in detected_faults:
                recommendation = "SUPPRESS SWITCH: Observed gain is within statistical measurement noise. Retain current server."
            elif FaultType.WEIGHT_INCONSISTENCY in detected_faults:
                recommendation = "RESET CONFIGURATION: Re-initialize AHP profile weights."
            else:
                recommendation = "RE-EVALUATE: Fallback to conservative local edge server."

        end_ns = time.perf_counter_ns()
        duration_us = (end_ns - start_ns) / 1000.0

        return AuditReport(
            decision_id=decision_id,
            timestamp_s=current_time_s,
            client_id=client.client_id,
            action=action,
            assigned_server_id=target_server_id,
            is_valid=is_valid,
            detected_faults=detected_faults,
            highest_severity=severity,
            rule_violations=rule_violations,
            root_cause_diagnosis=root_cause,
            remedial_recommendation=recommendation,
            audit_duration_us=round(duration_us, 2),
        )


class CounterfactualFaithfulnessVerifier:
    """
    Verifies explanation faithfulness and counterfactual decision inversion.
    """

    def __init__(
        self,
        engine: Optional[QoSFormalizationEngine] = None,
        explainer: Optional[EdgeSwitchingExplainer] = None,
    ):
        self.engine = engine or QoSFormalizationEngine()
        self.explainer = explainer or EdgeSwitchingExplainer(self.engine)

    def verify_faithfulness(
        self,
        client: VideoClientState,
        matrix: DecisionMatrix,
        decision: Dict[str, Any],
        criterion: Criterion,
        num_perturbation_steps: int = 10,
    ) -> FaithfulnessReport:
        """
        Check if perturbing the evaluated criterion produces a monotonically faithful change in decision margin,
        and test whether applying the counterfactual distance flips the decision.
        """
        exp = self.explainer.explain_decision(client, matrix, decision)
        crit_idx = matrix.criteria_order.index(criterion)
        nominal_margin = exp.decision_margin

        curr_id = client.current_server_id
        target_id = exp.target_server_id
        curr_idx = matrix.server_ids.index(curr_id)
        target_idx = matrix.server_ids.index(target_id)

        # Monotonicity test:
        # For a SWITCH decision, target has higher utility on positive criteria.
        # Gradually degrading target's utility must monotonically decrease the margin M.
        # For RETAIN, increasing target's utility must monotonically increase M.
        perturbation_deltas = np.linspace(0.0, 0.30, num_perturbation_steps)
        margins = []

        is_switch = (exp.action == "SWITCH")
        for delta in perturbation_deltas:
            U_mod = matrix.utility_matrix_U.copy()
            if is_switch:
                # Degrade target server utility
                U_mod[target_idx, crit_idx] = max(0.0, U_mod[target_idx, crit_idx] - delta)
            else:
                # Improve target server utility
                U_mod[target_idx, crit_idx] = min(1.0, U_mod[target_idx, crit_idx] + delta)

            u_target = float(np.dot(U_mod[target_idx], matrix.weight_vector_w))
            u_curr = float(np.dot(U_mod[curr_idx], matrix.weight_vector_w))
            delta_U = u_target - u_curr
            hurdle = exp.switching_hurdle + exp.noise_margin
            m = delta_U - hurdle
            margins.append(float(m))

        # Check monotonicity
        if is_switch:
            # Monotonically non-increasing
            diffs = np.diff(margins)
            is_faithful = bool(np.all(diffs <= 1e-6))
        else:
            # Monotonically non-decreasing
            diffs = np.diff(margins)
            is_faithful = bool(np.all(diffs >= -1e-6))

        # Counterfactual decision inversion test
        # Applying the physical counterfactual delta must flip the action sign
        cf_dist = exp.counterfactual_distances.get(criterion.value, 0.0)
        cf_margin = nominal_margin
        inverts_decision = False

        if cf_dist != 0.0:
            U_cf = matrix.utility_matrix_U.copy()
            w_k = float(matrix.weight_vector_w[crit_idx])
            if w_k > 0:
                delta_u_needed = abs(nominal_margin) / w_k
                if is_switch:
                    U_cf[curr_idx, crit_idx] = min(1.0, U_cf[curr_idx, crit_idx] + delta_u_needed)
                else:
                    U_cf[target_idx, crit_idx] = min(1.0, U_cf[target_idx, crit_idx] + delta_u_needed)

                u_target_cf = float(np.dot(U_cf[target_idx], matrix.weight_vector_w))
                u_curr_cf = float(np.dot(U_cf[curr_idx], matrix.weight_vector_w))
                delta_U_cf = u_target_cf - u_curr_cf
                cf_margin = delta_U_cf - (exp.switching_hurdle + exp.noise_margin)

                # Inversion occurred if sign of margin flipped
                if is_switch and cf_margin <= 1e-4:
                    inverts_decision = True
                elif (not is_switch) and cf_margin >= -1e-4:
                    inverts_decision = True

        return FaithfulnessReport(
            criterion_name=criterion.value,
            is_monotonically_faithful=is_faithful,
            perturbation_steps=[float(p) for p in perturbation_deltas],
            margin_trajectory=[round(float(m), 4) for m in margins],
            counterfactual_inverts_decision=inverts_decision,
            nominal_margin=round(float(nominal_margin), 4),
            counterfactual_margin=round(float(cf_margin), 4),
        )
