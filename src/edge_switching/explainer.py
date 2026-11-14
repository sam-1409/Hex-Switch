from dataclasses import dataclass, field
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

@dataclass
class ExplanationResult:
    action: str                                     # "SWITCH", "RETAIN", "EMERGENCY_SWITCH"
    current_server_id: str
    target_server_id: str
    decision_margin: float                         # Net margin over hurdle + noise
    switching_hurdle: float                        # lambda_switch
    noise_margin: float                            # k * sigma
    feature_attributions: Dict[str, float]         # Criterion name -> phi_k
    dominant_criterion: str                        # Criterion contributing most to decision
    counterfactual_distances: Dict[str, float]     # Min raw physical delta to flip decision
    natural_language_summary: str                  # Human-readable justification
    is_noise_suppressed: bool
    is_emergency: bool


@dataclass
class PreDecisionExplanation:
    server_rankings: List[str]                             # Ranked candidate server IDs
    composite_scores: Dict[str, float]                     # Server ID -> overall utility U(s)
    criterion_contributions: Dict[str, Dict[str, float]]   # Server ID -> {Criterion -> w_k * u_k(s)}
    feasible_candidates: List[str]                         # Viable candidates passing hard SLO constraints
    infeasible_candidates: List[str]                       # Candidates rejected by feasibility filter


@dataclass
class PostDecisionComparison:
    selected_server_id: str
    alternative_server_ids: List[str]
    action: str
    qos_parameter_values: Dict[str, Dict[str, float]]      # Server ID -> {latency, bandwidth, load, loss, availability}
    qos_utility_values: Dict[str, Dict[str, float]]        # Server ID -> {criterion -> normalized utility}
    counterfactual_distances: Dict[str, float]             # Minimal parameter delta to flip decision
    is_auditor_verified: bool
    audit_diagnosis: str
    recommended_feedback: Optional[str] = None

class EdgeSwitchingExplainer:
    def __init__(
        self,
        engine: Optional[QoSFormalizationEngine] = None,
        k_noise: float = 1.65,
        window_size: int = 10,
    ):
        self.engine = engine or QoSFormalizationEngine()
        self.k_noise = k_noise
        self.window_size = window_size

    def explain_decision(
        self,
        client: VideoClientState,
        matrix: DecisionMatrix,
        decision: Dict[str, Any],
        telemetry_history: Optional[List[Dict[str, ServerTelemetry]]] = None,
    ) -> ExplanationResult:
        curr_id = client.current_server_id
        target_id = str(decision.get("target_server", curr_id))
        action = str(decision.get("action", "RETAIN"))

        curr_idx = matrix.server_ids.index(curr_id) if curr_id in matrix.server_ids else -1
        target_idx = matrix.server_ids.index(target_id) if target_id in matrix.server_ids else -1

        curr_feas = (curr_idx >= 0 and matrix.feasible_mask[curr_idx])
        target_feas = (target_idx >= 0 and matrix.feasible_mask[target_idx])

        hurdle = float(decision.get("switching_hurdle", self.engine.config.base_switching_cost))

        # Estimate noise standard deviation
        sigma_est = 0.015
        if telemetry_history and len(telemetry_history) >= 3 and target_id in matrix.server_ids:
            recent = [h[target_id].latency_ms for h in telemetry_history[-self.window_size:] if target_id in h]
            if len(recent) >= 3:
                sigma_est = max(0.01, float(np.std(recent)) / 100.0)

        noise_margin = self.k_noise * sigma_est
        total_hurdle = hurdle + noise_margin

        # Check for emergency / constraint failure
        is_emergency = (action == "SWITCH" and not curr_feas)
        is_noise_suppressed = ("within telemetry noise margin" in decision.get("reason", "").lower())

        # 1. Exact Additive Feature Attribution
        attributions: Dict[str, float] = {}
        if curr_idx >= 0 and target_idx >= 0:
            u_curr = matrix.utility_matrix_U[curr_idx, :]
            u_target = matrix.utility_matrix_U[target_idx, :]
            w = matrix.weight_vector_w

            for k, crit in enumerate(matrix.criteria_order):
                phi = float(w[k] * (u_target[k] - u_curr[k]))
                attributions[crit.value] = phi
        else:
            for crit in matrix.criteria_order:
                attributions[crit.value] = 0.0

        # Dominant criterion
        if is_emergency:
            dominant_crit = Criterion.AVAILABILITY.value
        elif action == "SWITCH":
            pos_att = {k: v for k, v in attributions.items() if v > 0}
            dominant_crit = max(pos_att.keys(), key=lambda k: pos_att[k]) if pos_att else max(attributions.keys(), key=lambda k: abs(attributions[k]))
        else:
            dominant_crit = max(attributions.keys(), key=lambda k: abs(attributions[k])) if attributions else "none"

        # 2. Counterfactual Explanations
        static_gain = sum(attributions.values())
        net_margin = float(static_gain - total_hurdle)

        cf_distances: Dict[str, float] = {}
        for k, crit in enumerate(matrix.criteria_order):
            wk = matrix.weight_vector_w[k]
            if wk <= 1e-6:
                cf_distances[crit.value] = 0.0
                continue

            if action == "SWITCH" and not is_emergency:
                delta_u_needed = max(0.0, net_margin / wk)
                if crit == Criterion.LATENCY:
                    cf_raw = delta_u_needed * (self.engine.config.latency_max_ms - self.engine.config.latency_min_ms)
                    cf_distances[crit.value] = round(float(cf_raw), 1)
                elif crit == Criterion.EFFECTIVE_BW:
                    cf_raw = delta_u_needed * (self.engine.config.top_bitrate_mbps - self.engine.config.min_bandwidth_mbps)
                    cf_distances[crit.value] = round(float(cf_raw), 1)
                else:
                    cf_distances[crit.value] = round(float(delta_u_needed), 3)

            elif action == "RETAIN":
                shortfall_u = max(0.0, total_hurdle - static_gain) / wk
                if crit == Criterion.LATENCY:
                    cf_raw = shortfall_u * (self.engine.config.latency_max_ms - self.engine.config.latency_min_ms)
                    cf_distances[crit.value] = round(float(cf_raw), 1)
                elif crit == Criterion.EFFECTIVE_BW:
                    cf_raw = shortfall_u * (self.engine.config.top_bitrate_mbps - self.engine.config.min_bandwidth_mbps)
                    cf_distances[crit.value] = round(float(cf_raw), 1)
                else:
                    cf_distances[crit.value] = round(float(shortfall_u), 3)
            else:
                cf_distances[crit.value] = 0.0

        # 3. Natural Language Narrative Synthesis
        narrative = self._synthesize_narrative(
            action=action,
            curr_id=curr_id,
            target_id=target_id,
            attributions=attributions,
            dominant_crit=dominant_crit,
            static_gain=static_gain,
            hurdle=hurdle,
            noise_margin=noise_margin,
            net_margin=net_margin,
            is_emergency=is_emergency,
            is_noise_suppressed=is_noise_suppressed,
            cf_distances=cf_distances,
        )

        return ExplanationResult(
            action=action,
            current_server_id=curr_id,
            target_server_id=target_id,
            decision_margin=net_margin,
            switching_hurdle=hurdle,
            noise_margin=noise_margin,
            feature_attributions=attributions,
            dominant_criterion=dominant_crit,
            counterfactual_distances=cf_distances,
            natural_language_summary=narrative,
            is_noise_suppressed=is_noise_suppressed,
            is_emergency=is_emergency,
        )

    def _synthesize_narrative(
        self,
        action: str,
        curr_id: str,
        target_id: str,
        attributions: Dict[str, float],
        dominant_crit: str,
        static_gain: float,
        hurdle: float,
        noise_margin: float,
        net_margin: float,
        is_emergency: bool,
        is_noise_suppressed: bool,
        cf_distances: Dict[str, float],
    ) -> str:
        if is_emergency:
            return (
                f"EMERGENCY SWITCH: Migrated from {curr_id} to {target_id} because "
                f"{curr_id} violated operational feasibility (telemetry heartbeat lost or SLO failed). "
                f"Candidate {target_id} selected as best feasible fallback."
            )

        if action == "SWITCH":
            gains = [f"{k} (+{v:.2f})" for k, v in attributions.items() if v > 0.01]
            sacrifices = [f"{k} ({v:.2f})" for k, v in attributions.items() if v < -0.01]

            gain_str = ", ".join(gains) if gains else "higher general utility"
            sac_str = f" despite sacrifices in {', '.join(sacrifices)}" if sacrifices else ""

            cf_bw = cf_distances.get("effective_bw", 0.0)
            cf_lat = cf_distances.get("latency", 0.0)
            cf_text = f" Current server {curr_id} could retain client with +{cf_bw} Mbps bandwidth or -{cf_lat} ms latency." if cf_bw > 0 else ""

            return (
                f"SWITCH: Switched from {curr_id} to {target_id} driven by {gain_str}{sac_str}. "
                f"Net gain (+{static_gain:.3f}) exceeded switching hurdle ({hurdle:.2f}) + "
                f"noise threshold ({noise_margin:.3f}) by margin +{net_margin:.3f}.{cf_text}"
            )

        else:
            if is_noise_suppressed:
                return (
                    f"RETAIN: Kept connection on {curr_id}. Alternative {target_id} offered static gain "
                    f"(+{static_gain:.3f}), but this fell within the 95% telemetry noise band "
                    f"(±{noise_margin:.3f}). Handover suppressed to eliminate ping-pong churn."
                )
            elif curr_id == target_id:
                return f"RETAIN: {curr_id} remains the highest-utility feasible candidate."
            else:
                return (
                    f"RETAIN: Kept connection on {curr_id}. Migration gain (+{static_gain:.3f}) did not "
                    f"exceed switching cost hurdle ({hurdle:.2f}). Playout continuity preserved."
                )

    def explain_pre_decision(self, matrix: DecisionMatrix) -> PreDecisionExplanation:
        composite_scores: Dict[str, float] = {}
        criterion_contributions: Dict[str, Dict[str, float]] = {}
        feasible_candidates: List[str] = []
        infeasible_candidates: List[str] = []

        w = matrix.weight_vector_w
        for i, s_id in enumerate(matrix.server_ids):
            is_feas = bool(matrix.feasible_mask[i])
            if is_feas:
                feasible_candidates.append(s_id)
            else:
                infeasible_candidates.append(s_id)

            u_vec = matrix.utility_matrix_U[i, :]
            score = float(np.dot(w, u_vec)) if is_feas else -1.0
            composite_scores[s_id] = round(score, 4)

            contribs = {}
            for k, crit in enumerate(matrix.criteria_order):
                contribs[crit.value] = round(float(w[k] * u_vec[k]), 4)
            criterion_contributions[s_id] = contribs

        ranked_feasible = sorted(feasible_candidates, key=lambda s: composite_scores[s], reverse=True)
        server_rankings = ranked_feasible + infeasible_candidates

        return PreDecisionExplanation(
            server_rankings=server_rankings,
            composite_scores=composite_scores,
            criterion_contributions=criterion_contributions,
            feasible_candidates=feasible_candidates,
            infeasible_candidates=infeasible_candidates,
        )

    def explain_post_decision(
        self,
        client: VideoClientState,
        matrix: DecisionMatrix,
        decision: Dict[str, Any],
        telemetry_dict: Dict[str, ServerTelemetry],
        audit_report: Optional[Any] = None,
    ) -> PostDecisionComparison:
        selected_id = str(decision.get("target_server", client.current_server_id))
        action = str(decision.get("action", "RETAIN"))
        alternative_ids = [s for s in matrix.server_ids if s != selected_id]

        qos_params: Dict[str, Dict[str, float]] = {}
        qos_utilities: Dict[str, Dict[str, float]] = {}

        for i, s_id in enumerate(matrix.server_ids):
            telem = telemetry_dict.get(s_id)
            eff_bw = float(matrix.raw_matrix_X[i, 1]) if i < matrix.raw_matrix_X.shape[0] else (
                self.engine.compute_effective_bandwidth(telem) if telem else 0.0
            )
            if telem:
                qos_params[s_id] = {
                    "latency_ms": round(float(telem.latency_ms), 2),
                    "effective_bw_mbps": round(eff_bw, 2),
                    "active_users_load": int(telem.active_users),
                    "packet_loss_rate": round(float(telem.packet_loss), 4),
                    "is_available": 1.0 if telem.is_available else 0.0,
                }
            else:
                qos_params[s_id] = {
                    "latency_ms": 0.0,
                    "effective_bw_mbps": 0.0,
                    "active_users_load": 0,
                    "packet_loss_rate": 0.0,
                    "is_available": 0.0,
                }

            u_vec = matrix.utility_matrix_U[i, :]
            u_dict = {}
            for k, crit in enumerate(matrix.criteria_order):
                u_dict[crit.value] = round(float(u_vec[k]), 4)
            qos_utilities[s_id] = u_dict

        exp = self.explain_decision(client, matrix, decision)

        auditor_verified = bool(audit_report.is_valid) if audit_report else True
        diagnosis = str(audit_report.root_cause_diagnosis) if audit_report else "Audited: Compliant with all invariants."
        recommendation = str(audit_report.remedial_recommendation) if audit_report else None

        return PostDecisionComparison(
            selected_server_id=selected_id,
            alternative_server_ids=alternative_ids,
            action=action,
            qos_parameter_values=qos_params,
            qos_utility_values=qos_utilities,
            counterfactual_distances=exp.counterfactual_distances,
            is_auditor_verified=auditor_verified,
            audit_diagnosis=diagnosis,
            recommended_feedback=recommendation,
        )

    def generate_weight_feedback(
        self,
        current_weights: Dict[Criterion, float],
        audit_report: Optional[Any] = None,
        user_feedback_action: Optional[str] = None,
        learning_rate: float = 0.10,
    ) -> Dict[Criterion, float]:
        updated = dict(current_weights)

        # 1. Closed-loop adaptation based on Auditor warnings
        if audit_report and not audit_report.is_valid:
            fault_names = [getattr(f, "value", str(f)) for f in audit_report.detected_faults]
            if "buffer_starvation_hazard" in fault_names:
                # Playout stall risk: Increase latency and availability priority, reduce aggressive bandwidth pursuit
                updated[Criterion.LATENCY] = updated.get(Criterion.LATENCY, 0.2) + learning_rate
                updated[Criterion.AVAILABILITY] = updated.get(Criterion.AVAILABILITY, 0.2) + learning_rate
                updated[Criterion.EFFECTIVE_BW] = max(0.05, updated.get(Criterion.EFFECTIVE_BW, 0.3) - learning_rate)
            elif "spurious_handover" in fault_names:
                # Sensitive to noise: Moderately boost availability and stability
                updated[Criterion.AVAILABILITY] = updated.get(Criterion.AVAILABILITY, 0.2) + (0.5 * learning_rate)

        # 2. Closed-loop adaptation based on user preference steering
        if user_feedback_action:
            action_lower = user_feedback_action.lower()
            if "quality" in action_lower or "bandwidth" in action_lower:
                updated[Criterion.EFFECTIVE_BW] = updated.get(Criterion.EFFECTIVE_BW, 0.3) + learning_rate
            elif "latency" in action_lower or "interactive" in action_lower:
                updated[Criterion.LATENCY] = updated.get(Criterion.LATENCY, 0.2) + learning_rate
            elif "reliability" in action_lower or "loss" in action_lower:
                updated[Criterion.PACKET_LOSS] = updated.get(Criterion.PACKET_LOSS, 0.15) + learning_rate
                updated[Criterion.AVAILABILITY] = updated.get(Criterion.AVAILABILITY, 0.15) + learning_rate

        # Re-normalize weights strictly to sum to 1.0
        total_w = sum(updated.values())
        if total_w > 0:
            for k in updated:
                updated[k] = float(np.round(updated[k] / total_w, 4))
            # Handle rounding residual on dominant criterion
            diff = 1.0 - sum(updated.values())
            updated[Criterion.EFFECTIVE_BW] = round(updated[Criterion.EFFECTIVE_BW] + diff, 4)

        return updated