"""
XAI Explanation Pipeline for QoS-Aware Edge Server Switching.
Target Venue: IEEE PerCom 2027 (A4E Workshop)

Implements:
1. Exact Additive Feature Attribution (satisfying the Shapley efficiency axiom: sum phi_k = Delta U).
2. Counterfactual Explanations:
   - Noise-normalized decision margin M = Delta U - (lambda_switch + k * sigma).
   - Minimal physical perturbation delta_k required on current or target server to invert the decision.
3. Contrastive & Natural Language Synthesis:
   - Human-readable narrative detailing trade-offs (e.g. throughput gain vs latency penalty).
   - Emergency failure explanations and noise-suppression notifications.
"""

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
    """Structured container holding all explainability artifacts for a switching decision."""
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


class EdgeSwitchingExplainer:
    """
    Computes exact, closed-form explanations for edge server switching decisions.
    """

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
        """
        Generate complete feature attribution, counterfactual margin,
        and natural language narrative for the given switching decision.
        """
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

        # -------------------------------------------------------------
        # 1. Exact Additive Feature Attribution (Efficiency Axiom)
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 2. Counterfactual Explanations
        # -------------------------------------------------------------
        # Decision Margin M = Delta U - total_hurdle
        static_gain = sum(attributions.values())
        net_margin = float(static_gain - total_hurdle)

        cf_distances: Dict[str, float] = {}
        # What physical improvement on s_curr would prevent a switch?
        # Or what improvement on target is needed to trigger a switch?
        for k, crit in enumerate(matrix.criteria_order):
            wk = matrix.weight_vector_w[k]
            if wk <= 1e-6:
                cf_distances[crit.value] = 0.0
                continue

            if action == "SWITCH" and not is_emergency:
                # Required utility increase on current server to erase margin
                delta_u_needed = max(0.0, net_margin / wk)
                if crit == Criterion.LATENCY:
                    # delta_u = -delta_L / (L_max - L_min) => delta_L_ms = -delta_u * range
                    cf_raw = delta_u_needed * (self.engine.config.latency_max_ms - self.engine.config.latency_min_ms)
                    cf_distances[crit.value] = round(float(cf_raw), 1)
                elif crit == Criterion.EFFECTIVE_BW:
                    cf_raw = delta_u_needed * (self.engine.config.top_bitrate_mbps - self.engine.config.min_bandwidth_mbps)
                    cf_distances[crit.value] = round(float(cf_raw), 1)
                else:
                    cf_distances[crit.value] = round(float(delta_u_needed), 3)

            elif action == "RETAIN":
                # Shortfall needed on target server to trigger handover
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

        # -------------------------------------------------------------
        # 3. Natural Language Narrative Synthesis
        # -------------------------------------------------------------
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
        """Construct concise, precise natural language justification."""
        if is_emergency:
            return (
                f"EMERGENCY SWITCH: Migrated from {curr_id} to {target_id} because "
                f"{curr_id} violated operational feasibility (telemetry heartbeat lost or SLO failed). "
                f"Candidate {target_id} selected as best feasible fallback."
            )

        if action == "SWITCH":
            # Identify top driver and top trade-off
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
