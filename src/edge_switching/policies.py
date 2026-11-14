from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from edge_switching.qos_model import (
    Criterion,
    DecisionMatrix,
    QoSFormalizationEngine,
    ServerTelemetry,
    SystemConfig,
)
from edge_switching.simulator import VideoClientState

class BaseSelectionPolicy(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def decide(
        self,
        client: VideoClientState,
        matrix: DecisionMatrix,
        engine: QoSFormalizationEngine,
        telemetry_history: Optional[List[Dict[str, ServerTelemetry]]] = None,
    ) -> Dict[str, Any]:
        pass


class NearestServerPolicy(BaseSelectionPolicy):
    def __init__(self, target_server_id: str = "s1"):
        super().__init__("Nearest-Server")
        self.target_server_id = target_server_id

    def decide(self, client, matrix, engine, telemetry_history=None):
        target = self.target_server_id
        avail_col = matrix.criteria_order.index(Criterion.AVAILABILITY)
        target_idx = matrix.server_ids.index(target) if target in matrix.server_ids else -1

        if target_idx >= 0 and matrix.raw_matrix_X[target_idx, avail_col] > 0.5:
            action = "RETAIN" if client.current_server_id == target else "SWITCH"
            return {"action": action, "target_server": target, "decision_margin": 0.0, "policy_name": self.name}

        online_indices = [i for i in range(len(matrix.server_ids)) if matrix.raw_matrix_X[i, avail_col] > 0.5]
        if online_indices:
            fallback = matrix.server_ids[online_indices[0]]
            action = "RETAIN" if client.current_server_id == fallback else "SWITCH"
            return {"action": action, "target_server": fallback, "decision_margin": 0.0, "policy_name": self.name}

        return {"action": "EMERGENCY_RETAIN", "target_server": client.current_server_id, "decision_margin": 0.0, "policy_name": self.name}


class MinLatencyPolicy(BaseSelectionPolicy):
    def __init__(self):
        super().__init__("Min-Latency")

    def decide(self, client, matrix, engine, telemetry_history=None):
        lat_col = matrix.criteria_order.index(Criterion.LATENCY)
        avail_col = matrix.criteria_order.index(Criterion.AVAILABILITY)
        online_indices = [i for i in range(len(matrix.server_ids)) if matrix.raw_matrix_X[i, avail_col] > 0.5]
        if not online_indices:
            return {"action": "EMERGENCY_RETAIN", "target_server": client.current_server_id, "decision_margin": 0.0, "policy_name": self.name}

        best_idx = min(online_indices, key=lambda i: matrix.raw_matrix_X[i, lat_col])
        target = matrix.server_ids[best_idx]
        action = "RETAIN" if target == client.current_server_id else "SWITCH"
        return {"action": action, "target_server": target, "decision_margin": 0.0, "policy_name": self.name}

class MaxTotalBandwidthPolicy(BaseSelectionPolicy):
    def __init__(self):
        super().__init__("Max-Total-BW")

    def decide(self, client, matrix, engine, telemetry_history=None):
        avail_col = matrix.criteria_order.index(Criterion.AVAILABILITY)
        online_indices = [i for i in range(len(matrix.server_ids)) if matrix.raw_matrix_X[i, avail_col] > 0.5]
        if not online_indices:
            return {"action": "EMERGENCY_RETAIN", "target_server": client.current_server_id, "decision_margin": 0.0, "policy_name": self.name}

        if getattr(matrix, "raw_server_capacity", None) is not None:
            best_idx = max(online_indices, key=lambda i: matrix.raw_server_capacity[i])
        else:
            best_idx = max(online_indices, key=lambda i: matrix.raw_matrix_X[i, 1])
        target = matrix.server_ids[best_idx]
        action = "RETAIN" if target == client.current_server_id else "SWITCH"
        return {"action": action, "target_server": target, "decision_margin": 0.0, "policy_name": self.name}


class MaxEffectiveBandwidthPolicy(BaseSelectionPolicy):
    def __init__(self):
        super().__init__("Max-Effective-BW")

    def decide(self, client, matrix, engine, telemetry_history=None):
        bw_col = matrix.criteria_order.index(Criterion.EFFECTIVE_BW)
        feasible_indices = [i for i, f in enumerate(matrix.feasible_mask) if f]
        if not feasible_indices:
            return {"action": "EMERGENCY_RETAIN", "target_server": client.current_server_id, "decision_margin": 0.0, "policy_name": self.name}

        best_idx = max(feasible_indices, key=lambda i: matrix.raw_matrix_X[i, bw_col])
        target = matrix.server_ids[best_idx]
        action = "RETAIN" if target == client.current_server_id else "SWITCH"
        return {"action": action, "target_server": target, "decision_margin": 0.0, "policy_name": self.name}


class AhpTopsisPolicy(BaseSelectionPolicy):

    def __init__(self):
        super().__init__("AHP-TOPSIS")

    def decide(self, client, matrix, engine, telemetry_history=None):
        feasible_indices = [i for i, f in enumerate(matrix.feasible_mask) if f]
        if not feasible_indices:
            return {"action": "EMERGENCY_RETAIN", "target_server": client.current_server_id, "decision_margin": 0.0, "policy_name": self.name}

        sub_U = matrix.utility_matrix_U[feasible_indices, :]
        V = sub_U * matrix.weight_vector_w

        a_plus = np.max(V, axis=0)
        a_minus = np.min(V, axis=0)

        # Euclidean separation distances
        d_plus = np.sqrt(np.sum((V - a_plus) ** 2, axis=1))
        d_minus = np.sqrt(np.sum((V - a_minus) ** 2, axis=1))

        denom = d_plus + d_minus
        closeness = np.where(denom > 0, d_minus / denom, 0.5)

        best_local_idx = int(np.argmax(closeness))
        best_global_idx = feasible_indices[best_local_idx]
        target = matrix.server_ids[best_global_idx]
        action = "RETAIN" if target == client.current_server_id else "SWITCH"

        return {
            "action": action,
            "target_server": target,
            "decision_margin": float(closeness[best_local_idx] - (np.partition(closeness, -2)[-2] if len(closeness) > 1 else 0.0)),
            "policy_name": self.name,
            "closeness_scores": {matrix.server_ids[feasible_indices[k]]: float(closeness[k]) for k in range(len(feasible_indices))}
        }

class HysteresisOnlyPolicy(BaseSelectionPolicy):
    def __init__(self, hysteresis_threshold: float = 0.08):
        super().__init__("Static-Hysteresis")
        self.theta_hyst = hysteresis_threshold

    def decide(self, client, matrix, engine, telemetry_history=None):
        static_utils = matrix.utility_matrix_U @ matrix.weight_vector_w
        feasible_indices = [i for i, f in enumerate(matrix.feasible_mask) if f]
        if not feasible_indices:
            return {"action": "EMERGENCY_RETAIN", "target_server": client.current_server_id, "decision_margin": 0.0, "policy_name": self.name}

        curr_id = client.current_server_id
        curr_idx = matrix.server_ids.index(curr_id) if curr_id in matrix.server_ids else -1
        curr_is_feasible = (curr_idx in feasible_indices)

        # Best feasible candidate
        best_idx = max(feasible_indices, key=lambda i: static_utils[i])
        best_target = matrix.server_ids[best_idx]

        if not curr_is_feasible:
            # Force emergency migration
            return {"action": "SWITCH", "target_server": best_target, "decision_margin": 1.0, "policy_name": self.name}

        delta_u = static_utils[best_idx] - static_utils[curr_idx]
        if delta_u > self.theta_hyst:
            return {"action": "SWITCH", "target_server": best_target, "decision_margin": float(delta_u - self.theta_hyst), "policy_name": self.name}

        return {"action": "RETAIN", "target_server": curr_id, "decision_margin": float(self.theta_hyst - delta_u), "policy_name": self.name}

class QoEGreedyPolicy(BaseSelectionPolicy):
    def __init__(self):
        super().__init__("QoE-Greedy")

    def decide(self, client, matrix, engine, telemetry_history=None):
        feasible_indices = [i for i, f in enumerate(matrix.feasible_mask) if f]
        if not feasible_indices:
            return {"action": "EMERGENCY_RETAIN", "target_server": client.current_server_id, "decision_margin": 0.0, "policy_name": self.name}

        bw_col = matrix.criteria_order.index(Criterion.EFFECTIVE_BW)
        lat_col = matrix.criteria_order.index(Criterion.LATENCY)

        def expected_qoe(idx):
            bw = matrix.raw_matrix_X[idx, bw_col]
            lat = matrix.raw_matrix_X[idx, lat_col]
            return bw - (0.05 * lat)

        best_idx = max(feasible_indices, key=expected_qoe)
        target = matrix.server_ids[best_idx]
        action = "RETAIN" if target == client.current_server_id else "SWITCH"
        return {"action": action, "target_server": target, "decision_margin": 0.0, "policy_name": self.name}

class ProposedMarginPolicy(BaseSelectionPolicy):
    def __init__(
        self,
        base_hurdle: float = 0.05,
        k_noise: float = 1.65,
        min_dwell_steps: int = 4,
        window_size: int = 10,
    ):
        super().__init__("Proposed-Margin-Policy")
        self.base_hurdle = base_hurdle
        self.k_noise = k_noise
        self.min_dwell_steps = min_dwell_steps
        self.window_size = window_size
        self.dwell_counters: Dict[str, int] = {} 

    def decide(self, client, matrix, engine, telemetry_history=None):
        curr_id = client.current_server_id
        dwell = self.dwell_counters.get(client.client_id, 0)
        self.dwell_counters[client.client_id] = dwell + 1

        curr_idx = matrix.server_ids.index(curr_id) if curr_id in matrix.server_ids else -1
        curr_feasible = (curr_idx >= 0 and matrix.feasible_mask[curr_idx])

        # Evaluate formal decision engine
        raw_decision = engine.evaluate_switching_decision(
            matrix,
            current_server_id=curr_id,
            buffer_occupancy_s=client.buffer_occupancy_s,
        )

        target = str(raw_decision["target_server"])
        action = str(raw_decision["action"])
        hurdle = float(raw_decision["switching_hurdle"])

        if not curr_feasible:
            self.dwell_counters[client.client_id] = 0
            return {
                "action": "SWITCH",
                "target_server": target,
                "decision_margin": 1.0,
                "switching_hurdle": hurdle,
                "policy_name": self.name,
                "reason": "Emergency migration: current server violates feasibility",
            }

        if action == "RETAIN":
            return {
                "action": "RETAIN",
                "target_server": curr_id,
                "decision_margin": float(raw_decision["decision_margin"]),
                "switching_hurdle": hurdle,
                "policy_name": self.name,
                "reason": "Utility gain does not exceed switching hurdle",
            }

        if dwell < self.min_dwell_steps:
            return {
                "action": "RETAIN",
                "target_server": curr_id,
                "decision_margin": float(raw_decision["decision_margin"]),
                "switching_hurdle": hurdle,
                "policy_name": self.name,
                "reason": f"Dwell constraint holding server ({dwell}/{self.min_dwell_steps} steps)",
            }

        # Estimate telemetry noise standard deviation from recent observations
        sigma_est = 0.015  # Default empirical baseline
        if telemetry_history and len(telemetry_history) >= 3:
            recent_lats = [h[target].latency_ms for h in telemetry_history[-self.window_size:] if target in h]
            if len(recent_lats) >= 3:
                sigma_est = max(0.01, float(np.std(recent_lats)) / 100.0)

        noise_margin_required = hurdle + (self.k_noise * sigma_est)
        static_utils = matrix.utility_matrix_U @ matrix.weight_vector_w
        target_idx = matrix.server_ids.index(target)
        gain = float(static_utils[target_idx] - static_utils[curr_idx])

        if gain > noise_margin_required:
            # Valid switch: reset dwell counter
            self.dwell_counters[client.client_id] = 0
            return {
                "action": "SWITCH",
                "target_server": target,
                "decision_margin": float(gain - noise_margin_required),
                "switching_hurdle": hurdle,
                "policy_name": self.name,
                "reason": f"Gain {gain:.3f} exceeds noise-calibrated hurdle {noise_margin_required:.3f}",
            }

        # Noise-suppressed retention
        return {
            "action": "RETAIN",
            "target_server": curr_id,
            "decision_margin": float(noise_margin_required - gain),
            "switching_hurdle": hurdle,
            "policy_name": self.name,
            "reason": f"Gain {gain:.3f} within telemetry noise margin ({noise_margin_required:.3f})",
        }


def get_baseline_suite(default_target: str = "s1") -> Dict[str, BaseSelectionPolicy]:
    return {
        "Nearest": NearestServerPolicy(target_server_id=default_target),
        "Min-Latency": MinLatencyPolicy(),
        "Max-Total-BW": MaxTotalBandwidthPolicy(),
        "Max-Effective-BW": MaxEffectiveBandwidthPolicy(),
        "AHP-TOPSIS": AhpTopsisPolicy(),
        "Static-Hysteresis": HysteresisOnlyPolicy(hysteresis_threshold=0.08),
        "QoE-Greedy": QoEGreedyPolicy(),
        "Proposed-Margin": ProposedMarginPolicy(),
    }