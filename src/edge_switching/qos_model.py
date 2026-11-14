from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class Criterion(str, Enum):
    LATENCY = "latency"          
    EFFECTIVE_BW = "effective_bw" 
    SERVER_LOAD = "server_load"
    PACKET_LOSS = "packet_loss"      
    AVAILABILITY = "availability"


@dataclass(frozen=True)
class SystemConfig:
    latency_slo_ms: float = 120.0       
    min_bandwidth_mbps: float = 2.0     
    max_server_load: float = 0.95       
    max_stale_age_s: float = 2.0        
    max_packet_loss: float = 0.10       

    latency_min_ms: float = 10.0        
    latency_max_ms: float = 150.0       
    top_bitrate_mbps: float = 25.0      
    packet_loss_decay: float = 15.0     

    base_switching_cost: float = 0.05   
    switch_delay_s: float = 0.15        
    starvation_penalty_mult: float = 3.5 


@dataclass
class ServerTelemetry:
    server_id: str
    latency_ms: float
    server_capacity_mbps: float
    active_users: int
    path_bottleneck_mbps: float
    server_load: float
    packet_loss: float
    is_available: bool
    timestamp_s: float

    def compute_effective_bandwidth(self) -> float:
        if not self.is_available or self.active_users < 0:
            return 0.0
        fair_share_mbps = self.server_capacity_mbps / float(self.active_users + 1)
        return float(min(self.path_bottleneck_mbps, fair_share_mbps))


@dataclass
class DecisionMatrix:
    server_ids: List[str]
    raw_matrix_X: np.ndarray        # Shape: (M, K)
    utility_matrix_U: np.ndarray    # Shape: (M, K)
    feasible_mask: np.ndarray       # Shape: (M,) bool
    weight_vector_w: np.ndarray     # Shape: (K,)
    raw_server_capacity: Optional[np.ndarray] = None  # Shape: (M,) raw provisioned server capacity
    criteria_order: List[Criterion] = field(default_factory=lambda: [
        Criterion.LATENCY,
        Criterion.EFFECTIVE_BW,
        Criterion.SERVER_LOAD,
        Criterion.PACKET_LOSS,
        Criterion.AVAILABILITY
    ])


class QoSFormalizationEngine:
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()

    def check_feasibility(self, telemetry: ServerTelemetry, current_time_s: float) -> Tuple[bool, List[str]]:
        violations = []
        if not telemetry.is_available:
            violations.append("Server reported unavailable (A=0)")

        age_s = current_time_s - telemetry.timestamp_s
        if age_s > self.config.max_stale_age_s:
            violations.append(f"Telemetry stale (age={age_s:.2f}s > {self.config.max_stale_age_s}s)")

        if telemetry.latency_ms > self.config.latency_slo_ms:
            violations.append(f"Latency SLO violated ({telemetry.latency_ms:.1f}ms > {self.config.latency_slo_ms}ms)")

        b_eff = telemetry.compute_effective_bandwidth()
        if b_eff < self.config.min_bandwidth_mbps:
            violations.append(f"Effective bandwidth insufficient ({b_eff:.2f}Mbps < {self.config.min_bandwidth_mbps}Mbps)")

        if telemetry.server_load > self.config.max_server_load:
            violations.append(f"Server overloaded (load={telemetry.server_load:.2f} > {self.config.max_server_load})")

        if telemetry.packet_loss > self.config.max_packet_loss:
            violations.append(f"Packet loss excessive ({telemetry.packet_loss:.3f} > {self.config.max_packet_loss})")

        return len(violations) == 0, violations

    def normalize_criterion(self, criterion: Criterion, raw_val: float) -> float:
        if criterion == Criterion.LATENCY:
            u = (self.config.latency_max_ms - raw_val) / (self.config.latency_max_ms - self.config.latency_min_ms)
            return float(np.clip(u, 0.0, 1.0))

        elif criterion == Criterion.EFFECTIVE_BW:
            if raw_val <= self.config.min_bandwidth_mbps:
                return 0.0
            u = math.log(1.0 + raw_val) / math.log(1.0 + self.config.top_bitrate_mbps)
            return float(np.clip(u, 0.0, 1.0))

        elif criterion == Criterion.SERVER_LOAD:
            u = 1.0 - (raw_val ** 2)
            return float(np.clip(u, 0.0, 1.0))

        elif criterion == Criterion.PACKET_LOSS:
            u = math.exp(-self.config.packet_loss_decay * raw_val)
            return float(np.clip(u, 0.0, 1.0))

        elif criterion == Criterion.AVAILABILITY:
            return 1.0 if raw_val >= 1.0 else 0.0

        raise ValueError(f"Unknown criterion: {criterion}")

    def build_matrices(
        self,
        telemetries: List[ServerTelemetry],
        weights: Dict[Criterion, float],
        current_time_s: float
    ) -> DecisionMatrix:
        criteria = [
            Criterion.LATENCY,
            Criterion.EFFECTIVE_BW,
            Criterion.SERVER_LOAD,
            Criterion.PACKET_LOSS,
            Criterion.AVAILABILITY
        ]
        M = len(telemetries)
        K = len(criteria)

        X = np.zeros((M, K), dtype=float)
        U = np.zeros((M, K), dtype=float)
        feasible_mask = np.zeros(M, dtype=bool)

        total_w = sum(weights.get(c, 0.0) for c in criteria)
        if total_w <= 0:
            raise ValueError("Weight sum must be strictly positive.")
        w_vec = np.array([weights.get(c, 0.0) / total_w for c in criteria], dtype=float)

        server_ids = []
        for i, telem in enumerate(telemetries):
            server_ids.append(telem.server_id)
            is_feas, _ = self.check_feasibility(telem, current_time_s)
            feasible_mask[i] = is_feas

            b_eff = telem.compute_effective_bandwidth()
            raw_vals = [
                telem.latency_ms,
                b_eff,
                telem.server_load,
                telem.packet_loss,
                1.0 if telem.is_available else 0.0
            ]
            for j, c in enumerate(criteria):
                X[i, j] = raw_vals[j]
                U[i, j] = self.normalize_criterion(c, raw_vals[j])

        raw_caps = np.array([t.server_capacity_mbps for t in telemetries], dtype=float)
        return DecisionMatrix(
            server_ids=server_ids,
            raw_matrix_X=X,
            utility_matrix_U=U,
            feasible_mask=feasible_mask,
            weight_vector_w=w_vec,
            raw_server_capacity=raw_caps,
            criteria_order=criteria
        )

    def evaluate_switching_decision(
        self,
        matrix: DecisionMatrix,
        current_server_id: str,
        buffer_occupancy_s: float = 2.0
    ) -> Dict[str, object]:
        M = len(matrix.server_ids)
        static_utilities = matrix.utility_matrix_U @ matrix.weight_vector_w

        curr_idx = None
        for i, sid in enumerate(matrix.server_ids):
            if sid == current_server_id:
                curr_idx = i
                break

        hurdle = self.config.base_switching_cost
        if buffer_occupancy_s < self.config.switch_delay_s:
            hurdle *= self.config.starvation_penalty_mult

        net_utilities = np.copy(static_utilities)
        for i in range(M):
            if not matrix.feasible_mask[i]:
                net_utilities[i] = -float("inf")
            elif curr_idx is not None and i != curr_idx:
                net_utilities[i] -= hurdle

        best_idx = int(np.argmax(net_utilities))
        best_server_id = matrix.server_ids[best_idx]
        best_is_feasible = matrix.feasible_mask[best_idx]

        if not best_is_feasible or net_utilities[best_idx] == -float("inf"):
            action = "EMERGENCY_RETAIN"
            target_server_id = current_server_id
            margin = 0.0
        elif curr_idx is not None and best_idx == curr_idx:
            action = "RETAIN"
            target_server_id = current_server_id
            other_utilities = [net_utilities[j] for j in range(M) if j != curr_idx and matrix.feasible_mask[j]]
            second_best = max(other_utilities) if other_utilities else 0.0
            margin = float(net_utilities[curr_idx] - second_best)
        else:
            action = "SWITCH"
            target_server_id = best_server_id
            curr_val = net_utilities[curr_idx] if (curr_idx is not None and matrix.feasible_mask[curr_idx]) else 0.0
            margin = float(net_utilities[best_idx] - curr_val)

        return {
            "action": action,
            "target_server": target_server_id,
            "current_server": current_server_id,
            "static_utilities": {matrix.server_ids[i]: float(static_utilities[i]) for i in range(M)},
            "net_utilities": {matrix.server_ids[i]: float(net_utilities[i]) for i in range(M)},
            "feasible_candidates": [matrix.server_ids[i] for i in range(M) if matrix.feasible_mask[i]],
            "decision_margin": margin,
            "switching_hurdle": hurdle,
            "buffer_occupancy_s": buffer_occupancy_s
        }