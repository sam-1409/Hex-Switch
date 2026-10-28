"""
Formal Evaluation Metrics Suite for Edge Server Switching and Video Streaming.

Implements PerCom-standard quantitative metrics:
1. Linear Video QoE (ITU-T P.1203 proxy: bitrate - 4.3*stall - 0.5*delta_r - 1.5*switch).
2. QoS Violation Rate (% of steps violating latency SLO or experiencing stall).
3. Switching Rate (switches / minute).
4. Ping-Pong Rate (% of switches returning to previous server within 2 epochs).
5. Mean Dwell Time (average continuous duration in seconds on a server).
6. Regret versus Hindsight Oracle (QoE_oracle - QoE_policy).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_linear_qoe(
    bitrates_mbps: List[float],
    stalls_s: List[float],
    switches: List[bool],
    dt_s: float = 0.5,
    mu_stall: float = 4.3,
    gamma_smoothness: float = 0.5,
    eta_switch: float = 1.5,
) -> float:
    """
    Compute cumulative linear video QoE over time series:
        QoE = sum_t [ r(t)*dt - mu * stall(t) - gamma * |r(t) - r(t-1)| - eta * I(switch) ]
    """
    total_qoe = 0.0
    prev_r = bitrates_mbps[0] if bitrates_mbps else 3.0

    for i in range(len(bitrates_mbps)):
        r = bitrates_mbps[i]
        stall = stalls_s[i]
        sw = switches[i]

        step_qoe = (r * dt_s) - (mu_stall * stall) - (gamma_smoothness * abs(r - prev_r))
        if sw:
            step_qoe -= eta_switch

        total_qoe += step_qoe
        prev_r = r

    return float(total_qoe)


def compute_qos_violation_rate(
    latencies_ms: List[float],
    stalls_s: List[float],
    latency_slo_ms: float = 120.0,
) -> float:
    """
    Compute QoS Violation Rate:
    Fraction of steps where either Latency > SLO or rebuffering stall occurs.
    """
    if not latencies_ms:
        return 0.0
    violations = sum(
        1 for lat, stall in zip(latencies_ms, stalls_s)
        if lat > latency_slo_ms or stall > 0.0
    )
    return float(violations / len(latencies_ms))


def compute_switch_rate_per_minute(total_switches: int, duration_s: float) -> float:
    """Calculate switching frequency per minute."""
    if duration_s <= 0.0:
        return 0.0
    return float((total_switches / duration_s) * 60.0)


def compute_ping_pong_rate(server_sequence: List[str]) -> float:
    """
    Compute Ping-Pong Switching Rate:
    Fraction of switches where client returns to the server held 2 steps ago.
    """
    if len(server_sequence) < 3:
        return 0.0

    switch_events = 0
    ping_pong_events = 0

    for i in range(1, len(server_sequence)):
        if server_sequence[i] != server_sequence[i - 1]:
            switch_events += 1
            if i >= 2 and server_sequence[i] == server_sequence[i - 2]:
                ping_pong_events += 1

    if switch_events == 0:
        return 0.0
    return float(ping_pong_events / switch_events)


def compute_mean_dwell_time(server_sequence: List[str], dt_s: float = 0.5) -> float:
    """Calculate average continuous duration (in seconds) spent connected to a server."""
    if not server_sequence:
        return 0.0

    dwell_durations = []
    current_run = 1

    for i in range(1, len(server_sequence)):
        if server_sequence[i] == server_sequence[i - 1]:
            current_run += 1
        else:
            dwell_durations.append(current_run * dt_s)
            current_run = 1

    dwell_durations.append(current_run * dt_s)
    return float(np.mean(dwell_durations))


def compute_scenario_metrics(df: pd.DataFrame, dt_s: float = 0.5, oracle_qoe: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance dictionary from simulation run DataFrame.
    """
    duration_s = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0] + dt_s) if len(df) > 0 else 60.0
    server_seq = list(df["client_server"])
    switches_mask = list(df["action"] == "SWITCH")
    total_switches = sum(switches_mask)

    # Reconstruct or extract bitrate & stall lists
    stalls = list(df["stall_s"]) if "stall_s" in df else [0.0] * len(df)
    # Estimate bitrate from effective bandwidth
    bw_vals = list(df["effective_bw_mbps"])
    bitrates = [min(25.0, max(1.5, b * 0.85)) for b in bw_vals]

    latencies = list(df["client_latency_ms"]) if "client_latency_ms" in df else [30.0] * len(df)

    qoe = compute_linear_qoe(bitrates, stalls, switches_mask, dt_s)
    switch_rate = compute_switch_rate_per_minute(total_switches, duration_s)
    ping_pong_rate = compute_ping_pong_rate(server_seq)
    mean_dwell = compute_mean_dwell_time(server_seq, dt_s)
    qos_viol_rate = compute_qos_violation_rate(latencies, stalls)

    regret = float(oracle_qoe - qoe) if oracle_qoe is not None else 0.0

    return {
        "cumulative_qoe": qoe,
        "total_switches": total_switches,
        "switch_rate_per_min": switch_rate,
        "ping_pong_rate": ping_pong_rate,
        "mean_dwell_time_s": mean_dwell,
        "qos_violation_rate": qos_viol_rate,
        "regret_vs_oracle": regret,
    }
