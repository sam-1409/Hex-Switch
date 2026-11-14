from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

def compute_linear_qoe(
    bitrates_mbps: Sequence[float],
    stalls_s: Sequence[float],
    switches: Sequence[bool],
    dt_s: float = 0.5,
    mu_stall: float = 4.3,
    gamma_smoothness: float = 0.5,
    eta_switch: float = 1.5,
) -> float:
    n = len(bitrates_mbps)
    if not (len(stalls_s) == len(switches) == n):
        raise ValueError("bitrates_mbps, stalls_s, and switches must have equal lengths")
    if n == 0:
        return 0.0
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    total = 0.0
    prev_r = float(bitrates_mbps[0])
    for i, (rate, stall, switched) in enumerate(zip(bitrates_mbps, stalls_s, switches)):
        r, stall_s = float(rate), max(0.0, float(stall))
        if not np.isfinite(r) or not np.isfinite(stall_s):
            raise ValueError("QoE inputs must be finite")
        total += r * dt_s - mu_stall * stall_s
        if i > 0:
            total -= gamma_smoothness * abs(r - prev_r)
        if bool(switched):
            total -= eta_switch
        prev_r = r
    return float(total)

def compute_qos_violation_rate(
    latencies_ms: Sequence[float],
    stalls_s: Sequence[float],
    latency_slo_ms: float = 100.0,
    effective_bw_mbps: Optional[Sequence[float]] = None,
    min_effective_bw_mbps: float = 1.5,
    packet_loss: Optional[Sequence[float]] = None,
    max_packet_loss: float = 0.05,
    availability: Optional[Sequence[bool]] = None,
) -> float:
    n = len(latencies_ms)
    if n == 0:
        return 0.0
    arrays = {"stalls_s": stalls_s, "effective_bw_mbps": effective_bw_mbps,
              "packet_loss": packet_loss, "availability": availability}
    for name, values in arrays.items():
        if values is not None and len(values) != n:
            raise ValueError(f"{name} length must match latencies_ms")
    violated = np.asarray(latencies_ms, dtype=float) > latency_slo_ms
    violated |= np.asarray(stalls_s, dtype=float) > 0.0
    if effective_bw_mbps is not None:
        violated |= np.asarray(effective_bw_mbps, dtype=float) < min_effective_bw_mbps
    if packet_loss is not None:
        violated |= np.asarray(packet_loss, dtype=float) > max_packet_loss
    if availability is not None:
        violated |= ~np.asarray(availability, dtype=bool)
    return float(np.mean(violated))

def compute_switch_rate_per_minute(total_switches: int, duration_s: float) -> float:
    return float((total_switches / duration_s) * 60.0) if duration_s > 0 else 0.0

def compute_ping_pong_rate(server_sequence: Sequence[str]) -> float:
    if len(server_sequence) < 3:
        return 0.0
    switches = sum(server_sequence[i] != server_sequence[i - 1] for i in range(1, len(server_sequence)))
    ping_pongs = sum(
        server_sequence[i] != server_sequence[i - 1] and server_sequence[i] == server_sequence[i - 2]
        for i in range(2, len(server_sequence))
    )
    return float(ping_pongs / switches) if switches else 0.0

def compute_mean_dwell_time(server_sequence: Sequence[str], dt_s: float = 0.5) -> float:
    if not server_sequence:
        return 0.0
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    durations: List[float] = []
    run = 1
    for i in range(1, len(server_sequence)):
        if server_sequence[i] == server_sequence[i - 1]:
            run += 1
        else:
            durations.append(run * dt_s)
            run = 1
    durations.append(run * dt_s)
    return float(np.mean(durations))

def compute_scenario_metrics(
    df: pd.DataFrame,
    dt_s: float = 0.5,
    oracle_qoe: Optional[float] = None,
    latency_slo_ms: float = 100.0,
    min_effective_bw_mbps: float = 1.5,
    max_packet_loss: float = 0.05,
) -> Dict[str, float]:
    if df.empty:
        raise ValueError("Cannot evaluate an empty scenario DataFrame")
    required = {"client_server", "action", "effective_bw_mbps"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Scenario log missing required columns: {sorted(missing)}")

    duration = (float(df["time_s"].iloc[-1] - df["time_s"].iloc[0] + dt_s)
                if "time_s" in df and len(df) > 1 else len(df) * dt_s)
    server_seq = df["client_server"].astype(str).tolist()
    switches = df["action"].astype(str).eq("SWITCH").tolist()
    stalls = df["stall_s"].astype(float).tolist() if "stall_s" in df else [0.0] * len(df)
    bw = df["effective_bw_mbps"].astype(float).tolist()
    bitrates = (df["bitrate_mbps"].astype(float).tolist() if "bitrate_mbps" in df
                else [min(25.0, max(1.5, x * 0.85)) for x in bw])
    latencies = df["client_latency_ms"].astype(float).tolist() if "client_latency_ms" in df else [float("nan")] * len(df)
    packet_loss = df["client_packet_loss"].astype(float).tolist() if "client_packet_loss" in df else None
    availability = df["client_available"].astype(bool).tolist() if "client_available" in df else None

    if not np.all(np.isfinite(latencies)):
        raise ValueError("client_latency_ms is missing or non-finite; regenerate scenario logs with telemetry fields")

    qoe = compute_linear_qoe(bitrates, stalls, switches, dt_s)
    violation_rate = compute_qos_violation_rate(
        latencies, stalls, latency_slo_ms, bw, min_effective_bw_mbps,
        packet_loss, max_packet_loss, availability,
    )
    total_switches = int(sum(switches))
    result = {
        "cumulative_qoe_proxy": qoe,
        "cumulative_qoe": qoe,
        "total_switches": float(total_switches),
        "switch_rate_per_min": compute_switch_rate_per_minute(total_switches, duration),
        "ping_pong_rate": compute_ping_pong_rate(server_seq),
        "mean_dwell_time_s": compute_mean_dwell_time(server_seq, dt_s),
        "qos_violation_rate": violation_rate,
        "latency_violation_rate": float(np.mean(np.asarray(latencies) > latency_slo_ms)),
        "stall_epoch_rate": float(np.mean(np.asarray(stalls) > 0.0)),
        "bandwidth_violation_rate": float(np.mean(np.asarray(bw) < min_effective_bw_mbps)),
        "packet_loss_violation_rate": float(np.mean(np.asarray(packet_loss) > max_packet_loss)) if packet_loss is not None else float("nan"),
        "availability_violation_rate": float(np.mean(~np.asarray(availability, dtype=bool))) if availability is not None else float("nan"),
    }
    result["regret_vs_oracle"] = float(oracle_qoe - qoe) if oracle_qoe is not None else float("nan")
    return result