"""
Unit tests for Formal Evaluation Metrics (Step 3).

Verifies:
- Linear Video QoE computation (ITU-T P.1203 proxy).
- QoS violation rate.
- Switch rate and Ping-Pong rate.
- Mean dwell time.
- Comprehensive scenario metrics dictionary.
"""

import pytest
import numpy as np
import pandas as pd

from edge_switching.metrics import (
    compute_linear_qoe,
    compute_qos_violation_rate,
    compute_switch_rate_per_minute,
    compute_ping_pong_rate,
    compute_mean_dwell_time,
    compute_scenario_metrics,
)


def test_linear_qoe_formula():
    """Verify linear video QoE matches standard ITU-T P.1203 penalty coefficients."""
    # 3 epochs, dt=1.0s:
    # t=0: r=5.0, stall=0, switch=False -> 5.0
    # t=1: r=5.0, stall=1.0, switch=False -> 5.0 - 4.3*1.0 = 0.7
    # t=2: r=2.0, stall=0, switch=True -> 2.0 - 0.5*|2-5| - 1.5 = 2.0 - 1.5 - 1.5 = -1.0
    # Expected total = 5.0 + 0.7 - 1.0 = 4.7
    bitrates = [5.0, 5.0, 2.0]
    stalls = [0.0, 1.0, 0.0]
    switches = [False, False, True]

    qoe = compute_linear_qoe(bitrates, stalls, switches, dt_s=1.0)
    assert np.isclose(qoe, 4.7)


def test_qos_violation_rate():
    latencies = [30.0, 45.0, 140.0, 50.0]  # Latency > 120ms at index 2
    stalls = [0.0, 0.5, 0.0, 0.0]          # Stall at index 1
    # 2 violations out of 4 steps -> 50%
    viol_rate = compute_qos_violation_rate(latencies, stalls, latency_slo_ms=120.0)
    assert np.isclose(viol_rate, 0.50)


def test_ping_pong_rate():
    # Sequence with ping-pong: s1 -> s2 -> s1 -> s2
    # 3 switch events, all 2 switches (indices 2 and 3) are ping-pongs -> 2/3 ping-pong rate
    seq1 = ["s1", "s2", "s1", "s2"]
    pp1 = compute_ping_pong_rate(seq1)
    assert np.isclose(pp1, 2.0 / 3.0)

    # Sequence with zero ping-pong: s1 -> s2 -> s3 -> s4
    seq2 = ["s1", "s1", "s2", "s2", "s3", "s3"]
    pp2 = compute_ping_pong_rate(seq2)
    assert pp2 == 0.0


def test_mean_dwell_time():
    # Sequence: 4 steps on s1, then 2 steps on s2 (dt = 0.5s)
    # Dwell 1: 4 * 0.5 = 2.0s
    # Dwell 2: 2 * 0.5 = 1.0s
    # Mean dwell = (2.0 + 1.0) / 2 = 1.5s
    seq = ["s1", "s1", "s1", "s1", "s2", "s2"]
    mean_dwell = compute_mean_dwell_time(seq, dt_s=0.5)
    assert np.isclose(mean_dwell, 1.5)


def test_compute_scenario_metrics_dataframe():
    df = pd.DataFrame({
        "time_s": [0.0, 0.5, 1.0, 1.5],
        "client_server": ["s1", "s1", "s2", "s2"],
        "effective_bw_mbps": [20.0, 20.0, 25.0, 25.0],
        "stall_s": [0.0, 0.0, 0.0, 0.0],
        "action": ["RETAIN", "RETAIN", "SWITCH", "RETAIN"],
    })
    metrics = compute_scenario_metrics(df, dt_s=0.5, oracle_qoe=100.0)
    assert "cumulative_qoe" in metrics
    assert metrics["total_switches"] == 1
    assert metrics["ping_pong_rate"] == 0.0
    assert metrics["mean_dwell_time_s"] == 1.0
    assert metrics["regret_vs_oracle"] > 0.0
