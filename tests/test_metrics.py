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
    bitrates = [5.0, 5.0, 2.0]
    stalls = [0.0, 1.0, 0.0]
    switches = [False, False, True]

    qoe = compute_linear_qoe(bitrates, stalls, switches, dt_s=1.0)
    assert np.isclose(qoe, 4.7)

def test_qos_violation_rate():
    latencies = [30.0, 45.0, 140.0, 50.0]  # Latency > 120ms at index 2
    stalls = [0.0, 0.5, 0.0, 0.0]          # Stall at index 1
    viol_rate = compute_qos_violation_rate(latencies, stalls, latency_slo_ms=120.0)
    assert np.isclose(viol_rate, 0.50)

def test_ping_pong_rate():
    seq1 = ["s1", "s2", "s1", "s2"]
    pp1 = compute_ping_pong_rate(seq1)
    assert np.isclose(pp1, 2.0 / 3.0)

    seq2 = ["s1", "s1", "s2", "s2", "s3", "s3"]
    pp2 = compute_ping_pong_rate(seq2)
    assert pp2 == 0.0


def test_mean_dwell_time():
    seq = ["s1", "s1", "s1", "s1", "s2", "s2"]
    mean_dwell = compute_mean_dwell_time(seq, dt_s=0.5)
    assert np.isclose(mean_dwell, 1.5)

def test_compute_scenario_metrics_dataframe():
    df = pd.DataFrame({
        "time_s": [0.0, 0.5, 1.0, 1.5],
        "client_server": ["s1", "s1", "s2", "s2"],
        "effective_bw_mbps": [20.0, 20.0, 25.0, 25.0],
        "client_latency_ms": [30.0, 35.0, 40.0, 45.0],
        "client_packet_loss": [0.001, 0.001, 0.001, 0.001],
        "client_available": [True, True, True, True],
        "bitrate_mbps": [15.0, 15.0, 15.0, 15.0],
        "stall_s": [0.0, 0.0, 0.0, 0.0],
        "action": ["RETAIN", "RETAIN", "SWITCH", "RETAIN"],
    })
    metrics = compute_scenario_metrics(df, dt_s=0.5, oracle_qoe=100.0)
    assert "cumulative_qoe_proxy" in metrics
    assert "cumulative_qoe" in metrics
    assert metrics["total_switches"] == 1
    assert metrics["ping_pong_rate"] == 0.0
    assert metrics["mean_dwell_time_s"] == 1.0
    assert metrics["regret_vs_oracle"] > 0.0

def test_qos_violation_includes_availability_packet_loss_and_bandwidth():
    rate = compute_qos_violation_rate(
        [20.0, 20.0, 20.0, 20.0], [0.0, 0.0, 0.0, 0.0],
        effective_bw_mbps=[10.0, 10.0, 1.0, 10.0],
        packet_loss=[0.0, 0.06, 0.0, 0.0],
        availability=[True, True, True, False],
    )
    assert np.isclose(rate, 0.75)

def test_qos_metric_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_qos_violation_rate([20.0, 30.0], [0.0])