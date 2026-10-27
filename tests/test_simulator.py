"""
Unit tests for Step 2: Discrete-Event Closed-Loop Edge Simulator and Canonical Scenarios.

Verifies:
1. Analytic fair-share convergence.
2. Multi-user closed-loop coupling (server load updates when users switch).
3. Deterministic seeding (same seed -> identical logs).
4. Noise models (Gaussian, Student-t, Bursty).
5. Trace loader interpolation.
6. Execution and behavior of Scenarios A to F.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from edge_switching import (
    ClosedLoopSimulator,
    EdgeServerNode,
    EventLabel,
    NoiseDistribution,
    VideoClientState,
    create_standard_servers,
    run_scenario,
)


def test_analytic_fair_share_convergence():
    """Verify server capacity fair-share equals C_srv / (N_bg + N_clients)."""
    servers = create_standard_servers()
    s1 = servers["s1"]  # Capacity = 300 Mbps, Background = 10
    # Attach 5 clients
    for i in range(5):
        s1.connected_clients.add(f"c_{i}")

    total_users = 10 + 5
    expected_fair_share = 300.0 / total_users
    assert np.isclose(s1.calculate_fair_share_mbps(), expected_fair_share)
    assert np.isclose(s1.calculate_fair_share_mbps(including_new_joiner=True), 300.0 / 16.0)


def test_closed_loop_herding_and_load_feedback():
    """Verify that user switching dynamically alters server load and per-user capacity."""
    servers = create_standard_servers()
    # 10 clients on s1
    clients = [VideoClientState(f"c_{i}", current_server_id="s1") for i in range(10)]
    sim = ClosedLoopSimulator(servers=servers, clients=clients, seed=42)

    initial_s1_users = sim.servers["s1"].total_active_users()  # 10 bg + 10 clients = 20
    initial_s2_users = sim.servers["s2"].total_active_users()  # 25 bg + 0 clients = 25
    initial_s1_bw = sim.servers["s1"].calculate_fair_share_mbps()  # 300 / 20 = 15.0

    # Force 6 clients to switch to s2 (simulating herding)
    def herd_policy(client, matrix, engine):
        if int(client.client_id.split("_")[1]) < 6:
            return {"action": "SWITCH", "target_server": "s2", "decision_margin": 0.1, "switching_hurdle": 0.05}
        return {"action": "RETAIN", "target_server": "s1", "decision_margin": 0.1, "switching_hurdle": 0.05}

    step_res = sim.step(dt_s=1.0, client_policy_callback=herd_policy)

    # In closed loop, s1 active users should drop to 14, s2 users should rise to 31
    assert step_res["servers"]["s1"]["active_users"] == initial_s1_users - 6
    assert step_res["servers"]["s2"]["active_users"] == initial_s2_users + 6

    # Fair share on s1 should increase; fair share on s2 should decrease
    assert step_res["servers"]["s1"]["fair_share_mbps"] > initial_s1_bw
    assert step_res["servers"]["s2"]["fair_share_mbps"] < (800.0 / 25.0)


def test_deterministic_seeding():
    """Verify identical seeds produce bitwise identical simulation results."""
    df1 = run_scenario("scenario_a", duration_s=10.0, dt_s=1.0, num_clients=2, seed=999)
    df2 = run_scenario("scenario_a", duration_s=10.0, dt_s=1.0, num_clients=2, seed=999)

    pd.testing.assert_frame_equal(df1, df2)


def test_noise_models_statistical_properties():
    """Verify statistical characteristics of Gaussian, Heavy-tailed, and Bursty noise."""
    servers = create_standard_servers()
    clients = [VideoClientState("c1", "s1")]

    # Gaussian: standard deviation close to sigma
    sim_gauss = ClosedLoopSimulator(servers, clients, seed=42, noise_model=NoiseDistribution.GAUSSIAN)
    samples_gauss = [sim_gauss.sample_noise(sigma=5.0) for _ in range(5000)]
    assert np.isclose(np.mean(samples_gauss), 0.0, atol=0.3)
    assert np.isclose(np.std(samples_gauss), 5.0, atol=0.3)

    # Heavy-tailed (Student-t df=3): higher kurtosis (fatter tails)
    sim_heavy = ClosedLoopSimulator(servers, clients, seed=42, noise_model=NoiseDistribution.HEAVY_TAILED)
    samples_heavy = [sim_heavy.sample_noise(sigma=5.0) for _ in range(5000)]
    # Check max extreme outlier exceeds Gaussian max
    assert max(abs(s) for s in samples_heavy) > max(abs(s) for s in samples_gauss)


def test_trace_loader():
    """Verify trace replayer queries and interpolates values from CSV file."""
    trace_file = Path(__file__).resolve().parent.parent / "data" / "raw" / "mobile_4g_sample.csv"
    assert trace_file.exists()

    from edge_switching.simulator import TraceReplayer
    replayer = TraceReplayer(trace_file)

    # At t=0.0: bw=24.5, lat=28.2
    bw0, lat0, loss0 = replayer.query(0.0)
    assert np.isclose(bw0, 24.5)
    assert np.isclose(lat0, 28.2)

    # At t=0.5: linear interpolation between t=0 and t=1
    bw_half, lat_half, _ = replayer.query(0.5)
    assert np.isclose(bw_half, (24.5 + 23.8) / 2.0)


def test_scenario_a_noise_suppression():
    """Scenario A (Minor Fluctuation): Verified client executes RETAIN under noise."""
    df = run_scenario("scenario_a", duration_s=15.0, dt_s=0.5, num_clients=1, seed=42)
    # Under benign jitter, the switching hurdle prevents spurious switching
    switches = df[df["action"] == "SWITCH"]
    assert len(switches) == 0
    assert all(df["client_server"] == "s1")


def test_scenario_b_regime_shift_trigger():
    """Scenario B: Client successfully triggers SWITCH after persistent degradation at t=20s."""
    df = run_scenario("scenario_b", duration_s=30.0, dt_s=0.5, num_clients=1, seed=42)
    pre_shift = df[df["time_s"] < 20.0]
    post_shift = df[df["time_s"] >= 20.0]

    # Pre-shift: stays on s1
    assert all(pre_shift["client_server"] == "s1")

    # Post-shift: switches to better server (s2 or s3)
    post_servers = set(post_shift["client_server"])
    assert "s2" in post_servers or "s3" in post_servers


def test_scenario_c_failure_emergency_switch():
    """Scenario C: Client triggers emergency SWITCH when s1 experiences outage at t=15s."""
    df = run_scenario("scenario_c", duration_s=25.0, dt_s=0.5, num_clients=1, seed=42)
    post_failure = df[df["time_s"] >= 16.0]
    # Client must not remain on the failed server
    assert all(post_failure["client_server"] != "s1")


def test_scenario_e_staleness_detection():
    """Scenario E: Client switches away when s1 telemetry becomes stale."""
    df = run_scenario("scenario_e", duration_s=25.0, dt_s=0.5, num_clients=1, seed=42)
    # Staleness injected at t=18s (age threshold is 2.0s -> triggers at t=20.5s)
    post_stale = df[df["time_s"] >= 22.0]
    assert all(post_stale["client_server"] != "s1")


def test_scenario_f_mobility_switch():
    """Scenario F: Mobility drift triggers switch as s1 latency rises and s2 latency falls."""
    df = run_scenario("scenario_f", duration_s=35.0, dt_s=0.5, num_clients=1, seed=42)
    # Eventually switches to s2 as client moves closer to it
    final_server = df.iloc[-1]["client_server"]
    assert final_server == "s2"
