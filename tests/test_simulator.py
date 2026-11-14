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
    servers = create_standard_servers()
    s1 = servers["s1"]  # Capacity = 300 Mbps, Background = 10
    for i in range(5):
        s1.connected_clients.add(f"c_{i}")

    total_users = 10 + 5
    expected_fair_share = 300.0 / total_users
    assert np.isclose(s1.calculate_fair_share_mbps(), expected_fair_share)
    assert np.isclose(s1.calculate_fair_share_mbps(including_new_joiner=True), 300.0 / 16.0)

def test_closed_loop_herding_and_load_feedback():
    servers = create_standard_servers()
    clients = [VideoClientState(f"c_{i}", current_server_id="s1") for i in range(10)]
    sim = ClosedLoopSimulator(servers=servers, clients=clients, seed=42)

    initial_s1_users = sim.servers["s1"].total_active_users()  # 10 bg + 10 clients = 20
    initial_s2_users = sim.servers["s2"].total_active_users()  # 25 bg + 0 clients = 25
    initial_s1_bw = sim.servers["s1"].calculate_fair_share_mbps()  # 300 / 20 = 15.0

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
    df1 = run_scenario("scenario_a", duration_s=10.0, dt_s=1.0, num_clients=2, seed=999)
    df2 = run_scenario("scenario_a", duration_s=10.0, dt_s=1.0, num_clients=2, seed=999)

    pd.testing.assert_frame_equal(df1, df2)

def test_noise_models_statistical_properties():
    servers = create_standard_servers()
    clients = [VideoClientState("c1", "s1")]
    sim_gauss = ClosedLoopSimulator(servers, clients, seed=42, noise_model=NoiseDistribution.GAUSSIAN)
    samples_gauss = [sim_gauss.sample_noise(sigma=5.0) for _ in range(5000)]
    assert np.isclose(np.mean(samples_gauss), 0.0, atol=0.3)
    assert np.isclose(np.std(samples_gauss), 5.0, atol=0.3)
    sim_heavy = ClosedLoopSimulator(servers, clients, seed=42, noise_model=NoiseDistribution.HEAVY_TAILED)
    samples_heavy = [sim_heavy.sample_noise(sigma=5.0) for _ in range(5000)]
    assert max(abs(s) for s in samples_heavy) > max(abs(s) for s in samples_gauss)

def test_trace_loader():
    trace_file = Path(__file__).resolve().parent.parent / "data" / "raw" / "mobile_4g_sample.csv"
    assert trace_file.exists()
    from edge_switching.simulator import TraceReplayer
    replayer = TraceReplayer(trace_file)
    bw0, lat0, loss0 = replayer.query(0.0)
    assert np.isclose(bw0, 24.5)
    assert np.isclose(lat0, 28.2)
    bw_half, lat_half, _ = replayer.query(0.5)
    assert np.isclose(bw_half, (24.5 + 23.8) / 2.0)

def test_scenario_a_noise_suppression():
    df = run_scenario("scenario_a", duration_s=15.0, dt_s=0.5, num_clients=1, seed=42)
    switches = df[df["action"] == "SWITCH"]
    assert len(switches) == 0
    assert all(df["client_server"] == "s1")

def test_scenario_b_regime_shift_trigger():
    df = run_scenario("scenario_b", duration_s=30.0, dt_s=0.5, num_clients=1, seed=42)
    pre_shift = df[df["time_s"] < 20.0]
    post_shift = df[df["time_s"] >= 20.0]
    assert all(pre_shift["client_server"] == "s1")
    post_servers = set(post_shift["client_server"])
    assert "s2" in post_servers or "s3" in post_servers

def test_scenario_c_failure_emergency_switch():
    df = run_scenario("scenario_c", duration_s=25.0, dt_s=0.5, num_clients=1, seed=42)
    post_failure = df[df["time_s"] >= 16.0]
    assert all(post_failure["client_server"] != "s1")

def test_scenario_e_staleness_detection():
    df = run_scenario("scenario_e", duration_s=25.0, dt_s=0.5, num_clients=1, seed=42)
    post_stale = df[df["time_s"] >= 22.0]
    assert all(post_stale["client_server"] != "s1")

def test_scenario_f_mobility_switch():
    df = run_scenario("scenario_f", duration_s=35.0, dt_s=0.5, num_clients=1, seed=42)
    final_server = df.iloc[-1]["client_server"]
    assert final_server == "s2"

def test_scenario_low_buffer_discriminates_multiplier():
    from edge_switching.scenarios import create_tripolar_servers
    from edge_switching.policies import ProposedMarginPolicy
    from edge_switching.metrics import compute_scenario_metrics

    policy = ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65, min_dwell_steps=4)
    df = run_scenario("scenario_low_buffer", duration_s=15.0, dt_s=0.5, num_clients=1, seed=42, policy=policy)
    metrics = compute_scenario_metrics(df, dt_s=0.5)
    assert metrics["stall_epoch_rate"] == 0.0

def test_scenario_qos_stress_and_tripolar_servers():
    from edge_switching.scenarios import create_tripolar_servers
    from edge_switching.policies import ProposedMarginPolicy, MinLatencyPolicy
    from edge_switching.metrics import compute_scenario_metrics
    tripolar = create_tripolar_servers()
    assert len(tripolar) == 3
    assert tripolar["s1"].base_latency_ms < tripolar["s2"].base_latency_ms
    assert tripolar["s2"].capacity_mbps > tripolar["s1"].capacity_mbps
    proposed = ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65, min_dwell_steps=4)
    df_prop = run_scenario("scenario_qos_stress", duration_s=20.0, dt_s=0.5, num_clients=1, seed=42, policy=proposed)
    m_prop = compute_scenario_metrics(df_prop, dt_s=0.5)
    assert m_prop["qos_violation_rate"] <= 0.05