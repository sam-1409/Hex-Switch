import sys
import time
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import pandas as pd

from edge_switching.qos_model import (
    Criterion,
    DecisionMatrix,
    QoSFormalizationEngine,
    ServerTelemetry,
    SystemConfig,
)
from edge_switching.simulator import (
    ClosedLoopSimulator,
    EdgeServerNode,
    NoiseDistribution,
    VideoClientState,
)
from edge_switching.scenarios import create_standard_servers
from edge_switching.policies import (
    AhpTopsisPolicy,
    MaxTotalBandwidthPolicy,
    MinLatencyPolicy,
    ProposedMarginPolicy,
)
from edge_switching.weighting import get_literature_ahp_profile, STANDARD_CRITERIA
from edge_switching.explainer import EdgeSwitchingExplainer
from edge_switching.auditor import PostDecisionAuditor

LOGS_DIR = ROOT_DIR / "logs" / "scalability"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def run_client_scaling_herding():
    print("1. Running Multi-User Contention & Herding Sweep (N=1 to 100)...")
    client_counts = [1, 5, 10, 20, 35, 50, 75, 100]
    policies = {
        "Proposed-Margin": ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65, min_dwell_steps=4),
        "AHP-TOPSIS": AhpTopsisPolicy(),
        "Min-Latency": MinLatencyPolicy(),
        "Max-Total-BW": MaxTotalBandwidthPolicy(),
    }
    w_video, _, _ = get_literature_ahp_profile("video_streaming")
    duration_s = 20.0
    dt_s = 0.5
    num_seeds = 30
    records = []
    seed_level_records = []

    for n_clients in client_counts:
        print(f"  Evaluating N = {n_clients} clients...")
        for pol_name, policy in policies.items():
            qoe_list = []
            switch_rates = []
            ping_pong_rates = []
            bw_list = []
            viol_list = []

            for seed in range(num_seeds):
                servers = create_standard_servers()
                server_ids = list(servers.keys())
                clients = [
                    VideoClientState(f"c{i}", server_ids[i % len(server_ids)], buffer_occupancy_s=3.0)
                    for i in range(n_clients)
                ]

                sim = ClosedLoopSimulator(
                    servers=servers,
                    clients=clients,
                    seed=seed,
                    noise_model=NoiseDistribution.GAUSSIAN,
                    noise_sigma_latency=3.0,
                    noise_sigma_bw=1.5,
                )

                steps = int(duration_s / dt_s)
                total_switches = 0
                ping_pong_switches = 0
                step_qoes = []
                step_bws = []
                step_viols = 0
                total_client_steps = 0

                prev_servers = {c.client_id: c.current_server_id for c in clients}
                prev_prev_servers = {c.client_id: c.current_server_id for c in clients}

                for step_idx in range(steps):
                    sim.step(dt_s)
                    telemetries = [sim.latest_telemetry[sid] for sid in servers if sid in sim.latest_telemetry]
                    if not telemetries:
                        continue
                    matrix = sim.engine.build_matrices(telemetries, w_video, sim.current_time_s)

                    for client in sim.clients:
                        decision = policy.decide(
                            client=client,
                            matrix=matrix,
                            engine=sim.engine,
                            telemetry_history=sim.telemetry_history,
                        )

                        action = decision.get("action", "RETAIN")
                        target = decision.get("target_server", client.current_server_id)

                        curr_s = client.current_server_id
                        if action == "SWITCH" and target != curr_s:
                            total_switches += 1
                            # Check ping pong
                            if target == prev_prev_servers[client.client_id]:
                                ping_pong_switches += 1

                            # Update client server
                            sim.servers[curr_s].connected_clients.discard(client.client_id)
                            sim.servers[target].connected_clients.add(client.client_id)
                            client.current_server_id = target

                        prev_prev_servers[client.client_id] = prev_servers[client.client_id]
                        prev_servers[client.client_id] = client.current_server_id

                        # Measure observed QoS for this client
                        active_srv = sim.servers[client.current_server_id]
                        n_on_srv = len(active_srv.connected_clients)
                        fair_bw = active_srv.capacity_mbps / max(1, n_on_srv)
                        eff_bw = min(50.0, fair_bw)

                        # Video QoE proxy
                        step_qoe = 0.8 * (eff_bw / 25.0) - (0.005 * active_srv.base_latency_ms) - (0.5 if action == "SWITCH" else 0.0)
                        step_qoes.append(step_qoe)
                        step_bws.append(eff_bw)

                        if active_srv.base_latency_ms > 120.0 or eff_bw < 2.0:
                            step_viols += 1
                        total_client_steps += 1

                qoe_list.append(float(np.mean(step_qoes)))
                seed_level_records.append({
                    "num_clients": n_clients, "policy": pol_name, "seed": seed,
                    "mean_step_qoe_proxy": float(np.mean(step_qoes)),
                    "switch_rate_per_min": (total_switches / max(1, n_clients)) / (duration_s / 60.0),
                    "ping_pong_rate_pct": (ping_pong_switches / max(1, total_switches)) * 100.0 if total_switches else 0.0,
                    "mean_effective_bw_mbps": float(np.mean(step_bws)),
                    "qos_violation_rate": step_viols / max(1, total_client_steps),
                })
                sw_rate_per_min = (total_switches / max(1, n_clients)) / (duration_s / 60.0)
                switch_rates.append(sw_rate_per_min)
                pp_rate = (ping_pong_switches / max(1, total_switches)) * 100.0 if total_switches > 0 else 0.0
                ping_pong_rates.append(pp_rate)
                bw_list.append(float(np.mean(step_bws)))
                viol_list.append((step_viols / max(1, total_client_steps)) * 100.0)

            records.append({
                "num_clients": n_clients,
                "policy": pol_name,
                "mean_qoe": round(float(np.mean(qoe_list)), 4),
                "std_qoe": round(float(np.std(qoe_list)), 4),
                "mean_switch_rate": round(float(np.mean(switch_rates)), 2),
                "std_switch_rate": round(float(np.std(switch_rates)), 2),
                "mean_ping_pong_pct": round(float(np.mean(ping_pong_rates)), 2),
                "std_ping_pong_pct": round(float(np.std(ping_pong_rates)), 2),
                "mean_effective_bw_mbps": round(float(np.mean(bw_list)), 2),
                "qos_violation_pct": round(float(np.mean(viol_list)), 2),
            })

    pd.DataFrame(seed_level_records).to_csv(LOGS_DIR / "data_client_scaling_herding_seed_level.csv", index=False)
    df_scale = pd.DataFrame(records)
    out_csv = LOGS_DIR / "data_client_scaling_herding.csv"
    df_scale.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")


def run_server_scaling_latency():
    print("2. Measuring Algorithmic Execution Latency Scaling (M=3 to 32 servers)...")
    server_counts = [3, 5, 8, 12, 16, 24, 32]
    engine = QoSFormalizationEngine()
    policy = ProposedMarginPolicy(base_hurdle=0.05, k_noise=1.65)
    explainer = EdgeSwitchingExplainer(engine)
    auditor = PostDecisionAuditor()
    w_video, _, _ = get_literature_ahp_profile("video_streaming")
    now = 10.0

    records = []
    iterations = 200

    for m in server_counts:
        # Construct m synthetic server telemetries
        telemetries = []
        telem_dict = {}
        for s_idx in range(m):
            sid = f"srv_{s_idx}"
            lat = 15.0 + (s_idx * 4.0)
            cap = 100.0 + (s_idx * 50.0)
            users = 4 + (s_idx % 5)
            bottleneck = 20.0 + (s_idx * 2.0)
            load = 0.2 + (0.02 * (s_idx % 10))
            loss = 0.002 * (1 + (s_idx % 3))
            t = ServerTelemetry(sid, lat, cap, users, bottleneck, load, loss, True, now)
            telemetries.append(t)
            telem_dict[sid] = t

        client = VideoClientState("c0", "srv_0", buffer_occupancy_s=3.0)

        t_matrix_us = []
        t_policy_us = []
        t_explain_us = []
        t_audit_us = []
        t_total_us = []

        for _ in range(iterations):
            # 1. Matrix Construction O(M * K)
            t0 = time.perf_counter_ns()
            matrix = engine.build_matrices(telemetries, w_video, now)
            t1 = time.perf_counter_ns()

            # 2. Policy Selection O(M)
            decision = policy.decide(client, matrix, engine, None)
            t2 = time.perf_counter_ns()

            # 3. Explanation Generation O(K)
            exp = explainer.explain_decision(client, matrix, decision)
            t3 = time.perf_counter_ns()

            # 4. Post-Decision Audit O(1)
            audit = auditor.audit_decision("d_bench", client, matrix, decision, telem_dict, now)
            t4 = time.perf_counter_ns()

            t_matrix_us.append((t1 - t0) / 1000.0)
            t_policy_us.append((t2 - t1) / 1000.0)
            t_explain_us.append((t3 - t2) / 1000.0)
            t_audit_us.append((t4 - t3) / 1000.0)
            t_total_us.append((t4 - t0) / 1000.0)

        records.append({
            "num_servers_M": m,
            "matrix_build_us": round(float(np.mean(t_matrix_us)), 2),
            "policy_select_us": round(float(np.mean(t_policy_us)), 2),
            "explainer_us": round(float(np.mean(t_explain_us)), 2),
            "auditor_us": round(float(np.mean(t_audit_us)), 2),
            "total_pipeline_us": round(float(np.mean(t_total_us)), 2),
            "std_total_us": round(float(np.std(t_total_us)), 2),
            "p99_total_us": round(float(np.percentile(t_total_us, 99)), 2),
        })

    df_serv = pd.DataFrame(records)
    out_csv = LOGS_DIR / "data_server_scaling_latency.csv"
    df_serv.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")

def main():
    print("=" * 70)
    print("HEX-SWITCH: MULTI-USER SCALABILITY & HERDING BENCHMARK")
    print("=" * 70)

    run_client_scaling_herding()
    run_server_scaling_latency()

    print("\n[SUCCESS] Scalability benchmarks exported successfully.")

if __name__ == "__main__":
    main()