"""
Canonical Scenario Builders and Runner for PerCom 2027 Evaluation.

Implements Scenarios A to F:
- Scenario A: Minor Fluctuation (Noise Only)
- Scenario B: Meaningful Change (Regime Shift)
- Scenario C: Constraint Failure (SLO Violation / Outage)
- Scenario D: Multi-User Contention & Herding
- Scenario E: Telemetry Staleness (Sensor Drop)
- Scenario F: Client Mobility & Gradual Drift
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from edge_switching.qos_model import Criterion, SystemConfig
from edge_switching.simulator import (
    ClosedLoopSimulator,
    EdgeServerNode,
    EventLabel,
    NoiseDistribution,
    VideoClientState,
)


def create_standard_servers() -> Dict[str, EdgeServerNode]:
    """Create standard three-tier edge candidate servers."""
    return {
        "s1": EdgeServerNode(
            server_id="s1",
            name="Edge-Local-Alpha",
            capacity_mbps=300.0,
            base_latency_ms=25.0,
            background_users=10,
            max_capacity_users=50,
        ),
        "s2": EdgeServerNode(
            server_id="s2",
            name="Edge-Metro-Beta",
            capacity_mbps=800.0,
            base_latency_ms=45.0,
            background_users=25,
            max_capacity_users=100,
        ),
        "s3": EdgeServerNode(
            server_id="s3",
            name="Edge-Regional-Gamma",
            capacity_mbps=1500.0,
            base_latency_ms=80.0,
            background_users=60,
            max_capacity_users=200,
        ),
    }


def run_scenario(
    scenario_id: str,
    duration_s: float = 60.0,
    dt_s: float = 0.5,
    num_clients: int = 5,
    seed: int = 42,
    noise_model: NoiseDistribution = NoiseDistribution.GAUSSIAN,
    noise_sigma_lat: float = 3.0,
    noise_sigma_bw: float = 1.5,
    trace_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Execute a canonical evaluation scenario and return tabular log DataFrame.
    """
    servers = create_standard_servers()
    clients = [
        VideoClientState(client_id=f"client_{i}", current_server_id="s1")
        for i in range(num_clients)
    ]
    config = SystemConfig()
    sim = ClosedLoopSimulator(
        servers=servers,
        clients=clients,
        config=config,
        seed=seed,
        noise_model=noise_model,
        noise_sigma_latency=noise_sigma_lat,
        noise_sigma_bw=noise_sigma_bw,
    )

    root_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = root_dir / "data" / "raw"

    from edge_switching.simulator import TraceReplayer
    if scenario_id == "scenario_edge_infra":
        edge_dir = raw_data_dir / "edge-infrastructure-traces"
        if edge_dir.exists():
            sim.trace_replayers["s1"] = TraceReplayer(edge_dir)
            sim.servers["s1"].capacity_mbps = 850.0
            sim.servers["s1"].base_latency_ms = 15.0

    elif scenario_id == "scenario_puffer":
        puffer_dir = raw_data_dir / "puffer-fake-sample"
        if puffer_dir.exists():
            sim.trace_replayers["s1"] = TraceReplayer(puffer_dir)
            sim.servers["s1"].capacity_mbps = 50.0
            sim.servers["s1"].base_latency_ms = 40.0

    elif trace_path and trace_path.exists():
        sim.trace_replayers["s1"] = TraceReplayer(trace_path)

    records = []
    t = 0.0
    while t < duration_s:
        # Ground-truth event labeling and perturbation injection
        event_label = EventLabel.NOISE

        if scenario_id == "scenario_a":
            # Scenario A: Stationary noise only
            event_label = EventLabel.NOISE

        elif scenario_id == "scenario_b":
            # Scenario B: Regime shift on s1 at t >= 20.0s
            if t >= 20.0:
                event_label = EventLabel.REGIME_SHIFT
                sim.servers["s1"].capacity_mbps = 80.0     # -73% capacity drop
                sim.servers["s1"].base_latency_ms = 75.0   # +50ms latency inflation

        elif scenario_id == "scenario_c":
            # Scenario C: Server outage on s1 at t >= 15.0s
            if t >= 15.0:
                event_label = EventLabel.FAILURE
                sim.servers["s1"].is_available = False

        elif scenario_id == "scenario_d":
            # Scenario D: Flash crowd of 35 competing users joins s2 at t >= 20.0s
            if t >= 20.0:
                event_label = EventLabel.FLASH_CROWD
                sim.servers["s2"].background_users = 70

        elif scenario_id == "scenario_e":
            # Scenario E: Telemetry staleness on s1 at t >= 18.0s (heartbeat packet drop)
            if t >= 18.0:
                event_label = EventLabel.FAILURE
                sim.servers["s1"].heartbeat_active = False

        elif scenario_id == "scenario_f":
            # Scenario F: Client mobility drift (s1 latency increases, s2 decreases)
            if t >= 10.0:
                event_label = EventLabel.GRADUAL_DRIFT
                sim.servers["s1"].base_latency_ms = min(130.0, 25.0 + 1.8 * (t - 10.0))
                sim.servers["s2"].base_latency_ms = max(20.0, 45.0 - 0.7 * (t - 10.0))

        elif scenario_id == "scenario_edge_infra":
            event_label = EventLabel.NOISE

        elif scenario_id == "scenario_puffer":
            event_label = EventLabel.NOISE

        # Advance simulation
        step_result = sim.step(dt_s=dt_s, current_event_label=event_label)

        # Log primary client (client_0)
        c0 = step_result["clients"]["client_0"]
        d0 = step_result["decisions"]["client_0"]

        records.append({
            "time_s": step_result["time_s"],
            "scenario": scenario_id,
            "seed": seed,
            "event_label": event_label.value,
            "client_server": c0["server"],
            "effective_bw_mbps": c0["effective_bw_mbps"],
            "buffer_s": c0["buffer_s"],
            "stall_s": c0["stall_s"],
            "action": d0["action"],
            "target_server": d0["target_server"],
            "decision_margin": d0["decision_margin"],
            "switching_hurdle": d0["switching_hurdle"],
            "s1_active_users": step_result["servers"]["s1"]["active_users"],
            "s2_active_users": step_result["servers"]["s2"]["active_users"],
            "s3_active_users": step_result["servers"]["s3"]["active_users"],
            "s1_fair_share": step_result["servers"]["s1"]["fair_share_mbps"],
            "s2_fair_share": step_result["servers"]["s2"]["fair_share_mbps"],
            "client_qoe": c0["qoe"],
        })

        t += dt_s

    df = pd.DataFrame(records)
    return df
