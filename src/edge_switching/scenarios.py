from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from edge_switching.qos_model import Criterion, SystemConfig
from edge_switching.simulator import (
    ClosedLoopSimulator,
    EdgeServerNode,
    EventLabel,
    NoiseDistribution,
    VideoClientState,
)
_TRACE_CACHE: Dict[str, Any] = {}

def create_standard_servers() -> Dict[str, EdgeServerNode]:
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


def create_tripolar_servers() -> Dict[str, EdgeServerNode]:
    return {
        "s1": EdgeServerNode(
            server_id="s1",
            name="MicroEdge-UltraLowLatency",
            capacity_mbps=30.0,
            base_latency_ms=12.0,
            background_users=2,
            max_capacity_users=20,
        ),
        "s2": EdgeServerNode(
            server_id="s2",
            name="Regional-HighCapacityCloudlet",
            capacity_mbps=1200.0,
            base_latency_ms=55.0,
            background_users=10,
            max_capacity_users=100,
        ),
        "s3": EdgeServerNode(
            server_id="s3",
            name="Metro-ReliableBalancedNode",
            capacity_mbps=250.0,
            base_latency_ms=28.0,
            background_users=5,
            max_capacity_users=50,
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
    policy: Optional[Any] = None,
    weights: Optional[Dict[Criterion, float]] = None,
    servers: Optional[Dict[str, EdgeServerNode]] = None,
    initial_buffer_s: Optional[float] = None,
) -> pd.DataFrame:
    if servers is None:
        if scenario_id == "scenario_low_buffer":
            servers = {
                "s1": EdgeServerNode("s1", "Edge-Local-Modest", capacity_mbps=40.0, base_latency_ms=25.0, background_users=4, max_capacity_users=20),
                "s2": EdgeServerNode("s2", "Edge-Metro-HighCap", capacity_mbps=40.0, base_latency_ms=25.0, background_users=4, max_capacity_users=80),
                "s3": EdgeServerNode("s3", "Edge-Regional-Backup", capacity_mbps=500.0, base_latency_ms=70.0, background_users=25, max_capacity_users=100),
            }
        elif scenario_id == "scenario_tripolar":
            servers = create_tripolar_servers()
        else:
            servers = create_standard_servers()

    init_buf = initial_buffer_s if initial_buffer_s is not None else (0.20 if scenario_id == "scenario_low_buffer" else 3.0)
    clients = [
        VideoClientState(client_id=f"client_{i}", current_server_id="s1", buffer_occupancy_s=init_buf)
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
            k = str(edge_dir)
            if k not in _TRACE_CACHE:
                _TRACE_CACHE[k] = TraceReplayer(edge_dir)
            sim.trace_replayers["s1"] = _TRACE_CACHE[k]
            sim.servers["s1"].capacity_mbps = 850.0
            sim.servers["s1"].base_latency_ms = 15.0

    elif scenario_id == "scenario_puffer":
        puffer_dir = raw_data_dir / "puffer-fake-sample"
        if puffer_dir.exists():
            k = str(puffer_dir)
            if k not in _TRACE_CACHE:
                _TRACE_CACHE[k] = TraceReplayer(puffer_dir)
            sim.trace_replayers["s1"] = _TRACE_CACHE[k]
            sim.servers["s1"].capacity_mbps = 50.0
            sim.servers["s1"].base_latency_ms = 40.0

    elif trace_path and trace_path.exists():
        k = str(trace_path)
        if k not in _TRACE_CACHE:
            _TRACE_CACHE[k] = TraceReplayer(trace_path)
        sim.trace_replayers["s1"] = _TRACE_CACHE[k]

    records = []
    t = 0.0
    while t < duration_s:
        event_label = EventLabel.NOISE

        if scenario_id == "scenario_a":
            event_label = EventLabel.NOISE

        elif scenario_id == "scenario_b":
            if t >= 20.0:
                event_label = EventLabel.REGIME_SHIFT
                sim.servers["s1"].capacity_mbps = 80.0
                sim.servers["s1"].base_latency_ms = 75.0

        elif scenario_id == "scenario_c":
            if t >= 15.0:
                event_label = EventLabel.FAILURE
                sim.servers["s1"].is_available = False

        elif scenario_id == "scenario_d":
            if t >= 20.0:
                event_label = EventLabel.FLASH_CROWD
                sim.servers["s2"].background_users = 70

        elif scenario_id == "scenario_e":
            if t >= 18.0:
                event_label = EventLabel.FAILURE
                sim.servers["s1"].heartbeat_active = False

        elif scenario_id == "scenario_f":
            if t >= 10.0:
                event_label = EventLabel.GRADUAL_DRIFT
                sim.servers["s1"].base_latency_ms = min(130.0, 25.0 + 1.8 * (t - 10.0))
                sim.servers["s2"].base_latency_ms = max(20.0, 45.0 - 0.7 * (t - 10.0))

        elif scenario_id == "scenario_edge_infra":
            event_label = EventLabel.NOISE

        elif scenario_id == "scenario_puffer":
            event_label = EventLabel.NOISE

        elif scenario_id == "scenario_low_buffer":
            if 3.0 <= t < 3.5:
                event_label = EventLabel.REGIME_SHIFT
                sim.servers["s2"].capacity_mbps = 500.0
                for c in sim.clients:
                    c.buffer_occupancy_s = min(c.buffer_occupancy_s, 0.08)
            elif t >= 3.5:
                event_label = EventLabel.NOISE
                sim.servers["s2"].capacity_mbps = 500.0

        elif scenario_id == "scenario_qos_stress":
            sim.servers["s3"].base_latency_ms = 35.0
            if t >= 6.0:
                event_label = EventLabel.REGIME_SHIFT
                sim.servers["s1"].base_latency_ms = 135.0
                sim.servers["s1"].packet_loss = 0.08
            if t >= 8.0:
                event_label = EventLabel.FLASH_CROWD
                sim.servers["s2"].background_users = 85

        policy_cb = None
        if policy is not None:
            policy_cb = lambda c, m, eng: policy.decide(c, m, eng, sim.telemetry_history)

        step_result = sim.step(
            dt_s=dt_s,
            client_policy_callback=policy_cb,
            weights=weights,
            current_event_label=event_label,
        )

        c0 = step_result["clients"]["client_0"]
        d0 = step_result["decisions"]["client_0"]

        records.append({
            "time_s": step_result["time_s"],
            "scenario": scenario_id,
            "seed": seed,
            "event_label": event_label.value,
            "client_server": c0["server"],
            "effective_bw_mbps": c0["effective_bw_mbps"],
            "client_latency_ms": c0["latency_ms"],
            "client_packet_loss": c0["packet_loss"],
            "client_available": c0["available"],
            "bitrate_mbps": c0["bitrate_mbps"],
            "buffer_s": c0["buffer_s"],
            "stall_s": c0["stall_s"],
            "action": d0.get("action", "RETAIN"),
            "target_server": d0.get("target_server", c0["server"]),
            "decision_margin": float(d0.get("decision_margin", 0.0)),
            "switching_hurdle": float(d0.get("switching_hurdle", 0.0)),
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