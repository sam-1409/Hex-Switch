"""
Closed-Loop Multi-User Discrete-Event Edge Simulator for Video Streaming.

Key Features:
- Multi-user contention: Server fair share degrades dynamically as clients switch (+1 user effect and herding).
- Ground-truth event labeling: NOISE, REGIME_SHIFT, FAILURE, FLASH_CROWD, GRADUAL_DRIFT.
- Noise models: Gaussian jitter, Heavy-tailed (Student-t), Bursty Poisson.
- Video client playout dynamics: buffer evolution, rebuffering stalls, ABR adaptation, switching interruption.
- Trace-driven replay mode from CSV logs.
- Deterministic seeding for strict reproducibility.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from edge_switching.qos_model import (
    Criterion,
    DecisionMatrix,
    QoSFormalizationEngine,
    ServerTelemetry,
    SystemConfig,
)


class EventLabel(str, Enum):
    """Ground-truth event labels for evaluation (never leaked to the client policy)."""
    NOISE = "NOISE"                    # Benign telemetry jitter around stable mean
    REGIME_SHIFT = "REGIME_SHIFT"      # Persistent step change in capacity or latency
    FAILURE = "FAILURE"                # Server crash or hard constraint violation
    FLASH_CROWD = "FLASH_CROWD"        # Sudden influx of competing users
    GRADUAL_DRIFT = "GRADUAL_DRIFT"    # Slow monotonic physical drift (mobility)


class NoiseDistribution(str, Enum):
    """Telemetry measurement noise profiles."""
    GAUSSIAN = "gaussian"
    HEAVY_TAILED = "heavy_tailed"      # Student-t with df=3
    BURSTY = "bursty"                  # Occasional severe packet drop/latency spikes


@dataclass
class VideoClientState:
    """State of an individual streaming client in the closed-loop simulator."""
    client_id: str
    current_server_id: str
    buffer_occupancy_s: float = 3.0
    bitrate_ladder_mbps: List[float] = field(default_factory=lambda: [1.5, 3.0, 5.0, 8.0, 15.0, 25.0])
    selected_bitrate_mbps: float = 3.0
    total_stalls: int = 0
    total_stall_duration_s: float = 0.0
    total_switches: int = 0
    ping_pong_switches: int = 0
    recent_servers: List[str] = field(default_factory=list)
    cumulative_qoe: float = 0.0

    def step_playback(self, dt_s: float, effective_bw_mbps: float) -> Tuple[float, float]:
        """
        Simulate video playout and chunk downloading over time step dt_s.
        Returns (stall_duration_in_step, current_buffer).
        """
        # ABR selection: choose highest bitrate ladder rung that fits comfortably in effective_bw
        safe_bw = max(1.0, effective_bw_mbps * 0.85)
        suitable_rungs = [r for r in self.bitrate_ladder_mbps if r <= safe_bw]
        self.selected_bitrate_mbps = suitable_rungs[-1] if suitable_rungs else self.bitrate_ladder_mbps[0]

        # Download progress: chunk data received vs playout drain
        download_rate_ratio = effective_bw_mbps / self.selected_bitrate_mbps
        buffer_delta = (download_rate_ratio - 1.0) * dt_s

        new_buffer = self.buffer_occupancy_s + buffer_delta
        step_stall = 0.0
        if new_buffer < 0.0:
            step_stall = abs(new_buffer)
            self.total_stalls += 1
            self.total_stall_duration_s += step_stall
            self.buffer_occupancy_s = 0.0
        else:
            self.buffer_occupancy_s = min(30.0, new_buffer)

        # Simple linear QoE utility component (ITU-T P.1203 proxy):
        # QoE = Bitrate - 4.3 * StallDuration
        qoe_step = (self.selected_bitrate_mbps * dt_s) - (4.3 * step_stall)
        self.cumulative_qoe += qoe_step

        return step_stall, self.buffer_occupancy_s

    def apply_switch(self, target_server_id: str, switch_delay_s: float):
        """Execute server switch with playout penalty."""
        if target_server_id == self.current_server_id:
            return

        # Check for ping-pong switching (switching back to a server held in last 3 epochs)
        if target_server_id in self.recent_servers[-2:]:
            self.ping_pong_switches += 1

        self.recent_servers.append(self.current_server_id)
        if len(self.recent_servers) > 5:
            self.recent_servers.pop(0)

        self.current_server_id = target_server_id
        self.total_switches += 1

        # Drain buffer by switching delay (DNS + TLS + pipeline re-anchor)
        self.buffer_occupancy_s = max(0.0, self.buffer_occupancy_s - switch_delay_s)
        self.cumulative_qoe -= 1.5  # Switching subjective annoyance penalty


@dataclass
class EdgeServerNode:
    """State and capacity model of an edge server."""
    server_id: str
    name: str
    capacity_mbps: float
    base_latency_ms: float
    background_users: int = 10
    max_capacity_users: int = 100
    is_available: bool = True
    heartbeat_active: bool = True
    base_packet_loss: float = 0.001
    last_heartbeat_time_s: float = 0.0

    # Dynamic connected clients set for closed-loop load tracking
    connected_clients: Set[str] = field(default_factory=set)

    def total_active_users(self) -> int:
        return self.background_users + len(self.connected_clients)

    def calculate_fair_share_mbps(self, including_new_joiner: bool = False) -> float:
        """
        Compute max-min fair share per user accounting for closed-loop contention.
        """
        users = self.total_active_users() + (1 if including_new_joiner else 0)
        users = max(1, users)
        return float(self.capacity_mbps / users)

    def calculate_load_ratio(self) -> float:
        """Resource utilization fraction in [0, 1]."""
        return float(min(1.0, self.total_active_users() / float(self.max_capacity_users)))


class TraceReplayer:
    """Replays real-world throughput and latency traces (Cellular, Edge Infrastructure, Puffer)."""

    def __init__(self, trace_path: Optional[Path] = None):
        from edge_switching.trace_loaders import (
            EdgeInfrastructureTraceLoader,
            PufferTraceLoader,
        )
        self.edge_infra_loader: Optional[EdgeInfrastructureTraceLoader] = None
        self.puffer_loader: Optional[PufferTraceLoader] = None
        self.timestamps: List[float] = []
        self.bandwidths: List[float] = []
        self.latencies: List[float] = []
        self.packet_losses: List[float] = []

        if trace_path:
            path = Path(trace_path)
            if path.is_dir():
                if (path / "BW-MessageSize.csv").exists():
                    self.edge_infra_loader = EdgeInfrastructureTraceLoader(path)
                elif list(path.glob("video_sent_*.csv")):
                    self.puffer_loader = PufferTraceLoader(path)
            elif path.is_file():
                self._load_csv(path)

    def _load_csv(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        for line in lines[1:]:
            parts = [float(p) for p in line.split(",")]
            self.timestamps.append(parts[0])
            self.bandwidths.append(parts[1])
            self.latencies.append(parts[2])
            self.packet_losses.append(parts[3] if len(parts) > 3 else 0.0)

    def query(self, current_time_s: float) -> Tuple[float, float, float]:
        """Interpolate (bandwidth_mbps, latency_ms, packet_loss) at current_time_s."""
        if self.edge_infra_loader:
            bw, lat, is_outage = self.edge_infra_loader.query(current_time_s)
            loss = 1.0 if is_outage else 0.001
            return bw, lat, loss

        if self.puffer_loader:
            bw, lat = self.puffer_loader.query(current_time_s)
            return bw, lat, 0.001

        if not self.timestamps:
            return 25.0, 30.0, 0.001  # Default fallback

        max_t = self.timestamps[-1]
        t = current_time_s % max_t if max_t > 0 else 0.0
        bw = float(np.interp(t, self.timestamps, self.bandwidths))
        lat = float(np.interp(t, self.timestamps, self.latencies))
        loss = float(np.interp(t, self.timestamps, self.packet_losses))
        return bw, lat, loss


class ClosedLoopSimulator:
    """
    Discrete-event multi-user closed-loop simulator.
    Simulates dynamic edge conditions, user switching feedback, and logs reproducible telemetry.
    """

    def __init__(
        self,
        servers: Dict[str, EdgeServerNode],
        clients: List[VideoClientState],
        config: Optional[SystemConfig] = None,
        seed: int = 42,
        noise_model: NoiseDistribution = NoiseDistribution.GAUSSIAN,
        noise_sigma_latency: float = 3.0,
        noise_sigma_bw: float = 1.5,
    ):
        self.servers = servers
        self.clients = clients
        self.config = config or SystemConfig()
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.noise_model = noise_model
        self.noise_sigma_latency = noise_sigma_latency
        self.noise_sigma_bw = noise_sigma_bw
        self.engine = QoSFormalizationEngine(self.config)
        self.current_time_s = 0.0

        # Trace replayer instances per server (optional)
        self.trace_replayers: Dict[str, TraceReplayer] = {}

        # Telemetry delay buffer: server_id -> list of (deliver_at_t, ServerTelemetry)
        self.telemetry_queue: Dict[str, List[Tuple[float, ServerTelemetry]]] = {
            sid: [] for sid in self.servers
        }
        self.latest_telemetry: Dict[str, ServerTelemetry] = {}
        self.telemetry_history: List[Dict[str, ServerTelemetry]] = []

        # Sync initial client placements
        for c in self.clients:
            if c.current_server_id in self.servers:
                self.servers[c.current_server_id].connected_clients.add(c.client_id)

    def sample_noise(self, sigma: float) -> float:
        """Sample noise according to selected distribution."""
        if sigma <= 0.0:
            return 0.0
        if self.noise_model == NoiseDistribution.GAUSSIAN:
            return float(self.rng.normal(0.0, sigma))
        elif self.noise_model == NoiseDistribution.HEAVY_TAILED:
            # Student-t with df=3 has heavier tails than normal
            return float(self.rng.standard_t(df=3) * (sigma / math.sqrt(3.0)))
        elif self.noise_model == NoiseDistribution.BURSTY:
            # 10% chance of a severe 3x burst, otherwise mild Gaussian
            if self.rng.rand() < 0.10:
                return float(self.rng.normal(0.0, sigma * 3.5))
            return float(self.rng.normal(0.0, sigma * 0.7))
        return float(self.rng.normal(0.0, sigma))

    def generate_server_telemetry(self, server_id: str) -> ServerTelemetry:
        """Generate physical state + measurement noise."""
        srv = self.servers[server_id]

        # Base physical values (possibly driven by trace)
        if server_id in self.trace_replayers and self.trace_replayers[server_id].timestamps:
            trace_bw, trace_lat, trace_loss = self.trace_replayers[server_id].query(self.current_time_s)
            path_bottleneck = trace_bw
            latency_true = trace_lat
            loss_true = trace_loss
        else:
            path_bottleneck = 50.0  # Access link bottleneck
            latency_true = srv.base_latency_ms
            loss_true = srv.base_packet_loss

        # Add measurement noise
        noise_lat = self.sample_noise(self.noise_sigma_latency)
        noise_bw = self.sample_noise(self.noise_sigma_bw)
        observed_latency = max(5.0, latency_true + noise_lat)
        observed_path_bw = max(0.5, path_bottleneck + noise_bw)
        observed_load = srv.calculate_load_ratio()
        observed_loss = max(0.0, min(1.0, loss_true + abs(self.sample_noise(0.002))))

        return ServerTelemetry(
            server_id=srv.server_id,
            latency_ms=observed_latency,
            server_capacity_mbps=srv.capacity_mbps,
            active_users=srv.total_active_users(),
            path_bottleneck_mbps=observed_path_bw,
            server_load=observed_load,
            packet_loss=observed_loss,
            is_available=srv.is_available,
            timestamp_s=self.current_time_s,
        )

    def step(
        self,
        dt_s: float,
        client_policy_callback: Optional[callable] = None,
        weights: Optional[Dict[Criterion, float]] = None,
        current_event_label: EventLabel = EventLabel.NOISE,
    ) -> Dict[str, object]:
        """
        Advance simulation by time step dt_s.
        1. Emits new noisy telemetry.
        2. Evaluates switching policy for each client.
        3. Updates closed-loop server loads.
        4. Simulates video chunk playout.
        5. Returns full logged snapshot.
        """
        self.current_time_s += dt_s

        # 1. Update telemetry for all servers with active heartbeat stream
        for sid, srv in self.servers.items():
            if srv.heartbeat_active:
                srv.last_heartbeat_time_s = self.current_time_s
                telemetry = self.generate_server_telemetry(sid)
                self.latest_telemetry[sid] = telemetry

        self.telemetry_history.append(dict(self.latest_telemetry))

        # Compile telemetry list for policy evaluation
        active_telemetries = [self.latest_telemetry[sid] for sid in self.servers if sid in self.latest_telemetry]

        default_weights = weights or {
            Criterion.LATENCY: 0.35,
            Criterion.EFFECTIVE_BW: 0.30,
            Criterion.SERVER_LOAD: 0.15,
            Criterion.PACKET_LOSS: 0.15,
            Criterion.AVAILABILITY: 0.05,
        }

        # Build decision matrix
        matrix = self.engine.build_matrices(active_telemetries, default_weights, self.current_time_s)

        # 2. Evaluate actions for each client
        client_actions = {}
        for client in self.clients:
            if client_policy_callback:
                decision = client_policy_callback(client, matrix, self.engine)
            else:
                # Default policy: formal switching evaluation from Step 1
                decision = self.engine.evaluate_switching_decision(
                    matrix,
                    current_server_id=client.current_server_id,
                    buffer_occupancy_s=client.buffer_occupancy_s,
                )

            action = decision["action"]
            target = decision["target_server"]
            client_actions[client.client_id] = decision

            # If switching, update server membership in closed loop
            if action == "SWITCH" and target != client.current_server_id and target in self.servers:
                old_server = client.current_server_id
                if old_server in self.servers and client.client_id in self.servers[old_server].connected_clients:
                    self.servers[old_server].connected_clients.remove(client.client_id)
                self.servers[target].connected_clients.add(client.client_id)
                client.apply_switch(target, self.config.switch_delay_s)

        # 3. Simulate video chunk playout with updated closed-loop fair-share throughput
        client_metrics = {}
        for client in self.clients:
            srv = self.servers.get(client.current_server_id)
            if srv and srv.is_available:
                fair_share = srv.calculate_fair_share_mbps()
                path_bw = 50.0  # Default bottleneck
                if srv.server_id in self.trace_replayers and self.trace_replayers[srv.server_id].timestamps:
                    path_bw, _, _ = self.trace_replayers[srv.server_id].query(self.current_time_s)
                effective_bw = min(path_bw, fair_share)
            else:
                effective_bw = 0.0  # Zero throughput on outage

            curr_lat = 999.0
            if client.current_server_id in self.latest_telemetry:
                curr_lat = self.latest_telemetry[client.current_server_id].latency_ms
            elif srv:
                curr_lat = srv.base_latency_ms

            stall_s, buf_s = client.step_playback(dt_s, effective_bw)
            client_metrics[client.client_id] = {
                "server": client.current_server_id,
                "effective_bw_mbps": effective_bw,
                "latency_ms": curr_lat,
                "buffer_s": buf_s,
                "stall_s": stall_s,
                "total_stalls": client.total_stalls,
                "total_switches": client.total_switches,
                "ping_pong_switches": client.ping_pong_switches,
                "bitrate_mbps": client.selected_bitrate_mbps,
                "qoe": client.cumulative_qoe,
            }

        server_loads = {
            sid: {
                "active_users": srv.total_active_users(),
                "load_ratio": srv.calculate_load_ratio(),
                "fair_share_mbps": srv.calculate_fair_share_mbps(),
                "is_available": srv.is_available,
            }
            for sid, srv in self.servers.items()
        }

        return {
            "time_s": self.current_time_s,
            "event_label": current_event_label.value,
            "clients": client_metrics,
            "servers": server_loads,
            "decisions": client_actions,
        }
        