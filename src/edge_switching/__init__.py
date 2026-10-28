"""
edge_switching: Research codebase for HCI-Guided, XAI-Enabled Server Switching
for QoS-Aware Edge Computing Under Dynamic Network Conditions.

Target: IEEE PerCom 2027 A4E Workshop.
"""

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
    EventLabel,
    NoiseDistribution,
    VideoClientState,
)
from edge_switching.scenarios import (
    create_standard_servers,
    run_scenario,
)
from edge_switching.policies import (
    BaseSelectionPolicy,
    NearestServerPolicy,
    MinLatencyPolicy,
    MaxTotalBandwidthPolicy,
    MaxEffectiveBandwidthPolicy,
    AhpTopsisPolicy,
    HysteresisOnlyPolicy,
    QoEGreedyPolicy,
    ProposedMarginPolicy,
)
from edge_switching.metrics import (
    compute_linear_qoe,
    compute_qos_violation_rate,
    compute_switch_rate_per_minute,
    compute_ping_pong_rate,
    compute_mean_dwell_time,
    compute_scenario_metrics,
)

__version__ = "0.1.0"

__all__ = [
    "Criterion",
    "DecisionMatrix",
    "QoSFormalizationEngine",
    "ServerTelemetry",
    "SystemConfig",
    "ClosedLoopSimulator",
    "EdgeServerNode",
    "EventLabel",
    "NoiseDistribution",
    "VideoClientState",
    "create_standard_servers",
    "run_scenario",
    "BaseSelectionPolicy",
    "NearestServerPolicy",
    "MinLatencyPolicy",
    "MaxTotalBandwidthPolicy",
    "MaxEffectiveBandwidthPolicy",
    "AhpTopsisPolicy",
    "HysteresisOnlyPolicy",
    "QoEGreedyPolicy",
    "ProposedMarginPolicy",
    "compute_linear_qoe",
    "compute_qos_violation_rate",
    "compute_switch_rate_per_minute",
    "compute_ping_pong_rate",
    "compute_mean_dwell_time",
    "compute_scenario_metrics",
]
