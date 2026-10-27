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
]
