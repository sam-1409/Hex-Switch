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

__version__ = "0.1.0"

__all__ = [
    "Criterion",
    "DecisionMatrix",
    "QoSFormalizationEngine",
    "ServerTelemetry",
    "SystemConfig",
]
