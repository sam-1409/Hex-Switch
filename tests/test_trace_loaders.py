from pathlib import Path
import pytest
import numpy as np

from edge_switching.trace_loaders import (
    EdgeInfrastructureTraceLoader,
    PufferTraceLoader,
)
from edge_switching.simulator import TraceReplayer
from edge_switching.scenarios import run_scenario

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"

def test_edge_infrastructure_trace_loader():
    edge_dir = DATA_DIR / "edge-infrastructure-traces"
    assert edge_dir.exists(), f"Edge trace dir not found at {edge_dir}"

    loader = EdgeInfrastructureTraceLoader(edge_dir)
    assert len(loader.timestamps) > 0
    assert len(loader.bandwidths_mbps) == len(loader.timestamps)
    assert len(loader.latencies_ms) == len(loader.timestamps)

    # Query at t=0.0
    bw0, lat0, is_outage = loader.query(0.0)
    assert bw0 > 100.0  
    assert lat0 < 5.0   

def test_puffer_trace_loader():
    puffer_dir = DATA_DIR / "puffer-fake-sample"
    assert puffer_dir.exists(), f"Puffer trace dir not found at {puffer_dir}"

    loader = PufferTraceLoader(puffer_dir)
    assert len(loader.timestamps) > 0
    assert len(loader.bandwidths_mbps) == len(loader.timestamps)
    assert len(loader.latencies_ms) == len(loader.timestamps)

    # Query at t=0.0
    bw0, lat0 = loader.query(0.0)
    assert bw0 > 0.0    # Delivery rate converted to Mbps
    assert lat0 > 10.0  # Smoothed RTT in ms

def test_trace_replayer_directory_autodetect():
    edge_dir = DATA_DIR / "edge-infrastructure-traces"
    puffer_dir = DATA_DIR / "puffer-fake-sample"

    replayer_edge = TraceReplayer(edge_dir)
    assert replayer_edge.edge_infra_loader is not None
    bw_e, lat_e, loss_e = replayer_edge.query(10.0)
    assert bw_e > 0.0

    replayer_puffer = TraceReplayer(puffer_dir)
    assert replayer_puffer.puffer_loader is not None
    bw_p, lat_p, loss_p = replayer_puffer.query(5.0)
    assert bw_p > 0.0

def test_scenario_edge_infra_execution():
    df = run_scenario("scenario_edge_infra", duration_s=20.0, dt_s=0.5, num_clients=2, seed=42)
    assert len(df) == 40
    assert "effective_bw_mbps" in df.columns
    assert df["effective_bw_mbps"].mean() > 5.0

def test_scenario_puffer_execution():
    df = run_scenario("scenario_puffer", duration_s=20.0, dt_s=0.5, num_clients=2, seed=42)
    assert len(df) == 40
    assert "buffer_s" in df.columns
    assert "effective_bw_mbps" in df.columns