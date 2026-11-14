from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

class EdgeInfrastructureTraceLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.bw_file = data_dir / "BW-MessageSize.csv"
        self.lat_file = data_dir / "Latency.csv"
        self.timestamps: List[float] = []
        self.bandwidths_mbps: List[float] = []
        self.latencies_ms: List[float] = []
        self.outage_flags: List[bool] = []
        self._load()

    def _load(self):
        if not self.bw_file.exists() or not self.lat_file.exists():
            return
        import io
        with open(self.bw_file, "r", encoding="utf-8") as f:
            bw_lines = [line.strip().rstrip(";") for line in f if line.strip()]
        df_bw = pd.read_csv(io.StringIO("\n".join(bw_lines)), sep=";")

        with open(self.lat_file, "r", encoding="utf-8") as f:
            lat_lines = [line.strip().rstrip(";") for line in f if line.strip()]
        df_lat = pd.read_csv(io.StringIO("\n".join(lat_lines)), sep=";")

        df_bw.columns = [c.strip() for c in df_bw.columns]
        df_lat.columns = [c.strip() for c in df_lat.columns]

        d02_d01_bw = df_bw[(df_bw["Source device"] == "D02") & (df_bw["Destination device"] == "D01")].copy()
        d02_d01_lat = df_lat[(df_lat["Source device"] == "D02") & (df_lat["Destination device"] == "D01")].copy()

        min_len = min(len(d02_d01_bw), len(d02_d01_lat))
        for i in range(min_len):
            t_s = float(i * 10.0)  # 10-second intervals
            bw_val = float(d02_d01_bw.iloc[i]["BW (Mbps)"])
            is_outage = (bw_val < 0)
            clean_bw = 0.0 if is_outage else bw_val
            lat_val = float(d02_d01_lat.iloc[i]["Average latency (ms)"])

            self.timestamps.append(t_s)
            self.bandwidths_mbps.append(clean_bw)
            self.latencies_ms.append(lat_val)
            self.outage_flags.append(is_outage)

    def query(self, current_time_s: float) -> Tuple[float, float, bool]:
        if not self.timestamps:
            return 800.0, 1.5, False  # Baseline fallback
        max_t = self.timestamps[-1]
        t = current_time_s % max_t
        bw = float(np.interp(t, self.timestamps, self.bandwidths_mbps))
        lat = float(np.interp(t, self.timestamps, self.latencies_ms))
        idx = int(np.argmin([abs(ts - t) for ts in self.timestamps]))
        outage = self.outage_flags[idx]
        return bw, lat, outage

class PufferTraceLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sent_files = list(data_dir.glob("video_sent_*.csv"))
        self.buffer_files = list(data_dir.glob("client_buffer_*.csv"))
        self.timestamps: List[float] = []
        self.bandwidths_mbps: List[float] = []
        self.latencies_ms: List[float] = []
        self.chunk_sizes_bytes: List[int] = []
        self.buffers_s: List[float] = []
        self._load()

    def _load(self):
        if not self.sent_files:
            return

        df_sent = pd.read_csv(self.sent_files[0])
        df_sent.columns = [c.strip() for c in df_sent.columns]

        time_col = [c for c in df_sent.columns if "time" in c][0]
        df_sent = df_sent.sort_values(by=time_col).reset_index(drop=True)

        start_time_ns = df_sent[time_col].iloc[0]
        for _, row in df_sent.iterrows():
            t_s = float((row[time_col] - start_time_ns) / 1e9)
            deliv_rate_Bps = float(row.get("delivery_rate", 2000000))
            bw_mbps = (deliv_rate_Bps * 8.0) / 1e6
            rtt_us = float(row.get("rtt", 50000))
            lat_ms = rtt_us / 1000.0
            size_b = int(row.get("size", 250000))

            self.timestamps.append(t_s)
            self.bandwidths_mbps.append(max(0.5, bw_mbps))
            self.latencies_ms.append(max(5.0, lat_ms))
            self.chunk_sizes_bytes.append(size_b)

    def query(self, current_time_s: float) -> Tuple[float, float]:
        if not self.timestamps:
            return 12.0, 45.0
        max_t = self.timestamps[-1]
        t = current_time_s % max_t if max_t > 0 else 0.0
        bw = float(np.interp(t, self.timestamps, self.bandwidths_mbps))
        lat = float(np.interp(t, self.timestamps, self.latencies_ms))
        return bw, lat