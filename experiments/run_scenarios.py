import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from edge_switching.scenarios import run_scenario

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_STYLE = ROOT_DIR / "configs" / "plot_style.mplstyle"
LOGS_DIR = ROOT_DIR / "logs"
PLOTS_DIR = ROOT_DIR / "plots" / "generated"


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use(CONFIG_STYLE)

    scenarios = ["scenario_a", "scenario_b", "scenario_c", "scenario_d", "scenario_e", "scenario_f"]
    results = {}

    print("Running canonical evaluation scenarios (A through F)...")
    for sc in scenarios:
        df = run_scenario(sc, duration_s=40.0, dt_s=0.5, num_clients=3, seed=42)
        log_csv = LOGS_DIR / f"{sc}_run.csv"
        df.to_csv(log_csv, index=False)
        results[sc] = df
        switches = df[df["action"] == "SWITCH"]
        stalls = df[df["stall_s"] > 0]
        print(f"  [{sc}] completed: {len(df)} steps, {len(switches)} switches, {len(stalls)} stalls. Saved -> {log_csv.name}")

    # Generate multi-scenario timeline sanity check plot
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.0), sharex=True)
    axes = axes.flatten()

    titles = {
        "scenario_a": "Scenario A: Minor Jitter (Noise Only)",
        "scenario_b": "Scenario B: Persistent Regime Shift",
        "scenario_c": "Scenario C: Server Outage",
        "scenario_d": "Scenario D: Multi-User Herding",
        "scenario_e": "Scenario E: Telemetry Staleness",
        "scenario_f": "Scenario F: Client Mobility",
    }

    server_colors = {"s1": "#0072B2", "s2": "#D55E00", "s3": "#009E73"}

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        df = results[sc]
        t = df["time_s"]
        bw = df["effective_bw_mbps"]
        server = df["client_server"]

        for sid, col in server_colors.items():
            mask = (server == sid)
            if mask.any():
                ax.plot(t[mask], bw[mask], color=col, label=sid, linewidth=1.2)

        ax.set_title(titles[sc], fontsize=7, pad=2)
        ax.set_ylabel("Throughput (Mbps)", fontsize=6)
        if idx >= 4:
            ax.set_xlabel("Time (s)", fontsize=6)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=5, ncol=3)

    plt.tight_layout()
    plot_pdf = PLOTS_DIR / "scenarios_overview.pdf"
    plt.savefig(plot_pdf)
    print(f"Overview plot generated: {plot_pdf}")


if __name__ == "__main__":
    main()
