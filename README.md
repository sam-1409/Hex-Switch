# HCI-Guided, XAI-Enabled Server Switching for QoS-Aware Edge Computing

[![Target: IEEE PerCom 2027 A4E Workshop](https://img.shields.io/badge/Target-PerCom%202027%20A4E-blue.svg)](https://percom.org)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![Hardware: RTX 2050 4GB](https://img.shields.io/badge/Hardware-RTX%202050%204GB%20%7C%2024GB%20RAM-orange.svg)](#hardware-specifications)

Research codebase for: **"HCI-Guided, XAI-Enabled Server Switching for QoS-Aware Edge Computing Under Dynamic Network Conditions"**.

---

## 1. Project Status & Working Protocol
* **Current Step:** Step 0 — Foundations and Novelty Gate.
* **Target Venue:** IEEE PerCom 2027 Workshop on Analytics for Edge (A4E) — Submission Deadline: **November 17, 2026 (AoE)**.
* **Strict Gate Protocol:** No future components or simulation models are implemented in Step 0. Transition to Step 1 occurs only upon explicit co-author approval of the Contribution Statement (`docs/CONTRIBUTION_DRAFT.md`).

---

## 2. Directory Layout

```text
edge_switching_research/
├── configs/
│   └── plot_style.mplstyle      # Shared publication plotting style (Okabe-Ito, 3.5 in)
├── data/
│   ├── raw/                     # Raw network traces (FCC, mobile 4G/5G)
│   └── processed/               # Normalized scenario inputs
├── docs/
│   ├── CHANGELOG.md             # Version and step history
│   ├── CLAIMS.md                # Hypotheses-to-evidence matrix
│   ├── CONTRIBUTION_DRAFT.md    # Formal <=150-word contribution statement
│   ├── DECISIONS.md             # Architecture & research decisions log
│   ├── LITERATURE_MATRIX.md     # Peer-reviewed prior art & gap analysis
│   ├── OPEN_DECISIONS.md        # Active choices requiring gate sign-off
│   ├── REVIEWER_QUESTIONS.md    # Top 5 reviewer challenges & defenses
│   └── TIMELINE.md              # 21-day backward milestone plan
├── experiments/
│   └── runs/                    # Seed manifests and execution logs
├── logs/                        # Telemetry and audit output logs
├── plots/
│   └── generated/               # Production vector PDFs (fonttype 42)
├── references/                  # Peer-reviewed PDF papers & citations
├── src/
│   └── edge_switching/          # Pure Python typed package
│       ├── __init__.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   └── test_environment.py     # Environment sanity tests
├── environment.yml              # Conda environment for Anaconda Navigator
├── .gitignore                   # Ignore logs, data caches, temporary files
└── README.md                    # Project documentation
```

---

## 3. Environment Setup (Anaconda Navigator & Windows)

### Option A: Using Anaconda Navigator GUI
1. Open **Anaconda Navigator**.
2. Click on the **Environments** tab on the left.
3. Click the **Import** button at the bottom.
4. Select `Local file` and browse to `d:\Desktop\Research\PerCom2026\edge_switching_research\environment.yml`.
5. Enter Environment name: `edge-switching` and click **Import**.

### Option B: Using Anaconda Prompt (Command Line)
Open Anaconda Prompt and navigate to the project directory:
```powershell
cd d:\Desktop\Research\PerCom2026\edge_switching_research
conda env create -f environment.yml
conda activate edge-switching
```

### Validate Installation
Run the sanity tests:
```powershell
pytest -v tests/test_environment.py
```

---

## 4. Hardware Specifications
* **GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB GDDR6 VRAM, CUDA 12.1+ support).
* **RAM:** 24 GB DDR4/DDR5 system memory.
* **OS:** Windows 11 64-bit.
* **Execution Strategy:** Discrete-event simulation and matrix evaluations are vectorized on CPU for deterministic, seed-matched reproducibility. GPU acceleration (PyTorch CUDA) is reserved for neural baseline comparisons and heavy parallel inference.

---

## 5. Reproducibility & Research Standard
Every experimental run must log:
* UTC timestamp, unique run UUID, Git commit hash, and config SHA-256.
* Independent seeds ($\ge 30$) with mean and 95% bootstrap/Student-t confidence intervals.
* Non-parametric hypothesis tests (Mann-Whitney $U$, Wilcoxon signed-rank) and effect sizes (Cliff's $\delta$, Vargha-Delaney $\hat{A}_{12}$).
* Explicit reporting of negative results and Pareto trade-offs.
