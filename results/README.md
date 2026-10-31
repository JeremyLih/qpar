# Results — run record (protocol 12.1)

Everything in the paper comes from the files in this directory. Every decision and deviation from
the protocol, with its reason, is in `DECISIONS.md`; the sections cited below refer to it.

## Contents

| Path | What |
|---|---|
| `runs/<run_id>.json` | One per training run (840): config, seed, per-epoch log, best epoch, A, subset accuracies (noiseless, calibrated, each depolarizing p), parameter counts, G, G_variational, twoq_transpiled, timings |
| `noise/<run_id>.<part>.json` | The noisy-evaluation parts as run (3,150; also merged into the run JSONs) |
| `checkpoints/<run_id>.pt` | Best-validation weights of every run (not in the public repository; see `.gitignore`) |
| `latency.csv`, `latency_env.json` | D per configuration (168) and the Mac's environment |
| `runs.csv`, `summary.csv` | All runs flat; one row per configuration |
| `qpar_maxima.csv` | Q_max, D_max, G_max per dataset (8.1) |
| `sensitivity*.csv` | 8.2 (54 main configurations) and the supplement over the 27 train_frac = 1.0 configurations |
| `rank_comparison.csv`, `rank_moves.csv` | 8.3 |
| `stats_*.csv` | Section 13: hybrid vs cnn_matched (per cell and per topology), noise correlations |
| `tables/T1–T10.tex`, `figures/F1–F6.{pdf,png}` | 12.2, 12.3 (T8–T10 added: §14) |
| `appendix/random_topology_edges.txt`, `appendix/sweep_gate_counts.txt` | Paper appendix: Random edge lists for n = 6 and 8 (with the redraw rule; §14), and G and two-qubit gates after transpilation per sweep configuration; written by `scripts/appendix.py` from `configs/topologies.yaml` and `summary.csv` |
| `smoke/`, `sweep_selection_*.json` | Smoke tests (11.2) and the qubit-sweep selection (decision 6a) |
| `fleet/`, `logs/`, `STATUS` | Cloud records: plan, per-instance shards and queue states, per-run logs. The coordinator's state (`fleet/fleet_state.json`, `fleet/step1.json`) and event logs (`logs/coordinator.log`, `logs/fleet_events.log`) hold local paths and instance IP addresses and are not in the public repository |

## Commands

```bash
uv sync --frozen && uv run pytest                      # environment, unit tests (11.1)
uv run python scripts/queue.py configs                 # configs/matrix, configs/baselines, topologies
uv run python scripts/smoke.py archs                   # first smoke run and timing run, Mac (results/smoke/mac/)
uv run python scripts/smoke.py timing
uv run python scripts/queue.py step1                   # one Lambda instance: unit tests + smoke re-run (2a)
uv run python scripts/queue.py plan                    # fleet plan from the cloud smoke rates
caffeinate -is uv run python scripts/queue.py coordinate   # the fleet: training, sweep selection, noise, sync
OMP_NUM_THREADS=1 uv run python -m qpar.latency        # D, on the Mac (protocol 7)
uv run python -m qpar.analysis                         # CSVs, tables, figures
uv run python scripts/appendix.py                      # appendix/ (paper appendix exports)
```

`scripts/reproduce.sh cloud|local` chains all of them.

## Cloud

| | |
|---|---|
| Provider, type | Lambda Cloud `gpu_1x_a10`, 30 vCPUs (Intel Xeon Platinum 8358), us-east-1, $1.29/h (GPU unused) |
| Workers | One single-threaded worker per vCPU (decision 3c) |
| Instances | 12 over the run: 8 at launch, 1 replacement and 3 for the final noise parts (§13) |
| Instance-hours | 85.0 (+ the step-1 instance) |
| Wall-clock | 33.6 h, 2026-09-26 22:58 to 2026-09-28 08:35 (fleet launch to last verified termination) |
| Cost | **$110.37** (step-1 instance $0.74 + fleet $109.64); hard cap $200 |
| Jobs | 840 training runs, 3,150 noise parts; every instance terminated only after a checksum-verified sync |

Incidents (all in §13): a slow host (fleet-3, half speed) evacuated; two Mac network outages
(09-27 12:14–17:50, 23:10–05:00) with a false instance loss and duplicate work (≈ $5), resolved and
guarded against in the coordinator; 35 runs skipped 47 updates with a non-finite gradient (the
guard of §13); 60 runs moved between instances at epoch boundaries and resumed from their saved state.

## Library versions

From one lockfile (`uv.lock`), Python 3.11:

| | Cloud (all 840 runs) | Mac (latency, analysis) |
|---|---|---|
| Python | 3.11.16 | 3.11.15 |
| torch | 2.14.0+cpu | 2.14.0 |
| pennylane / pennylane-lightning | 0.45.1 / 0.45.0 | 0.45.1 / 0.45.0 |
| qiskit / qiskit-aer / qiskit-ibm-runtime | 2.5.2 / 0.17.2 / 0.50.0 | same |
| numpy | 2.4.6 | 2.4.6 |
| Machine | Linux x86_64 | macOS 26.6.2, Apple M3 Pro (arm64) |

## Noise backend (6.2)

`FakeAachen` from qiskit-ibm-runtime 0.50.0: IBM Heron r3, 156 qubits, basis gates cz, rz, sx, x
(+ id, delay, measure, reset, if_else), calibration date 2026-04-17 12:29:46+02:00; noise model
`NoiseModel.from_backend`, transpiled at optimization level 1 (seed 0), 1,024 shots (§4).

## Latency (7)

Measured 2026-09-28 15:45:46–15:46:31 on the Apple M3 Pro above: one process, one thread,
lightning.qubit analytic, seed-0 checkpoints, batch size 1, 20 warm-up and the median of 200 samples
(§13a).

## Deviations from the protocol

All recorded with reasons in `DECISIONS.md`; the main ones: canonical placement wiring without
dropout (1A, §3a); smoke re-run at train_frac = 0.2 (2a, §3c); one thread per worker and Aer's
density-matrix simulator for 6.3 (3c, §3e); Random topology redrawn while equal to Linear (4b);
qubit-sweep selection by validation accuracy (6a); the n = 8 amplitude sweep candidates started
speculatively (§12); non-finite gradient steps skipped (§13); tables T8–T10 added and the γ
supplement at train_frac = 1.0 (§14).
