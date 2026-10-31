# Q-Par Experiment Protocol — v3 (2026-09-22)

This document is the specification for the complete experimental study behind the paper
"A Novel Composite Metric for Comparing Hybrid CNN–QNN Models under NISQ Constraints"
(target: IEEE Access). It is written for Claude Code. Everything in Sections 1–13 is
binding. Where the document says **decide**, Claude Code chooses and records the choice in
`results/DECISIONS.md`. Where it says **STOP**, Claude Code stops and asks Jeremy before
continuing. Nothing in this document may be "improved" silently.

Root: `~/QML Paper/`. New code lives in `~/QML Paper/code/` (currently empty). The old code
in `~/QML Paper/archive/` is **read-only** and is consulted for exactly one purpose (Section
4.3). The old CSV and artifacts are provenance only; none of their numbers are reused.

---

## 0. What the study must produce

Evidence for five claims, in this priority order:

1. Every term of Q-Par varies across the study (σ_a via seeds, Q via a qubit sweep).
2. Q-Par ranks configurations differently from accuracy alone, and the differences are
   explainable by the cost terms.
3. Q-Par's ranking is stable enough under reasonable weight changes to be useful
   (sensitivity analysis).
4. A parameter-matched classical model anchors every hybrid result.
5. The study's scope is honest: simulation only; latency is a simulator-cost proxy;
   noise robustness is measured under a calibrated device noise model, not on hardware.

Everything else is secondary.

---

## 1. Ground rules

- **No silent design changes.** Any deviation from this document goes into
  `results/DECISIONS.md` with a reason, or triggers a STOP.
- **Every run is reproducible from a config file and a seed.** No hidden state, no manual
  steps, no interactive notebooks in the pipeline.
- **Write results immediately.** Each run writes its own JSON on completion. The runner is
  resumable and skips runs whose JSON exists and validates.
- **One codebase for everything.** Hybrid models, classical baselines, noisy evaluation,
  latency, analysis, tables and figures all come from `code/`.
- **The test set is touched once per trained model**, at the end of training, with the
  best-validation checkpoint. Never for model selection.
- **All numbers in the paper come from `results/`.** Nothing is hand-typed.
- **The original scripts in `archive/` are not edited, imported, or executed** except as
  described in 4.3.

---

## 2. Repository layout (`~/QML Paper/code/`)

```
code/
  pyproject.toml            # uv-managed; pinned versions; python 3.11
  README.md                 # how to reproduce everything with one command
  configs/
    base.yaml               # shared defaults (Section 5)
    matrix/                 # one YAML per (dataset, placement, encoding, topology, train_frac)
    baselines/              # one YAML per (dataset, baseline_type, placement, train_frac)
    sweep/                  # qubit sweep configs
  src/qpar/
    data.py                 # datasets, splits, subsets (Section 3)
    cnn.py                  # CNN body (4.1)
    quantum.py              # quantum block, encodings, topologies, gate counting (4.2, 4.4, 4.5, 7)
    models.py               # placements + baselines assembled from cnn.py and quantum.py (4.3, 4.6)
    train.py                # training loop (Section 5)
    evaluate.py             # noiseless / calibrated-noise / depolarizing evaluation (Section 6)
    latency.py              # latency measurement (7.4)
    qpar.py                 # Q-Par computation, normalization, sensitivity (Section 8)
    analysis.py             # summary tables, statistics, figures (Sections 12–13)
    cloud.py                # Lambda lifecycle, queue, sync (Section 10)
  scripts/
    run.py                  # run one config + seed
    queue.py                # build/resume the run queue
    smoke.py                # smoke tests (11.2)
    reproduce.sh            # one command: everything
  tests/                    # unit tests (11.1)
  results/
    runs/                   # one JSON per run
    runs.csv  summary.csv   # generated
    tables/  figures/       # generated (LaTeX, PDF, PNG)
    DECISIONS.md  STATUS  README.md
```

---

## 3. Data

| Item | Specification |
|---|---|
| Datasets | MNIST and Fashion-MNIST, via `torchvision`, 28×28 grayscale, 10 classes. Same protocol for both. |
| Preprocessing | Scale to [0,1], then normalize with the dataset's standard mean/std. No augmentation. |
| Validation split | 5,000 images drawn **stratified** from the 60,000 training images with `split_seed = 0`. Fixed for all runs of that dataset. The remaining 55,000 form the training pool. |
| Training subsets | `train_frac ∈ {0.2, 0.5, 1.0}` of the 55,000 pool, drawn **stratified with the run seed** (so each seed sees a different subset; this is intentional — σ_a then includes data-sampling variance). |
| Test set | The full official 10,000-image test set. Used for the noiseless accuracy A. |
| Noisy-evaluation subset | 2,000 test images, stratified, `subset_seed = 0`, identical for every config, seed, dataset, and noise condition. Used for all evaluations in 6.2 and 6.3. |

---

## 4. Models

### 4.1 CNN body (fixed for the whole study)

LeNet-class, chosen for defensibility and for leaving accuracy headroom so that placement and
encoding effects remain visible.

```
conv1: Conv2d(1, 8, kernel=5)   → ReLU → MaxPool(2)     # 28→24→12
conv2: Conv2d(8, 16, kernel=5)  → ReLU → MaxPool(2)     # 12→8→4
flatten: 16·4·4 = 256
fc1:   Linear(256, 32) → ReLU
fc2:   Linear(32, 10)
```

Trainable parameters: 208 + 3,216 + 8,224 + 330 = **11,978**. `models.py` must assert this
number in a unit test.

### 4.2 Quantum block QB(d_in → n)

A quantum block takes a `d_in`-dimensional classical vector and returns `n` real values
(one per qubit). It has three parts:

1. **Input projection** (classical, trainable): `Linear(d_in, k)` with
   `k = n` for angle encoding and `k = 2^n` for amplitude encoding.
2. **Encoding + variational circuit** (Section 4.4, 4.5) on `n` qubits, `L = 3` layers.
   Each layer: `Rot(φ, θ, ω)` on every qubit (3 trainable parameters per qubit), then the
   entangling CNOTs of the chosen topology. Trainable quantum parameters: `3·n·L` (= 36 at
   n = 4).
3. **Readout**: `⟨Z_i⟩` for each qubit `i` → `n` outputs.

Simulator for training and noiseless evaluation: `lightning.qubit` (fall back to
`default.qubit` only if lightning is unavailable; record which). Gradients via adjoint
differentiation. Analytic expectation values (no shots) in training.

### 4.3 Placements — mechanics come from `archive/`

The paper defines three placements: **Pre** (quantum block before the first convolution),
**Middle** (at the bottleneck after the convolutions), **Post** (as the classification
head). The CNN body (4.1) and quantum block (4.2) are standardized by this document, but
**how the quantum block is wired in at each placement must be taken from the original
scripts** in `archive/` so that the placements mean what the paper says they mean.

Procedure:

1. Read the three original scripts (pre / middle / post). Extract, for each placement:
   what tensor feeds the quantum block, how it is reduced to `d_in`, and how the block's
   `n` outputs re-enter the network.
2. Write the extracted wiring into `results/DECISIONS.md` under "Placement mechanics
   (from archive)".
3. Implement exactly that wiring on top of 4.1/4.2.
4. **STOP** if the archive is ambiguous or inconsistent for any placement, and present the
   extracted wiring plus the following canonical defaults for Jeremy to confirm:
   - Pre: `QB(784 → n)` on the flattened input; outputs expanded by `Linear(n, 784)`,
     reshaped to 1×28×28, and **added** to the input image (residual); the CNN then runs
     unchanged.
   - Middle: `QB(256 → n)` replaces `fc1`; followed by `Linear(n, 32) → ReLU → fc2`.
   - Post: `QB(32 → n)` replaces `fc2`; followed by `Linear(n, 10)`.

### 4.4 Encodings

| Name | Circuit | Feature map |
|---|---|---|
| Angle | `RY(z_i)` on qubit i | `z = π · tanh(Linear(d_in, n)(x))` |
| Amplitude | `AmplitudeEmbedding(v, normalize=True)` | `v = Linear(d_in, 2^n)(x)`; add `1e-8` to the first component before normalization to avoid the zero vector |

### 4.5 Topologies (entangling CNOTs per layer, n qubits)

| Name | Edges |
|---|---|
| Linear | `(0,1), (1,2), …, (n−2, n−1)` — `n−1` CNOTs |
| FullyConnected | all pairs `(i, j)`, `i < j` — `n(n−1)/2` CNOTs |
| Random | `n−1` distinct pairs drawn uniformly without replacement from all pairs, with `topology_seed = 2026`, **once per n**, stored in `configs/topologies.yaml`. The same graph is used by every run at that n. Random therefore has Linear's CNOT count and differs only in structure. |

### 4.6 Classical baselines

| Name | Definition | Runs |
|---|---|---|
| `base_cnn` | The CNN body of 4.1, unmodified. One model, independent of placement. | 2 datasets × 3 train_frac × 5 seeds = 30 |
| `cnn_matched` | For each placement and encoding, the hybrid model with the **encoding + variational circuit + readout** replaced by a classical map `k → n` whose parameter count is the closest achievable to the quantum part (`3·n·L = 36`) among: `Linear(k, n, bias=False)`, `Linear(k, n)`, `Linear(k, h) → tanh → Linear(h, n)` for `h ∈ {1, 2, 3}`. Input projection and everything else unchanged. Record the exact quantum-part and classical-part counts; the total-model difference will be < 0.5 % and must be reported as such. | 2 datasets × 3 placements × 2 encodings × 3 train_frac × 5 seeds = 180 |

Baselines are trained with the identical protocol (Section 5) and evaluated in all three
conditions (Section 6) where applicable (noise conditions do not apply to classical models;
record `NaN`).

---

## 5. Training protocol (identical for every run)

| Item | Value |
|---|---|
| Loss | Cross-entropy, 10 classes |
| Optimizer | Adam, lr = 1e-3, betas = (0.9, 0.999), no weight decay |
| Batch size | 64 |
| Epochs | 15, fixed. No early stopping. |
| Checkpoint | Best validation accuracy (evaluated after every epoch). Ties → earlier epoch. |
| Seeds | `run_seed ∈ {0, 1, 2, 3, 4}`. A seed sets `random`, `numpy`, `torch` (CPU and any accelerator), the DataLoader generator, the training-subset draw, and parameter initialization. |
| Initialization | PyTorch defaults for classical layers; quantum `Rot` parameters uniform in `[0, 2π)` from the seeded generator. |
| Threads | `OMP_NUM_THREADS = 2` per worker (Section 10). |
| Logged per epoch | train loss, train accuracy, validation accuracy, wall-clock seconds (informational only; never used in any metric). |

Convergence statistic: `epochs_to_threshold` = first epoch at which validation accuracy
reaches the dataset threshold — **MNIST 0.95, Fashion-MNIST 0.80** — or `NaN` if never
reached within 15 epochs. Wall-clock time-to-threshold is **not** a reported metric.

---

## 6. Evaluation conditions

### 6.1 Noiseless (primary)

Best-validation checkpoint, full 10,000 test images, analytic expectation values on the
training simulator. Produces the accuracy **A** used in Q-Par.

### 6.2 Calibrated device noise model

- Backend: the newest **Heron-class** fake backend available in the installed
  `qiskit-ibm-runtime` fake provider (fall back to `FakeTorino`). Record backend name,
  qubit count, basis gates, and the calibration date embedded in the fake backend.
- Build the noise model with `NoiseModel.from_backend(backend)` (Qiskit Aer): gate errors,
  thermal relaxation, readout errors as provided.
- Transpile each circuit structure once to the backend's basis gates and coupling map with
  `optimization_level = 1`, seeded; bind per-sample parameters afterwards. Record the
  transpiled depth and two-qubit-gate count per configuration (supplementary to G).
- Execution: `shots = 1024` per circuit. The classical layers run unchanged in PyTorch;
  only the quantum block is executed through Aer.
- Data: the 2,000-image noisy-evaluation subset (Section 3).
- Output per run: `acc_noise_calibrated`. Degradation `Δ_cal = A_subset_noiseless −
  acc_noise_calibrated`, where `A_subset_noiseless` is the analytic accuracy on the same
  2,000 images (compute and store it too).
- **Validation gate** (11.3): the same pipeline with `noise_model = None` and
  `shots = 1024` must reproduce `A_subset_noiseless` within ±1.0 percentage point on at
  least three configurations before any noisy evaluation is launched.

### 6.3 Depolarizing sweep (sensitivity to noise strength)

- Simulator: `default.mixed`, analytic (density matrix).
- Channel: `DepolarizingChannel(p)` after every single-qubit gate and
  `DepolarizingChannel(10·p)` on both qubits after every two-qubit gate, applied to the
  variational layers **and** the encoding gates.
- `p ∈ {0.001, 0.005, 0.01, 0.02}`.
- Data: the 2,000-image subset. Output per run and per p: `acc_depol_p`.

Noise evaluations apply to every trained hybrid model (main matrices and qubit sweep). They
are inference only; no retraining.

---

## 7. Measured quantities

| Symbol | Definition | Where measured |
|---|---|---|
| **A** | Noiseless test accuracy (6.1), per run; per configuration: mean over the 5 seeds | cloud |
| **σ_a** | Sample standard deviation (ddof = 1) of A over the 5 seeds of a configuration | analysis |
| **Q** | Number of qubits in the quantum block | config |
| **G** | Logical gate count of the quantum block's circuit at full decomposition into elementary gates, via `qml.specs` with the template expanded — **includes the encoding gates** (amplitude embedding decomposes to many gates; that cost is real and must count). Computed once per architecture. Store separately: `G_variational` (Rot + CNOT only) and `twoq_transpiled` (6.2). | analysis |
| **D** | Inference latency per sample: **measured on Jeremy's Mac, not in the cloud**, with the seed-0 checkpoint of each configuration: batch size 1, `lightning.qubit` analytic, 20 warm-up samples, then the **median** over 200 test samples, in milliseconds. Same machine, same process settings, same library versions for every configuration, measured in one uninterrupted session with nothing else running. Also measured for `base_cnn` and `cnn_matched`. | local |
| `params_total`, `params_quantum`, `params_classical` | Exact trainable parameter counts | config |
| `epochs_to_threshold` | Section 5 | cloud |
| `acc_noise_calibrated`, `Δ_cal`, `acc_depol_p` | Section 6 | cloud |

---

## 8. Q-Par

### 8.1 Definition (unchanged structure; Q normalized like D and G)

```
Q-Par = A^α · (1 − σ_a)^β  /  [ (Q/Q_max)^γ · (D/D_max)^δ · (G/G_max)^ε ]
```

- Equal weights `α = β = γ = δ = ε = 1` are the reported baseline.
- `Q_max`, `D_max`, `G_max` are the maxima **over all hybrid configurations of that
  dataset (main matrix + qubit sweep)**. Compute and record them. Classical baselines have
  no Q-Par (Q = 0); they are reported on A, D and parameter count only.
- Validation gate (11.1): with `σ_a = 0`, `Q = Q_max`, `D = D_max`, `G = G_max` the
  function must return `A`; with the paper's original v2 numbers for one row and the
  original raw-Q form, a separately implemented `qpar_legacy` must reproduce the v2 CSV's
  Q-Par to `1e-4` (this only checks the arithmetic; legacy numbers are not used anywhere).

### 8.2 Sensitivity

1. One exponent at a time: each of α, β, γ, δ, ε ∈ {0.5, 1, 2}, others at 1 (11 settings
   including the baseline).
2. Three presets: accuracy-first (α = 2), latency-first (δ = 2), resource-first
   (γ = ε = 2).
3. For each setting and dataset: the full ranking of the 54 main configurations, Kendall τ
   against the equal-weight ranking, and the identity of the top 5.

### 8.3 Rank comparison

Per dataset: Kendall τ and a rank–rank scatter between the ranking by A alone and the
ranking by equal-weight Q-Par; list every configuration whose rank moves by ≥ 10 places,
with which cost term drove the move.

---

## 9. Experiment matrix and run order

| Block | Configs | Seeds | Runs | Order |
|---|---|---|---|---|
| M1 MNIST main matrix | 3 placements × 2 encodings × 3 topologies × 3 train_frac = 54 | 5 | 270 | 1 |
| B1 MNIST baselines | `base_cnn` (3 train_frac) + `cnn_matched` (3 × 2 × 3) = 21 | 5 | 105 | 2 |
| M2 Fashion-MNIST main matrix | 54 | 5 | 270 | 3 |
| B2 Fashion-MNIST baselines | 21 | 5 | 105 | 4 |
| S Qubit sweep | per dataset, the best configuration **per placement** by mean noiseless A at `train_frac = 1.0` in M1/M2, re-trained at `n ∈ {2, 4, 6, 8}` (n = 4 reuses the M1/M2 runs), `train_frac = 1.0` only → 2 datasets × 3 placements × 3 new n = 18 | 5 | 90 | 5 |
| N Noisy evaluation (6.2, 6.3) | every hybrid run in M1, M2, S | — | inference only | 6 |
| D Latency (7) | every configuration in M1, M2, S, B1, B2; seed-0 checkpoint | — | local | 7 |

Within a block, order runs by `train_frac` descending (1.0 first) so the most important
results exist earliest if the budget is cut.

Total training runs: 840. Amplitude encoding at n = 8 uses a 256-dimensional projection;
at n = 2, a 4-dimensional one. Angle at n = 8 uses 8 features.

---

## 10. Compute plan

- **Local (Mac):** scaffolding, unit tests, smoke tests, latency measurement, analysis,
  figures. Nothing else trains locally.
- **Cloud (Lambda):** all training and all noisy evaluation. `cloud.py` creates the
  instance through the Lambda API, installs the environment from the lockfile, pulls
  `code/`, runs the queue, syncs results back, and terminates the instance.
- **Instance choice:** the workload is small statevector circuits (n ≤ 8) plus a tiny CNN;
  **GPU is irrelevant; vCPU count is the lever.** Choose the available instance type that
  maximizes vCPUs per dollar-hour (a low-end GPU instance with many vCPUs is fine). Record
  the choice and hourly rate.
- **Parallelism:** one worker process per 2 vCPUs, `OMP_NUM_THREADS = 2`, each worker
  pulling the next run from the queue. Noisy evaluation is parallelized the same way.
- **Sync:** `rsync` `results/runs/` and logs to the Mac every 10 minutes and at the end.
  `results/STATUS` (one line: block, runs done/total, ETA, spend so far) is updated on
  every sync.
- **Budget guard:** hard cap **US$150**. Before launching, compute projected hours from
  the smoke-test timing and the worker count; if projected cost > $100, **STOP** and
  report. During the run, terminate at the cap, on completion, or after 30 minutes idle.
- **Terminate only after a verified final sync.** The Mac-side monitor performs the
  termination: when the queue reports completion (or the cap/idle condition is hit), run a
  final `rsync`, verify that every run JSON on the instance exists locally with a matching
  checksum, and only then call the Lambda terminate API. If the Mac is unreachable, the
  instance keeps running (bounded by the cap); it never self-terminates with unsynced
  results.
- **Credentials:** Lambda API key and SSH key from environment variables
  (`LAMBDA_API_KEY`, `LAMBDA_SSH_KEY_NAME`, `LAMBDA_SSH_PRIVATE_KEY_PATH`). Never written
  to disk inside the repo, never printed.
- Pin all package versions in `pyproject.toml` / lockfile; the cloud environment must be
  built from the same lockfile as the Mac.

---

## 11. Gates before the cloud launch

### 11.1 Unit tests (`tests/`)
- CNN body parameter count == 11,978.
- Quantum-part parameter count == 3·n·L for every n used.
- Topology edge lists: Linear and FullyConnected exact; Random reproducible from
  `topology_seed = 2026` and has n−1 edges.
- Amplitude encoding: projection output normalized; angle encoding: features in (−π, π).
- `G` counting: for a hand-built 2-qubit, 1-layer angle/linear circuit, G equals the
  hand count.
- Q-Par: identities in 8.1; `qpar_legacy` reproduces one v2 row to 1e-4.
- Stratified splits: class proportions within ±0.5 % of uniform; validation and test sets
  disjoint from every training subset; the same `split_seed` gives byte-identical splits.
- Seeding: two runs of the same config and seed for 20 batches give identical losses.

### 11.2 Smoke tests (`scripts/smoke.py`, Mac)
- Every architecture combination in M1 (18) plus one config per sweep n and every
  baseline type: train 1 epoch on 2 % of the pool, evaluate on 500 test images, assert
  accuracy > 0.20 (double chance) and no NaNs.
- Time one full run (n = 4, amplitude, FullyConnected, `train_frac = 1.0`, 15 epochs) on
  the Mac with 2 threads; use it to project cloud wall-clock and cost (Section 10).
- Noisy pipeline: run 6.2 with `noise_model = None`, `shots = 1024` on three smoke models
  and check agreement with analytic accuracy within ±1.0 point; run 6.3 at p = 0 and
  check exact agreement.

### 11.3 Launch checklist
All 11.1 and 11.2 green; projected cost ≤ $100 (else STOP); `results/DECISIONS.md` contains
the placement mechanics and every decision made so far; Jeremy has confirmed the placement
mechanics if a STOP was raised.

---

## 12. Outputs

### 12.1 Files
- `results/runs/<run_id>.json`: config, seed, per-epoch log, best epoch, A, subset
  accuracies (noiseless, calibrated, each depolarizing p), parameter counts, G,
  G_variational, twoq_transpiled, timings (informational).
- `results/runs.csv`: one row per run, all fields flat.
- `results/summary.csv`: one row per configuration: mean and std of A, σ_a, 95 % CI,
  Q, D, G, Q-Par (equal weights), Q-Par rank, A rank, mean Δ_cal, mean acc_depol_p,
  mean epochs_to_threshold, parameter counts.
- `results/sensitivity.csv`: Section 8.2.
- `results/DECISIONS.md`, `results/README.md` (exact commands, instance type, hours, cost,
  library versions, backend name and calibration date, deviations), `results/STATUS`.

### 12.2 Tables (LaTeX, IEEE two-column, `booktabs`, `\begin{table*}` where wide)
- T1: MNIST, `train_frac = 1.0`, 18 configurations — A (mean ± std), σ_a, G, D, Q-Par,
  Q-Par rank, A rank.
- T2: Fashion-MNIST, same layout.
- T3: Best hybrid per placement vs `cnn_matched` vs `base_cnn`, both datasets, three
  train_frac — A (mean ± std), parameters, D.
- T4: Qubit sweep — per dataset and placement: n, A, σ_a, G, D, Q-Par.
- T5: Noise — per configuration at `train_frac = 1.0`: A_subset_noiseless,
  acc_noise_calibrated, Δ_cal, acc_depol at each p, G, twoq_transpiled.
- T6: Sensitivity — Kendall τ vs equal weights for every setting; top-5 per setting.
- T7 (appendix): full 54-row matrices for both datasets with A mean ± std and Q-Par.

### 12.3 Figures (PDF + 300 dpi PNG; single-column 3.5 in or double-column 7.16 in wide;
fonts ≥ 8 pt; colorblind-safe palette; error bars = ± 1 std over seeds)
- F1: A vs train_frac — best hybrid per placement, `cnn_matched`, `base_cnn`; one panel
  per dataset.
- F2: Rank–rank scatter, A rank vs Q-Par rank, 54 points per dataset, labelled outliers
  (rank change ≥ 10).
- F3: Δ_cal vs G, one point per configuration, with Spearman ρ and p in the legend; one
  panel per dataset.
- F4: acc_depol vs p, one line per placement (mean over encodings/topologies), both
  datasets.
- F5: Qubit sweep — A, D, G, Q-Par vs n, one panel each, lines per placement.
- F6: Sensitivity heatmap — Kendall τ for each exponent setting.

---

## 13. Statistics (all in `analysis.py`)

- Per configuration: mean, std (ddof = 1), 95 % CI from the t-distribution with n = 5.
- Hybrid vs `cnn_matched`, per dataset × placement × encoding × train_frac: paired over
  seeds — paired t-test and Wilcoxon signed-rank; report p and Cohen's d. With n = 5 the
  Wilcoxon p-floor is 0.0625; report it, do not overstate.
- Rankings: Kendall τ (A vs Q-Par; each sensitivity setting vs equal weights).
- Noise: Spearman ρ between G and Δ_cal, and between `twoq_transpiled` and Δ_cal, across
  configurations at `train_frac = 1.0`.
- Every table cell that is a mean carries its std or CI. No single-run numbers anywhere in
  the tables.

---

## 14. Reproducibility package

- `scripts/reproduce.sh` regenerates everything from raw data: environment → tests → smoke
  → (cloud or local) queue → sync → latency (prompts for the Mac step) → analysis →
  tables → figures.
- `README.md`: purpose, layout, one-command reproduction, expected cost and time, how to
  run a subset locally.
- After the study is complete and Jeremy approves: create the **public GitHub repository**
  `qpar-hybrid-cnn-qnn` from `code/` with results (`runs/` JSON, CSVs, tables, figures)
  included; checkpoints (`.pt`) excluded or in Git LFS; MIT license; `CITATION.cff`
  placeholder. The archive is **not** included. Report the URL.

---

## 15. STOP-and-ask list

Stop and ask Jeremy before continuing if any of the following occurs:

1. The archive's placement wiring is ambiguous or contradicts the paper (4.3).
2. Projected cloud cost > $100 or projected wall-clock > 48 h (10, 11.2).
3. Any configuration fails the smoke test, or during the main run any configuration at
   `train_frac = 1.0` ends with mean A < 0.5 (something is wrong with that architecture,
   not with the data).
4. The noisy pipeline fails its validation gate (6.2, 11.2).
5. No Heron-class fake backend is available and `FakeTorino` is also unavailable.
6. Lambda credentials are missing or instance creation fails twice.
7. Anything in this document turns out to be internally inconsistent or impossible to
   implement as written.

When stopping, report: what was found, the options, and a recommendation. Do not pick.

---

## Appendix A — run_id and config naming

`<dataset>_<placement>_<encoding>_<topology>_n<qubits>_tf<train_frac>_s<seed>`
e.g. `mnist_middle_amplitude_linear_n4_tf1.0_s0`;
baselines: `<dataset>_base_cnn_tf0.5_s2`, `<dataset>_matched_post_angle_tf0.2_s4`.

## Appendix B — what is deliberately NOT in this study

Real quantum hardware; training under noise; error mitigation; data augmentation; early
stopping; hyperparameter search; datasets beyond MNIST and Fashion-MNIST; changes to the
Q-Par functional form beyond the Q normalization in 8.1. Each of these is future work and
must not be added.
