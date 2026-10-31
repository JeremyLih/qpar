# DECISIONS — Q-Par experiment (protocol v3, 2026-09-22)

Every choice the protocol leaves open ("decide"), every interpretation of an ambiguity, and every
deviation is recorded here with its reason. Entries marked **PENDING JEREMY** were raised as a STOP
(protocol Section 15) and are implemented provisionally so scaffolding, tests and smoke tests could
run; nothing has been launched on them.

Status legend: DECIDED · PROVISIONAL (awaiting confirmation) · FACT (a measured/observed fact that
constrains the implementation).

## 0. Jeremy's answers to the first STOP (2026-09-26)

- **1A-no-dropout** — canonical placement defaults (4.3) for Pre, Middle and Post; no Dropout, no
  BatchNorm. §3a is now DECIDED.
- **2a** — smoke tests re-run with 1 epoch at tf = 0.2 (11,000 images, 172 steps) instead of 2 %;
  criteria unchanged (accuracy > 0.20 on 500 test images, no NaN). Run on the first cloud
  instance, in parallel across its cores (also validates the cloud code).
- **3c** — depolarizing sweep (6.3) computed with Qiskit Aer `method="density_matrix"`, exact
  expectation values (no shots), same channels (`depolarizing_error(4p/3)` = PennyLane
  `DepolarizingChannel(p)`); PennyLane `default.mixed` kept as the test reference. And one worker
  per physical core (1 thread per worker) instead of one per 2 vCPUs.
- **4b** — Random topology: pre-declared rejection rule — draw with `default_rng(2026)` until the
  edge set differs from Linear's (impossible only at n = 2, which has a single pair).
- **5a** — stratification test against the source split's class proportions (±0.5 pp).
- **6a** — the qubit sweep selects by mean best-validation accuracy (not test accuracy); the same
  rule is used for "best hybrid per placement" in T3/F1.
- **Credentials** — `LAMBDA_API_KEY`, `LAMBDA_SSH_KEY_NAME` and `LAMBDA_SSH_PRIVATE_KEY_PATH`
  exported from the shell profile (the key name and its local path were removed from this entry
  for publication, 2026-10-31). Keys are never printed or copied.
- **New priorities and cloud plan** — wall-clock first; hard cap **$200** total; proceed without
  asking if the projection is ≤ $150, otherwise STOP. Up to 8 Lambda instances, GPU types up to
  and including H100, chosen by total vCPUs; cheapest fleet whose projected wall-clock is within
  ~1.5× of the floor set by the longest run. Step 1: one instance runs the smoke re-run; if green,
  scale to the fleet immediately. Shards finish together (longest first, rebalance laggards);
  training before noisy evaluation on the same fleet; resumable; relaunch dead instances;
  continuous sync; an instance terminates only after its shard is verified synced. Latency is
  not started without Jeremy.

---

## 1. Environment

- **DECIDED** — Python 3.11 (uv-managed CPython 3.11.15), dependencies pinned with `==` in
  `pyproject.toml` and locked in `uv.lock`: torch 2.14.0, torchvision 0.29.0, pennylane 0.45.1,
  pennylane-lightning 0.45.0, qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0,
  numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, pyyaml 6.0.3, matplotlib 3.11.2, pytest 9.1.1.
- **DECIDED** — On Linux (cloud) torch/torchvision resolve from the PyTorch CPU wheel index
  (`2.14.0+cpu`); on macOS from PyPI. Same `uv.lock` for both. Reason: the protocol states GPU is
  irrelevant; CPU wheels avoid a multi-GB CUDA download on every instance.
- **FACT** — `lightning.qubit` is available on the Mac; no fallback to `default.qubit` was needed.

## 2. Quantum block implementation

- **FACT / DECIDED — float64 into the circuit.** With float32 circuit inputs, PennyLane's Möttönen
  decomposition (used for amplitude encoding under adjoint differentiation) produces non-finite
  input gradients for ~1.6 % of samples at n = 8 (10 of 640 in a test): `2·arcsin(√(num/den))`
  rounds to `arcsin(1)` whose derivative is infinite. With float64 inputs: 0 of 640. The feature
  map output and the Rot weights are therefore cast to float64 before the QNode; circuit outputs
  are cast back to float32 for the classical layers. The trainable `Rot` parameters are stored as
  float32 `nn.Parameter`s (PyTorch default) and cast inside `forward`.
- **DECIDED — decomposition of amplitude embedding during training is applied once per batch.**
  Adjoint differentiation cannot differentiate a state preparation, so PennyLane itself replaces
  `AmplitudeEmbedding` by `MottonenStatePreparation` (RY/RZ/CNOT) during training — by default once
  *per sample* after splitting the batch. The training QNode is wrapped in PennyLane's own
  `qml.transforms.decompose(gate_set={RY, RZ, CNOT, Rot, GlobalPhase})`, which applies the same
  decomposition once per batch. Verified bit-identical (max |Δ| = 0.0 in outputs, input gradients
  and weight gradients at n = 4, 6, 8); 2.3× faster at n = 4. The circuit is unchanged; this is
  purely where PennyLane performs its own decomposition. Evaluation (no gradients) uses the
  undecomposed QNode, where lightning loads the state natively.

## 3. Gate count G

- **DECIDED — elementary gate set {RX, RY, RZ, CNOT}.** G is counted with `qml.specs` after
  `qml.transforms.decompose` to this set: `Rot` → RZ·RY·RZ (3 gates), `AmplitudeEmbedding` → full
  Möttönen structure. `GlobalPhase` (emitted by Möttönen) is excluded: it is not a physical gate
  and has no observable effect. This matches the archive's own convention of counting Rot as 3
  single-qubit gates (its CSV G = 52 = 4 + 3·(3·4 + 4)).
- **DECIDED — "full structure" for amplitude encoding.** PennyLane skips Möttönen rotations whose
  angle is zero only when the input does not require gradients; during training nothing is
  skipped. G is counted with a gradient-carrying input, i.e. the complete, input-independent
  structure: 2^n−1 RY + 2^n−1 RZ + 2(2^n−2) CNOT for the encoding.
- **DECIDED — G_variational** = the variational layers at the same decomposition level:
  L·(3n + |E|). G = G_encoding + G_variational.

- **FACT — angle feature map range.** π·tanh(u) lies in (−π, π) mathematically; in float64,
  tanh(u) rounds to exactly ±1 for |u| > ~18.4, so z = ±π exactly for such inputs. RY(π) and RY(−π)
  differ only by a global phase, so there is no physical discontinuity. The 11.1 test checks the
  open interval for |u| ≤ 18 and |z| ≤ π for all inputs.

## 3a. Placement mechanics (from archive) — **DECIDED: canonical defaults, no dropout (1A)**

Source: `archive/v1_2025-09/code/hybrid_cnn_qnn_mnist_{pre,middle,post}.py`, read only; no other
archive code was opened. Paper definition (`paper/v2/sections/methodology.tex`): "two initial
convolutional layers …, a variational quantum circuit …, and a final fully connected neural
network"; Pre = "inserted before the first convolution", Middle = "at the bottleneck after the
convolution layers", Post = "used as the final classification head"; "all model variants share the
same convolutional components".

Elements common to all three scripts: before the circuit, `Linear(d, k) → Tanh → BatchNorm1d(k)`
(k = n angle, 2^n amplitude); after the circuit, `Dropout(0.1) → Linear(n, 10)` producing the
logits. Archive CNN: `Conv2d(1,16,3,p=1)→ReLU→MaxPool2→Conv2d(16,32,3,p=1)→ReLU→MaxPool2→Flatten`
(1,568 features). (Its topologies — Linear as a ring, Random as a per-layer Erdős–Rényi graph —
and feature maps are superseded by protocol 4.4/4.5 and are not wiring.)

| Placement | What feeds the block | Reduction to d_in | How the n outputs re-enter |
|---|---|---|---|
| Pre (`HybridPreQNN_MPS`) | flattened normalized image (784) | `Linear(784, k)` | directly into `Linear(n, 10)` → logits. **The model contains no convolution at all.** |
| Middle (`HybridCNNQNN_MPS`) | flattened conv features (bottleneck) | `Linear(1568, k)` | directly into `Linear(n, 10)` → logits |
| Post (`HybridPostQNN_MPS`) | output of a hidden FC `Linear(1568,128)→ReLU` | `Linear(128, k)` | directly into `Linear(n, 10)` → logits |

Mapped onto the 4.1 body (convs → 256; fc1 256→32; fc2 32→10), with total trainable parameters at
n = 4 (angle / amplitude):

| Placement | Archive wiring on 4.1/4.2 | params | Canonical default (4.3) | params |
|---|---|---|---|---|
| Pre | `flatten → QB(784→n) → Linear(n,10)`; no CNN | 3,226 / 12,646 | `x + reshape(Linear(n,784)(QB(784→n)(flatten x)))` → full CNN body | 19,074 / 28,494 |
| Middle | `convs → QB(256→n) → Linear(n,10)`; fc1 and fc2 removed | 4,538 / 7,622 | `convs → QB(256→n) → Linear(n,32) → ReLU → fc2` | 4,978 / 8,062 |
| Post | `convs → fc1 → ReLU → QB(32→n) → Linear(n,10)` | 11,866 / 12,262 | identical | 11,866 / 12,262 |

Findings: (1) **Pre contradicts the paper**: the archive's Pre has no convolutional layers, so the
block is not "before the first convolution" and the variants do not "share the same convolutional
components". It cannot be implemented "on top of 4.1" without deciding whether a CNN follows.
(2) **Middle is unambiguous but differs from the canonical default**: the archive feeds the block's
outputs straight to the logits layer, so in the archive Middle and Post both end in
`QB → Linear(n,10)` and differ only in whether one classical hidden layer precedes the block.
(3) **Post matches the canonical default exactly.** (4) `Tanh → BatchNorm1d` before the circuit is
superseded by the protocol's feature maps (4.4). `Dropout(0.1)` before the head is in none of 4.1,
4.2 or the canonical defaults; it is ambiguous whether it is "wiring".

**Implementation:** canonical defaults for all three placements, no Dropout, no BatchNorm
(`src/qpar/models.py`) — confirmed by Jeremy (1A).

## 3b. Other inconsistencies found — **DECIDED: 5a, 4b, 6a**

- **Stratified splits vs "±0.5 % of uniform" (3 vs 11.1).** Stratified sampling reproduces the
  source class distribution. MNIST is not balanced (train: class 1 = 11.24 %, class 5 = 9.04 %;
  test: 11.35 % / 8.92 %), so every stratified MNIST split is 1.24–1.35 pp from uniform and cannot
  pass the test as written; Fashion-MNIST is exactly balanced and passes. PROVISIONAL: the test
  checks ±0.5 pp against the source split's proportions (measured max deviation: 0.02 pp).
- **Random topology coincides with Linear at n = 4 (4.5).** With `numpy.default_rng(2026)` drawing
  3 of the 6 pairs, the draw is {(0,1),(1,2),(2,3)} — the Linear chain (probability 1/20). So at
  n = 4, i.e. in all of M1/M2, "Random … differs only in structure" is false: the 36 Random configs
  would be bit-identical re-runs of the Linear configs (same seeds, same data, same init). n = 6 and
  n = 8 draws differ from Linear. PROVISIONAL: implemented as drawn (no re-draw, no RNG change —
  either would be a post-hoc choice after seeing the outcome). Confirmed empirically: the smoke
  runs of Pre/Random and Pre/Linear are bit-identical (A = 0.578, loss = 2.189 both).
- **Sweep selection uses the test set (9 vs 1).** Block S selects "the best configuration per
  placement by mean noiseless A", and A is test accuracy (6.1, 7); Section 1 says the test set is
  never used for model selection. PROVISIONAL: nothing implemented yet (block S runs later).

## 3c. Smoke-test failures — **DECIDED: re-run at tf = 0.2 (2a)**

`scripts/smoke.py archs` (report: `results/smoke/archs.json`): **31 of 43 configurations pass,
12 fail** the "1 epoch on 2 % of the pool (1,100 images = 17 Adam steps), accuracy > 0.20 on 500
test images" criterion. No NaN anywhere.

| Failing configuration | A | | Failing configuration | A |
|---|---|---|---|---|
| middle / angle / linear / n4 | 0.142 | | middle / angle / FC / n6 | 0.148 |
| middle / angle / FC / n4 | 0.150 | | middle / angle / FC / n8 | 0.098 |
| middle / angle / random / n4 | 0.142 | | post / angle / FC / n8 | 0.194 |
| post / angle / FC / n4 | 0.200 | | matched / middle / amplitude | 0.098 |
| middle / angle / FC / n2 | 0.174 | | matched / post / angle | 0.090 |
| post / angle / FC / n2 | 0.142 | | matched / post / amplitude | 0.090 |

Diagnosis (scratchpad experiments, not part of the pipeline; same seed and 2 % subset):
- Nothing is broken. Trained longer on the same 1,100 images, every failing model passes 0.20
  within 2–4 epochs (34–68 steps) and keeps improving (validation accuracy after 136 steps:
  middle/angle 0.34, post/angle 0.44, middle/amplitude 0.65, post/amplitude 0.68,
  matched middle/amplitude 0.40, matched post/angle 0.40; base_cnn 0.90, pre/angle 0.89).
- The failures track the **narrow bottleneck**, not the quantum circuit: three purely classical
  cnn_matched baselines fail too. Pre passes easily because its CNN body is intact (the quantum
  path is a residual).
- On the full pool (860 steps/epoch) the weakest ones are fine: middle/angle/FC validation
  accuracy 0.935 after epoch 1, 0.951 after epoch 2; post/angle/FC 0.916 → 0.973; matched
  post/angle 0.867 → 0.887. So the "mean A < 0.5 at tf = 1.0" stop is unlikely to trigger.
- Accuracy bug found by the smoke run and fixed: accuracies were float32 means, so 100/500
  compared as 0.2000000030 > 0.20 and post/angle/FC/n4 was wrongly marked as passing. All
  accuracies are now exact integer ratios (`train.fraction_equal`; regression test added).

## 3e. Projected cost — **DECIDED: Aer for 6.3 + one worker per core (3c)**

- **Timing run** (11.2; `results/smoke/timing.json`): pre/amplitude/FC, n = 4, tf = 1.0, 15 epochs,
  2 threads, Mac: **50.1 min**; A = 0.9867, epochs_to_threshold = 1. The smoke-derived per-sample
  rates predict it to within 2.3 % (calibration factor 0.977).
- **Noisy-pipeline gates pass** (`results/smoke/noise.json`; STOP 15.4 not triggered). 2,000-image
  subset; shots-only = 6.2 pipeline with `noise_model=None`, 1,024 shots:

| model | analytic | shots, no noise | Δ (pt) | depol p = 0 |
|---|---|---|---|---|
| pre/amplitude/FC (smoke) | 0.5860 | 0.5870 | +0.10 | exact (100 % same predictions) |
| middle/angle/random (smoke) | 0.1495 | 0.1455 | −0.40 | exact |
| post/amplitude/linear (smoke) | 0.2400 | 0.2385 | −0.15 | exact |
| pre/amplitude/FC (15-epoch) | 0.9890 | 0.9895 | +0.05 | exact |

- **Per-image noisy-evaluation cost** (Mac, FullyConnected; transpiled two-qubit gates on
  FakeAachen): n = 4 angle 36 CZ, Aer 10 ms, default.mixed 3.0 ms per p; n = 4 amplitude 115 CZ,
  18 ms, 6.6 ms; n = 6 amplitude 439 CZ, 112 ms, 225 ms; n = 8 angle 294 CZ, 162 ms, 1.5 s;
  **n = 8 amplitude 1,624 CZ (depth 5,235), Aer 0.87 s, default.mixed 8.9 s per p**.
- **Projection** (`results/smoke/projection.json`): list-scheduling of the real job graph
  (Section 9 order and dependencies) on vCPU/2 workers; "slowdown" = cloud-vCPU / Mac per-sample
  time, unknown until measured on the instance. The sweep's winners are unknown, so both extremes:

| sweep winners | Mac worker-h | gpu_1x_gh200 (32 workers) slowdown 1.5 | slowdown 2.0 |
|---|---|---|---|
| all angle | 369 | 17.8 h, $41 | 23.6 h, $54 |
| all amplitude | 1,181 | **57.1 h, $131** | **76.0 h, $174** |

  In the amplitude case both limits (> $100, > 48 h) are exceeded. Cause: the 6.3 depolarizing
  sweep of the thirty n = 8 amplitude sweep models on default.mixed, ~20 h each on the Mac —
  ~606 of the 1,181 worker-hours. (gpu_1x_a10: 79–158 h, $102–204.)
- **Scratchpad benchmark of an alternative (not implemented):** Aer `method="density_matrix"`
  with exact `save_expectation_value` (no shots) and the same channels
  (PennyLane `DepolarizingChannel(p)` = Qiskit `depolarizing_error(4p/3, 1)`; 10·p on each qubit of
  every CNOT) reproduces default.mixed to ≤ 1e-9 and is 9–16× faster (n = 8 amplitude: 0.81 s per
  image per p). Re-projected: amplitude case $69 / 30 h (slowdown 1.5) to $92 / 40 h (2.0); angle case
  $30–39. With one worker per vCPU as well (GH200 has no SMT, so each vCPU is a core): amplitude
  $52–69, angle $16–22.

## 3d. Credentials — **RESOLVED (see §0)**

`LAMBDA_API_KEY`, `LAMBDA_SSH_KEY_NAME`, `LAMBDA_SSH_PRIVATE_KEY_PATH` are not set in this shell
(presence checked; values never read or printed). Required before the cloud queue step.

## 4. Calibrated noise backend (6.2)

- **DECIDED — FakeAachen.** Heron-class fake backends installed in qiskit-ibm-runtime 0.50.0:
  FakeTorino (r1), FakeFez, FakeMarrakesh, FakeKingston (r2), FakePittsburgh, FakeBoston,
  FakeAachen (r3). "Newest" = highest Heron revision, then latest embedded calibration date:
  FakeAachen, Heron r3, 156 qubits, calibration `2026-04-17 12:29:46+02:00`, basis gates
  {cz, rz, sx, x, id, delay, measure, reset, if_else}.
- **DECIDED** — noise model `NoiseModel.from_backend(FakeAachen())` (gate errors, thermal
  relaxation, readout errors as provided); `transpile(optimization_level=1, seed_transpiler=0)`
  once per architecture; Aer `seed_simulator = 0 + offset` of each 250-image chunk within its
  500-image `predict` batch, so the seeds repeat every 500 images; `max_parallel_threads = 1`
  (both corrected 2026-10-31, see the end of §14). Aer drops idle
  qubits itself and chose its density-matrix method. ⟨Z_i⟩ = (N₀ − N₁)/shots from the counts of
  clbit i (Qiskit bitstrings are little-endian; unit-tested).
- **DECIDED — the Qiskit circuit is the lightning circuit.** Built gate-for-gate from the same
  elementary decomposition that G counts (one parameter per rotation, bound per sample; weights
  bound as well). Unit tests check, for n ∈ {2, 4}, both encodings and all topologies, that its
  noise-free statevector reproduces lightning's ⟨Z_i⟩ to 1e-6 and that its gate count equals G.

## 5. Data, naming, run files

- **DECIDED — ids.** Datasets `mnist`, `fashionmnist` (no underscore: `_` separates run-id
  fields); topologies `linear`, `fullyconnected`, `random`; placements `pre`, `middle`, `post`;
  encodings `angle`, `amplitude`. Run ids follow Appendix A, e.g.
  `fashionmnist_middle_amplitude_linear_n4_tf1.0_s0`, `mnist_base_cnn_tf0.5_s2`,
  `mnist_matched_post_angle_tf0.2_s4`. Config files are named by the run id without `_s<seed>`.
- **DECIDED — normalization constants** (standard values over the 60k training images):
  MNIST (0.1307, 0.3081), Fashion-MNIST (0.2860, 0.3530).
- **DECIDED — stratified sampling.** Per-class quotas by the largest-remainder method (sum is
  exact; remainder ties to the lower class index); within a class, `numpy.random.default_rng`
  `choice` without replacement; indices returned sorted. Validation: `default_rng(split_seed=0)`;
  pool = the other 55,000; training subsets: `default_rng(run_seed)` on the pool (11,000 / 27,500;
  tf = 1.0 is the whole pool, so for tf = 1.0 seeds vary initialization and batch order only);
  noisy-evaluation subset: `default_rng(subset_seed=0)` on the test set. Smoke tests' 500 test
  images: stratified, `default_rng(0)`.
- **DECIDED — run files.** `results/runs/<run_id>.json` and `results/checkpoints/<run_id>.pt`
  (best-validation state dict). The checkpoint is written first and the JSON atomically last, so an
  existing, valid JSON implies a checkpoint. A run with a non-finite training loss raises; it never
  continues silently. Per-epoch log also records `train_seconds` (informational).
- **DECIDED — A_subset_noiseless** is taken from the same full-test-set predictions as A (the 2,000
  images are a subset of the 10,000): same model, same analytic simulator, no second pass.
- **DECIDED — classical baselines' noise fields** are `null` in the JSON and NaN in the CSVs.
- **DECIDED — accuracies are exact k/N** (see §3c for the float32 bug this replaced).

## 6. cnn_matched (4.6)

- **DECIDED — input of the classical map** = exactly the tensor the circuit would consume: the
  feature-map output (angle: z = π·tanh(Wx); amplitude: the normalized v). Everything classical
  before the circuit (projection, feature map) is unchanged; encoding gates + variational circuit +
  readout are replaced. Ties between candidate maps → first in the protocol's listed order (none
  occur).
- **FACT — selected maps and totals** (quantum part = 36):

| placement / encoding | hybrid total | matched total | map (params) | difference |
|---|---|---|---|---|
| pre / angle | 19,074 | 19,069 | mlp_h3 (31) | −5 (−0.026 %) |
| pre / amplitude | 28,494 | 28,504 | mlp_h2 (46) | +10 (+0.035 %) |
| middle / angle | 4,978 | 4,973 | mlp_h3 (31) | −5 (−0.100 %) |
| middle / amplitude | 8,062 | 8,072 | mlp_h2 (46) | +10 (+0.124 %) |
| post / angle | 11,866 | 11,861 | mlp_h3 (31) | −5 (−0.042 %) |
| post / amplitude | 12,262 | 12,272 | mlp_h2 (46) | +10 (+0.082 %) |

  All within the protocol's < 0.5 %. Note: `mlp_h3` / `mlp_h2` route the signal through 3 / 2
  hidden units — a narrower bottleneck than the quantum block's n = 4 outputs.

## 7. Depolarizing sweep (6.3)

- **DECIDED — "every single-qubit gate" means every elementary gate** of the decomposition G
  counts: each Rot contributes RZ·RY·RZ, each followed by `DepolarizingChannel(p)`; every CNOT is
  followed by `DepolarizingChannel(10·p)` on both qubits; encoding gates (RY, or the Möttönen
  RY/RZ/CNOT) included; GlobalPhase not executed. This keeps "noise per counted gate" consistent
  with G. Qiskit Aer `method="density_matrix"`, exact expectation values, 250 images per Aer job
  (decision 3c; corrected 2026-10-31, see the end of §14).

## 8. Smoke tests (11.2)

- **DECIDED — coverage** (43 configurations, a superset of the protocol's list): all 18 M1
  architectures; for each sweep n ∈ {2, 6, 8}, all 3 placements × 2 encodings with
  FullyConnected (the most CNOTs; the sweep's winners are not known yet); base_cnn and all six
  cnn_matched variants.
- **DECIDED — timing run** placement: Pre (the protocol names n, encoding, topology and tf but not
  the placement; Pre has the largest classical part, so it is the conservative choice).
- **DECIDED — noisy-pipeline validation models**: pre/amplitude/FC, middle/angle/random,
  post/amplitude/linear (all placements, both encodings, all topologies), plus the 15-epoch timing
  model as a well-trained fourth.

## 9. Cloud stage — superseded where §10 says so

- **DECIDED — dispatch order.** Section 9 block order (M1, B1, M2, B2, S, N); within a block
  train_frac descending; ties: larger n first (longest-processing-time first — without it, run-id
  order would start the sweep's n = 8 runs last and lengthen the paid wall-clock), then run id.
  Noise jobs (N) depend on their training run; S jobs on every tf = 1.0 run of their dataset's
  main block (the selection input). N jobs fill workers only when no earlier-block job is ready.
- **DECIDED — "pulls code/"**: there is no git remote, so the Mac rsyncs `code/` (without `.venv`
  and `results/`, with the downloaded datasets) to the instance; the instance builds its
  environment with `uv sync --frozen` from the same `uv.lock` and runs the unit tests before the
  queue starts.
- **DECIDED — failed jobs** are retried once, then reported (`failed` in `queue_state.json` and in
  STATUS) and not retried again; never silently dropped.
- **DECIDED — termination** follows 10 exactly: only the Mac monitor terminates, after a final
  rsync whose SHA-256 of every remote run JSON and checkpoint matches the local copy; otherwise
  STATUS says "FINAL SYNC NOT VERIFIED — instance NOT terminated". The instance never terminates
  itself. The Mac monitor should run under `caffeinate` so the Mac does not sleep.
- **DECIDED — instance rule**: among instance types with capacity at launch time, maximize vCPUs
  per dollar-hour (from the API's `price_cents_per_hour`, `specs.vcpus`). On the public price list
  (2026-09-26) that is `gpu_1x_gh200` (64 vCPU, $2.29/h, 27.9 vCPU/$·h; ARM — `uv.lock` has aarch64
  wheels for torch+cpu, pennylane-lightning, qiskit-aer, numpy, scipy), then `gpu_1x_a10`
  (30 vCPU, $1.29/h, 23.3).
- **FACT — CPU use per worker.** A training process runs at ~100 % of one core with
  OMP_NUM_THREADS = 2: per-sample cost is single-threaded PennyLane overhead. The protocol's
  2 vCPUs per worker is followed; 1 worker per vCPU would roughly double throughput per dollar
  (a deviation — not done without approval).

## 10. Fleet implementation (after Jeremy's plan, 2026-09-26)

- **FACT — availability at planning time.** `gpu_1x_gh200` had no capacity; available allowed
  types, best vCPU per dollar first: `gpu_1x_a10` (30 vCPU, $1.29/h), `gpu_1x_a100_sxm4`
  (30, $1.99/h), `gpu_8x_a100_80gb_sxm4` (240, $22.32/h), `gpu_1x_h100_pcie` (26, $3.29/h),
  `gpu_8x_a100` (124, $15.92/h). The Lambda API sits behind Cloudflare, which rejects Python's
  default User-Agent (error 1010); requests send `User-Agent: qpar-experiment/0.1`.
- **DECIDED — "GPU types up to and including H100"**: every `gpu_*` type except B200 (GH200 counts
  as Hopper generation, i.e. H100-class). CPU-only types are not GPU types and are not used.
- **DECIDED — one single-threaded worker per physical core** (`lscpu` on the instance). On x86 the
  vCPUs are hyperthreads; two workers per core would lengthen the longest runs, which set the
  wall-clock floor. On GH200 physical cores = vCPUs, as in decision 3c.
- **DECIDED — noisy evaluation split into parts** (`cal`, `depol<p>` for each p): five independent
  jobs per hybrid run, each writing `results/noise/<run_id>.<part>.json`, so the long n = 8 parts
  run in parallel. They are folded into `results/runs/<run_id>.json` (12.1 layout; baselines get
  null noise fields) only after every instance is terminated, so the checksum verification always
  compares unmodified files.
- **DECIDED — resumable training**: after every epoch the full state (model, Adam moments, the
  shuffling generator, log, best checkpoint) goes to `results/partial/<run_id>.pt`; a restarted
  run continues bit-identically (unit test: 1 + 1 epochs = 2 epochs, all tensors equal). The run
  JSON records `resumed_after_epochs`.
- **DECIDED — dispatch order (wall-clock first, "longest runs first")**: phase 0 = M1/M2 tf = 1.0
  training (M1 first; the sweep selection needs these), phase 1 = sweep training (as soon as its
  selection exists), phase 2 = all other training, phase 3 = noisy evaluation; longest first
  within a phase. This departs from the Section 9 block order (whose purpose was partial results if
  the budget were cut) because Jeremy prioritized wall-clock; the set of runs is unchanged.
  Noise parts only fill workers that no training job can use ("training before noisy evaluation").
- **DECIDED — fleet sizing.** Floor = makespan with unlimited workers (the critical path: M
  tf = 1.0 runs → selection → longest sweep run → its noise parts). Initial fleet: the fewest
  instances of the step-1 type whose projected makespan is ≤ 1.5 × floor in the worst case (all
  sweep winners amplitude). After both selections, the remaining work is re-planned and instances
  are added (≤ 8 live) if that brings it within 1.5 × its floor and the projected total ≤ $150.
- **DECIDED — shards and rebalancing.** Run bundles (training + its five noise parts) are assigned
  longest-first to the instance with the least projected load per worker. Every 15 min (and when
  an instance starts) unstarted jobs move from the instance projected to finish last to the one
  finishing first while the gap > max(30 min, 15 %), in two phases: the source first drops them
  from its shard; only jobs it had not started move, and a noise part moves only with its
  training run or once that run is done (outputs pushed). Unit tests cover these cases.
- **DECIDED — termination and failures.** An instance is terminated only when it owns no
  unfinished job, both selections exist, no other instance holds work it could take, and every
  run JSON, checkpoint and noise result on it exists on the Mac with the same SHA-256. Unreachable
  for 30 min → declared dead, its unfinished jobs go to a replacement (resuming synced partials);
  never terminated before sync (a dead instance still billed after 2 h is a STOP notice). A launch
  failure is retried once with a replacement; a second consecutive creation failure is STOP 15.6.
- **DECIDED — mid-run STOP conditions notify Jeremy and the fleet keeps running** (a stopped
  fleet would bill idle or lose progress), except the cost cap: at $195 spent every instance is
  terminated after verified sync. Projected total > $150 or wall-clock > 48 h, and any tf = 1.0
  configuration with mean A < 0.5, are raised once each.
- **FACT — local integration test** (scratchpad copy, tiny non-study config n = 2, tf = 0.02;
  outputs discarded): the runner kept noise parts waiting for their training run, the training
  run was killed after epoch 3 and resumed bit-identically, and the merge produced the 12.1 layout.

## 11. Cloud step 1 (2026-09-26 22:07–22:41) — **DECIDED: (a), launch**

- **FACT — instance.** qpar-1 = `gpu_1x_a10`, us-east-1, $1.29/h; Intel Xeon Platinum 8358
  @ 2.60 GHz exposed as 30 single-thread vCPUs → 30 workers. Provisioned from `uv.lock` in 6 min;
  the full unit-test suite passed there. Cost $0.74 (34 min); terminated after verified sync.
- **FACT — smoke re-run (2a)**: 42/43 pass in 17.8 min on 30 workers (`results/smoke/archs.json`).
  Noise gates pass: shots-only −0.45 to +0.05 pt; Aer depolarizing at p = 0 exact
  (`results/smoke/noise.json`).
- **FAIL: post / angle / FullyConnected / n = 2**, A = 0.198 (99/500). Diagnostic
  (`results/smoke/diag/`): not broken — validation accuracy 0.201 → 0.409 over 6 epochs at tf = 0.2,
  then 0.415 → 0.419 over 2 epochs at tf = 1.0: a capacity plateau (two ⟨Z⟩ values feed
  `Linear(2, 10)`). Middle/angle/n = 2 (0.366) and post/amplitude/n = 2 (0.280) passed but are low.
  Consequence: if the sweep selects angle encoding for Post (or Middle), its n = 2 point will very
  likely end with mean A < 0.5, which is the mid-run STOP 15.3 condition.
- **FACT — cloud speed.** Per-sample training is 2.7× the Mac under full load (median of 40
  configurations) and 2.1–2.3× on an idle machine: Ice Lake cores are ~2.2× slower than M3
  performance cores for this Python-bound workload; 30 concurrent workers add ~1.2×.
- **FACT — plan from measured cloud rates** (`results/fleet/plan.json`): worst case (all sweep
  winners amplitude) floor 24.8 h; 1 × A10 51.7 h / $67 (2.07 × floor); **2 × A10 29.9 h / $77
  (1.19 × floor) — chosen**; 3 × 27.0 h / $104. Angle case: floor 5.4 h; the post-selection re-plan
  would go to ~4 instances, ~6.8 h / $35. Projected total with step 1: $78 (≤ $150, ≤ 48 h).

## 12. Jeremy's answers to the step-1 STOP (2026-09-26)

- **(a)** The post/angle/FC/n = 2 smoke failure is accepted as a capacity limit, not a defect. The
  mid-run 15.3 condition (tf = 1.0 configuration with mean A < 0.5) applies to every configuration
  except the qubit sweep's n = 2 points, which are reported as they come out.
- **GH200 counts as H100-class** (allowed).
- **Speculative n = 8 amplitude candidates.** At launch, the n = 8 amplitude sweep run is started for
  every (dataset, placement, topology) × 5 seeds — 90 runs — so the ~25 h floor starts immediately.
  After the validation-based selection (6a), candidates that do not match the selected (encoding,
  topology) are cancelled and discarded (partials deleted; logged); a matching candidate *is* the
  sweep's n = 8 run (same config, same seed, hence the same result as starting it after the
  selection; the selection only reads M tf = 1.0 validation accuracies, so there is no leakage).
- **DECIDED — speculation mechanics.** Candidates (`configs/sweep_candidates/`) are dispatched before
  everything else (phase −1), grouped by topology on dedicated instances (30 per A10) so survivors
  of a common topology share an instance. At each dataset's selection the non-matching candidates
  are cancelled on their instance (process killed, partial deleted) and recorded under
  `discarded` in `results/fleet/fleet_state.json` with the epochs they had completed. A surviving
  candidate's noise parts are assigned only when its training finishes (inputs delivered to
  whichever instance takes them).
- **DECIDED — consolidation.** After both selections, surviving candidates are consolidated onto
  the fewest instances (those already holding most of them): a survivor elsewhere is withdrawn
  from its shard and asked to yield at its next epoch boundary (`results/yield/<run_id>`; the
  process exits 75 after saving its state), its partial is moved, and it resumes on the keeper
  without losing a completed epoch. Other instances then terminate when idle and verified synced.
  (Runner yield/cancel/resume paths tested locally on a tiny non-study config.)
- **DECIDED — fleet (plan with speculation, measured cloud rates, `results/fleet/plan.json`).**
  Cost integrates the busy-core profile assuming idle instances are consolidated and terminated.
  Worst case (all sweep winners amplitude): floor 22.5 h; 3 × A10 32.5 h / $124; 4 × 23.6 h / $86;
  **8 × A10 22.8 h (1.00 × floor), ~59 instance-hours, $76 — chosen (cheapest within 1.5 × floor;
  more instances also finish the selection-critical runs sooner, which cuts speculative waste)**.
  Angle case on 8 × A10: 5.7 h, $44. Projected total with step 1: $76 (≤ $150, ≤ 48 h).

## 13. Fleet run — incidents and fixes (2026-09-26/27)

- **22:58 launch**: 8 × gpu_1x_a10 (us-east-1). Coordinator bugs found on the live fleet and fixed
  without losing work: (1) a runtime cost projection that assumed every instance bills until
  the longest candidate ends raised a spurious "STOP 15.2 $224" (the plan projects $76; the
  flag was cleared and the projection now uses the plan until both selections exist, then a
  consolidating simulation like the planner); (2) `results/` did not exist on fresh instances, so
  shard pushes and pulls failed (and sync failures were swallowed) — result dirs are now created
  at start and sync errors are logged; (3) a runner restart command matched its own `pkill`
  pattern — stop and launch are now separate ssh calls; (4) the ssh known-hosts path contained a
  space, which ssh splits (a stray `~/QML` file was created and removed) — now
  `~/.cache/qpar/known_hosts`; (5) the heartbeat check now requires a recent successful sync, so
  a sync problem can never make the coordinator restart healthy runners. All 240 workers were
  verified running throughout; no runner was restarted.
- **FACT — NaN in two amplitude runs.** `mnist_middle_amplitude_random_n4_tf1.0_s4` and
  `fashionmnist_post_amplitude_fullyconnected_n4_tf1.0_s4` (both on fleet-8) crashed twice each in
  epoch 1: the weights became NaN after an update, then Möttönen state preparation rejected the
  NaN input. Not reproducible on the Mac (different floating-point path); no kernel hardware
  errors on fleet-8.
- **DECIDED — skip ill-defined updates.** If any gradient is non-finite after `backward()`, that
  optimizer step is skipped (logged per epoch as `skipped_steps`; `skipped_steps_total` in the run
  JSON). A run that never produces a non-finite gradient is unaffected, so runs trained before and
  after the deployment are directly comparable. Suspected source: PennyLane's Möttönen angle
  `2·arcsin(√r)` has an infinite derivative at r ∈ {0, 1} although the function (= 2·atan2) is
  smooth there. Deployed at 23:49 to all instances (new job processes load it; running jobs were
  not interrupted). Unit test: an injected NaN gradient skips exactly one update.
- **DECIDED — retry once elsewhere.** A job that failed twice on one instance is reassigned once to
  another instance (with its partial state); a second failure there is final and reported. The two
  jobs above were reassigned to fleet-5 and fleet-4 at 23:51.
- **FACT — two candidate crashes predate the guard.** `fashionmnist_middle_amplitude_fullyconnected_n8_s1`
  (fleet-1) and `fashionmnist_pre_amplitude_linear_n8_s4` (fleet-2) failed twice with the same NaN;
  their last crashes are logged at 00:14/00:17, ~30 min into second attempts that began before the
  23:49 guard deployment. They were retried at 00:19 on fleet-8/fleet-7 (restarting epoch 1;
  if either survives the selection the floor moves ~1.2 h). Every process started after 23:49
  has the guard; retries will show `skipped_steps` instead of crashing.
- **Fixed (00:19)** — rsync exit 24 (a partial deleted by a run finishing mid-sync) no longer
  counts as a sync failure (it could have tripped the 30-min "unreachable" rule); each instance's
  queue state is pulled first.
- **Fixed (01:05)** — the rebalancer weighed candidate hours: a retried candidate made fleet-8
  look 19 h "late", and 492 of its jobs were moved to fleet-5 at 00:34 (a correct two-phase move,
  just unnecessary). It now balances only non-candidate work over the cores candidates leave free
  and never sends work to an instance whose cores are all held by candidates (regression test).
  Job logs are now pulled per instance (`results/logs/instances/<name>/`): before, a retry's empty
  log overwrote the failed attempt's log on the Mac (the originals were read on the instances).
- **Fixed (01:33, 01:44)** — rebalance sizing and timing. Moves were sized by the sender's free
  cores only, so work went to candidate instances with a single free core (01:09: 330 jobs
  fleet-5 → fleet-1) and then back out (01:27). A move now transfers the work that equalizes
  both finish times, gap / (1/F_late + 1/F_early) core-seconds (regression test), and rebalancing
  also runs immediately when an instance has idle cores and nothing queued (fleet-8 sat mostly idle
  from ~01:20 to 01:45). Two-phase handoffs meant no job ever ran twice.
- **FACT — sweep selection (6a), 02:25.** Mean best-validation accuracy, tf = 1.0, 5 seeds
  (`results/sweep_selection_{mnist,fashionmnist}.json`). MNIST: pre angle/FC (0.9902), middle
  amplitude/random (0.9892), post amplitude/random (0.9902). Fashion-MNIST: pre amplitude/linear,
  middle amplitude/FC, post amplitude/FC. Margins between the top configurations are ~0.1 pp.
  Speculative candidates: 25 kept (the selected amplitude configurations), **65 discarded** (0–2
  epochs each; cancelled on fleet-1/2/3 — processes killed, partials deleted — logged under
  `discarded` in `results/fleet/fleet_state.json`). The MNIST Pre sweep point at n = 8 runs angle
  (not speculated; ~3 h). 515 new sweep jobs were created (n = 2, 6 and the survivors' noise parts).
- **FACT — consolidation.** fleet-3 (10 survivors) is the keeper; 15 survivors on fleet-1 (9),
  fleet-2 (4), fleet-7 (1) and fleet-8 (1) yield at their next epoch boundary and resume there.
  Post-selection re-plan: remaining floor 22.5 h, no additional instances, projected total $85.
- **Fixed (02:29)** — survivors migrating to a keeper reserve its cores in the coordinator's
  accounting, so rebalancing cannot fill them with other work. fleet-3 had exactly 15 free slots
  for the 15 arrivals (its other queued work is noise parts waiting for their training runs).
- **FACT — the NaN was a deterministic numerical singularity, not hardware.** The retried
  `fashionmnist_pre_amplitude_linear_n8_tf1.0_s4` (it had crashed twice before the guard) recorded
  `skipped_steps = 1` in epoch 1 and trained on normally (validation 0.8636 after epoch 1): the same
  run on the same platform produces a non-finite gradient at the same point, which the guard now
  skips. This rules out a fault on fleet-8.
- **FACT — first survivor migration (02:44).** The same run yielded on fleet-7 after epoch 1 and
  resumed on fleet-3 from its partial state (migration verified end to end).
- **DECIDED (03:34) — consolidation of all long runs.** Once no instance had startable work queued,
  every unfinished training run with > 1 h left (the 25 survivors, 30 n = 6 sweep runs and the
  five MNIST Pre n = 8 angle runs) was gathered onto the fewest instances (keepers fleet-3 and
  fleet-2); runs elsewhere yielded at their next epoch boundary (≈25 min for n = 6) and resumed
  there, taking their noise parts along. Non-keepers then terminated after verified sync:
  fleet-8 03:48 ($6.24), fleet-6 03:52 ($6.33), fleet-7 04:15 ($6.81), fleet-5 04:18 ($6.88),
  fleet-1 04:34 ($7.24).
- **Fixed (04:33) — a migrating run that crashed stayed in limbo.** Survivor
  `fashionmnist_post_amplitude_fullyconnected_n8_s0` (started 23:09 on the pre-guard code) hit the
  NaN at 02:26 in epoch 3, the minute it was withdrawn for migration; a withdrawn job is not
  restarted by its runner and a single crash is not "failed", so it sat unassigned for 2 h and kept
  fleet-1 alive. Migration now also completes for a run that stopped on its source without
  yielding, from its latest partial (epoch 2; no skipped steps before the crash, so the resumed
  run equals a guarded one). Cost: ~2 h of that run's progress; it becomes the last run to finish.
  Regression test added; a test fixture that omitted a still-running job was corrected.
- **DECIDED (04:4x) — end-game capacity (wall-clock first).** The 25 survivors' 125 noise parts
  (~0.8 h each at n = 8, ~100 core-hours) become ready only as each survivor finishes; on the
  keeper alone they would queue for ~3.3 h. When the first survivor is within 1 h of finishing
  (from its own epoch times), the coordinator launches enough extra instances for the parts to run
  in about 1 h, only if the projected total stays ≤ $150; those instances are not idle-terminated
  before they have done some work. Unit test added.

- **FACT (06:30) — fleet-3 is a slow host; DECIDED (06:40) — evacuate it to fleet-4.** The
  synced partial states showed the MNIST n = 8 survivors at 3.0 h per epoch against 1.5 h for the
  Fashion-MNIST ones. On the instance: 30 single-threaded run processes, load 30.0 on 30 vCPUs,
  no orphans. Per instance, the mean epoch time of never-migrated runs relative to the same
  configuration elsewhere (full load everywhere): fleets 4–8 1.00 (n = 83–223 runs each), fleet-3
  1.99; its n = 6 runs took 0.84 h per epoch against 0.40 h on fleet-2 (20 processes) and 0.36 h on
  fleet-4 (1 process). fleet-3 had held the survivors since launch, so all 25 ran at half speed:
  about 36–39 h left for the MNIST ones, a projected ≈ 47 h total, close to the 48 h STOP (15.2).
  The coordinator's projection uses the smoke rates and did not see it. Rebalancing a lagging
  shard is within Jeremy's plan, so without a STOP: fleet-3's 29 training runs (25 survivors,
  4 n = 6 runs) move to fleet-4 — a normal-speed host with 29 idle workers, finishing its last
  run — at their next epoch boundary (the migration mechanism of the consolidation), with their
  queued noise parts; fleet-3 keeps only noise parts, receives no rebalanced work and terminates
  when empty. No new instance; cheaper than before (fleet-3 stops in about 2 h instead of ~40 h).
  Expected: survivors done ≈ 24–27 h after launch instead of ≈ 46 h. Also: the end game now
  estimates each survivor's remaining time from its latest epoch (the current host's speed)
  instead of its mean epoch. Unit test added (`test_evacuate_moves_training_runs_off_a_slow_instance`).

- **FACT (07:13) — the evacuation works.** The first runs moved at 06:46; the first epoch on
  fleet-4 of fashionmnist_middle_amplitude_fullyconnected_n6_tf1.0_s1 took 0.40 h against
  0.82–0.89 h for its five epochs on fleet-3. By 07:02, 14 of the 29 runs had moved.

- **FACT (17:55) — the Mac was offline 12:14–17:50; fleet-4 was never lost.** From 12:14 every
  sync of fleet-4 failed (ssh "Bad file descriptor") and, at 12:44, so did the Lambda API calls
  (`URLError(OSError(9))`): the Mac itself had no network (asleep or offline after the Claude app
  was quit; caffeinate -is does not prevent sleep on battery or with the lid closed). After 30 min
  the coordinator declared fleet-4 dead, handed its 25 runs to replacements (fleet-9, fleet-10)
  whose launches failed for the same reason, and raised STOP 15.6 (instance creation failed twice)
  and, at 14:48, "fleet-4 unreachable for 2 h". Meanwhile fleet-4 (Lambda status active) kept
  training all 25 survivors without a fault (25 processes, no failures). No instance was created,
  so nothing was billed twice. **Fixed (17:58):** fleet-4 revived in the coordinator state (its
  25 runs reassigned to it, the abandoned fleet-9/10 left as records, the two false STOPs
  cleared); the first sync pulled the missing 5.5 h. An instance is now declared dead only when the
  Lambda API answers (if the API is unreachable too, the Mac is offline and the coordinator waits);
  unit test added. Jeremy was told (the STOP was a false alarm caused by the Mac, not by Lambda).
- **DECIDED (18:05) — no extra instances to speed up the last 25 runs.** Jeremy asked whether more
  instances could finish the study within 3 h. They cannot: each remaining run is one sequential
  training job already on its own core, with 5–7 epochs left at ≈ 1.6 h each (≈ 7–10 h). Spreading
  the runs over more instances would only reduce CPU contention (n = 6 evidence: 0.36 h/epoch alone
  vs 0.45 h with 30 processes on the same host), saving ≈ 1 h for ≈ $25 — and ≈ 12 % per new host
  of landing on a slow one like fleet-3, which would cost more than it saves (expected gain ≈ 0.5 h).
  Not worth it; what matters for the finish is that the Mac stays awake and online (end-game
  noise instances, verified termination).

- **FACT (06:51, 09-28) — the Mac was offline again 23:10–05:00; a false loss followed.** The new
  rule held while the network was down (every half hour: "the Mac seems offline; waiting"), but at
  05:00, with the network back, the first ssh attempt timed out while the Lambda API answered
  "active": fleet-4 was declared dead on its stale last contact (23:10), fleet-11 was launched and
  restarted the 25 survivors from their 23:10 partial states (duplicate work, 05:08–06:55), and
  STOPs were raised ("fleet-4 unreachable for 2 h"; 15.2 "projected wall-clock 55.8 h > 48 h",
  from re-running the survivors). Meanwhile fleet-4 finished all 25 survivors (840/840 training
  runs), then idled — the end game never triggered while the Mac was offline, so none of the 125
  survivor noise parts had started. **Fixed (06:55–07:05):** fleet-11's duplicate runs stopped and
  their partial states deleted (it had produced nothing); fleet-4 synced (checksums verified; all
  840 run JSONs valid with checkpoints) and revived; fleet-11 kept for noise parts; three more
  instances (fleet-12–14, marked end game) for the 125 parts to run in about one wave (≈ 1 h,
  ≈ $5) instead of ≈ 2.4 h on two; the false STOPs cleared (real wall-clock ≈ 34 h at the finish).
  Code: while the Mac is offline, each instance's 30-min unreachable window restarts (unit test);
  instances added to the state as planned are launched by the coordinator. Cost of the incident:
  ≈ 2 h of fleet-11 plus fleet-4 idle 04:40–07:05, ≈ $5.

- **DECIDED (07:16) — six queued noise parts moved to the idle fleet-14 by hand.** fleet-12–14 came up
  after fleet-4 and fleet-11 had started 30 parts each; the rebalancer moved the not-yet-started
  ones, except six (4 on fleet-4, 2 on fleet-11) that it judged balanced, because its finish estimate
  spreads queued work evenly over busy cores (fluid approximation) while a queued part actually waits
  ≈ 45 min for a core. Moved to fleet-14 with the coordinator's own two-phase handoff; from 07:20 all
  125 parts ran at once (fleet-4 30, fleet-11 30, fleet-12 30, fleet-13 28, fleet-14 7).

- **FACT (08:35, 09-28) — cloud phase complete.** DONE: all 3,990 jobs (840 training runs, 3,150
  noise parts) complete, every instance terminated after a checksum-verified sync. Total cost
  $110.37 (step-1 instance $0.74 + fleet $109.64; 85.0 instance-hours of gpu_1x_a10 at $1.29/h,
  12 instances over the run), wall-clock 33.6 h from the fleet launch (2026-09-26 22:58).
  `check_complete` passes on the Mac: 840 runs, all noise results merged into the run JSONs.
  Next: latency on Jeremy's Mac (protocol 7), then the analysis.

## 13a. Latency D (protocol 7) — measured 2026-09-28 15:45:46–15:46:31 (Jeremy's go)

- **FACT — measured as specified.** Jeremy's Mac (Apple M3 Pro, 6 performance + 6 efficiency cores,
  on AC power, other applications closed), one process, `OMP_NUM_THREADS=1`, torch threads 1,
  lightning.qubit analytic, seed-0 checkpoints of all 168 configurations, batch size 1, first 220
  images of the noisy-evaluation subset: 20 warm-up, median of 200 (`results/latency.csv`,
  `results/latency_env.json`, log `results/logs/latency.log`). The fleet coordinator and monitors
  were stopped first. macOS background daemons (media analysis) were active at background priority;
  load average ≈ 4 before and after (recorded in latency_env.json; fields `cpu` and load averages
  were added to it for this record).
- **FACT — values.** base_cnn 0.05 ms; cnn_matched 0.06–0.08 ms; hybrids at n = 4 1.31–1.75 ms
  (angle median 1.35, amplitude 1.40); sweep n = 2 0.84–0.95, n = 6 1.81–2.67, n = 8 2.25–4.05 ms.
  The same architecture at its three train_frac differs by 1.1 % (median; max 10.6 %): D is a
  property of the architecture, as expected.
- **FACT — D measurement spread.** For the same n = 4 architecture measured six times (two datasets ×
  three train_frac), max/min D has median 1.045 and maximum 1.125 (post/angle/FC); e.g.
  fashionmnist_pre_angle_random_n4 1.386–1.394 ms at tf 0.2/0.5 but 1.498 ms at tf 1.0
  (`results/latency.csv`). Among configurations with the same G, Q-Par is ordered by D differences
  of this size.
- **FACT — D and G measure different costs for amplitude encoding.** G counts the amplitude
  embedding at full decomposition (Möttönen; protocol 7), whereas inference on lightning.qubit
  prepares the state natively (StatePrep), so amplitude circuits cost little more than angle ones
  in D. D is the simulator-cost proxy the protocol defines (claim 5); to be discussed in the paper.

## 14. Analysis (Sections 8, 12, 13) — `src/qpar/analysis.py`, written while the fleet runs

Tested on a synthetic complete study (`tests/test_analysis.py`) and on the synced runs; the final
run needs every run, every noise part and `results/latency.csv`, and refuses otherwise.

- **DECIDED — rows.** Only the study's 168 configurations (`configs/matrix`, `baselines`, `sweep`)
  are analysed; a run JSON of any other configuration (a discarded sweep candidate) is ignored.
  Noise results are read from the merged run JSON or, before the merge, from `results/noise/`.
  Runs trained before the non-finite-gradient guard (§13) have no skip count and are counted as 0
  (a non-finite gradient would have crashed them). `runs.csv` flattens every field (per-epoch
  log as `eNN.field`).
- **DECIDED — summary.** Mean, std (ddof = 1), 95 % CI half-width (t, 4 dof) over seeds; σ_a = that
  std. Q = n for hybrids, 0 for baselines (no Q-Par for baselines, 8.1). `epochs_to_threshold`:
  mean over the seeds that reached the threshold, plus how many did. G and `twoq_transpiled` are
  checked to be identical across seeds.
- **DECIDED — normalization and ranks.** Q_max, D_max, G_max over all 63 hybrid configurations of
  the dataset (54 main + 9 sweep, 8.1), recorded in `qpar_maxima.csv` and in the T1/T2 captions.
  Ranks (A and Q-Par; 1 = highest, ties share the best rank) are among the dataset's 54 main
  configurations — the population of 8.2/8.3; T1/T2 show these ranks. Sweep configurations get
  Q-Par but no rank.
- **DECIDED — sensitivity (8.2).** The 14 settings of `qpar.sensitivity_settings`; Kendall τ-b
  against the equal-weight scores. `sensitivity.csv` holds every setting's full ranking (long
  format); `sensitivity_summary.csv`/T6 hold τ, p and the top 5.
- **FACT — γ cannot move the 8.2/8.3 rankings.** Every main-matrix configuration has n = 4, so
  Q/Q_max is one constant for all 54: γ = 0.5 and γ = 2 give τ = 1, resource-first (γ = ε = 2)
  ranks exactly like ε = 2, and Q never drives a rank move. Q's effect on Q-Par is visible only
  in the qubit sweep (T4, F5). Not a STOP (implementable as written; the design makes it so); to
  be raised with Jeremy with the latency request.
- **DECIDED — rank-move driver (8.3).** A configuration that rises by ≥ 10 places passed
  configurations with a higher A; since log Q-Par = log A + the cost terms, each pass is explained
  exactly by the terms. The driver is the term (σ_a, Q, D, G) with the largest mean log advantage
  over the configurations passed (rising) or passing it (falling); all four advantages are in
  `rank_moves.csv`. First tried and rejected: "the term whose removal (exponent 0) undoes most of
  the move" — D and G are strongly correlated, so removing either alone leaves the other driving
  the move and the attribution fell to noise terms (seen on synthetic data).
- **DECIDED — T9 and numbered labels in F2.** 8.3's list needs a place in the paper that T1–T7 do
  not provide: T9 lists every move ≥ 10 with its driver. Q-Par will reorder many configurations
  (D and G vary far more than A), so F2 labels the moved points with their T9 numbers instead of
  text (text labels collided even on synthetic data); a shaded band marks moves < 10.
- **DECIDED — hybrid vs cnn_matched (13).** Primary: the protocol's 36 cells (dataset × placement ×
  encoding × train_frac); the hybrid's value per seed is the mean over its three topologies (the
  matched model is shared by all three), paired with the matched model by seed (same training
  subset and batch order). Per-topology comparisons (108) are in
  `stats_hybrid_vs_matched_per_topology.csv`. Paired t-test; Wilcoxon signed-rank exact over all
  2^n sign flips (scipy permutation method: zeros dropped, ties mid-ranked, never the normal
  approximation, so p ≥ 0.0625 with 5 seeds); Cohen's d_z = mean difference / std of differences.
  p-values unadjusted for multiple comparisons, as specified. T8 (appendix) reports the 36 cells.
- **DECIDED — noise population (T5, F3, Spearman).** "Configurations at train_frac = 1.0" = all 27
  hybrid configurations per dataset at tf = 1.0 (18 main + 9 sweep; the sweep's noise results
  would otherwise appear nowhere). Spearman ρ per dataset (F3 has a panel per dataset), plus the
  main matrix alone (n = 4 fixed) as a check that the correlation is not just the qubit count.
- **DECIDED — F4.** Main matrix at tf = 1.0; per placement and seed, the mean over its six
  encoding × topology configurations; error bars = std of those means over seeds. p = 0 is the
  noiseless accuracy on the same 2,000 images; symlog p-axis (linear below 0.001).
- **DECIDED — best hybrid per placement (T3, F1).** The sweep selection (6a: mean best-validation
  accuracy at tf = 1.0), tracked across train_frac; its cnn_matched is the matched model of the
  same placement and encoding. T3's D is that of the tf = 1.0 configuration; the maximum
  total-parameter gap between matched and hybrid models is computed into the caption (4.6).
- **DECIDED — D in tables.** D is the per-configuration measurement 7 defines (seed-0 checkpoint,
  median of 200 samples); 13's "no single-run numbers" is read as applying to training outcomes.
  D, G and Q-Par are configuration-level and carry no std.
- **DECIDED — formatting.** Accuracies in percent (2 decimals) with ± std; σ_a and Δ_cal in
  percentage points; Q-Par 3 significant digits; D in ms (3 decimals); p < 0.001 printed as
  "< 0.001". T1/T2 list "A (mean ± std)" and σ_a separately as the protocol asks, although the two
  numbers coincide. Tables are `table*` fragments with booktabs for `\input{}`; all nine compile in
  an IEEEtran document without overfull boxes (TeX Live 2026).
- **DECIDED — figures.** Okabe–Ito blue/vermillion/bluish green for Pre/Middle/Post, validated
  with the dataviz palette validator on a white surface, all pairs: worst CVD ΔE 11.0 (deutan),
  normal-vision ΔE 18.7, contrast ≥ 3:1 for all three. Marker shape per placement as the second
  encoding (never color alone); single-hue sequential blue for the F6 heatmap. Arial (DejaVu Sans
  where Arial is missing) ≥ 8 pt; TrueType fonts embedded (pdf.fonttype 42: IEEE PDF eXpress
  rejects Type 3); PDF + 300 dpi PNG at 3.5 in / 7.16 in width. Error bars ±1 std over seeds
  wherever there is seed variation (A, Δ_cal, noisy accuracies).
- **FACT (05:50) — the calibrated noise model is applied; its effect on accuracy is small.** On the
  synced runs so far, Δ_cal averages ≈ 0.05 pp (range −0.4 to +0.6 pp; 47 of 116 configurations
  slightly negative) and ρ(G, Δ_cal) ≈ 0, while the depolarizing sweep collapses accuracy at
  p ≥ 0.005 for n ≤ 4 (e.g. MNIST amplitude 97.4 % → 50 % at p = 0.005). Check on the Mac with the
  production code path (`AerCore`, FakeAachen, seed-0 checkpoint of
  mnist_middle_amplitude_fullyconnected_n4_tf1.0, 40 images, 200,000 shots): ideal Aer reproduces
  the exact ⟨Z⟩ to 0.006, and the noisy ⟨Z⟩ ≈ 0.82 × exact (least squares; mean |Δ| 0.087) on a
  circuit transpiled to 115 two-qubit gates, depth 394. The noise model therefore acts, mostly as
  a near-uniform contraction of the quantum block's outputs, which the trained heads' argmax
  largely tolerates; at 1,024 shots, shot noise then dominates Δ_cal. Pre-placement models add the
  quantum block as a residual to the image, so even strong depolarizing noise barely changes them
  (MNIST Pre n = 6/8: 98.9 % at p = 0.02). Findings, not deviations; for the paper's discussion.
- **DECIDED by Jeremy (2026-09-28) — γ supplement.** "Yes to the supplement, but restrict it to
  train_frac = 1.0: the 18 main-grid configs plus the 9 sweep points (27 total), so it is
  like-for-like. In the main-grid sensitivity, state explicitly that γ is inert by construction and
  point to the supplement." Implemented: `sensitivity(summary, "tf1")` over the 27 hybrid
  configurations at train_frac = 1.0 per dataset (Q-Par with the unchanged 8.1 normalization),
  `results/sensitivity_supplement_tf1.csv` (+ `_summary.csv`), table T10
  (`tab:sensitivity_supplement`); the T6 caption states that γ is inert on the main matrix and
  refers to T10; F6 marks the γ settings and resource-first with † and a footnote.
- **FACT (2026-10-25) — the n = 6 and n = 8 Random graphs are not connected.** The kept seed-2026
  draws (`configs/topologies.yaml`; the first draw at both sizes, §3b, 4b) are n = 6
  {(0,1),(0,3),(1,2),(1,5),(2,3)} and n = 8 {(0,1),(0,3),(0,5),(1,4),(1,7),(2,5),(3,4)}. The rule
  (4.5) draws n − 1 pairs with no connectivity condition; each draw contains a 4-cycle, so qubit 4
  (n = 6) and qubit 6 (n = 8) receive no CNOT in the variational layers. This concerns the sweep
  configurations with the Random topology, MNIST Middle/amplitude and Post/amplitude at n = 6 and 8
  (20 runs). Their amplitude encoding still applies CNOTs to that qubit (64 at n = 6, 256 at n = 8,
  Möttönen decomposition), and the readout measures ⟨Z⟩ on every qubit. G and twoq_transpiled are
  counted from the circuits as run, so no result changes. The n = 4 star and the n = 2 edge are
  connected. Found while exporting the appendix edge lists (`appendix/random_topology_edges.txt`);
  a finding, not a deviation.
- **Fixed (2026-10-31) — exact zeros printed as "−0.00".** Two Δ_cal means are exactly zero (MNIST
  Post/Angle/Random and Post/Amplitude/FC, n = 4, tf = 1.0) but come out of floating point as
  −2.2e-17, and T5 printed them as "−0.00 ± 0.04", so counting negatives gave 13 instead of 11.
  `fixed()` and `pct()` now print a value that rounds to zero without a sign (regression test added).
  The analysis was regenerated on a copy of the inputs: every CSV, every other table and every PNG
  are byte-identical (PDFs differ only in their creation date); only those two T5 cells changed.
- **Corrected (2026-10-31) — three stale statements in §4 and §7** (found in the code audit; nothing
  re-run; the results in `results/` were produced as described here). (1) §4 said Aer's
  `seed_simulator = 0 + offset` of each 250-image chunk, "so no two images share a sampling seed".
  The offset is `start` ∈ {0, 250}, counted within each 500-image batch that `predict` passes to the
  model, so it restarts every 500 images: in each 2,000-image evaluation, images i, i+500, i+1000
  and i+1500 share a sampling seed (within one Aer job each image gets its own seed). Expected values
  are unaffected: each image is still sampled from its own circuit; images that share a seed only
  reuse the same random-number stream. (2) §4 said `max_parallel_threads = 2` (one worker = 2
  threads); it is 1, because the noisy evaluation passes `training.threads` = 1 from `base.yaml` (one
  single-threaded worker per core, decision 3c). (3) §7 said the depolarizing sweep ran on
  `default.mixed`, analytic, broadcast over chunks of 100 images; it ran on Qiskit Aer
  `method="density_matrix"` with exact expectation values (decision 3c), 250 images per Aer job, and
  `default.mixed` is only the reference in `tests/test_evaluate.py`.
