#!/usr/bin/env bash
# One command for the whole study (protocol Section 14):
# environment -> tests -> smoke -> queue (cloud or local) -> sync -> latency (Mac, prompted) -> analysis.
#
#   scripts/reproduce.sh cloud           # Lambda fleet; needs LAMBDA_API_KEY, LAMBDA_SSH_KEY_NAME and
#                                        # LAMBDA_SSH_PRIVATE_KEY_PATH in the environment
#   scripts/reproduce.sh local [WORKERS] # this machine, WORKERS single-threaded runs in parallel
#
# Every step is resumable: finished runs and noise parts are skipped, so the script can be re-run
# after an interruption. A subset of the study runs locally with queue.py jobs (see README.md).
set -euo pipefail
cd "$(dirname "$0")/.."

mode=${1:-}
workers=${2:-$(sysctl -n hw.physicalcpu 2>/dev/null || nproc)}
export OMP_NUM_THREADS=1  # base.yaml: one thread per worker

uv sync --frozen                         # the pinned environment (pyproject.toml + uv.lock)
uv run python scripts/queue.py configs   # configs/matrix, configs/baselines, configs/topologies.yaml
uv run pytest -q                         # unit tests (11.1)

case "$mode" in
  cloud)
    uv run python scripts/queue.py step1       # one instance: unit tests + smoke re-run there (11.2, 2a)
    uv run python scripts/queue.py plan        # fleet plan from the smoke rates measured there (10)
    uv run python scripts/queue.py coordinate  # training, sweep selection, noisy evaluation, continuous
                                               # sync, terminate each instance after a verified sync
    ;;
  local)
    uv run python scripts/smoke.py archs --workers "$workers"  # smoke (11.2, 2a)
    uv run python scripts/smoke.py noise
    stage() {  # one stage of the queue: its incomplete jobs, $workers at a time
      uv run python scripts/queue.py jobs --stage "$1" |
        xargs -P "$workers" -L 1 sh -c 'uv run python scripts/run.py "$0" --seed "$1" --task "$2"'
    }
    stage main                               # M1, B1, M2, B2 (9)
    uv run python scripts/queue.py select    # qubit-sweep configurations (9, decision 6a)
    stage sweep                              # S
    stage noise                              # N: 6.2 and 6.3 for every hybrid run
    ;;
  *)
    echo "usage: scripts/reproduce.sh cloud|local [WORKERS]" >&2
    exit 2
    ;;
esac

cat <<'EOF'

Latency D (protocol 7) is measured on the paper's Mac: seed-0 checkpoints, batch size 1, in one
uninterrupted session with nothing else running (close other applications, keep the Mac on power).
EOF
read -r -p "Press Enter to measure latency now (Ctrl-C to stop here and resume later) " _
uv run python -m qpar.latency            # results/latency.csv, results/latency_env.json
uv run python -m qpar.analysis           # CSVs, tables/T1-T10.tex, figures/F1-F6.{pdf,png} (8, 12, 13)
uv run python scripts/appendix.py        # results/appendix/ (paper appendix exports)
