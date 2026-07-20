#!/bin/bash
# ============================================================================
# Karolina EVALUATION job — Slurm sbatch script, CPU only (house rule: evals
# run on CPU). Karolina runs Slurm (migrated off PBS).
#   sbatch  jobs/eval_template.slurm
#   salloc -A PROJECT-ID -p qcpu_exp             # <1 h interactive smoke
# Docs: https://docs.it4i.cz/general/job-submission-and-execution/
#
# Accounting is NODE-HOURS and a CPU job takes a WHOLE node (128 cores) charged
# as 1 node-hour per hour whether used or not — so pack many eval lanes into
# one job (each lane uses ~8 threads; a node fits ~16 lanes) instead of one
# lane per job. Use qcpu_exp (1 h) for smokes.
# ============================================================================
#SBATCH --job-name=rr-eval
#SBATCH -A PROJECT-ID                 # MANDATORY — ask the owner for the real ID
#SBATCH -p qcpu                       # qcpu_exp = 1 h test; qcpu_long up to 144 h
#SBATCH --nodes=1                     # whole node: 2x AMD EPYC 7763 = 128 cores
#SBATCH --time=24:00:00               # qcpu default 24 h, max 48 h
#SBATCH --output=%x-%j.out

set -euo pipefail

ml purge
ml Python                             # adjust per `ml av Python`

REPO=/scratch/project/PROJECT-ID/karolina_bundle/MCTS_evolution   # adjust to your /scratch unpack path
VENV=/scratch/project/PROJECT-ID/venv                             # adjust (built on a login node)
source "$VENV/bin/activate"

cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export OMP_NUM_THREADS=8              # per-lane thread cap (house convention)
export CUDA_VISIBLE_DEVICES=""

# --- Payload ----------------------------------------------------------------
# 128 cores, ~8 per lane: run lanes in parallel with & and `wait`, exactly like
# karolina_bundle/launcher_patterns/final_lanes.sh (the two scaling lanes still
# unfinished at transfer time — reuse it nearly verbatim, fixing the absolute
# paths at its top and dropping its scratch-cache scan). Those exact rerun
# commands are also spelled out in KAROLINA_PROMPT.md work-queue item 1.
#
# Single-lane example (base-450, extended backward + forward, full budgets):
# python3 -m eval.compare \
#     --instances eval/data/bench450.jsonl --expansions 1200 --k 5 \
#     --backward-policy checkpoints_backward/policy_b1.ckpt \
#     --backward-value  checkpoints_backward/value_b1.ckpt \
#     --backward-anytime --backward-b1 \
#     --device cpu --out eval/results/my_run.json --md /dev/null
#
# Long CPU evals buffer stdout (prompt pitfall 8): check CPU% via sstat/sacct
# before assuming a lane hung.

echo "EDIT ME: put the real eval lanes here (see comments above)" >&2
exit 1
