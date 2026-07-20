#!/bin/bash
# ============================================================================
# Karolina TRAINING job — Slurm sbatch script, one A100 (1/8 of a GPU node).
#
# Karolina runs Slurm (it migrated off PBS). Submit / monitor / interactive:
#   sbatch  jobs/train_template.slurm
#   squeue --me
#   salloc -A PROJECT-ID -p qgpu_exp --gpus 1     # 1 h interactive test shell
# Docs: https://docs.it4i.cz/general/job-submission-and-execution/
#       https://docs.it4i.cz/general/resource-allocation-and-job-execution/
#
# Accounting is NODE-HOURS: 1 GPU-hour = 0.125 node-hour (a GPU node has 8
# A100s; one GPU = 1/8 node). It is charged against the allocation whether the
# GPU is busy or idle. Ask the owner for the PROJECT-ID and the node-hour
# budget BEFORE any large run. Smoke-test on qgpu_exp (1 h) first. Only request
# a whole node (--gpus 8) when a run actually parallelizes across 8 GPUs.
# ============================================================================
#SBATCH --job-name=rr-train
#SBATCH -A PROJECT-ID                 # MANDATORY — ask the owner for the real ID
#SBATCH -p qgpu                       # qgpu_exp = 1 h test; qgpu_free = low priority
#SBATCH --nodes=1
#SBATCH --gpus=1                      # 1 A100-40GB + 16 cores + 1/8 node RAM (~128 GB)
#SBATCH --time=24:00:00               # qgpu default 24 h, max 48 h
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- Environment ------------------------------------------------------------
# Compute nodes have NO outbound internet. The venv/conda env and any pip/cargo
# installs must ALREADY exist, built earlier on a LOGIN node. Prefer Karolina
# modules where they match (find versions with `ml av Python CUDA PyTorch`).
ml purge
ml Python                             # adjust to the exact module from `ml av`
# ml CUDA                             # only if not using a torch wheel that bundles it

REPO=/scratch/project/PROJECT-ID/karolina_bundle/MCTS_evolution   # adjust to your unpack path on /scratch
VENV=/scratch/project/PROJECT-ID/venv                             # adjust (built on a login node)
source "$VENV/bin/activate"

cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export OMP_NUM_THREADS=16

# --- Payload ----------------------------------------------------------------
# Replace with the real training launch. Known-good launcher patterns from the
# origin machine are in the bundle at  karolina_bundle/launcher_patterns/ :
#   train_fwd_lowlr*.py  — forward nets for scaling configs (low LR + SPLITS
#                          rebinding; REQUIRED at >=6 robots, prompt pitfall 4)
#   b1_retrain_chain.sh  — B1 policy (cold) + value (warm-start) retrain recipe
# For scaling configs set RR_GRID / RR_ROBOTS / RR_WALLS / RR_ENV_DIR first
# (values printed by: python3 -m scaling.run_config --config <cfg>).
# Batch-size caps that already OOMed a 40 GB card: 24x24 forward <=32-64;
# 32x32 forward 16; 32x32 backward value batch 2 / group 8.

echo "EDIT ME: put the real training command here (see comments above)" >&2
exit 1
