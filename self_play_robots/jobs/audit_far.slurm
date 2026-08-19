#!/bin/bash
# M5 far-size zero-shot exact audit (PROBLEM.md 8/M5: "net trained via self-play
# at <=32 benched zero-shot at 40-64 against exact ground truth"): spr.audit of a
# (policy, value) pair on the exact-labeled decision corpora at 32 (test split
# of scaling/data/g32r4/backward.jsonl) and 40/48/56/64 (backward_audit.rust.jsonl,
# the labeler's fidelity-curve sets, FINDINGS 53/70) -> results/audit/<TAG>.json.
#   sbatch -J spr-audit-<TAG> self_play_robots/jobs/audit_far.slurm <TAG> <POLICY> <VALUE> [--byref]
# Cost: ~30-60 min on 1 GPU (~0.1 nh).
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
TAG="$1"; P="$2"; V="$3"; EXTRA="${4:-}"
OUT="$REPO/self_play_robots/results/audit/${TAG}.json"; mkdir -p "$(dirname "$OUT")"
echo "SPR AUDITFAR $TAG start $(date +%F' '%T) job=$SLURM_JOB_ID policy=$P value=$V extra=$EXTRA"
python -m spr.audit --policy "$P" --value "$V" --split test $EXTRA --batch 4 --device cuda \
  --data g32r4=scaling/data/g32r4/backward.jsonl \
  --data g40r4=scaling/data/g40r4/backward_audit.rust.jsonl \
  --data g48r4=scaling/data/g48r4/backward_audit.rust.jsonl \
  --data g56r4=scaling/data/g56r4/backward_audit.rust.jsonl \
  --data g64r4=scaling/data/g64r4/backward_audit.rust.jsonl \
  --out "$OUT" || { echo "SPR AUDITFAR FAILED $TAG"; exit 1; }
echo "SPR AUDITFAR DONE $TAG $(date +%F' '%T)"
