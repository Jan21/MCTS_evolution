#!/bin/bash
# Stage 2 diagnostic: the pre-registered config (pe=none, recurrence=12) sat on
# the constant-value plateau for 28 epochs in job 4863358 -- val_spearman ~ 0,
# val_top1 at the random rate, the reachability head frozen at the 82.4% base
# rate. That is the collapse mode FINDINGS 53/61 already record for this net
# family, not evidence about what a goal-conditioned cost net can learn, so the
# question "is the configuration broken or is the task unlearnable" has to be
# answered before any gate verdict.
#
# Four short arms, 4 epochs each (~4 min each at the measured 59 s/epoch), on
# the SAME training corpus and the same held-out val corpus. Only two knobs
# move, both structural:
#   pe          none | sin2d   -- with pe=none and 2 input channels every
#                                 non-robot cell starts from an identical
#                                 embedding, so the readout may have no way to
#                                 tell two goal cells apart
#   recurrence  12 | 4         -- 12 weight-tied steps over a near-uniform field
#                                 is the classic over-smoothing setting
#
#   sbatch -J spr-sg-s2diag self_play_robots/jobs/subgoal_stage2_diag.slurm
# Cost: 1 GPU on qgpu_exp for ~20 min = 0.042 node-hours.
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" \
       PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
D="$REPO/self_play_robots/results/subgoal/stage2"
B="$REPO/runs/spr/subgoal/stage2_diag"
mkdir -p "$B"
echo "SPR SG-S2DIAG start $(date +%F' '%T) job=${SLURM_JOB_ID:-none}"
nvidia-smi -L

arm () {   # <tag> <pe> <recurrence>
  if [ -s "$B/$1/best.txt" ]; then echo "SPR SG-S2DIAG skip $1 (exists)"; return; fi
  t0=$SECONDS
  python -m subgoal.stage2 train --train "$D/train_b1000-1699.npz" \
      --val "$D/val_b1800-1849.npz" --run "$B/$1" --epochs 4 --batch 16 \
      --workers 8 --max-minutes 8 --pe "$2" --recurrence "$3" \
      || echo "SPR SG-S2DIAG FAILED $1"
  echo "SPR SG-S2DIAG $1 took $((SECONDS-t0))s"
}
arm pe-none_rec12  none  12
arm pe-sin2d_rec12 sin2d 12
arm pe-none_rec4   none  4
arm pe-sin2d_rec4  sin2d 4

echo "SPR SG-S2DIAG summary"
python - <<'PY'
import csv, glob, json, os
B = os.environ["REPO"] + "/runs/spr/subgoal/stage2_diag"
for d in sorted(glob.glob(B + "/*/lightning_logs/version_*/metrics.csv")):
    arm = d.split("/")[-4]
    rows = [r for r in csv.DictReader(open(d)) if r.get("val_spearman")]
    print(arm, [(r["epoch"], round(float(r["val_spearman"]), 3),
                 round(float(r["val_mae"]), 2),
                 round(float(r.get("val_group_spread") or 0), 3),
                 round(float(r["val_reach_acc"]), 3)) for r in rows], flush=True)
PY
echo "SPR SG-S2DIAG DONE $(date +%F' '%T)"
