#!/bin/bash
# F-M0 -- FORWARD arena parity (PROBLEM.md section 8 M0, for the primitive-move arm):
# re-run the recorded forward rows through THIS folder's harness
# (spr.fwd.arena -> spr.fwd.bench chunked 8-wide -> eval/merge_compare_shards.py
# -> eval.replay_validate) and check PER-ROW parity (solved, moves, expansions,
# d_star) against eval/results/comparison_forward.json. The search itself is
# `move_planner.evaluate.nn_astar` UNCHANGED, so any difference is wiring.
#
#   sbatch -J spr-fwd-m0 self_play_robots/jobs/fwd_m0_parity.slurm [DEV] [CKPT ...]
# defaults: DEV=cuda (the device the reference was recorded on),
#           CKPT = move_planner/checkpoints/candidate_scored.ckpt (450/450, regret
#                  0.0667, 94.2% optimal, mean 36.0 expansions -- the g16r4 forward
#                  baseline of record) and move_planner_v2/runs_warm/iter5.ckpt
#                  (439/450 -- exercises the UNSOLVED path too).
# Env: OUT_ROOT (default runs/spr/fwd), WIDTH (8), THREADS (2).
#
# Cost basis: recorded mean 0.99 s/instance (candidate_scored) and 3.3 s
# (iter5) on cuda; 450 instances / 8-wide -> ~1 + ~4 min plus model loads.
# One GPU for <= 1 h  ->  <= 0.125 node-hours (expect ~0.03).
# Idempotent/resumable: per-chunk skip-if-exists keyed on the run stamp.
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
DEV="${1:-cuda}"; shift || true
CKPTS=("$@")
[ ${#CKPTS[@]} -gt 0 ] || CKPTS=(move_planner/checkpoints/candidate_scored.ckpt \
                                move_planner_v2/runs_warm/iter5.ckpt)
OUT_ROOT="${OUT_ROOT:-$REPO/runs/spr/fwd}"; WIDTH="${WIDTH:-8}"; THREADS="${THREADS:-2}"
mkdir -p "$REPO/runs/spr" "$OUT_ROOT/m0"
echo "SPR FWD M0 start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} dev=$DEV ckpts=${CKPTS[*]}"
nvidia-smi -L 2>/dev/null || true
rc_all=0
for CK in "${CKPTS[@]}"; do
  TAG=$(echo "$CK" | tr '/' '_' | sed 's/\.ckpt$//')
  OUT="$OUT_ROOT/m0/astar_${TAG}_${DEV}.json"
  t0=$SECONDS
  python -m spr.fwd.arena bench \
      --name "fwd_m0_astar_${TAG}_${DEV}" --ckpt "$CK" \
      --instances eval/data/bench450.jsonl --search astar \
      --expansions 1200 --k 5 --width "$WIDTH" --threads "$THREADS" --device "$DEV" \
      --run-root "$OUT_ROOT/runs" --out "$OUT" \
      --parity-ref eval/results/comparison_forward.json --parity-system "$CK" \
      --parity-out "$OUT_ROOT/m0/astar_${TAG}_${DEV}.parity.json" \
    || { echo "SPR FWD M0 PARITY FAILED $CK"; rc_all=1; }
  echo "SPR FWD M0 arm $CK took $((SECONDS-t0))s"
done
echo "SPR FWD M0 DONE rc=$rc_all $(date +%F' '%T)"
exit $rc_all
