#!/bin/bash
# Smoke + timing probe for the v07 hybrid on the SUPERVISED (per-size) nets.
# Loads the g24r4 exact-taught PolicyTF/LoopedValueNet pair through
# spr.nets --arch persize and runs a handful of graded-exam instances through
# each of the three arms, so the real job's walltime can be sized honestly.
#   sbatch -J spr-supsmoke self_play_robots/jobs/v07_supnets_smoke.slurm
# Cost: qgpu_exp, <=30 min on 1 GPU = <=0.06 node-hours.
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
export REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
SM="$REPO/runs/spr/supnets_smoke"; mkdir -p "$SM"
P="$REPO/supervised_valuenet/scaling/runs/g24r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1654.ckpt"
V="$REPO/supervised_valuenet/scaling/runs/g24r4/backward-value/lightning_logs/version_0/checkpoints/epoch=22-step=38019.ckpt"
GRADED="$REPO/supervised_valuenet/scaling/data/g24r4/bench.solved.jsonl"
GED=$(python -c "from scaling.configs import get; print(get('g24r4').env_dir_abs)")
N=6
echo "SPR SUPSMOKE start $(date +%F' '%T) job=$SLURM_JOB_ID N=$N"
nvidia-smi -L
head -$N "$GRADED" > "$SM/smoke.jsonl"
for spec in "d1 1 600 6 100" "d2 2 500 8 80"; do
  set -- $spec; tag=$1; d=$2; b0=$3; tm=$4; sub=$5
  t0=$SECONDS
  RR_ENV_DIR="$GED" RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 \
  python -m variants.v07_hybrid_actions --instances "$SM/smoke.jsonl" --policy "$P" --value "$V" \
      --arch persize --boards pkl --device cuda --prefix-depth "$d" --b0 "$b0" --top-m "$tm" --sub "$sub" \
      --out "$SM/hyb_$tag.json" || { echo "SPR SUPSMOKE FAILED hyb $tag"; exit 1; }
  echo "SPR SUPSMOKE hyb $tag: $((SECONDS-t0))s for $N -> $(( (SECONDS-t0)*232/N/60 )) min projected (1 proc)"
  RR_ENV_DIR="$GED" RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 \
  python -m eval.replay_validate --compare "$SM/hyb_$tag.json" --env-dir "$GED" 2>&1 | tail -3
done
cd "$REPO/self_play_robots"
t0=$SECONDS
python -m spr.arena bench --arm "supsmoke_ctl" --config g24r4 --instances scaling/data/g24r4/bench.solved.jsonl \
    --policy "$P" --value "$V" --driver spr \
    --flags "--search mcts --best-at-budget --vocab b2 --anytime --arch persize" \
    --width 6 --threads 2 --device cuda --limit $N --out "$SM/ctl.json" \
    || { echo "SPR SUPSMOKE FAILED ctl"; exit 1; }
echo "SPR SUPSMOKE ctl: $((SECONDS-t0))s for $N (6-wide)"
python - <<'PY'
import json, os
sm = os.environ["REPO"] + "/runs/spr/supnets_smoke"
for f in ("hyb_d1", "hyb_d2", "ctl"):
    d = json.load(open(f"{sm}/{f}.json"))
    for n, s in d["systems"].items():
        a = s["aggregate"]
        print(f"  {f}: {n[:70]} | solved {a['solved']}/{a['n']} mean_moves {a['mean_moves']:.2f} "
              f"mean_regret {a['mean_regret']:.2f} mean_exp {a['mean_expansions']:.0f} "
              f"mean_s {a['mean_seconds']:.1f}")
PY
echo "SPR SUPSMOKE DONE $(date +%F' '%T)"
