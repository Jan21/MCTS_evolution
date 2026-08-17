#!/bin/bash
# M1 -- bench every trained size-free pair on BOTH pinned exams with the arena
# protocol (1200 expansions, k=5, prefix-check, strict realization,
# replay-certified; spr.arena -> spr.bench -> eval.compare._nn_astar_backward)
# and compute the M1 gate vs the per-size supervised base-vocabulary pairs.
# Pairs = every runs/spr/m1/<ds>_policy_s<S> x runs/spr/m1/<ds>_value_<warm|cold>_s<S>
# with the SAME ds (mixed / g16 / g24); each benched at g16r4 AND g24r4 (the
# single-size pairs' other size = zero-shot cross-size transfer).
#
#   sbatch -J spr-m1-bench [--dependency=afterany:<train jobs>] self_play_robots/jobs/m1_bench.slurm [seed] [device]
# Outputs results/m1/<ds>_<value arm>_s<S>_<cfg>.json + .gate.json.
# Cost basis: base450 3-5 min; g24r4 232 inst ~10-30 min 8-wide; up to 8 combos
# -> 2-4 h (walltime 8 h; chunks resumable, skip-if-exists per combo).
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
SEED="${1:-21}"; DEV="${2:-cuda}"
best_ckpt() { ls -t "$1"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1; }
echo "SPR M1 BENCH start $(date +%F' '%T) job=$SLURM_JOB_ID seed=$SEED dev=$DEV"
mkdir -p results/m1
rc_all=0
for ds in mixed g16 g24; do
  PDIR="$REPO/runs/spr/m1/${ds}_policy_s${SEED}"
  [ -f "$PDIR/DONE" ] || { echo "SPR M1 BENCH: no policy for $ds yet"; continue; }
  BP=$(best_ckpt "$PDIR")
  for varm in value_warm value_cold; do
    VDIR="$REPO/runs/spr/m1/${ds}_${varm}_s${SEED}"
    [ -f "$VDIR/DONE" ] || continue
    BV=$(best_ckpt "$VDIR")
    TAG="${ds}_${varm}_s${SEED}"
    echo "SPR M1 BENCH pair $TAG policy=$BP value=$BV"
    printf 'policy=%s\nvalue=%s\n' "$BP" "$BV" > "results/m1/${TAG}.nets.txt"
    for cfg in g16r4 g24r4; do
      out="results/m1/${TAG}_${cfg}.json"
      if [ -s "$out" ]; then echo "SPR M1 BENCH SKIP $TAG $cfg"; else
        INST=eval/data/bench450.jsonl; [ "$cfg" = g24r4 ] && INST=scaling/data/g24r4/bench.solved.jsonl
        t0=$SECONDS
        python -m spr.arena bench --arm "m1_${TAG}_${cfg}" --config "$cfg" --instances "$INST" \
            --policy "$BP" --value "$BV" --driver spr --flags "--search arena_astar --prefix-check" \
            --width 8 --threads 2 --device "$DEV" --out "$out" || { echo "SPR M1 BENCH FAILED $TAG $cfg"; rc_all=1; continue; }
        echo "SPR M1 BENCH $TAG $cfg took $((SECONDS-t0))s"
      fi
      REF=g16r4_v2_prefix; [ "$cfg" = g24r4 ] && REF=g24r4_exact_prefix
      python -m spr.gate m1 --new "$out" --ref-arm "$REF" --out "results/m1/${TAG}_${cfg}.gate.json" | tail -1
    done
  done
done
echo "SPR M1 BENCH DONE rc=$rc_all $(date +%F' '%T)"
exit $rc_all
