#!/bin/bash
# Generic arena bench of one (policy, value) pair: several exams x searches.
#   sbatch -J spr-bench self_play_robots/jobs/bench_pair.slurm <TAG> <POLICY> <VALUE> <OUTDIR> <EXTRA_FLAGS> [cfg:instances ...]
# e.g. ... "--vocab b2 --anytime" g24r4:scaling/data/g24r4/bench.solved.jsonl g24r4:scaling/data/g24r4/bench.unsolved.jsonl
# Each exam is benched with arena_astar (prefix-check) and mcts (best-at-budget).
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
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
TAG="$1"; BP="$2"; BV="$3"; OUTD="$4"; EXTRA="$5"; shift 5
mkdir -p "$OUTD"
echo "SPR BENCHPAIR $TAG start $(date +%F' '%T) job=$SLURM_JOB_ID extra='$EXTRA'"
for spec in "$@"; do
  cfg="${spec%%:*}"; inst="${spec#*:}"; base=$(basename "$inst" .jsonl | tr '.' '_')
  for s in astar mcts; do
    PF="--prefix-check"; case "$EXTRA" in *b2*|*b1*) PF="";; esac   # prefix_key cannot sort B2 (park/by-ref) plans; the recorded B2 rows ran without it
    fl="--search arena_astar $PF $EXTRA"; [ "$s" = mcts ] && fl="--search mcts $PF --best-at-budget $EXTRA"
    out="$OUTD/${TAG}_${cfg}_${base}_${s}.json"; [ -s "$out" ] && continue
    t0=$SECONDS
    python -m spr.arena bench --arm "bp_${TAG}_${cfg}_${base}_${s}" --config "$cfg" --instances "$inst" \
        --policy "$BP" --value "$BV" --driver spr --flags "$fl" --width 8 --threads 2 --device cuda --out "$out" \
        || echo "SPR BENCHPAIR FAILED $cfg $base $s"
    echo "SPR BENCHPAIR $cfg $base $s took $((SECONDS-t0))s"
  done
done
echo "SPR BENCHPAIR DONE $TAG $(date +%F' '%T)"
