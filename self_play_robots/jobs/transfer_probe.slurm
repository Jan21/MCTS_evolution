#!/bin/bash
# Where is the headroom? Zero-shot benches of the M1 mixed size-free pair (never
# trained on 32x32 or 8 robots) with the arena A* and MCTS on the untouched
# exams, plus the base-language moves ceiling there:
#   g24r4 frontier (bench.unsolved, 218), g32r4 graded (175) + frontier (275),
#   g24r8 graded + frontier; ceiling probes g32r4/g24r8 graded.
# Recorded supervised rows: scaling/results/{g32r4,g24r8}/comparison*.json.
#   sbatch -J spr-transfer self_play_robots/jobs/transfer_probe.slurm <POLICY.ckpt> <VALUE.ckpt>
# ~1-1.5 h on 1 GPU (32x32 benches slower) = 0.15-0.2 nh.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
BP="$1"; BV="$2"
OUTD="${3:-results/transfer}"; mkdir -p "$OUTD" results/ceiling
echo "SPR TRANSFER start $(date +%F' '%T) job=$SLURM_JOB_ID policy=$BP value=$BV"
bench() {  # bench <cfg> <instances> <tag> <flags>
  local out="$OUTD/${1}_${3}.json"; [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  local t0=$SECONDS
  python -m spr.arena bench --arm "tr_${1}_${3}" --config "$1" --instances "$2" --policy "$BP" --value "$BV" \
      --driver spr --flags "$4" --width 8 --threads 2 --device cuda --out "$out" || echo "SPR TRANSFER FAILED $1 $3"
  echo "SPR TRANSFER bench $1 $3 took $((SECONDS-t0))s"
}
A="--search arena_astar --prefix-check"; M="--search mcts --prefix-check --best-at-budget"
bench g24r4 scaling/data/g24r4/bench.unsolved.jsonl frontier_astar "$A"
bench g24r4 scaling/data/g24r4/bench.unsolved.jsonl frontier_mcts  "$M"
bench g32r4 scaling/data/g32r4/bench.solved.jsonl   graded_astar   "$A"
bench g32r4 scaling/data/g32r4/bench.solved.jsonl   graded_mcts    "$M"
bench g24r8 scaling/data/g24r8/bench.solved.jsonl   graded_astar   "$A"
bench g24r8 scaling/data/g24r8/bench.solved.jsonl   graded_mcts    "$M"
bench g32r4 scaling/data/g32r4/bench.unsolved.jsonl frontier_astar "$A"
bench g24r8 scaling/data/g24r8/bench.unsolved.jsonl frontier_astar "$A"
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
[ -s results/ceiling/g32r4_base.json ] || python -m spr.ceiling --config g32r4 --instances scaling/data/g32r4/bench.solved.jsonl --vocab base --workers 16 --time-cap 120 --out results/ceiling/g32r4_base.json
[ -s results/ceiling/g24r8_base.json ] || python -m spr.ceiling --config g24r8 --instances scaling/data/g24r8/bench.solved.jsonl --vocab base --workers 16 --time-cap 120 --out results/ceiling/g24r8_base.json
echo "SPR TRANSFER DONE $(date +%F' '%T)"
