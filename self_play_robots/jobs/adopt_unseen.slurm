#!/bin/bash
# Wave-4 item 1: the missing measurement of entry 11 -- adopt_mainline's nets
# vs their seed (mix_b2mix_iter3's nets) on the UNSEEN exam (arena A*, B2).
#   sbatch -J spr-var-adoptun self_play_robots/jobs/adopt_unseen.slurm
# Projected: ~30 min = 0.06 nh.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
RES="$REPO/self_play_robots/results/variants/adopt_mainline"
EXAM="$REPO/self_play_robots/results/variants/exam/g24r4_unseen.jsonl"
EXAMB="$REPO/self_play_robots/results/variants/exam/boards_g24r4"
AP=$(grep '^policy=' "$RES/nets.txt" | cut -d= -f2-); AV=$(grep '^value=' "$RES/nets.txt" | cut -d= -f2-)
MP=$(grep '^policy=' "$REPO/self_play_robots/results/selfplay/mix_b2mix_iter3/nets.txt" | cut -d= -f2-)
MV=$(grep '^value=' "$REPO/self_play_robots/results/selfplay/mix_b2mix_iter3/nets.txt" | cut -d= -f2-)
echo "SPR ADOPTUN start $(date +%F' '%T) job=$SLURM_JOB_ID"
run() { local out="$1" p="$2" v="$3" arm="$4"; [ -s "$out" ] && { echo "SKIP $arm"; return; }
  python -m spr.arena bench --arm "$arm" --config g24r4 --instances "$EXAM" \
      --policy "$p" --value "$v" --driver spr \
      --flags "--search arena_astar --vocab b2 --anytime --boards lean" \
      --env-dir "$EXAMB" --width 8 --threads 2 --device cuda --out "$out" \
    || echo "SPR ADOPTUN FAILED $arm"; }
run "$RES/bench_unseen_astar.json" "$AP" "$AV" adoptun_adopt
run "$RES/bench_unseen_mixit3.json" "$MP" "$MV" adoptun_mixit3
python -m spr.gate compare --a "$RES/bench_unseen_astar.json" --b "$RES/bench_unseen_mixit3.json" \
    > "$RES/gate_unseen_vs_mixit3.json" || true
echo "SPR ADOPTUN gate: $(tr -d '\n' < "$RES/gate_unseen_vs_mixit3.json" | cut -c1-240)"
echo "SPR ADOPTUN DONE $(date +%F' '%T)"
