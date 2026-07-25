#!/bin/bash
# 2026-07-25: repointed from the retired karolina_bundle to the working
# repo (cold-start audit). HISTORICAL recipe: it uses the PRE-RETRAIN
# nets and the old output filenames. For the Track 1 retrained rows,
# author a fresh script per the repo-root prompt.md (exact filenames
# are pinned by eval/report_data.py).
# Canonical Lever B2 probe runs with the FINAL code (cycle guard in) and
# generous caps:
#  P1  base: all 11 B1-residual structural instances
#      -> analysis/artifacts/ceiling_probe_results_b2.json  (report reads this)
#  P2  base: 40-instance solved sample (superset re-verification)
#  P3  g16r6: the 14 not-yet-recovered instances (2 proven-impossible under
#      B1 + 12 frontier-bound INCONCLUSIVE)
#      -> scaling/results/g16r6/ceiling_probe_b2.json
#SBATCH --job-name=rr-b2probe
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 4
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/b2/%x-%j.out

set -euo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

BUNDLE=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$BUNDLE/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
RUN="$BUNDLE/runs/b2"; mkdir -p "$RUN"

( unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
  python3 analysis/artifacts/ceiling_probe.py \
    analysis/artifacts/b2_probe_instances.json \
    analysis/artifacts/ceiling_probe_results_b2.json \
    --b2 --time-cap 900 --max-iters 400000 --max-frontier 2000000 \
    > "$RUN/p1_base_residual.log" 2>&1 ) &
( unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
  python3 analysis/artifacts/ceiling_probe.py \
    analysis/artifacts/b1_solved40_instances.json \
    "$RUN/probe_b2_solved40_final.json" \
    --b2 --time-cap 300 --max-frontier 2000000 \
    > "$RUN/p2_solved40.log" 2>&1 ) &
( export RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48 RR_ENV_DIR="$SV/environments_g16r6"
  python3 analysis/artifacts/ceiling_probe.py \
    scaling/results/g16r6/b2_probe_instances.json \
    scaling/results/g16r6/ceiling_probe_b2.json \
    --b2 --time-cap 900 --max-iters 400000 --max-frontier 2500000 \
    > "$RUN/p3_g16r6.log" 2>&1 ) &
wait

echo "=== P1 base residual ==="; tail -8 "$RUN/p1_base_residual.log"
echo "=== P2 solved40 ===";      tail -5 "$RUN/p2_solved40.log"
echo "=== P3 g16r6 ===";         tail -8 "$RUN/p3_g16r6.log"
echo "B2PROBE DONE"
