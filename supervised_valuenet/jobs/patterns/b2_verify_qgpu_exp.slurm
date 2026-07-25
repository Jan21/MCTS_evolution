#!/bin/bash
# 2026-07-25: repointed from the retired karolina_bundle to the working
# repo (cold-start audit). HISTORICAL recipe: it uses the PRE-RETRAIN
# nets and the old output filenames. For the Track 1 retrained rows,
# author a fresh script per the repo-root prompt.md (exact filenames
# are pinned by eval/report_data.py).
# Lever B2 verification bundle (backward planner only, CPU):
#  V1  probe --b1 on the 49-instance base failure set -> must match
#      analysis/artifacts/ceiling_probe_results_b1.json per-row (B1 regression)
#  V2  realizer A/B: 249 stored plans, realize.py@HEAD vs realize.py+B2
#      -> 0 mismatches expected (B2 realizer changes are byref-gated)
#  V3  probe --b2 on the 11 residual structural instances (THE measurement)
#  V4  probe --b2 on the 40-instance solved sample -> all stay REALIZABLE
#SBATCH --job-name=rr-b2verify
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 2
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/b2/%x-%j.out

set -euo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

BUNDLE=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$BUNDLE/supervised_valuenet"
export PYTHONPATH=. OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
RUN="$BUNDLE/runs/b2"; mkdir -p "$RUN"

# V2 needs the pre-change realizer: take it from git HEAD
git show HEAD:supervised_valuenet/eval/realize.py > "$RUN/realize_pre_b2.py"

python3 analysis/artifacts/ceiling_probe.py \
    analysis/artifacts/ceiling_probe_instances.json "$RUN/probe_b1_ab.json" --b1 \
    > "$RUN/v1_b1_ab.log" 2>&1 &
python3 analysis/artifacts/b1_realizer_ab.py "$RUN/realize_pre_b2.py" \
    "$RUN/realizer_b2_ab.json" > "$RUN/v2_realizer_ab.log" 2>&1 &
python3 analysis/artifacts/ceiling_probe.py \
    analysis/artifacts/b2_probe_instances.json "$RUN/probe_b2_residual.json" \
    --b2 --time-cap 240 > "$RUN/v3_b2_residual.log" 2>&1 &
python3 analysis/artifacts/ceiling_probe.py \
    analysis/artifacts/b1_solved40_instances.json "$RUN/probe_b2_solved40.json" \
    --b2 --time-cap 120 > "$RUN/v4_b2_solved40.log" 2>&1 &
wait

echo "=== V1 (b1 regression) ==="; tail -5 "$RUN/v1_b1_ab.log"
echo "=== V2 (realizer A/B) ===";  tail -5 "$RUN/v2_realizer_ab.log"
echo "=== V3 (b2 residual) ===";   tail -6 "$RUN/v3_b2_residual.log"
echo "=== V4 (b2 solved40) ===";   tail -5 "$RUN/v4_b2_solved40.log"
echo "B2VERIFY DONE"
