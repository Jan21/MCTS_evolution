#!/bin/bash
# M1 -- bench a size-free (policy, value) pair on BOTH pinned exams with the
# arena protocol (1200 expansions, k=5, prefix-check, strict realization,
# replay-certified) via spr.arena -> spr.bench -> eval.compare._nn_astar_backward,
# then compute the M1 gate vs the per-size supervised base-vocabulary pairs.
#
#   sbatch -J spr-m1-bench self_play_robots/jobs/m1_bench.slurm <TAG> <POLICY_DIR> <VALUE_DIR> [device]
# POLICY_DIR/VALUE_DIR = spr.train --out-dir (best epoch=*.ckpt is picked; the
# best-by-val_regret checkpoint is the one ModelCheckpoint kept). device
# defaults to cuda (8 chunk processes share the GPU); pass cpu for the exact
# arena setting. Outputs results/m1/<TAG>_{g16r4,g24r4}.json + gate JSONs.
# Cost basis: base450 3-5 min; g24r4 232 inst ~10-30 min 8-wide (bad nets burn
# the full budget: walltime 6 h; chunks are resumable).
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
TAG="$1"; PDIR="$2"; VDIR="$3"; DEV="${4:-cuda}"
best_ckpt() {  # newest best epoch ckpt written by ModelCheckpoint (top_k=1)
  ls -t "$1"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1; }
BP=$(best_ckpt "$PDIR"); BV=$(best_ckpt "$VDIR")
[ -n "$BP" ] && [ -n "$BV" ] || { echo "FATAL: missing checkpoints in $PDIR / $VDIR"; exit 1; }
echo "SPR M1 BENCH $TAG start $(date +%F' '%T) job=$SLURM_JOB_ID policy=$BP value=$BV dev=$DEV"
mkdir -p results/m1
rc_all=0
run_arm() {  # run_arm <cfg> <instances> <ref-arm>
  local cfg="$1" inst="$2" ref="$3" out="results/m1/${TAG}_${cfg}.json"
  local t0=$SECONDS
  python -m spr.arena bench --arm "m1_${TAG}_${cfg}" --config "$cfg" --instances "$inst" \
      --policy "$BP" --value "$BV" --driver spr --flags "--search arena_astar --prefix-check" \
      --width 8 --threads 2 --device "$DEV" --out "$out" || { echo "SPR M1 BENCH FAILED $cfg"; rc_all=1; return; }
  echo "SPR M1 BENCH $cfg took $((SECONDS-t0))s"
  python -m spr.gate m1 --new "$out" --ref-arm "$ref" --out "results/m1/${TAG}_${cfg}.gate.json" || true
}
run_arm g16r4 eval/data/bench450.jsonl              g16r4_v2_prefix
run_arm g24r4 scaling/data/g24r4/bench.solved.jsonl g24r4_exact_prefix
echo "SPR M1 BENCH DONE $TAG rc=$rc_all $(date +%F' '%T)"
exit $rc_all
