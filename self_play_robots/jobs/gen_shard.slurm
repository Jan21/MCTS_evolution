#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:58:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/gen-%x-%j.out
# One shard of the self-play generation phase of jobs/selfplay_iter.slurm
# (same flags), for a board-id range small enough to finish inside the fast
# partition. Env: CFG K VOCAB TAGV POLICY VALUE SHARD IDS WORKERS.
# Outputs $IT/records_$SHARD.jsonl; merge_gen.slurm joins the shards.
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=${CFG:-g64r4}; K=${K:-1}; VOCAB=${VOCAB:-b2}; TAGV=${TAGV:-_b2}
IT="$REPO/runs/spr/selfplay/${CFG}${TAGV}_iter${K}"; mkdir -p "$IT"
P=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/selfplay/mix_b2mix_iter3
POL=${POLICY:-$P/policy/lightning_logs/version_0/checkpoints/epoch=2-step=2361.ckpt}
VAL=${VALUE:-$P/value/lightning_logs/version_0/checkpoints/epoch=0-step=1573.ckpt}
echo "GEN SHARD $SHARD cfg=$CFG tag=$TAGV boards $IDS policy=$POL value=$VAL $(date +%T)"
python -m spr.selfplay --config "$CFG" --policy "$POL" --value "$VAL" \
    --boards-dir "$REPO/runs/spr/boards/${CFG}${TAGV}_iter${K}" --board-ids "$IDS" --vocab "$VOCAB" \
    --per-board 8 --seed "$K" --iter "$K" --expansions 600 --stop-after 150 \
    --root-noise 0.25 --complete-siblings --root-all --workers "${WORKERS:-4}" --threads 2 --device cuda \
    --timeout 600 --min-expansions 0 --out "$IT/records_$SHARD.jsonl" || { echo "GEN SHARD $SHARD FAILED"; exit 1; }
echo "GEN SHARD $SHARD DONE $(date +%T)"
