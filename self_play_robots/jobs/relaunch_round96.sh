#!/bin/bash
# relaunch_round96.sh <TAGV> <POLICY> <VALUE> <BENCHTAG> [dep ids colon-separated]
# merge (>= 58 boards) + 3 chained links + 13 ladder benches for a 96x96 round.
set -u
TAGV=$1; POL=$2; VAL=$3; BT=$4; DEP=${5:-}
cd /scratch/project/open-37-42/janhula/mcts_stage5_probe/self_play_robots
D=""; [ -n "$DEP" ] && D="--dependency=afterany:$DEP"
M=$(sbatch --parsable --job-name merge96$TAGV $D --export=ALL,CFG=g96r4,K=1,TAGV=$TAGV,NBOARDS=58 jobs/merge_gen.slurm)
prev=$M; dep=afterok; links=""
for i in 1 2 3; do
  prev=$(sbatch --parsable -J sp96$TAGV --partition=qgpu_exp --time=00:58:00 --dependency=$dep:$prev \
    --export=ALL,VOCAB=b2,TAGV_OVERRIDE=$TAGV,WORKERS=4,BOARDS=60,VBS=1,VMPG=2,PBS=2 jobs/selfplay_iter.slurm g96r4 1 "$POL" "$VAL")
  dep=afterany; links="$links $prev"; done
LAST=$prev
sbatch --parsable --dependency=afterany:$LAST --job-name np64$BT --export=ALL,ROUND=g96r4${TAGV}_iter1,CFG=g64r4,LIM=40,OFF=0,TAG=$BT jobs/bench_newpair.slurm >/dev/null
for OFF in 0 2 4 6 8 10 12 14 16 18 20 22; do
  sbatch --parsable --dependency=afterany:$LAST --job-name np96$BT-$OFF --export=ALL,ROUND=g96r4${TAGV}_iter1,CFG=g96r4,LIM=2,OFF=$OFF,TAG=$BT jobs/bench_newpair.slurm >/dev/null; done
echo "ARM $TAGV: merge $M links$links, 13 benches"
