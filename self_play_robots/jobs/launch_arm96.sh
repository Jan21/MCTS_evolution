#!/bin/bash
# launch_arm96.sh <TAGV> <POLICY> <VALUE> <BENCHTAG>: a full B2 self-play round at 96x96
# from the given pair, as shards + merge + chained links + ladder-protocol benches.
set -u
TAGV=$1; POL=$2; VAL=$3; BT=$4
cd /scratch/project/open-37-42/janhula/mcts_stage5_probe/self_play_robots
ids=""
for s in $(seq 0 29); do a=$((8000+2*s)); b=$((a+1))
  j=$(sbatch --parsable --job-name g96${TAGV}-$s --export=ALL,CFG=g96r4,K=1,VOCAB=b2,TAGV=$TAGV,POLICY=$POL,VALUE=$VAL,SHARD=$s,IDS=$a-$b jobs/gen_shard.slurm); ids="$ids:$j"; done
ids=${ids#:}
M=$(sbatch --parsable --job-name merge96$TAGV --dependency=afterany:$ids --export=ALL,CFG=g96r4,K=1,TAGV=$TAGV,NSHARDS=30 jobs/merge_gen.slurm)
prev=$M; dep=afterok; links=""
for i in 1 2 3; do
  prev=$(sbatch --parsable -J sp96$TAGV --partition=qgpu_exp --time=00:58:00 --dependency=$dep:$prev \
    --export=ALL,VOCAB=b2,TAGV_OVERRIDE=$TAGV,WORKERS=4,BOARDS=60,VBS=1,VMPG=2,PBS=2 jobs/selfplay_iter.slurm g96r4 1 "$POL" "$VAL")
  dep=afterany; links="$links $prev"; done
LAST=$prev; bench=""
b=$(sbatch --parsable --dependency=afterany:$LAST --job-name np64$BT --export=ALL,ROUND=g96r4${TAGV}_iter1,CFG=g64r4,LIM=40,OFF=0,TAG=$BT jobs/bench_newpair.slurm); bench="$bench $b"
for OFF in 0 2 4 6 8 10 12 14 16 18 20 22; do
  b=$(sbatch --parsable --dependency=afterany:$LAST --job-name np96$BT-$OFF --export=ALL,ROUND=g96r4${TAGV}_iter1,CFG=g96r4,LIM=2,OFF=$OFF,TAG=$BT jobs/bench_newpair.slurm); bench="$bench $b"; done
echo "ARM $TAGV: shards ${ids//:/ }"
echo "ARM $TAGV: merge $M links$links benches$bench"
