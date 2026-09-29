#!/bin/bash
# launch_arm96_fill.sh <TAGV> <POLICY> <VALUE> <BENCHTAG>: re-run the generation
# shards that did not finish, as ONE board per job (a/b halves of the original
# 2-board shard), then merge + chained round + ladder benches, dependency-ordered.
set -u
TAGV=$1; POL=$2; VAL=$3; BT=$4
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
cd $REPO/self_play_robots
IT=$REPO/runs/spr/selfplay/g96r4${TAGV}_iter1
ids=""; n=0
for s in $(seq 0 29); do
  [ -s "$IT/records_$s.jsonl" ] && continue
  a=$((8000+2*s)); b=$((a+1))
  for half in a b; do
    [ $half = a ] && id=$a || id=$b
    j=$(sbatch --parsable --job-name g96${TAGV}-${s}${half} --export=ALL,CFG=g96r4,K=1,VOCAB=b2,TAGV=$TAGV,POLICY=$POL,VALUE=$VAL,SHARD=${s}${half},IDS=$id-$id jobs/gen_shard.slurm)
    ids="$ids:$j"; n=$((n+1))
  done
done
ids=${ids#:}
M=$(sbatch --parsable --job-name merge96$TAGV --dependency=afterany:$ids --export=ALL,CFG=g96r4,K=1,TAGV=$TAGV,NBOARDS=60 jobs/merge_gen.slurm)
prev=$M; dep=afterok; links=""
for i in 1 2 3; do
  prev=$(sbatch --parsable -J sp96$TAGV --partition=qgpu_exp --time=00:58:00 --dependency=$dep:$prev \
    --export=ALL,VOCAB=b2,TAGV_OVERRIDE=$TAGV,WORKERS=4,BOARDS=60,VBS=1,VMPG=2,PBS=2 jobs/selfplay_iter.slurm g96r4 1 "$POL" "$VAL")
  dep=afterany; links="$links $prev"; done
LAST=$prev
sbatch --parsable --dependency=afterany:$LAST --job-name np64$BT --export=ALL,ROUND=g96r4${TAGV}_iter1,CFG=g64r4,LIM=40,OFF=0,TAG=$BT jobs/bench_newpair.slurm >/dev/null
for OFF in 0 2 4 6 8 10 12 14 16 18 20 22; do
  sbatch --parsable --dependency=afterany:$LAST --job-name np96$BT-$OFF --export=ALL,ROUND=g96r4${TAGV}_iter1,CFG=g96r4,LIM=2,OFF=$OFF,TAG=$BT jobs/bench_newpair.slurm >/dev/null; done
echo "ARM $TAGV: $n one-board shards resubmitted, merge $M, links$links, 13 benches queued"
