#!/bin/bash
# The missing 2x2 cell (analysis/publishability.md objection 1.3): base-B1 nets
# running OLD-vocabulary search on the g16r6/g16r8 pinned sets.
#
# The 16x16 rungs' published rows are
#   comparison.json     : per-config nets  + old vocabulary
#   comparison_b2.json  : base-B1 nets     + full (B2) vocabulary
# so "old vs full language" and "per-config vs base nets" are confounded. This
# job fills the fourth cell -- base-B1 nets + old vocabulary -- which isolates
# the LANGUAGE effect at fixed nets (against comparison_b2.json) and the NETS
# effect at fixed language (against comparison.json).
#
# Flags are comparison_b2.json's verbatim MINUS --backward-b2, so the only
# difference between the two rows is the plan language. Note that dropping the
# B2 flag also drops park repairs (they require b1+anytime) -- that is correct:
# park repair is part of the extended language, not an independent knob.
#
# --dump-moves is on (it costs no extra BFS since 2026-07-25) so every claimed
# solve here is replay-certifiable; --count-slides is deliberately OFF so the
# recorded seconds stay comparable to the rows this is contrasted with.
#
# Backward-only, existing checkpoints, so it depends on nothing in the
# retraining chain. Four lanes in parallel inside one 1-GPU job; measured B2
# counterparts totalled 2.5 CPU-hours, so ~1.5 h wall, ~0.2 node-hours.
# Idempotent: a lane whose output exists is skipped, writes are tmp+mv.
#
#   sbatch jobs/patterns/basenets_oldvocab.slurm
#
#SBATCH --job-name=rr-2x2cell
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
# WALLTIME 16 h, not 24: the cooling reservation (MOTD, daily 10:00-18:00 CEST)
# blocks any job requesting more than 16 hours from starting while it is in
# effect. Every lane here is chunked/idempotent, so a 16 h cap costs at most
# one chunk and a resubmit.
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/cell2x2/%x-%j.out

set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1

RUN="$REPO/runs/cell2x2"
mkdir -p "$RUN"
BP=checkpoints_backward/policy_b1.ckpt
BV=checkpoints_backward/value_b1.ckpt

run_one() {                     # run_one <cfg> <graded|frontier>
  local cfg="$1" set="$2"
  local inst out
  if [ "$set" = graded ]; then
    inst="scaling/data/$cfg/bench.solved.jsonl"
    out="scaling/results/$cfg/comparison_basenets_oldvocab.json"
  else
    inst="scaling/data/$cfg/bench.unsolved.jsonl"
    out="scaling/results/$cfg/comparison_ungraded_basenets_oldvocab.json"
  fi
  [ -s "$out" ] && { echo "skip $cfg/$set (done)"; return 0; }
  local robots=${cfg#g16r}
  export RR_GRID=16 RR_ROBOTS="$robots" RR_WALLS=48 \
         RR_ENV_DIR="$SV/environments_$cfg"
  echo "start $cfg/$set $(date +%H:%M:%S)"
  if python3 -m eval.compare --instances "$inst" --expansions 1200 --k 5 \
       --backward-policy "$BP" --backward-value "$BV" \
       --backward-anytime --forward-ckpts "" --device cpu --dump-moves \
       --out "$out.tmp.$$" --md /dev/null > "$RUN/$cfg.$set.log" 2>&1; then
    mv "$out.tmp.$$" "$out"
    echo "done  $cfg/$set $(date +%H:%M:%S)"
  else
    echo "FAIL  $cfg/$set (see $RUN/$cfg.$set.log)"; return 1
  fi
}
export -f run_one
export SV RUN BP BV

printf '%s\n' "g16r6 graded" "g16r6 frontier" "g16r8 graded" "g16r8 frontier" \
  | xargs -P 4 -n 1 -I{} bash -c 'run_one $@' _ {}

# independent replay certification of every row this job produced
for cfg in g16r6 g16r8; do
  for f in "scaling/results/$cfg/comparison_basenets_oldvocab.json" \
           "scaling/results/$cfg/comparison_ungraded_basenets_oldvocab.json"; do
    [ -s "$f" ] || continue
    echo "== replay_validate $f =="
    python3 -m eval.replay_validate --compare "$f" \
      --env-dir "$SV/environments_$cfg" 2>&1 | tail -1
  done
done
echo "CELL2X2 DONE"
