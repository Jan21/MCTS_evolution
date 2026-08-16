#!/bin/bash
# Submission plan for the two experiments in scratchpad ideas/impl_seeds_forward_plan.md.
# DRY-RUN by default: prints every sbatch it would issue. Pass --go to submit.
#   jobs/patterns/seeds_forward_submit.sh [--go] [A16] [A32] [B16] [B24]
# Groups (default: A16 A32 B16 -- B24 is the expensive optional tier):
#   A16 = 3 seed replicates of the g16r8 headline pair (B1 base nets)      ~1.5 nh
#   A32 = 3 seed replicates of the g32r4 headline pair (per-config nets)   ~5 nh
#   B16 = forward rescue grid at g16r8 (9 trainings + select + bench)      ~8 nh
#   B24 = forward rescue grid at g24r4 (9 trainings + select + bench)     ~22 nh  (opt-in)
# Every group is a chain: job -> afterany resume link (same command; skips
# finished stages) -> (B only) select+bench afterany the resume link.
set -uo pipefail
cd /scratch/project/open-37-42/petrhyner/MCTS_evolution/supervised_valuenet
GO=0; PLAN_GROUPS=()
for a in "$@"; do case "$a" in --go) GO=1 ;; A16|A32|B16|B24) PLAN_GROUPS+=("$a") ;; *) echo "unknown arg $a"; exit 1 ;; esac; done
[ ${#PLAN_GROUPS[@]} -gt 0 ] || PLAN_GROUPS=(A16 A32 B16)
mkdir -p ../runs/seedhl ../runs/fwdgrid

submit() {   # submit <name> <sbatch args...>  -> echoes job id (or a fake id in dry-run)
  local name="$1"; shift
  if [ "$GO" = 1 ]; then
    local out; out=$(sbatch --parsable -J "$name" "$@") || { echo "sbatch failed: $name" >&2; exit 1; }
    echo "${out%%;*}"
  else
    echo "DRY: sbatch -J $name $*" >&2; echo "DRY-$name"
  fi
}

for g in "${PLAN_GROUPS[@]}"; do
  case "$g" in
    A16|A32)
      cfg=g16r8; [ "$g" = A32 ] && cfg=g32r4
      for s in 21 37 53; do
        j1=$(submit "rr-seedhl-$cfg-s$s"   jobs/patterns/seed_headline_pair.slurm "$cfg" "$s")
        j2=$(submit "rr-seedhl-$cfg-s$s-r" --dependency=afterany:"$j1" jobs/patterns/seed_headline_pair.slurm "$cfg" "$s")
        echo "$g seed $s: $j1 -> resume $j2"
      done ;;
    B16|B24)
      cfg=g16r8; [ "$g" = B24 ] && cfg=g24r4
      j1=$(submit "rr-fwdgrid-$cfg"   --array=0-8 jobs/patterns/fwd_rescue_grid.slurm "$cfg")
      j2=$(submit "rr-fwdgrid-$cfg-r" --array=0-8 --dependency=afterany:"$j1" jobs/patterns/fwd_rescue_grid.slurm "$cfg")
      j3=$(submit "rr-fwdsel-$cfg"    --dependency=afterany:"$j2" jobs/patterns/fwd_rescue_select_bench.slurm "$cfg")
      j4=$(submit "rr-fwdsel-$cfg-r"  --dependency=afterany:"$j3" jobs/patterns/fwd_rescue_select_bench.slurm "$cfg")
      echo "$g: grid $j1 -> resume $j2 -> select+bench $j3 -> resume $j4"
      [ "$g" = B24 ] && echo "  (B24 note: fwd_rescue_select_bench frontier lane at g24r4 is ~25-30 h; expect the -r link to be needed, or set FWD_BENCH_SETS=graded)" ;;
  esac
done
[ "$GO" = 1 ] || echo "(dry run -- nothing submitted; add --go)"
