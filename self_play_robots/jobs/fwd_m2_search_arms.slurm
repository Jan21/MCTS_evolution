#!/bin/bash
# F-M2 -- does SEARCH beat GREED at inference, in the primitive-move action space
# (PROBLEM.md section 8 M2, forward arm)? Same net, same pinned exam
# (eval/data/bench450.jsonl), same budget (1200 expansions, k=5,
# replay-certified), different searches:
#   astar          move_planner.evaluate.nn_astar UNCHANGED  <- the reference arm
#   greedy_value   nn_greedy_value UNCHANGED (beam 1 on the value head) <- "greed"
#   greedy_policy  nn_greedy_policy UNCHANGED (beam 1 on the policy head)
#   mcts_min       PUCT c=1.5, min backup, FIRST goal returned
#   mcts_mean      PUCT c=1.5, mean backup, FIRST goal returned
#   mcts_best      PUCT c=1.5, min backup, BEST-AT-BUDGET (shortest path found)
# then paired tables (McNemar on solved + sign test on moves, spr.gate.paired via
# the spr.fwd.gate shim) of every arm vs astar and vs greedy_value.
#
#   sbatch -J spr-fwd-m2 self_play_robots/jobs/fwd_m2_search_arms.slurm [TAG] [CKPT] [DEV]
# defaults: TAG=cand, CKPT=move_planner/checkpoints/candidate_scored.ckpt, DEV=cuda
# Env: OUT_ROOT (runs/spr/fwd), WIDTH (8), THREADS (2), EXP (1200), K (5), ARMS.
#
# Cost basis (recorded cuda rates, 8-wide): astar ~1 min, greedy arms <1 min,
# mcts_min/mean ~5-10 min, mcts_best worst case 450 x 1200 expansions at
# ~0.027 s/expansion / 8 = ~31 min  ->  ~1-1.5 h expected on 1 GPU
# = ~0.13-0.19 node-hours; the 6 h walltime cap bounds it at 0.75 nh.
# Idempotent: an arm whose output file exists is skipped.
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
TAG="${1:-cand}"; CK="${2:-move_planner/checkpoints/candidate_scored.ckpt}"; DEV="${3:-cuda}"
OUT_ROOT="${OUT_ROOT:-$REPO/runs/spr/fwd}"; WIDTH="${WIDTH:-8}"; THREADS="${THREADS:-2}"
EXP="${EXP:-1200}"; K="${K:-5}"
ARMS="${ARMS:-astar greedy_value greedy_policy mcts_min mcts_mean mcts_best}"
mkdir -p "$REPO/runs/spr" "$OUT_ROOT/m2"
echo "SPR FWD M2 $TAG start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} ckpt=$CK dev=$DEV arms=$ARMS"
nvidia-smi -L 2>/dev/null || true
search_of() { case "$1" in astar) echo astar;; greedy_value) echo greedy_value;;
  greedy_policy) echo greedy_policy;; mcts_*) echo mcts;; esac; }
flags_of()  { case "$1" in
  mcts_min)  echo "--mcts-backup min";;
  mcts_mean) echo "--mcts-backup mean";;
  mcts_best) echo "--mcts-backup min --best-at-budget";;
  *) echo "";; esac; }
rc_all=0
for arm in $ARMS; do
  out="$OUT_ROOT/m2/${TAG}_${arm}.json"
  [ -s "$out" ] && { echo "SPR FWD M2 SKIP $arm"; continue; }
  t0=$SECONDS
  python -m spr.fwd.arena bench --name "fwd_m2_${TAG}_${arm}" --ckpt "$CK" \
      --instances eval/data/bench450.jsonl --search "$(search_of "$arm")" \
      --flags "$(flags_of "$arm")" --expansions "$EXP" --k "$K" \
      --width "$WIDTH" --threads "$THREADS" --device "$DEV" \
      --run-root "$OUT_ROOT/runs" --out "$out" \
    || { echo "SPR FWD M2 BENCH FAILED $arm"; rc_all=1; continue; }
  echo "SPR FWD M2 bench $arm took $((SECONDS-t0))s"
done
# paired tables: the M2 gate is "strictly better mean moves at equal-or-better
# solve rate" vs greed; astar is the reference the whole forward track is graded on.
for base in astar greedy_value; do
  b="$OUT_ROOT/m2/${TAG}_${base}.json"; [ -s "$b" ] || continue
  for arm in $ARMS; do
    [ "$arm" = "$base" ] && continue
    a="$OUT_ROOT/m2/${TAG}_${arm}.json"; [ -s "$a" ] || continue
    python -m spr.fwd.gate compare --a "$a" --b "$b" \
        --out "$OUT_ROOT/m2/${TAG}_${arm}.vs_${base}.json" >/dev/null \
      || echo "SPR FWD M2 WARN gate $arm vs $base failed"
  done
done
# and every arm against the FROZEN recorded baseline row for the same ckpt
REFJ="$REPO/supervised_valuenet/eval/results/comparison_forward.json"
for arm in $ARMS; do
  a="$OUT_ROOT/m2/${TAG}_${arm}.json"; [ -s "$a" ] || continue
  python -m spr.fwd.gate compare --a "$a" --b "$REFJ" --b-system "$CK" \
      --out "$OUT_ROOT/m2/${TAG}_${arm}.vs_recorded.json" >/dev/null 2>&1 \
    || echo "SPR FWD M2 note: no recorded row for $CK (skipped $arm)"
done
python - "$OUT_ROOT/m2" "$TAG" <<'EOF'
import json, sys, pathlib
d, tag = pathlib.Path(sys.argv[1]), sys.argv[2]
print("| arm | solved | mean regret | % optimal | mean moves | mean expansions | s/inst |")
print("|---|---|---|---|---|---|---|")
for f in sorted(d.glob(f"{tag}_*.json")):
    if ".vs_" in f.name: continue
    p = json.loads(f.read_text())
    for n, s in p["systems"].items():
        a = s.get("aggregate") or {}
        print(f"| {f.stem.replace(tag+'_','')} | {a.get('solved')}/{a.get('n')} | "
              f"{a.get('mean_regret')} | {a.get('pct_optimal')} | {a.get('mean_moves')} | "
              f"{a.get('mean_expansions')} | {a.get('mean_seconds')} |")
EOF
echo "SPR FWD M2 DONE $TAG rc=$rc_all $(date +%F' '%T)"
exit $rc_all
