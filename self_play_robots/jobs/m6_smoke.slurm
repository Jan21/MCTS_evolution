#!/bin/bash
# M6 Phase-0 GPU smoke (see m6/smoke_parts.py + M6_DESIGN.md): feasibility
# numbers for self-play at 80x80/96x96 -- net latency + memory at 64/80/96,
# per-expansion search cost (subgoal MCTS + hybrid) at 80 and 96, a 3-instance
# generation probe at 80 (ctg magnitudes vs the 96-bin clamp), and one real
# training step at 80 (policy batch 1; value 1 group x 2 records).
#   sbatch -J spr-m6-smoke self_play_robots/jobs/m6_smoke.slurm
# Cost: <= 1 h on 1 GPU = <= 0.125 nh.
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
W="$REPO/runs/spr/m6smoke"; mkdir -p "$W"
OUT="$REPO/self_play_robots/results/m6/smoke.jsonl"; mkdir -p "$(dirname "$OUT")"
P=$(grep '^policy=' results/selfplay/mix_b2mix_iter3/nets.txt | cut -d= -f2-)
V=$(grep '^value=' results/selfplay/mix_b2mix_iter3/nets.txt | cut -d= -f2-)
echo "M6 SMOKE start $(date +%F' '%T) job=$SLURM_JOB_ID nets=$P|$V"
nvidia-smi -L

python -m m6.smoke_parts boards --work "$W" --out "$OUT" --sv "$REPO/supervised_valuenet" || echo "M6 SMOKE FAILED boards"
python -m m6.smoke_parts micro --work "$W" --out "$OUT" --sv "$REPO/supervised_valuenet" --policy "$P" || echo "M6 SMOKE WARN micro"

bench_one() {  # bench_one <cfg> <n> <search-args> <tag> <exp>
  local cfg="$1" tag="$4"
  ( export RR_GRID="$2" RR_ROBOTS=4 RR_WALLS=$(python -c "print(round(48*($2/16)**2))") RR_ENV_DIR="$W/boards_$cfg"
    timeout 2400 python -m spr.bench --instances "$W/inst_$cfg.jsonl" --expansions "$5" --k 5 \
      --policy "$P" --value "$V" $3 --vocab b2 --anytime --boards lean --device cuda --dump-moves \
      --out "$W/bench_${cfg}_${tag}.json" --md /dev/null ) \
    && python - "$W/bench_${cfg}_${tag}.json" "$cfg" "$tag" "$OUT" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); a=list(d["systems"].values())[0]["aggregate"]
row={"part":"bench","config":sys.argv[2],"tag":sys.argv[3],"solved":a["solved"],"n":a["n"],
     "mean_expansions":round(a["mean_expansions"],1),"mean_seconds":round(a["mean_seconds"],1),
     "s_per_exp":round(a["mean_seconds"]/max(a["mean_expansions"],1),2),
     "mean_strict":a.get("mean_realized_strict"),"abstract":a.get("mean_plan_cost_abstract")}
open(sys.argv[4],"a").write(json.dumps(row)+"\n"); print("[m6smoke]",json.dumps(row))
PYEOF
  [ $? -eq 0 ] || echo "M6 SMOKE WARN bench $cfg $tag"
}
bench_one g80r4 80 "--search mcts --best-at-budget" mcts40 40
bench_one g80r4 80 "--search arena_astar" astar 40
bench_one g96r4 96 "--search mcts --best-at-budget" mcts25 25

# hybrid probe at 80 (root slides, small budgets)
( export RR_GRID=80 RR_ROBOTS=4 RR_WALLS=1200 RR_ENV_DIR="$W/boards_g80r4"
  timeout 1500 python -m variants.v07_hybrid_actions bench --instances "$W/inst_g80r4.jsonl" \
    --policy "$P" --value "$V" --b0 120 --top-m 4 --sub 30 --boards lean --device cuda \
    --limit 2 --out "$W/hybrid_g80.json" ) \
  && python - "$W/hybrid_g80.json" "$OUT" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); a=list(d["systems"].values())[0]["aggregate"]
row={"part":"hybrid","config":"g80r4","solved":a["solved"],"n":a["n"],
     "mean_expansions":round(a["mean_expansions"],1),"mean_seconds":round(a["mean_seconds"],1)}
open(sys.argv[2],"a").write(json.dumps(row)+"\n"); print("[m6smoke]",json.dumps(row))
PYEOF
[ $? -eq 0 ] || echo "M6 SMOKE WARN hybrid"

# mini generation at 80 (1 board x 3 instances) -> records for the training probe + ctg range
timeout 2000 python -m spr.selfplay --config g80r4 --policy "$P" --value "$V" \
  --boards-dir "$W/genboards_g80r4" --board-ids 30950-30950 --vocab b2 --per-board 3 \
  --seed 11 --iter 0 --expansions 60 --stop-after 30 --root-noise 0.25 --complete-siblings --root-all \
  --workers 1 --threads 4 --device cuda --timeout 900 --out "$W/gen80.jsonl" \
  || echo "M6 SMOKE WARN generation"
python - "$W/gen80.jsonl" "$OUT" <<'PYEOF'
import json,sys
try:
    recs=[json.loads(l) for l in open(sys.argv[1])]
    ctg=[int(round(float(r["cost_to_go"]))) for r in recs]
    st=[r.get("strict_total") for r in recs if r.get("strict_total") is not None]
    m=json.load(open(sys.argv[1]+".manifest.json"))
    row={"part":"gen","config":"g80r4","records":len(recs),
         "ctg_max":max(ctg) if ctg else None,"ctg_over95":sum(1 for c in ctg if c>95),
         "strict_max":max(st) if st else None,
         "instances":m["instances"],"solved":m["solved"],"mean_sec_per_inst":round(m["mean_seconds"],1),
         "mean_expansions":round(m["mean_expansions"],1)}
except Exception as e:
    row={"part":"gen","error":str(e)}
open(sys.argv[2],"a").write(json.dumps(row)+"\n"); print("[m6smoke]",json.dumps(row))
PYEOF

python -m m6.smoke_parts trainprobe --work "$W" --out "$OUT" --sv "$REPO/supervised_valuenet" \
  --policy "$P" --value "$V" --records "$W/gen80.jsonl" --value-records 2 \
  || echo "M6 SMOKE WARN trainprobe"
echo "M6 SMOKE DONE $(date +%F' '%T)"
