#!/bin/bash
#SBATCH --job-name=sg5-front24
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/sg5-front24-%j.out
# Pre-registered in self_play_robots/subgoal/STAGE5_PROBE_FRONTIER.md.
set -euo pipefail

REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
BOARDS=/scratch/project/open-37-42/petrhyner/MCTS_evolution/supervised_valuenet/environments_g24r4
OUT=$REPO/self_play_robots/results/subgoal/stage5_frontier
SHARDS=16

cd "$REPO"; mkdir -p "$OUT"
export OMP_NUM_THREADS=1
export PYTHONPATH=supervised_valuenet:self_play_robots
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is)"
git log -1 --pretty="code at %h %s"

for j in $(seq 0 $((SHARDS-1))); do
  python3 -m subgoal.space \
    --size 24 --env-dir "$BOARDS" \
    --bench supervised_valuenet/scaling/data/g24r4/bench.unsolved.jsonl \
    --base-ceiling self_play_robots/results/ceiling/g24r4_frontier_base.json \
    --b2-ceiling  self_play_robots/results/ceiling/g24r4_frontier_b2.json \
    --stride 7 --n 32 --shards $SHARDS --shard "$j" \
    --time-cap 900 --max-pops 500000 \
    --out "results/subgoal/stage5_frontier/shard_${j}.json" \
    > "$OUT/shard_${j}.log" 2>&1 &
done
wait
echo "all shards done, $(date -Is)"

python3 - <<'PYEOF'
import json, glob, statistics as st
from pathlib import Path
out = Path("self_play_robots/results/subgoal/stage5_frontier")
rows, meta = [], None
for f in sorted(glob.glob(str(out / "shard_*.json"))):
    d = json.load(open(f))
    meta = meta or {k: v for k, v in d.items()
                    if k not in ("rows", "summary", "instance_indices", "shard")}
    rows.extend(d["rows"])
rows.sort(key=lambda r: r["idx"])

res  = [r for r in rows if r.get("certified") and r.get("proved_optimal")]
subg = [r["n_subgoals"] for r in res]
# falsification test: our optimum must never exceed the hand-written best
viol = [(r["idx"], r["certified_moves"], r["b2_moves"]) for r in res
        if r.get("b2_moves") is not None and r["certified_moves"] > r["b2_moves"]]
strict = [(r["idx"], r["certified_moves"], r["b2_moves"]) for r in res
          if r.get("b2_moves") is not None and r["certified_moves"] < r["b2_moves"]]
both = [r for r in res if r.get("b2_moves") is not None]

summary = dict(
    n_instances=len(rows), n_resolved=len(res), n_capped=sum(1 for r in rows if r["capped"]),
    subgoals_mean=(sum(subg) / len(subg)) if subg else None,
    subgoals_median=st.median(subg) if subg else None,
    subgoals_min=min(subg) if subg else None, subgoals_max=max(subg) if subg else None,
    subgoals_hist={str(k): subg.count(k) for k in sorted(set(subg))},
    new_optima_established=len(res),
    moves_mean_resolved=(sum(r["certified_moves"] for r in res) / len(res)) if res else None,
    moves_max_resolved=max((r["certified_moves"] for r in res), default=None),
    moves_per_subgoal=(sum(r["certified_moves"] for r in res) / sum(subg)) if subg else None,
    falsification_violations=viol,
    n_both_have_answer=len(both),
    n_state_strictly_shorter=len(strict),
    handwritten_extra_on_shared=(
        sum(r["b2_moves"] - r["certified_moves"] for r in both) / len(both)) if both else None,
    handwritten_b2_resolved_count=sum(1 for r in rows if r.get("b2_moves") is not None),
    mean_seconds=sum(r["seconds"] for r in rows) / len(rows),
    median_seconds=st.median([r["seconds"] for r in rows]),
    mean_pops=sum(r["pops"] for r in rows) / len(rows),
)
meta["summary"] = summary; meta["rows"] = rows
Path(out / "merged.json").write_text(json.dumps(meta, indent=1) + "\n")
print(json.dumps(summary, indent=1))
print("MERGED", out / "merged.json")
PYEOF
