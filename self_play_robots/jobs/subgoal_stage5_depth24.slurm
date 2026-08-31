#!/bin/bash
#SBATCH --job-name=sg5-depth24
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/sg5-depth24-%j.out
# Stage 5 probe, pre-registered in self_play_robots/subgoal/STAGE5_PROBE.md.
# CPU-only work; the GPU is requested only because the allocation has no CPU
# partition. 16 shards of the same pre-registered instance list, run in
# parallel, then merged.
set -euo pipefail

REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
BOARDS=/scratch/project/open-37-42/petrhyner/MCTS_evolution/supervised_valuenet/environments_g24r4
OUT=$REPO/self_play_robots/results/subgoal/stage5_probe
SHARDS=16

cd "$REPO"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1
export PYTHONPATH=supervised_valuenet:self_play_robots

echo "job $SLURM_JOB_ID on $(hostname), $(date -Is)"
git log -1 --pretty="code at %h %s"

for j in $(seq 0 $((SHARDS-1))); do
  python3 -m subgoal.space \
    --size 24 \
    --env-dir "$BOARDS" \
    --bench supervised_valuenet/scaling/data/g24r4/bench.solved.jsonl \
    --base-ceiling self_play_robots/results/ceiling/g24r4_base.json \
    --b2-ceiling  self_play_robots/results/ceiling/g24r4_b2.json \
    --stride 8 --n 29 --shards $SHARDS --shard "$j" \
    --time-cap 900 --max-pops 200000 \
    --out "results/subgoal/stage5_probe/shard_${j}.json" \
    > "$OUT/shard_${j}.log" 2>&1 &
done
wait
echo "all shards done, $(date -Is)"

python3 - <<'PYEOF'
import json, glob, statistics as st
from pathlib import Path
out = Path("self_play_robots/results/subgoal/stage5_probe")
files = sorted(glob.glob(str(out / "shard_*.json")))
rows, meta = [], None
for f in files:
    d = json.load(open(f))
    meta = meta or {k: v for k, v in d.items() if k not in ("rows", "summary", "instance_indices", "shard")}
    rows.extend(d["rows"])
rows.sort(key=lambda r: r["idx"])

res    = [r for r in rows if r.get("certified") and r.get("proved_optimal")]
capped = [r for r in rows if r.get("capped")]
extra  = [r["certified_moves"] - r["d_star"] for r in res]
subg   = [r["n_subgoals"] for r in res]
b2     = [r["b2_moves"] - r["d_star"] for r in rows if r.get("b2_moves") is not None]
base   = [r["base_moves"] - r["d_star"] for r in rows if r.get("base_moves") is not None]

summary = dict(
    n_instances=len(rows), n_resolved=len(res), n_capped=len(capped),
    state_mean_extra=(sum(extra) / len(extra)) if extra else None,
    state_n_optimal=sum(1 for v in extra if v == 0),
    state_any_below_dstar=sum(1 for v in extra if v < 0),
    subgoals_mean=(sum(subg) / len(subg)) if subg else None,
    subgoals_median=st.median(subg) if subg else None,
    subgoals_min=min(subg) if subg else None,
    subgoals_max=max(subg) if subg else None,
    subgoals_hist={str(k): subg.count(k) for k in sorted(set(subg))},
    handwritten_b2_mean_extra=(sum(b2) / len(b2)) if b2 else None,
    handwritten_base_mean_extra=(sum(base) / len(base)) if base else None,
    mean_d_star_all=sum(r["d_star"] for r in rows) / len(rows),
    mean_d_star_resolved=(sum(r["d_star"] for r in res) / len(res)) if res else None,
    mean_seconds=sum(r["seconds"] for r in rows) / len(rows),
    max_seconds=max(r["seconds"] for r in rows),
    mean_pops=sum(r["pops"] for r in rows) / len(rows),
)
meta["summary"] = summary
meta["rows"] = rows
Path(out / "merged.json").write_text(json.dumps(meta, indent=1) + "\n")
print(json.dumps(summary, indent=1))
print("MERGED", out / "merged.json")
PYEOF
