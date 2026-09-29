#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/merge-%x-%j.out
# Join the generation shards of jobs/gen_shard.slurm into the round's
# records.jsonl and mark the generation phase done. Env: CFG K TAGV NSHARDS.
# Fails (and so blocks the chained round) if any shard is missing.
set -uo pipefail
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
CFG=${CFG:-g64r4}; K=${K:-1}; TAGV=${TAGV:-_b2}; N=${NSHARDS:-15}
IT="$REPO/runs/spr/selfplay/${CFG}${TAGV}_iter${K}"; RES="$REPO/self_play_robots/results/selfplay/${CFG}${TAGV}_iter${K}"
mkdir -p "$RES"
# every records_*.jsonl present (shards may be split into a/b halves); the
# check is board coverage: NBOARDS distinct env_ids, not shard indices
NB=${NBOARDS:-60}
files=$(ls "$IT"/records_*.jsonl 2>/dev/null | grep -v instances)
[ -n "$files" ] || { echo "MERGE FAILED: no shard files"; exit 1; }
cat $files > "$IT/records.jsonl.tmp"
nb=$(python3 -c "import json,sys; print(len({json.loads(l)['env_id'] for l in open(sys.argv[1])}))" "$IT/records.jsonl.tmp")
echo "MERGE boards covered: $nb of $NB, from $(echo $files | wc -w) shard files"
[ "$nb" -ge "$NB" ] || { rm -f "$IT/records.jsonl.tmp"; echo "MERGE FAILED: only $nb boards"; exit 1; }
mv "$IT/records.jsonl.tmp" "$IT/records.jsonl"
cat $(for f in $files; do echo "$f.instances.jsonl"; done) > "$IT/records.jsonl.instances.jsonl" 2>/dev/null || true
python3 - "$IT" <<'PY'
import json, sys, glob
it = sys.argv[1]
files = sorted(f for f in glob.glob(f"{it}/records_*.jsonl") if "instances" not in f)
shards = [json.load(open(f + ".manifest.json")) for f in files]
recs = sum(1 for _ in open(f"{it}/records.jsonl"))
m = dict(shards[0]); m["merged_from_shards"] = len(files); m["records"] = recs
for k in ("instances", "solved"):
    if all(isinstance(s.get(k), (int, float)) for s in shards): m[k] = sum(s[k] for s in shards)
json.dump(m, open(f"{it}/records.jsonl.manifest.json", "w"), indent=1)
print("MERGE records:", recs, {k: m.get(k) for k in ("instances", "solved")})
PY
cp "$IT/records.jsonl.manifest.json" "$RES/generation.manifest.json"
touch "$IT/DONE.gen"; echo "MERGE DONE gen marked $(date +%T)"
