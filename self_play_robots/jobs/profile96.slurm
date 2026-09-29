#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/prof96-%j.out
# Where the time goes inside one search iteration at 96x96: cProfile around
# spr.bench on one hard ladder instance (idx 1, never solved), 300 expansions,
# with the full value net and with the split (per-decision) value net.
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
NETS=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/selfplay/mix_b2mix_iter3
POL=$NETS/policy/lightning_logs/version_0/checkpoints/epoch=2-step=2361.ckpt
VALF=$NETS/value/lightning_logs/version_0/checkpoints/epoch=0-step=1573.ckpt
VALS=$REPO/self_play_robots/results/subgoal/split/dec0-sw-r12-lr1e-4-x8-distill.pt
CFG=g96r4
eval "$(python -c "
from scaling.configs import get, env
for k,v in env(get('$CFG')).items(): print(f'export {k}={v}')")"
export RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
sed -n "${IDX:-2}p" "$REPO/supervised_valuenet/scaling/data/$CFG/ladder.jsonl" > /tmp/prof_inst.jsonl
mkdir -p results/subgoal/prof
if [ -n "${CHECK:-}" ]; then   # equality check of the batched proposer readout against the original loop
  echo "=== CHECK batched policy readout, ${CHECK} calls, 20 expansions $(date +%T)"
  RR_POLICY_CHECK=$CHECK python -m spr.bench --instances /tmp/prof_inst.jsonl --policy "$POL" --value "$VALS" --arch split \
    --search arena_astar --vocab b2 --anytime --expansions 20 --k 5 --boards lean --device cuda \
    --out results/subgoal/prof/check.json 2>&1 | grep -E "policy-check|solved [0-9]+/" | tail -${CHECK}
fi
for ARM in full split; do
  if [ $ARM = full ]; then VAL=$VALF; ARCH=sizefree; else VAL=$VALS; ARCH=split; fi
  echo "=== PROFILE arm=$ARM idx=${IDX:-2} expansions=${EXP:-300} $(date +%T)"
  /usr/bin/time -f "wall %e s, maxrss %M KB" python -m cProfile -o results/subgoal/prof/g96_${ARM}${TAG:-}.pstats -m spr.bench \
    --instances /tmp/prof_inst.jsonl --policy "$POL" --value "$VAL" --arch $ARCH \
    --search arena_astar --vocab b2 --anytime --expansions ${EXP:-300} --k 5 \
    --boards lean --device cuda --out results/subgoal/prof/g96_${ARM}${TAG:-}.json 2>&1 | grep -vE "^W0|UserWarning|warnings.warn" | tail -4
  python - <<PY
import pstats, json
st = pstats.Stats("results/subgoal/prof/g96_${ARM}${TAG:-}.pstats")
rows = json.load(open("results/subgoal/prof/g96_${ARM}${TAG:-}.json"))["systems"]
r = list(rows.values())[0]["rows"][0]
print("bench row: solved", r["solved"], "expansions", r["expansions"], "seconds %.1f" % r["seconds"], "acct", {k: v for k, v in r["accounting"].items() if v})
want = ["_nn_astar_backward", "_value_cost", "_policy_logp", "_policy_logp_loop", "astar", "expand", "propose", "propose_subgoal_states", "_apply", "policy_logp", "value_ctg", "_encode", "logp_map",
        "forced_fixes", "compute_exact_shortest_path_length", "_expand", "cand_record", "hidx", "_context", "_fixed_g", "_segment",
        "strict_moves", "repairs", "park_repairs", "node_features", "key_indices", "policy_features", "heappush", "heappop",
        "scaled_dot_product_attention", "from_env", "adjacency_from_graph", "_reference_helpers"]
tot = {}
for (file, line, fn), (cc, nc, tt, ct, callers) in st.stats.items():
    if fn in want:
        k = fn if fn not in tot or ct > tot[fn][1] else None
        if k: tot[fn] = (nc, ct, tt, file.split("/")[-1])
print("function: calls, cumulative s, own s, file")
for fn, (nc, ct, tt, f) in sorted(tot.items(), key=lambda x: -x[1][1]):
    print(f"  {fn:38s} {nc:7d} {ct:9.2f} {tt:8.2f}  {f}")
PY
done
echo "=== PROFILE DONE $(date +%T)"
