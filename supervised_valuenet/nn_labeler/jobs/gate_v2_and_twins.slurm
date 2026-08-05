#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-gtwin
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/gate_v2_twins_%j.out
#
# Two things, in order:
#
# 1. GATE the v2 production net (prod2, job 4616677) exactly like the capstone
#    gated v1 (FINDINGS 59): replay 600 test instances at g24r4 + g32r4,
#    audit against exact labels. Pick the better labeler by argmin agreement
#    at g32r4 (tie-break: mean gap) and bank it. v1's numbers to beat:
#    argmin 92.6% (24), 91.5% (32); mean gap 0.566 / 0.553.
#
# 2. TWIN CORPORA for the downstream-equivalence experiment (the headline):
#    replay the FULL instance set of the exact corpora at g24r4/g24r8/g32r4
#    with the winning net -- same boards, same instances, only the label
#    source differs. Planners retrained on these twins vs the exact-trained
#    planners (already benched in scaling/results/*/comparison.json) isolate
#    exactly one variable: does label provenance matter downstream?
#    All three cells are leakage-free: neither v1 nor v2 ever trained on
#    g24r4/g24r8/g32r4 (their mix is g8r4-g15r4 + g16r6 + g16r8).
#
# Chunked by graph range (descent writes its output once, at the end -- a
# walltime kill must lose at most one chunk, pitfall #9), idempotent, and the
# corpus each twin replays comes from twin_map.env read at RUNTIME.
#
#   sbatch supervised_valuenet/nn_labeler/jobs/gate_v2_and_twins.slurm
set -u
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi -L

source nn_labeler/jobs/twin_map.env

V1=nn_labeler/banked/prod_v1_s11.ckpt
V2DIR=nn_labeler/runs/prod2_mix8to16_none_s11_lr1e-4

# ---- 1. gate v2 at both large rungs (same protocol as the capstone) ---------
V2=$(ls -t "$V2DIR"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
if [ -z "$V2" ]; then
    echo "GTWIN NO V2 CKPT -- falling back to banked v1"
    WINNER=$V1
else
    echo "GTWIN V2 CKPT $V2"
    for cfg in g24r4 g32r4; do
        exact=$(eval echo "\$TWIN_CORPUS_${cfg}")
        dg="nn_labeler/results/v2gate_${cfg}.jsonl"
        if [ ! -f "$dg" ]; then
            python -m nn_labeler.descent --config "$cfg" --ckpt "$V2" \
                --instances-from "$exact" --graphs 900-1049 --limit-instances 600 \
                --device cuda --timeout 180 --out "$dg" \
                || { echo "GTWIN V2GATE-DESCENT FAILED $cfg"; break; }
        fi
        python -m nn_labeler.audit_descent --exact "$exact" --descent "$dg" \
            --out "nn_labeler/results/v2gate_${cfg}.json" \
            || echo "GTWIN V2GATE-AUDIT FAILED $cfg"
    done
    WINNER=$(python - "$V2" <<'PY'
import json, os, sys
v2_ckpt = sys.argv[1]
def load(p):
    return json.load(open(p))["summary"] if os.path.exists(p) else None
v1 = load("nn_labeler/results/capgate_g32r4.json")
v2 = load("nn_labeler/results/v2gate_g32r4.json")
def key(d):
    # argmin agreement first (the load-bearing metric, FINDINGS 59a),
    # then smaller mean gap
    return (d["argmin_agreement"], -d["gap_mean"])
if v2 is None:
    print("nn_labeler/banked/prod_v1_s11.ckpt")
elif key(v2) > key(v1):
    print(v2_ckpt)
else:
    print("nn_labeler/banked/prod_v1_s11.ckpt")
PY
)
fi
echo "GTWIN WINNER $WINNER"

# bank v2 if it won (idempotent; sha recorded like v1's banking)
if [ "$WINNER" != "$V1" ] && [ ! -f nn_labeler/banked/prod_v2_s11.ckpt ]; then
    cp "$WINNER" nn_labeler/banked/prod_v2_s11.ckpt
    python - "$WINNER" <<'PY'
import hashlib, json, sys
src = sys.argv[1]
sha = hashlib.sha256(open("nn_labeler/banked/prod_v2_s11.ckpt","rb").read()).hexdigest()[:16]
json.dump({"source_ckpt": src, "sha256_16": sha,
           "train_mix": "g8r4-g15r4 rust + g16r6 + g16r8 (same as v1)",
           "gate": "v2gate_g24r4.json / v2gate_g32r4.json vs capgate_* (v1)",
           "banked_by": "gate_v2_and_twins.slurm"},
          open("nn_labeler/banked/prod_v2_s11.json","w"), indent=1)
print("GTWIN BANKED v2 sha", sha)
PY
    WINNER=nn_labeler/banked/prod_v2_s11.ckpt
fi
echo "GTWIN LABELER OF RECORD $WINNER"

# ---- 2. twin corpora: full replay, chunked by graph range -------------------
for cfg in g24r4 g24r8 g32r4; do
    exact=$(eval echo "\$TWIN_CORPUS_${cfg}")
    [ -f "$exact" ] || { echo "GTWIN MISSING EXACT $cfg $exact"; continue; }
    final="nn_labeler/results/twin_${cfg}.jsonl"
    if [ -f "$final" ]; then echo "GTWIN SKIP twin $cfg (exists)"; continue; fi
    ok=1
    for range in 0-349 350-699 700-1049; do
        chunk="nn_labeler/results/twin_${cfg}.${range}.jsonl"
        [ -f "$chunk" ] && { echo "GTWIN SKIP chunk $cfg $range"; continue; }
        t0=$SECONDS
        python -m nn_labeler.descent --config "$cfg" --ckpt "$WINNER" \
            --instances-from "$exact" --graphs "$range" \
            --device cuda --timeout 180 --out "$chunk" \
            || { echo "GTWIN TWIN-CHUNK FAILED $cfg $range"; ok=0; break; }
        echo "GTWIN CHUNK $cfg $range $(wc -l < "$chunk") records $((SECONDS-t0))s"
    done
    [ "$ok" = 1 ] || continue
    cat "nn_labeler/results/twin_${cfg}.0-349.jsonl" \
        "nn_labeler/results/twin_${cfg}.350-699.jsonl" \
        "nn_labeler/results/twin_${cfg}.700-1049.jsonl" > "${final}.tmp" \
        && mv "${final}.tmp" "$final"
    echo "GTWIN TWIN $cfg $(wc -l < "$final") records (exact corpus: $(wc -l < "$exact"))"
    # corpus-level label-quality gate for the paper's table (CPU, cheap)
    python -m nn_labeler.audit_descent --exact "$exact" --descent "$final" \
        --out "nn_labeler/results/twin_gate_${cfg}.json" \
        || echo "GTWIN TWIN-GATE FAILED $cfg"
done
echo "NNLAB GTWIN DONE"
