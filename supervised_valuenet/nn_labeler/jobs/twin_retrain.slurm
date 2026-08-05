#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-twinrt
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/twin_retrain_%j.out
#
# THE DOWNSTREAM-EQUIVALENCE EXPERIMENT (the headline of the NN-labeler track).
#
# gate_v2_and_twins.slurm builds the TWIN corpora: the exact corpora's own
# instances, on the exact corpora's own boards, relabeled by the banked NN
# labeler instead of the exact solver (nn_labeler/results/twin_<cfg>.jsonl).
# This job retrains the two BACKWARD planners on those twins and benches them
# with the protocol that produced scaling/results/<cfg>/comparison.json, so the
# twin row and the exact row differ in exactly one variable: label provenance.
#
# The forward planner is NOT retrained (descent stores no primitive-move
# sequences; see nn_labeler/TWIN_WIRING.md section 5), so every lane here is
# backward-only and every forward row in the study stands untouched.
#
# Vocabulary: BASE, everywhere. Nothing here reads or writes a b2 path and no
# run/result name contains "b2" (house rule: never mix vocabularies).
#
# HYPERPARAMETERS are the ORIGINAL runs' verbatim, per config -- provenance and
# the evidence for each number are in nn_labeler/TWIN_WIRING.md section 2:
#   g24r4  policy 30ep bs8      | value 30ep bs4 mpg16 warmup0   (g24r4_chain.sh)
#   g24r8  policy 30ep bs8      | value 30ep bs4 mpg16 warmup0   (g24r8_chain.sh)
#   g32r4  policy 30ep bs8      | value 25ep bs2 mpg8  warmup0   (g32r4_chain.sh
#                                                       + g32r4_retrain.sh)
# lr is the trainer default 3e-4 in all six (the >=6-robot low-lr launcher
# pattern was NOT used at these rungs -- hparams.yaml says lr: 0.0003), value
# bins are the default 50 in all three (g32r4's checkpoint of record is 50-class
# despite run_config's 96-bin recommendation), and no net is warm-started
# because none of the exact-arm nets was.
#
# COST: ~25 h of work for 6 trainings + 6 bench lanes (TWIN_WIRING.md section 2
# has the per-step arithmetic), against a 16 h walltime cap. That is deliberate:
# every step is skip-if-exists and every training resumes from its own
# last.ckpt (RR_RESUME=1), so a walltime kill costs at most one epoch and
# RESUBMITTING THE SAME COMMAND CONTINUES. Expect 2 submissions. Configs run
# cheapest-first so the first submission lands complete rows, not six halves.
#
#   sbatch supervised_valuenet/nn_labeler/jobs/twin_retrain.slurm
#
# Optional env overrides (both default to "the production twin arm"):
#   TWIN_TAG=-seed21 TWIN_SEED=21 sbatch ...   # a seed replicate; run dirs and
#                                              # result files gain the tag, so a
#                                              # replicate cannot overwrite the
#                                              # production row (the b2
#                                              # seed-study pattern)
set -u
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
cd "$SV"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi -L

TAG="${TWIN_TAG:-}"
SEEDARG=""
[ -n "${TWIN_SEED:-}" ] && SEEDARG="--torch-seed ${TWIN_SEED}"
echo "TWINRT tag='${TAG}' seed='${TWIN_SEED:-none}' $(date +%F' '%T)"

# A config's RR_* must never leak from one lane into the next. scaling.train
# sets them in-process from the config, but eval.compare reads them from the
# environment, so they are exported per bench lane and cleared here.
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

STEPS_OK=0; STEPS_FAILED=0; STEPS_SKIPPED_MISSING=0

# ---------------------------------------------------------------------------
# training: one system, one config. Idempotent on a DONE sentinel; resumable
# from its own last.ckpt; refuses to resume across a corpus change.
# ---------------------------------------------------------------------------
train_one() {                       # train_one <cfg> <backward-policy|backward-value> <extra args...>
    local cfg="$1" system="$2"; shift 2
    local data="nn_labeler/results/twin_${cfg}.jsonl"
    local dir="scaling/runs/${cfg}/${system}-nntwin${TAG}"
    if [ ! -s "$data" ]; then
        echo "TWINRT MISSING TWIN CORPUS $cfg $data -- gate_v2_and_twins.slurm has not produced it yet"
        STEPS_SKIPPED_MISSING=$((STEPS_SKIPPED_MISSING + 1)); return 2
    fi
    if [ -f "$dir/DONE" ]; then
        echo "TWINRT SKIP train $cfg/$system (DONE)"; STEPS_OK=$((STEPS_OK + 1)); return 0
    fi
    mkdir -p "$dir"
    # Stale-resume guard (FINDINGS 38's silent-success class): the run dir
    # records the corpus it was started on. A corpus that changed underneath a
    # half-finished run must stop the job, not silently continue.
    local stamp; stamp=$(sha256sum "$data" | cut -c1-16)
    if [ -f "$dir/DATA.txt" ]; then
        if ! grep -qx "sha256_16=$stamp" "$dir/DATA.txt"; then
            echo "TWINRT FAILED train $cfg/$system: $data changed under a partial run"
            echo "                 (expected $(grep '^sha256_16=' "$dir/DATA.txt"), got $stamp)"
            echo "                 remove $dir and resubmit to retrain from scratch"
            STEPS_FAILED=$((STEPS_FAILED + 1)); return 1
        fi
    else
        printf 'data=%s\nsha256_16=%s\nrecords=%s\nstarted=%s\n' \
            "$data" "$stamp" "$(wc -l < "$data")" "$(date -Is)" > "$dir/DATA.txt"
    fi
    local resume=""
    ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
    echo "TWINRT START train $cfg/$system resume=${resume:-0} $(date +%T)  args: $*"
    local t0=$SECONDS
    RR_RESUME=${resume:-0} python -m scaling.train --config "$cfg" \
        --system "$system" --run-dir "$dir" $SEEDARG -- --data "$data" "$@"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "TWINRT FAILED train $cfg/$system rc=$rc after $((SECONDS - t0))s"
        STEPS_FAILED=$((STEPS_FAILED + 1)); return 1
    fi
    # A trainer that exits 0 without ever writing a monitored checkpoint is a
    # silent failure (pitfall: a resume already at max_epochs runs no epoch).
    if ! ls "$dir"/lightning_logs/version_*/checkpoints/epoch=*.ckpt >/dev/null 2>&1; then
        echo "TWINRT FAILED train $cfg/$system: exit 0 but no epoch=*.ckpt under $dir"
        STEPS_FAILED=$((STEPS_FAILED + 1)); return 1
    fi
    touch "$dir/DONE"
    echo "TWINRT DONE train $cfg/$system in $((SECONDS - t0))s"
    STEPS_OK=$((STEPS_OK + 1)); return 0
}

ckpt_of() {                         # newest monitored best under a run dir (never last.ckpt)
    ls -t "$1"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1
}

# ---------------------------------------------------------------------------
# bench: eval.compare with comparison.json's protocol verbatim, backward only.
# Sharded 8-wide over the job's 16 cores, chunk-level idempotent, merged, then
# independently replay-certified.
# ---------------------------------------------------------------------------
bench_one() {                       # bench_one <cfg> <graded|frontier>
    local cfg="$1" set="$2"
    local pdir="scaling/runs/${cfg}/backward-policy-nntwin${TAG}"
    local vdir="scaling/runs/${cfg}/backward-value-nntwin${TAG}"
    if [ ! -f "$pdir/DONE" ] || [ ! -f "$vdir/DONE" ]; then
        echo "TWINRT SKIP bench $cfg/$set (nets not both trained yet)"; return 2
    fi
    # BP/BV/RUN are deliberately NOT `local`: they are exported to the xargs
    # lanes below, and only one bench lane runs at a time.
    BP=$(ckpt_of "$pdir"); BV=$(ckpt_of "$vdir")
    if [ -z "$BP" ] || [ -z "$BV" ]; then
        echo "TWINRT FAILED bench $cfg/$set: checkpoint missing (policy='$BP' value='$BV')"
        STEPS_FAILED=$((STEPS_FAILED + 1)); return 1
    fi
    local INST OUT
    if [ "$set" = graded ]; then
        INST="scaling/data/${cfg}/bench.solved.jsonl"
        OUT="scaling/results/${cfg}/comparison_nntwin${TAG}.json"
    else
        INST="scaling/data/${cfg}/bench.unsolved.jsonl"
        OUT="scaling/results/${cfg}/comparison_ungraded_nntwin${TAG}.json"
    fi
    [ -s "$INST" ] || { echo "TWINRT FAILED bench $cfg/$set: instances missing $INST"
                        STEPS_FAILED=$((STEPS_FAILED + 1)); return 1; }
    # Accept an existing row only if it parses AND carries rows: a walltime kill
    # inside the in-place merge can leave truncated JSON that the skip gate
    # would otherwise wave through.
    if [ -s "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
rows=[len(s.get('rows') or []) for s in d['systems'].values() if s.get('rows')]
sys.exit(0 if rows and min(rows)>0 else 1)" 2>/dev/null; then
        echo "TWINRT SKIP bench $cfg/$set (exists: $OUT)"; STEPS_OK=$((STEPS_OK + 1)); return 0
    elif [ -s "$OUT" ]; then
        echo "TWINRT bench $cfg/$set: existing $OUT truncated -- redoing the merge"; rm -f "$OUT"
    fi

    local ENVDIR="$SV/environments_${cfg}"
    case "$cfg" in
        g24r4) export RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 ;;
        g24r8) export RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108 ;;
        g32r4) export RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192 ;;
        *) echo "TWINRT FAILED bench: unknown config $cfg"
           STEPS_FAILED=$((STEPS_FAILED + 1)); return 1 ;;
    esac
    export RR_ENV_DIR="$ENVDIR"

    # Chunk reuse is keyed on checkpoint identity, so a rerun after a retrain
    # can never merge rows computed with the previous nets.
    local STAMP
    STAMP=$(printf '%s\n%s\n' "$BP" "$BV" | sha256sum | cut -c1-12)
    RUN="$REPO/runs/nnlab/twin.${cfg}.${set}${TAG}.${STAMP}"
    mkdir -p "$RUN"
    printf 'policy=%s\nvalue=%s\ninstances=%s\n' "$BP" "$BV" "$INST" > "$RUN/checkpoints.txt"
    # Chunks must stay in benchmark order -- merge_compare_shards.py requires it.
    [ -e "$RUN/chunk.000.jsonl" ] || \
        split -d -a 3 -l 8 "$INST" "$RUN/chunk." --additional-suffix=.jsonl

    echo "TWINRT START bench $cfg/$set $(date +%T)  policy=$BP  value=$BV"
    local t0=$SECONDS
    run_chunk() {
        local chunk="$1"; local base; base=$(basename "$chunk" .jsonl)
        local out="$RUN/$base.json"
        [ -s "$out" ] && return 0
        if OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" python3 -m eval.compare \
                --instances "$chunk" --expansions 1200 --k 5 \
                --backward-policy "$BP" --backward-value "$BV" \
                --backward-prefix-check --forward-ckpts "" --device cpu \
                --dump-moves --out "$out.tmp.$$" --md /dev/null \
                > "$RUN/$base.log" 2>&1; then
            mv "$out.tmp.$$" "$out"
        else
            echo "TWINRT CHUNK FAIL $base (see $RUN/$base.log)"; return 1
        fi
    }
    export -f run_chunk
    export RUN BP BV
    ls "$RUN"/chunk.*.jsonl | xargs -P 8 -n 1 -I{} bash -c 'run_chunk "$@"' _ {}

    local NC ND
    NC=$(ls "$RUN"/chunk.*.jsonl | wc -l)
    ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
    if [ "$ND" -ne "$NC" ]; then
        echo "TWINRT PARTIAL bench $cfg/$set: $ND/$NC chunks done after $((SECONDS - t0))s (resubmit to resume)"
        return 2
    fi
    python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
        --instances "$INST" --out "$OUT" --md /dev/null
    if [ $? -ne 0 ]; then
        echo "TWINRT FAILED bench $cfg/$set: merge failed"
        STEPS_FAILED=$((STEPS_FAILED + 1)); return 1
    fi
    echo "== replay certification $cfg/$set =="
    python3 -m eval.replay_validate --compare "$OUT" --env-dir "$ENVDIR"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        # Quarantine: left at $OUT it would satisfy the skip gate on the next
        # submission and never be certified again, and $OUT is exactly what the
        # report reads.
        mv -f "$OUT" "$OUT.uncertified" 2>/dev/null
        echo "TWINRT FAILED bench $cfg/$set REPLAY rc=$rc (quarantined at $OUT.uncertified)"
        STEPS_FAILED=$((STEPS_FAILED + 1)); return 1
    fi
    echo "TWINRT DONE bench $cfg/$set in $((SECONDS - t0))s -> $OUT"
    STEPS_OK=$((STEPS_OK + 1)); return 0
}

# ---------------------------------------------------------------------------
# per-config steps, cheapest config first. Proposal net before value net: it is
# the cheap and stable one, so a broken corpus or recipe surfaces in minutes
# instead of after a multi-hour value fit (the b2_retrain_one ordering).
# ---------------------------------------------------------------------------
for cfg in g24r4 g24r8 g32r4; do
    echo "===== TWINRT CONFIG $cfg $(date +%F' '%T) ====="
    case "$cfg" in
        g24r4) PEPOCHS=30; PBATCH="--batch-size 8"
               VEPOCHS=30; VBATCH="--batch-size 4 --max-per-group 16" ;;
        g24r8) PEPOCHS=30; PBATCH="--batch-size 8"
               VEPOCHS=30; VBATCH="--batch-size 4 --max-per-group 16" ;;
        g32r4) PEPOCHS=30; PBATCH="--batch-size 8"
               VEPOCHS=25; VBATCH="--batch-size 2 --max-per-group 8" ;;
    esac
    unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

    train_one "$cfg" backward-policy --epochs "$PEPOCHS" $PBATCH
    train_one "$cfg" backward-value  --epochs "$VEPOCHS" --warmup 0 $VBATCH

    bench_one "$cfg" graded
    bench_one "$cfg" frontier
    unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
done

echo "===== TWINRT SUMMARY ok=$STEPS_OK failed=$STEPS_FAILED missing_corpus=$STEPS_SKIPPED_MISSING ====="
# 12 steps: 3 configs x (policy, value, graded bench, frontier bench).
if [ "$STEPS_FAILED" -ne 0 ] || [ "$STEPS_OK" -ne 12 ]; then
    # The grep-able DONE marker must never print over an incomplete run: the
    # 2026-07-27 g32r4 b2 retrain announced DONE over a failed value net and the
    # failure was only noticed when a downstream lane could not find the net.
    echo "NNLAB TWINRT PARTIAL ($STEPS_OK/12 steps complete) -- resubmit the same command to continue"
    exit 1
fi
echo "NNLAB TWINRT DONE"
