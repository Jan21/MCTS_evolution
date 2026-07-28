#!/bin/bash
# Pipeline driver v2 (2026-07-28, successor session). Carries the cap-20000
# arm to completion: base shards -> merge -> QC -> retrain -> bank -> Track 1,
# plus g24r8/g32r4 retrain -> bank -> Track 1, plus the g32r4 cap-5000
# contrast lanes once its (independent) retrain completes.
#
# v1's fatal gate is fixed here: v1's retrain_done() fired on the EXISTENCE of
# a Lightning checkpoint file, which appears minutes into value training. It
# banked g16r8 mid-training at 11:11 -- the manifest pointed at
# epoch=1-step=4166.ckpt, a file save_top_k later deleted. Every transition
# below is gated on `sacct` reporting COMPLETED for a recorded job id, and
# banking happens only after that. (FINDINGS: the house rule is "a completion
# marker must be gated on the thing it claims".)
set -uo pipefail
module purge 2>/dev/null
module load Python/3.11.5-GCCcore-13.2.0 bzip2/1.0.8-GCCcore-13.2.0 2>/dev/null
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd /scratch/project/open-37-42/petrhyner/MCTS_evolution/supervised_valuenet
export PYTHONPATH=. OMP_NUM_THREADS=2
MAN=scaling/runs/b2_banked_cap20000.json

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
state() { sacct -j "$1" --format=State --noheader -X 2>/dev/null | head -1 | tr -d ' '; }

# --- live-job registry (2026-07-28 22:00) ---
SHARD_BASE_ID=4598238                    # base shard s<i> = SHARD_BASE_ID + i
LABELS_G32R4=4598201                     # g32r4 cap-20000 label generation
CONTRAST_G32R4=4598168                   # g32r4 cap-5000 retrain (contrast arm)
declare -A RETRAIN=( [g24r8]=4599232 )   # g16r4/g32r4 added when submitted
declare -A T1SENT=()
CONTRAST_STATE=waiting                   # waiting | sent | lost
BASE_ABORT=""

all_base_shards_done() {
  local i f jid s
  for i in $(seq 0 31); do
    f="scaling/data/g16r4/backward_b2.cap20000.rust.shard${i}of32.jsonl"
    jid=$((SHARD_BASE_ID + i))
    s=$(state "$jid")
    if [ "$s" != COMPLETED ] || [ ! -s "$f" ]; then
      case "$s" in FAILED|TIMEOUT|CANCELLED*|NODE_FAIL)
        log "SHARD PROBLEM: s$i job $jid state=$s -- base merge blocked, intervene" ;;
      esac
      return 1
    fi
  done
  return 0
}

qc_share() {  # prints the by-reference share (float, percent) of a label file
  python3 scaling/qc_byref.py "$1" | tee -a "$0.qc.log" \
    | sed -n 's/.*= \([0-9.]*\)%.*/\1/p'
}

bank_and_verify() {  # bank_and_verify <cfg> -> 0 if cfg banked with real files
  python3 -m scaling.bank_b2 --label-set cap20000 --write --out "$MAN" \
    >> "$0.bank.log" 2>&1
  python3 - "$1" <<'PY'
import json, os, sys
cfg = sys.argv[1]
m = json.load(open("scaling/runs/b2_banked_cap20000.json"))
e = m.get(cfg) or sys.exit(1)
sys.exit(0 if os.path.exists(e["policy"]) and os.path.exists(e["value"]) else 1)
PY
}

submit_t1() {  # submit_t1 <cfg>  (cap-20000 lanes, manifest-suffixed outputs)
  if [ "$1" = g16r4 ]; then
    sbatch --export=ALL,BANK_MANIFEST=$MAN -J rr-t1-base-cap20k \
      jobs/patterns/track1_rows.slurm base base450
  else
    sbatch --export=ALL,BANK_MANIFEST=$MAN -J "rr-t1-$1-cap20k-grd" \
      jobs/patterns/track1_rows.slurm "$1" graded
    sbatch --export=ALL,BANK_MANIFEST=$MAN -J "rr-t1-$1-cap20k-frn" \
      jobs/patterns/track1_rows.slurm "$1" frontier
  fi
}

for round in $(seq 1 2000); do
  # --- base: 32 shards -> merge -> QC gate -> retrain ---
  if [ -z "${RETRAIN[g16r4]:-}" ] && [ -z "$BASE_ABORT" ]; then
    if [ ! -s scaling/data/g16r4/backward_b2.cap20000.rust.jsonl ]; then
      if all_base_shards_done; then
        log "merging 32 base cap-20000 shards"
        bash jobs/patterns/b2_labels_merge.sh g16r4 32 20000 || BASE_ABORT=mergefail
      fi
    fi
    if [ -s scaling/data/g16r4/backward_b2.cap20000.rust.jsonl ] && [ -z "$BASE_ABORT" ]; then
      SH=$(qc_share scaling/data/g16r4/backward_b2.cap20000.rust.jsonl)
      log "base QC by-reference share: ${SH}% (expect ~11; ~5 = depleted corpus)"
      if python3 -c "import sys; sys.exit(0 if float('${SH:-0}') >= 8.0 else 1)"; then
        JID=$(sbatch --parsable -J rr-b2rt-g16r4-cap20k \
              jobs/patterns/b2_retrain_one.slurm g16r4 cap20000)
        RETRAIN[g16r4]=$JID; log "base retrain submitted: $JID"
      else
        BASE_ABORT=qcfail
        log "BASE QC ABORT: share ${SH}% < 8% -- NOT retraining, intervene"
      fi
    fi
  fi

  # --- g32r4: labels COMPLETED -> QC -> retrain ---
  if [ -z "${RETRAIN[g32r4]:-}" ] && [ "$(state $LABELS_G32R4)" = COMPLETED ] \
     && [ -s scaling/data/g32r4/backward_b2.cap20000.rust.jsonl ]; then
    SH=$(qc_share scaling/data/g32r4/backward_b2.cap20000.rust.jsonl)
    log "g32r4 QC by-reference share: ${SH}%"
    JID=$(sbatch --parsable -J rr-b2rt-g32r4-cap20k \
          jobs/patterns/b2_retrain_one.slurm g32r4 cap20000)
    RETRAIN[g32r4]=$JID; log "g32r4 retrain submitted: $JID (QC ${SH}%)"
  fi
  case "$(state $LABELS_G32R4)" in FAILED|TIMEOUT|NODE_FAIL)
    [ -z "${RETRAIN[g32r4]:-}" ] && log "g32r4 LABELS DIED -- intervene" ;;
  esac

  # --- each retrain: COMPLETED -> rebank -> verify -> Track 1 ---
  for c in "${!RETRAIN[@]}"; do
    [ -n "${T1SENT[$c]:-}" ] && continue
    s=$(state "${RETRAIN[$c]}")
    case "$s" in
      COMPLETED)
        if bank_and_verify "$c"; then
          submit_t1 "$c" && T1SENT[$c]=1 && log "Track 1 submitted: $c cap20000"
        else
          log "BANK FAILED for $c after COMPLETED retrain -- intervene"
          T1SENT[$c]=bankfail
        fi ;;
      FAILED|TIMEOUT|NODE_FAIL|CANCELLED*)
        log "RETRAIN DIED: $c job ${RETRAIN[$c]} state=$s -- intervene"
        T1SENT[$c]=dead ;;
    esac
  done

  # --- g32r4 cap-5000 contrast: retrain COMPLETED -> default bank -> lanes ---
  if [ "$CONTRAST_STATE" = waiting ]; then
    s=$(state $CONTRAST_G32R4)
    case "$s" in
      COMPLETED)
        python3 -m scaling.bank_b2 --label-set "" --write >> "$0.bank.log" 2>&1
        if python3 -c "
import json,os,sys
e=json.load(open('scaling/runs/b2_banked.json')).get('g32r4') or sys.exit(1)
sys.exit(0 if os.path.exists(e['policy']) and os.path.exists(e['value']) else 1)"; then
          sbatch -J rr-t1-g32r4-grd jobs/patterns/track1_rows.slurm g32r4 graded
          sbatch -J rr-t1-g32r4-frn jobs/patterns/track1_rows.slurm g32r4 frontier
          CONTRAST_STATE=sent; log "g32r4 cap-5000 contrast lanes submitted"
        else
          CONTRAST_STATE=lost; log "g32r4 contrast bank failed -- intervene"
        fi ;;
      FAILED|TIMEOUT|NODE_FAIL|CANCELLED*)
        CONTRAST_STATE=lost
        log "g32r4 cap-5000 retrain died ($s) -- contrast arm lost; do NOT bank default mode (last ckpt is mid-training)" ;;
    esac
  fi

  n_t1=0; for c in g16r4 g24r8 g32r4; do [ -n "${T1SENT[$c]:-}" ] && n_t1=$((n_t1+1)); done
  log "heartbeat: t1sent=${!T1SENT[*]:-none} contrast=$CONTRAST_STATE base_abort=${BASE_ABORT:-no} retrains=${!RETRAIN[*]}"
  if [ "$n_t1" -eq 3 ] && [ "$CONTRAST_STATE" != waiting ]; then
    log "PIPELINE_V2_COMPLETE"; break
  fi
  sleep 300
done
log "PIPELINE_V2_EXIT t1=${!T1SENT[*]:-none} contrast=$CONTRAST_STATE"
