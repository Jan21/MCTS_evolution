#!/bin/bash
# Seed-study follow-through: as each rr-seed-g16r6-s* retrain COMPLETES,
# write its per-seed manifest and submit its Track 1 rows. Same gating rule
# as pipeline_driver_v2.sh: sacct COMPLETED on the recorded job id, never
# checkpoint existence (Lightning writes checkpoints mid-training).
set -uo pipefail
module purge 2>/dev/null
module load Python/3.11.5-GCCcore-13.2.0 bzip2/1.0.8-GCCcore-13.2.0 2>/dev/null
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd /scratch/project/open-37-42/petrhyner/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
state() { sacct -j "$1" --format=State --noheader -X 2>/dev/null | head -1 | tr -d ' '; }

declare -A JOB=( [21]=4599909 [37]=4599910 [53]=4599911 )
declare -A SENT=()

for round in $(seq 1 2000); do
  for s in 21 37 53; do
    [ -n "${SENT[$s]:-}" ] && continue
    st=$(state "${JOB[$s]}")
    case "$st" in
      COMPLETED)
        MAN="scaling/runs/b2_banked_cap20000_seed${s}.json"
        if python3 - "$s" "$MAN" <<'PY'
import glob, json, sys
seed, man = sys.argv[1], sys.argv[2]
entry = {}
for sysname, key in (("policy", "policy"), ("value", "value")):
    pat = (f"scaling/runs/g16r6/backward-{sysname}-b2-cap20000-seed{seed}/"
           "lightning_logs/version_*/checkpoints/*.ckpt")
    cks = sorted(glob.glob(pat))
    if len(cks) != 1:
        sys.exit(f"REFUSE: {len(cks)} ckpts for {pat} (want exactly 1)")
    entry[key] = cks[0]
json.dump({"g16r6": entry}, open(man, "w"), indent=1)
print(f"wrote {man}: {entry}")
PY
        then
          sbatch --export=ALL,BANK_MANIFEST="$MAN" -J "rr-t1-g16r6-c20ks${s}-grd" \
            jobs/patterns/track1_rows.slurm g16r6 graded
          sbatch --export=ALL,BANK_MANIFEST="$MAN" -J "rr-t1-g16r6-c20ks${s}-frn" \
            jobs/patterns/track1_rows.slurm g16r6 frontier
          SENT[$s]=1; log "seed $s: manifest + Track 1 lanes submitted"
        else
          SENT[$s]=refused; log "seed $s: manifest REFUSED -- intervene"
        fi ;;
      FAILED|TIMEOUT|NODE_FAIL|CANCELLED*)
        SENT[$s]=dead; log "seed $s retrain died ($st) -- intervene" ;;
    esac
  done
  [ "${#SENT[@]}" -ge 3 ] && { log "SEED_FOLLOWUP_COMPLETE ${SENT[*]}"; break; }
  log "heartbeat: sent=${!SENT[*]:-none}"
  sleep 600
done
log "SEED_FOLLOWUP_EXIT"
