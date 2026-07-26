#!/bin/bash
# Concatenate B2 label shards into the dataset the retrain jobs read.
# Mirrors the established convention: scaling/data/g16r6/backward.rust.jsonl is
# byte-exactly the concatenation of its 8 shard files.
#   bash jobs/patterns/b2_labels_merge.sh g16r4 8
set -euo pipefail
CFG="$1"; N="$2"
cd "$(dirname "$0")/../.."
OUT="scaling/data/$CFG/backward_b2.rust.jsonl"
SHARDS=()
for i in $(seq 0 $((N-1))); do
  f="scaling/data/$CFG/backward_b2.rust.shard${i}of${N}.jsonl"
  [ -s "$f" ] || { echo "MISSING $f -- not merging"; exit 1; }
  SHARDS+=("$f")
done
cat "${SHARDS[@]}" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
wc -l "$OUT"
echo "MERGED $CFG from $N shards"
