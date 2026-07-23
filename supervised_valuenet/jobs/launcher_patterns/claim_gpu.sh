#!/bin/bash
# claim_gpu.sh <tag>: prints a free 40GB GPU index. Global cap: 2 concurrent claims.
TAG=$1; LOCKDIR=/tmp/claude-1010/gpu_claims; mkdir -p $LOCKDIR
while true; do
  NCLAIMS=$(ls -d $LOCKDIR/gpu* 2>/dev/null | wc -l)
  if [ "$NCLAIMS" -lt 2 ]; then
    for G in $(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | awk -F', ' '$2+0 < 1000 && $3+0 > 30000 {print $1}'); do
      if mkdir $LOCKDIR/gpu$G 2>/dev/null; then echo $TAG > $LOCKDIR/gpu$G/owner; echo $G; exit 0; fi
    done
  fi
  sleep 600
done
