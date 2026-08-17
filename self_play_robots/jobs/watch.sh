#!/bin/bash
# Event stream for the orchestrator's Monitor: queue-state changes of all spr-*
# jobs + new milestone lines in runs/spr/spr-*.out (DONE/FAILED/PARITY/GATE/...).
D=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr
prev=""
while true; do
  cur=$(squeue --me -h -o "%i %j %T %R" 2>/dev/null | grep " spr-" | sort | tr '\n' ';')
  if [ "$cur" != "$prev" ]; then echo "QUEUE: ${cur:-none-left}"; prev="$cur"; fi
  for f in "$D"/spr-*.out; do
    [ -f "$f" ] || continue
    seen="$f.seen"; touch "$seen"
    grep -E "PARITY|DONE|FAILED|GATE|Traceback|Error|OOM|Killed|WARN" "$f" 2>/dev/null \
      | grep -v "SPR ITER PHASE" | grep -v -x -F -f "$seen" > "$f.new" 2>/dev/null
    if [ -s "$f.new" ]; then sed "s|^|$(basename "$f"): |" "$f.new"; cat "$f.new" >> "$seen"; fi
  done
  sleep 120
done
