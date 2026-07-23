# Battle-tested Slurm job patterns (Karolina, July 2026)

Working scripts from the scaling-completion campaign. Before reuse, update the
`BUNDLE=`/`SV=` path variables at the top of each (they point at the retired
`karolina_bundle/` working copy; the working repo is now
`/scratch/project/open-37-42/petrhyner/MCTS_evolution`), and keep walltimes at
24:00:00 (charge is elapsed-only; short walltimes killed near-complete chunks).
`lanes_one.slurm` is the sharded-eval workhorse: idempotent 15-instance chunks,
optional shard-glob argument for disjoint sub-jobs, auto-merge (via
eval/merge_compare_shards.py) by whichever job finishes its lane last.
