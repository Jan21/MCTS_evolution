# Variants lab — FINDINGS (results log)

Own numbering from 1 (max+1); every entry cites result files and Slurm job ids.
Protocol and portfolio rationale: `variants/DESIGN.md`. The orchestrator's main
log (`self_play_robots/FINDINGS.md`) cross-references this file; nothing here
is duplicated there.

---

1. **Lab opened; framework, unseen exam, and wave 1 built and smoke-tested
   (2026-08-20, login-node CPU only).** Registry/hook framework
   (`variants/__init__.py`; `--variant` plumbed into `spr.selfplay` [worker-side,
   ctx-passing], `spr.train`, `spr.bench`; `spr.arena --env-dir` for
   off-config board dirs), pinned unseen exam (200 instances on 50 fresh
   24×24 boards, ids 20000+, `results/variants/exam/`), runner
   (`jobs/variant_iter.slurm`), baselines job (`jobs/variant_baselines.slurm`),
   report tab (`report/gen_report.py` "Variants lab"). Wave-1 portfolio after
   the owner's "breakthrough, not knob-tweaking" directive: v00 control,
   v01 visit-policy targets, v04 deep-emit (the one knob sanity control),
   v06 Gumbel root + sequential halving (own `mcts_gumbel`), v08 cold-start
   (bootstrap control), v09 strict-moves value targets (train on the metric),
   v12 frontier-mining curriculum; v02/v03/v05 parked as incremental,
   v07 hybrid action space = documented stub (wave 2). CPU smokes: unseen
   bench path via lean boards + RR_ENV_DIR (2/2 solved), v06 and v12
   generation hooks (records produced, hooks logged), v01 and v09 train hooks
   (metrics finite; v09 rescaled 80/80 records to strict units). One real bug
   found and fixed by the smokes: under `python -m spr.selfplay` the live
   module is `__main__`, so a hook importing `spr.selfplay` saw a second,
   empty module — worker state is now passed to hooks explicitly
   (`apply_phase(..., ctx=_W)`).
   Sources: `variants/*.py`, `jobs/variant_*.slurm`.

2. **Wave 1 submitted (2026-08-20).** 7 arms × ~2.5 h (5 h walltime each) + 1
   baselines job (3 h walltime) ≈ projected 2.3 nh actual / 4.8 nh walltime
   ceiling (program cap 50 nh). Job ids in the manifest below (filled at
   submission). Per-arm results land in `results/variants/<vid>/` as
   bench/gate/manifest JSONs and render in the report's Variants tab
   ("pending" until then). Verdict entries follow as arms complete.
