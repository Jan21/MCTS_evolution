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

2. **Wave 1 submitted (2026-08-20 09:23).** Jobs (qgpu, 1 GPU each):
   baselines 4719424 (3 h), v00_control 4719425 (5 h), then after v00:
   v01 4719426, v04 4719427, v06 4719428, v08 4719429, v09 4719430,
   v12 4719431 (5 h each). Projected ≈2.3 nh actual (≈4.8 nh walltime
   ceiling; program cap 50 nh). Per-arm results land in
   `results/variants/<vid>/` (bench/gate/manifest JSONs) and render in the
   report's Variants tab ("pending" until then). Verdict entries follow as
   arms complete; any WIN is replicated at a second seed before being
   claimed.

3. **Wave-1 interim: unseen-bench bug found and fixed; baselines + control
   landed (2026-08-20).** (a) BUG: the runner's unseen bench ran without
   `--boards lean` (a text patch had silently not matched), so lean exam pkls
   went through `GridEnv.from_env`, whose reconstructed reachability matrix
   lacks entries for arbitrary endpoints -> KeyError in `_initial_plan`,
   all 25 chunks rc=1 (job 4719425's unseen leg; log
   `runs/spr/var_v00_control_unseen.46ba792d0401/chunk.000.log`). Fixed with
   an assert-verified patch (commit in tree); bench-only fix-up jobs
   4722745-4722751 resubmitted (runner is idempotent: gen/train/graded/
   frontier skip). Lesson recorded: verify patch application, not just
   patch-script exit. (b) Unseen-exam BASELINES (job 4719424,
   `results/variants/baselines/`): frozen seed nets 175/200, 14.49 mv,
   172 exp; supervised per-size backward pair 134/200, 14.78 mv, 30 exp.
   The seed nets' +41-solve margin on fresh boards says the self-play line's
   generalization edge over the supervised planner is large before any
   variant runs; the variants' bar is 175/200. (c) CONTROL (job 4719425):
   generation 320 instances / 289 solved / 3,359 certified records (1,900 s;
   5 OOM-dropped + 1 timeout, within the loop's norm); graded A* 228/232
   (regret 2.39, 33.8 exp) = the loop's ceiling level; frontier A* 160/218
   -- one standard iteration from the mix_b2mix_iter2 nets at g24r4-only,
   NOTE: above the 24-only B2 loop's best (158, main FINDINGS §19) --
   the mixed-curriculum warm start itself is worth frontier solves.
   Sources: `results/variants/{baselines,v00_control}/`, jobs
   4719424/4719425.
