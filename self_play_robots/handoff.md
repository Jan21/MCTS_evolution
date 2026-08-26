# self_play_robots — handoff (updated 2026-08-26; supersedes the 2026-08-19 version)

Read: `PROBLEM.md` → this → `FINDINGS.md` §1–§26+ → `variants/FINDINGS.md` (lab, entries 1–22)
→ `M6_DESIGN.md`/`M6_FINDINGS.md` (in progress) → `results/status.json`. Report:
`report/selfplay.html` (LOCAL ONLY). Tracker: pe-hy/experiment_tracker (74+ runs, lineage).

## State
- **M0–M5 DONE, M4 PASS, breakthrough achieved and replicated**: the flagship planner
  (v14-recipe nets + v07 hybrid depth-2/3 search) is the planner-of-record:
  graded 231/232 at regret 0.944 (d3: 0.861) — BELOW the pure-subgoal language floor
  (1.17, §3); frontier 177/218; unseen 188/200 (+1.47 vs perfect on the 137 exactly
  labeled). Ceiling break survives same-budget controls, two net families, and
  transfers zero-shot: g32r4 174/175 (17/0 moves p=1.5e-5), g24r8 159/161
  (23/0 p=2.4e-7), g24r8 frontier 276/289 (record). §26 + variants/FINDINGS 19–20.
- **Variants lab COMPLETE** (waves 1–5, ~28/50 nh): adopted v09 strict-value,
  v04 emit-all, v14 stack, v07 hybrid; killed v01/v12/v15/v16 (honest negatives);
  three lessons: train on the graded metric; keep everything explored; search in
  subgoals + slides. Lab in maintenance; two control legs may still be finishing.
- **M6 STARTED**: beyond-oracle self-play at 80×80 (certification only). M6 lead
  agent in Phase 0 feasibility (memory at 6401 tokens, 96-bin clamp check, 80-board
  gen, exam definition, optional 72×72 audit rung). Budget ≤20 nh; Phase 1 needs
  orchestrator green-light.
- **Paper**: `paper/DRAFT.md` + `paper/CLAIMS.md` being drafted (local only).
- **Coordination**: three persistent agents (variants lab = maintenance; tracker/data;
  M6 lead; + paper drafter, report-tables agent). Boundaries in their briefs.
- **Push**: VS Code askpass bridge is down more than up; commits queue locally,
  flush with `VSCODE_GIT_IPC_HANDLE=<owned socket> timeout 170 git push` (see memory:
  sockets can be live-but-slow; try each owned socket, newest-first is NOT reliable).
- Spend ≈50 nh of 1000. Cooling reservation daily 10:00–18:00 persists (≤8 h jobs).
