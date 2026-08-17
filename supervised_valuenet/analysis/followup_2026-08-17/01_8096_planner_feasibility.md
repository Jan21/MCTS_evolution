# Feasibility brief — downstream planner at 80×80 / 96×96 (2026-08-17)

Read-only code study (Opus agent, this session). Decides FINDINGS §72c's open
question: can the planner stack train/eval above the 64×64 oracle envelope?

## Verdict

- **80×80: feasible-with-changes** (~1.5–2.5 nh) but **statistically doomed** —
  see risks. **96×96: infeasible in fp32 on A100-40GB** (51.7 GB at batch-1,
  single candidate); needs bf16/SDPA rework (untested paths).
- **Decision taken: no compute.** The citable finding is architectural (below);
  an experiment would produce uninterpretable numbers.

## The deciding facts (file:line verified)

1. **The planner nets are NOT size-free.** Both carry a learned n²-sized
   positional table: `train/looped_pc.py:146` and `train/policy_tf.py:116` —
   `self.pos = nn.Parameter(torch.randn(grid*grid+1, d_model))`. The labeler
   removed exactly this (`nn_labeler/model.py:4`, `:110` — "a checkpoint of
   this net is valid at any n"). At n=80 the PE is 1.23 M params (~50% of the
   net) fit from ~190 training decisions (~6,500 params/example →
   memorization guaranteed). Warm-start impossible: strict `load_state_dict`
   fails on the pos shape.
2. **T = n²+1 tokens; attention scores materialized densely** at
   O(M·T²) (`looped_pc.py:123-126`; no flash/SDPA). Validated memory model
   ≈608·M·K² bytes reproduces the observed 13 GB/step OOM at g32r4. 80×80
   fits only at M=1; 96×96 not at all (fp32).
3. **M=1 silently disables the ranking loss** (`looped_pc.py:175-183` —
   log_softmax over a 1-candidate group ≡ 0) → not the same recipe as the
   other cells; a confound, not just a memory workaround.
4. **Eval path can't load lean boards** (`eval/compare.py:340` uses
   `GridEnv.from_env` → empty reachability on lean pkls) and has **no
   `torch.no_grad()`** (`eval/end2end.py:79-107`) → ~125 GB retained autograd
   per value call at n=80. CPU eval ~22 h/instance at 1200 expansions.
5. **Corpus stats:** g80 1,350 records / 272 groups (81% depth-0); g96
   1,031 / 223. `cost_to_go` reaches 76 → default 50 value bins would
   silently clamp 27% of the label range (`--num-classes 96` mandatory).

## What an experiment could/could not claim

Could: existence ("trains at 6,401 tokens, at the memory envelope"),
label-relative self-consistency, certified solve rate on 3 held-out boards.
Could NOT: anything about optimality, the fidelity dose-response, scaling
(fresh non-transferable PE per size = no continuity with the 17–64 curve).

## The publishable line (zero compute)

The labeler generalizes beyond its training sizes because it has no
positional table; the planner cannot, for a one-line architectural reason
(`looped_pc.py:146`). The 80/96 corpora remain what FINDINGS §72c scoped
them as: certified labels whose warranty is the flat 17–64 curve.
