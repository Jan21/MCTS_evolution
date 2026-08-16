# Reproducibility & code-health memo

**(1) Could a stranger clone this and regenerate the headline table? Partially.**
Code, the base 88MB `nn/data/combined.jsonl` labeled dataset, and 280 result
JSON/JSONL files are committed to git; `requirements.txt` and per-module
READMEs (root, `supervised_valuenet/`, `rust_datagen/`, `scaling/`) give real
commands. But three things are gitignored and unrecoverable from git alone:
(a) all `environments*/` board pickles (2GB+, 1295+ boards) — the README's
recovery path is `ssh jonathan '...'`, a private host outside the repo; (b)
every trained checkpoint (`*.ckpt`, `checkpoints_backward/`,
`nn_labeler/banked/*.ckpt` — note `banked/prod_v1_s11.ckpt` (11.8MB) is
absent from git while its `.json` metadata sidecar is tracked); (c) the
report builder (`eval/build_report.py`) hardcodes a Karolina venv path
(`/scratch/project/open-37-42/petrhyner/venv`). A stranger can retrain nets
(~1h/net, A100, documented) but cannot rebuild the exact scaling-study boards
or reproduce the exact checkpoint the headline numbers cite without that ssh
access. Worse: `git status` shows **137 untracked result JSON/JSONL files**
in `nn_labeler/results/` right now (recent ladder/coarse/gate/beyond-UNVERIFIABLE
runs) vs 280 tracked — recent headline-adjacent numbers are sitting only on
`/scratch`, which auto-purges after 90 days of inactivity, with no evidence
of `/mnt/proj` archival.

**(2) Hand-patched/one-off parts.** FINDINGS.md item 51 documents a
hand-picked checkpoint for g32r4 cap-20000 (`bank_b2.py` refused to
auto-select between two legitimate candidates; a human picked the
marker-backed one, recorded via a `_hand_banked` note in the manifest) — well
audited, but not a rerunnable script step. `eval/build_report.py` does
self-verify every headline number against registered sources, which is
good practice, but it can only verify what's checked in.

**(3) Next steps (release checklist), roughly ordered:**
1. **Commit or archive the 137 untracked result files now** (small) — before
   the 90-day scratch purge; `git add` + push.
2. **Replace the `ssh jonathan` board-fetch instruction** with an in-repo
   generator invocation or a hosted archive/DOI (medium) — `nn.gen_grids` can
   regenerate structurally, but exact pinned boards need one authoritative
   source.
3. **Publish checkpoints** (headline value/policy nets + the banked g32r4
   pick) to a durable store (Zenodo/HF Hub) with a manifest mapping
   checkpoint -> FINDINGS item -> exact training command (medium).
4. **Freeze one `env.yml`/lockfile** (exact torch/lightning/CUDA versions,
   not just `>=`) pinned to what actually produced the headline numbers
   (small).
5. **Make `bank_b2.py`'s human tie-break machine-checkable** — script the
   decision rule it currently defers to a person, or require a signed
   rationale file alongside every hand-banked pick (medium).
