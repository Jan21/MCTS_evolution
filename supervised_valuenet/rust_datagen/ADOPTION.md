# Adoption decision — Rust data generation engine

Date: 2026-07-15. Decision maker: the orchestrating session, under the project's
standing rules. Verdict: **ADOPTED for production data generation**, with the following
adjudication of VERIFICATION.md §4's blocker.

## The adjudication

The §5.11 zero-threshold gate measured 82/1,650 instances (≈5%) whose record streams
diverge in label content between engines. The verification proved the root cause is
that the **reference itself is under-determined**: at equal-score cut points the Python
labeler's outputs are a function of CPython's set-iteration order (Python's own labels
change across board representations), and the inner plan search's cost estimate is not
admissible (VERIFICATION §4 shape 2, committed repro), so its returned completion can
depend on heap tie order in either engine.

Given that:
1. per-decision labels on pinned contexts are exact (0 diffs / 258,999 labels, gate 3);
2. the field training actually consumes diverges 0 / 90,130 matched pairs (§5.11(i));
3. no canonical Python dataset exists to match (Python self-disagreement is proven);
4. the original port contract explicitly declared trajectory-level tie divergence out
   of scope — the strict-zero framing of §5.11 was an additional, stronger probe;

the measured divergence class is **accepted for production**. Training data from the
Rust engine is drawn from the same instance distribution with labels exactly as correct
as the Python labeler's for the trajectories generated — and in the one root-caused
value-divergence class, more correct.

## Obligations attached to this adoption

- The non-admissible-estimate finding is disclosed in the project's FINDINGS.md (it
  affects Python-era labels equally; measured incidence 12/71,200 records ≈ 0.017%).
- Any future work requiring a canonical order-free reference (e.g. exact cross-engine
  dataset reproduction) must implement VERIFICATION §4's option (b): content-based
  tie-breaks in the candidate sort, the plan-search heap, and dependent-support
  resolution — in BOTH engines — and regenerate reference corpora.
- Mixed-engine training sets are allowed (the divergence class is tie-content only),
  but each data file records its engine in the sidecar manifest for provenance.
