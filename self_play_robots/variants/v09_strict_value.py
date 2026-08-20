"""Train the value net on the BENCHMARK metric: realized strict moves-to-go
instead of abstract plan-cost units.

Structurally novel (owner directive 2026-08-20): the entire stack -- supervised
and self-play alike -- trains the value on ABSTRACT plan cost (segments +
support fixes), then evaluates on REALIZED STRICT MOVES. The two diverge
exactly where plans are subtle (support reuse, park repairs: §3 measured the
divergence as the language regret). Nobody has ever trained on the metric
itself. Self-play makes it possible for the first time: every certified record
carries the full plan's strict realization (`strict_total`), which the
supervised exact corpora do not have.

Mechanism (train-side hook): rewrite each self-play record's target to

    ctg_strict = cost_to_go * strict_total / abstract_total

-- the decision's abstract cost-to-go rescaled by the certified plan's global
strict/abstract ratio. This is proportional attribution (a per-decision strict
decomposition does not exist: strict realization is a global property of the
plan), so within-group RANKINGS are preserved while the value HEAD learns the
strict scale; across groups and depths the net now regresses toward realized
moves. Search then estimates f in strict units end to end (certified terminals
already score strict), removing the §19b unit tension where estimates are
abstract but the metric is not. `is_optimal` (policy target) is untouched.
"""
from variants import Variant

VARIANT = Variant(
    vid="v09_strict_value",
    axis="targets",
    title="Strict-moves value targets (train on the metric)",
    hypothesis="A value net regressing realized strict moves ranks plans by "
               "what the bench actually scores, cutting realized moves on "
               "shared solves -- the axis every previous iteration left flat.",
    mechanism="Train-side hook: cost_to_go *= strict_total/abstract_total on "
              "self-play records (proportional strict attribution; clamped to "
              "[0, 95]).",
    expected_failure="The global ratio is too coarse per decision -> extra "
                     "label noise with no ranking change; or strict-scale "
                     "estimates break A*'s admissibility-ish ordering and "
                     "solve rate drops.",
    hooks=("train",),
)


def apply_train():
    from nn_labeler import dataset

    _orig = dataset.load_corpus

    def load_corpus_strict(path, config, n, env_dir, limit=None):
        recs = _orig(path, config, n, env_dir, limit=limit)
        hit = skip = 0
        for r in recs:
            if r.get("label_source") != "spr_mcts":
                continue
            st, ab = r.get("strict_total"), r.get("abstract_total")
            if not st or not ab or ab <= 0:
                skip += 1
                continue
            r["cost_to_go"] = int(round(min(max(
                float(r["cost_to_go"]) * float(st) / float(ab), 0.0), 95.0)))
            hit += 1
        if hit or skip:
            print(f"[v09_strict_value] rescaled {hit} records to strict units "
                  f"({skip} lacked strict/abstract totals)", flush=True)
        return recs

    dataset.load_corpus = load_corpus_strict
