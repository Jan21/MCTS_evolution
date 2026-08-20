"""TD(lambda)-style value targets: blend the certified cost-to-go with the
generation net's own estimate (pure self-labels; no oracle involved).

The loop's value labels are `plan_cost(certified) - fixed_g` -- an outcome
target. Outcome targets are unbiased about the plan the search FOUND but noisy
about the plan the state DESERVES (the search's completion can be suboptimal,
which §19b measured as regret drift). MuZero/EfficientZero mix outcomes with
bootstrapped estimates to cut this variance. Records already carry `ctg_hat`
(the net's estimate at generation time), so the blend is a load-time rewrite:

    cost_to_go' = round(LAM * certified + (1 - LAM) * clamp(ctg_hat, 0, 95))

LAM=0.7. `is_optimal` is left as the search's argmin (policy targets unchanged);
only the value distribution's target bin moves.
"""
from variants import Variant

LAM = 0.7

VARIANT = Variant(
    vid="v02_td_blend",
    axis="targets",
    title="Bootstrapped value targets (lambda=0.7 blend)",
    hypothesis="Blending certified outcomes with the net's own bootstrap cuts "
               "label variance from suboptimal completions and lowers bench "
               "regret at unchanged solve rate.",
    mechanism="Train-side hook: dataset.load_corpus post-filter rewrites "
              "cost_to_go on self-play records (label_source=spr_mcts, ctg_hat "
              "present) to 0.7*certified + 0.3*bootstrap.",
    expected_failure="Bootstrap reinforces the net's own biases (self-"
                     "confirmation) -> value drifts optimistic, solve rate "
                     "drops on the frontier.",
    hooks=("train",),
    status="parked",  # incremental target smoothing (owner 2026-08-20: no knob tweaking)
)


def apply_train():
    from nn_labeler import dataset

    _orig = dataset.load_corpus

    def load_corpus_blend(path, config, n, env_dir, limit=None):
        recs = _orig(path, config, n, env_dir, limit=limit)
        hit = 0
        for r in recs:
            if r.get("label_source") == "spr_mcts" and r.get("ctg_hat") is not None:
                cert = float(r["cost_to_go"])
                boot = min(max(float(r["ctg_hat"]), 0.0), 95.0)
                r["cost_to_go"] = int(round(LAM * cert + (1.0 - LAM) * boot))
                hit += 1
        if hit:
            print(f"[v02_td_blend] rewrote cost_to_go on {hit}/{len(recs)} records "
                  f"(lambda={LAM})", flush=True)
        return recs

    dataset.load_corpus = load_corpus_blend
