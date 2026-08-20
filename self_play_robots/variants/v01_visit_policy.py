"""Policy targets from MCTS visit counts (AlphaZero's pi ~ N^(1/tau)) instead of
the labeler's cost-softmax.

The stack trains the policy on soft targets softmax(-cost_to_go/temp) -- a
DESCENT-style target that needs certified costs for every candidate and ignores
what the search actually preferred. AlphaZero's classic target is the root visit
distribution: it encodes the search's full exploration/exploitation judgement,
exists for every expanded decision (even ones with < 2 certified candidates),
and is the textbook "policy improvement operator" signal. Records already carry
`visits` (spr.selfplay provenance), so this is a pure train-side change.

Mechanism: patch `spr.nets.policy_meta` to keep each candidate's visit count and
`SizeFreePolicyNet._soft_targets` to build the bn/sup/helper target cascade from
pi = N^(1/tau)/sum (tau=1) whenever visits are present (falls back to the
cost-softmax for exact-corpus records, which have no visits).
"""
from variants import Variant

VARIANT = Variant(
    vid="v01_visit_policy",
    axis="targets",
    title="Visit-count policy targets (AlphaZero pi)",
    hypothesis="Visit-distribution targets carry more of the search's ranking "
               "than cost-softmax over certified candidates and improve the "
               "policy's top-k on unseen boards.",
    mechanism="Train-side hook: policy soft-targets = normalized N^(1/tau) over "
              "the decision's candidates (tau=1), cost-softmax fallback when no "
              "visits (exact corpora).",
    expected_failure="Visits at 300-expansion budgets are noisy and biased "
                     "toward the tree's first discoveries -> flat or worse "
                     "val_regret and no bench movement.",
    hooks=("train",),
)


def apply_train():
    import torch
    import torch.nn.functional as F
    from spr import nets

    _orig_meta = nets.policy_meta

    def policy_meta_visits(group, n, byref=False):
        m = _orig_meta(group, n, byref=byref)
        if m is None:
            return None
        # rebuild the candidate list the same way to align visit counts
        from spr.nets import flat
        g0 = group[0]
        hidx = {tuple(h[0]): i for i, h in enumerate(g0["helpers"])}
        cidx = {h[1]: i for i, h in enumerate(g0["helpers"])}
        vis = {}
        for r in group:
            hi = hidx.get(tuple(r["cand_helper"][0]))
            if hi is None and byref:
                hi = cidx.get(r["cand_helper"][1])
            if hi is None:
                continue
            key = (flat(r["cand_bottleneck"], n), flat(r["cand_support"], n), hi)
            v = r.get("visits")
            if v is not None:
                vis[key] = max(int(v), 0)
        m["visit_map"] = vis if vis and sum(vis.values()) > 0 else None
        return m

    nets.policy_meta = policy_meta_visits

    _orig_targets = nets.SizeFreePolicyNet._soft_targets

    def _soft_targets_visits(self, m, dev):
        vis = m.get("visit_map")
        if not vis:
            return _orig_targets(self, m, dev)
        cands = m["cands"]
        w = torch.tensor([float(vis.get((c[0], c[1], c[2]), 0)) for c in cands],
                         dtype=torch.float, device=dev)
        if w.sum() <= 0:
            return _orig_targets(self, m, dev)
        w = w / w.sum()
        vbn, tbn, tsp = m["valid_bn"], m["tgt_bn"], m["tgt_sup"]
        sup_list = m["sup_by_bn"][tbn]
        bn_t = torch.zeros(len(vbn), device=dev)
        sup_t = torch.zeros(len(sup_list), device=dev)
        help_t = torch.zeros(len(m["helper_cells"]), device=dev)
        for (bn, sp, hi_, _), wc in zip(cands, w):
            bn_t[vbn.index(bn)] += wc
            if bn == tbn and sp in sup_list:
                sup_t[sup_list.index(sp)] += wc
                if sp == tsp:
                    help_t[hi_] += wc
        z = lambda t: t / t.sum() if float(t.sum()) > 0 else torch.full_like(t, 1.0 / len(t))
        return z(bn_t), z(sup_t), z(help_t)

    nets.SizeFreePolicyNet._soft_targets = _soft_targets_visits
