"""The value-network benchmark: fixed splits + framework-agnostic metrics.

Two generalization axes:
  - cross-board: val / test live on graphs never seen in training.
  - iid (optional): held-out instances on the training graphs (generate with a
    different seed into a separate file).

Splits are by graph id so a model cannot memorize board geometry. Metrics take
plain Python predictions (no torch dependency here) so any architecture plugs in.

    from nn.benchmark import SPLITS, load, accuracy, value_mae, top1_optimal
"""
from __future__ import annotations

import json
from pathlib import Path

from nn.labels import to_class, expected_value


# Large generated val/test to tame top1_optimal variance (16 boards was too few).
# Test keeps the canonical stock held-out boards plus generated ones.
SPLITS = {
    "train": list(range(0, 96)) + list(range(1000, 1800)),
    "val":   list(range(1800, 2400)),                       # ~600 generated boards
    "test":  list(range(112, 128)) + list(range(2400, 3000)),
}


def load(path):
    return [json.loads(line) for line in open(path)]


def by_split(records, split):
    ids = set(SPLITS[split])
    return [r for r in records if r["env_id"] in ids]


def group_by_decision(records):
    """Group candidate records that belong to the same decision state.

    A decision is identified by (env_id, instance, segment). Useful for the
    policy/selection metric: did the model rank the optimal candidate first?
    """
    groups = {}
    for r in records:
        key = (r["env_id"], tuple(r["target"]),
               tuple(map(tuple, [r["target_robot"][0]])),
               tuple(r["seg_start"]), tuple(r["seg_end"]), r["depth"])
        groups.setdefault(key, []).append(r)
    return list(groups.values())


# -- metrics (predictions are scalar cost-to-go per record) -------------------

def _clip(x, num_classes):
    return max(0.0, min(float(x), num_classes - 1))


def value_mae(records, pred_scalar, num_classes=None):
    """Mean absolute error of predicted cost-to-go (clipped to the class range)."""
    def t(x):
        return _clip(x, num_classes) if num_classes else x
    errs = [abs(t(pred_scalar[i]) - t(r["cost_to_go"])) for i, r in enumerate(records)]
    return sum(errs) / len(errs) if errs else float("nan")


def accuracy(records, pred_scalar, num_classes):
    """Hard-bin classification accuracy."""
    hit = sum(to_class(round(pred_scalar[i]), num_classes) ==
              to_class(r["cost_to_go"], num_classes)
              for i, r in enumerate(records))
    return hit / len(records) if records else float("nan")


def top1_optimal(records, pred_scalar):
    """Fraction of decisions whose cheapest-predicted candidate is truly optimal.

    This is the metric that matters downstream: if the value net picks the
    optimal subgoal greedily, beam=1 reaches the optimum.
    """
    idx = {id(r): i for i, r in enumerate(records)}
    groups = group_by_decision(records)
    good = 0
    for g in groups:
        picked = min(g, key=lambda r: pred_scalar[idx[id(r)]])
        good += bool(picked["is_optimal"])
    return good / len(groups) if groups else float("nan")


def regret(records, pred_scalar):
    """Mean extra moves from greedily following the model.

    Per decision: the model commits its cheapest-predicted candidate; regret is
    that candidate's true cost-to-go minus the optimal (minimum) cost-to-go.
    This is the metric that matters downstream -- a near-optimal pick costs
    almost nothing, unlike the all-or-nothing top1_optimal.
    """
    idx = {id(r): i for i, r in enumerate(records)}
    total = 0.0
    groups = group_by_decision(records)
    for g in groups:
        pick = min(g, key=lambda r: pred_scalar[idx[id(r)]])
        opt = min(r["cost_to_go"] for r in g)
        total += pick["cost_to_go"] - opt
    return total / len(groups) if groups else float("nan")


def summary(records, pred_scalar, num_classes):
    return {
        "n": len(records),
        "mae": round(value_mae(records, pred_scalar, num_classes), 3),
        "top1_optimal": round(top1_optimal(records, pred_scalar), 3),
        "regret": round(regret(records, pred_scalar), 3),
    }


# -- baselines ----------------------------------------------------------------

def heuristic_baseline(records):
    """Predict cost-to-go = subgoal_score, recomputed from the record.

    Needs the env; recomputes the hand-coded score so the benchmark has a
    reference number every learned model must beat.
    """
    from GridEnv import GridEnv, Robot_at, Subgoal
    cache = {}
    preds = []
    for r in records:
        env = cache.get(r["env_id"])
        if env is None:
            env, _ = GridEnv.from_env(r["env_id"])
            cache[r["env_id"]] = env
        sg = Subgoal(
            bottleneck=Robot_at(tuple(r["cand_bottleneck"]), r["mover_color"]),
            support=Robot_at(tuple(r["cand_support"]), r["cand_helper"][1]),
            goal_pos=tuple(r["seg_end"]),
            target_robot=Robot_at(tuple(r["target_robot"][0]), r["target_robot"][1]),
            helper=Robot_at(tuple(r["cand_helper"][0]), r["cand_helper"][1]))
        preds.append(float(env.subgoal_score(sg)))
    return preds


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="nn/data/train.jsonl")
    p.add_argument("--num-classes", type=int, default=32)
    a = p.parse_args()
    recs = load(a.data)
    for split in ("train", "val", "test"):
        rs = by_split(recs, split)
        if not rs:
            continue
        base = heuristic_baseline(rs)
        print(f"{split:>5}: {summary(rs, base, a.num_classes)}  (heuristic baseline)")
