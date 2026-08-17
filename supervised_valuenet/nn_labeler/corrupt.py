"""Controlled label corruption for the causality arm (FINDINGS 80 design).

The dose-response campaign (FINDINGS 74) showed downstream utility tracking
label argmin-fidelity across three cells, but fidelity co-varied with corpus
size and robot count -- no cell isolates fidelity as the causal knob. This
tool manufactures the missing cells: it takes an EXACT corpus, subsamples
whole decision groups to twin size (the corpus-shrinkage control), then for
a calibrated fraction of multi-value groups redirects the argmin to the
group's second-best candidate (label + is_optimal surgery). Boards,
instances, robot count, corpus size and every unchanged candidate label stay
exact; only argmin agreement moves. All requested doses are emitted from the
SAME subsample so fidelity is the only variable between arms.

Corruption is structural, not iid noise: the argmin lands on the runner-up
candidate, and only in near-tie groups (second-best within --max-gap moves
of optimal) -- the labeler's own dominant error mode is small-gap
second-best picks (gate gap p90 = 2 at every 4-robot cell, FINDINGS 65/74),
so the injected errors match the real morphology in both location and size.

  python -m nn_labeler.corrupt --data scaling/data/g24r4/backward.jsonl \
      --match-records 39452 --doses 1.0,0.86,0.822 --seed 11 \
      --out-prefix nn_labeler/results/corrupt_g24r4

Writes <prefix>_d1000.jsonl, <prefix>_d860.jsonl, <prefix>_d822.jsonl and a
<prefix>.manifest.json with achieved doses; achieved argmin agreement is
computed on the emitted records (the exact corpus is its own reference).
"""
import argparse
import json
import random
from collections import OrderedDict


def gkey(r):
    return (r["env_id"], tuple(r["target"]), tuple(r["target_robot"][0]),
            tuple(r["seg_start"]), tuple(r["seg_end"]), r["depth"])


def load_groups(path):
    groups = OrderedDict()
    for line in open(path):
        r = json.loads(line)
        groups.setdefault(gkey(r), []).append(r)
    return groups


def corrupt_group(recs):
    """Redirect argmin to the second-best distinct value. Returns delta stats."""
    vals = sorted(set(r["cost_to_go"] for r in recs))
    m, s = vals[0], vals[1]
    moved = 0
    for r in recs:
        if r["cost_to_go"] == m:
            r["cost_to_go"] = s + 1     # old optima now rank below second-best
            r["is_optimal"] = False
            moved += 1
        elif r["cost_to_go"] == s:
            r["is_optimal"] = True      # new argmin = old runner-up
        else:
            r["is_optimal"] = False
    return moved, (s + 1 - m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--match-records", type=int, required=True,
                   help="subsample whole groups until record count reaches this")
    p.add_argument("--doses", default="1.0,0.86,0.822",
                   help="comma list of target argmin-agreement levels; 1.0 = "
                        "size control (no corruption)")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--max-gap", type=int, default=2,
                   help="corrupt only groups whose second-best is within this "
                        "many moves of optimal (the labeler's real error "
                        "morphology; gate gap p90 = 2)")
    p.add_argument("--out-prefix", required=True)
    a = p.parse_args()

    groups = load_groups(a.data)
    n_all = sum(len(g) for g in groups.values())
    rng = random.Random(a.seed)

    # -- shared subsample: whole groups, uniform order, stop at target size --
    keys = list(groups)
    rng.shuffle(keys)
    picked, count = [], 0
    for k in keys:
        if count >= a.match_records:
            break
        picked.append(k)
        count += len(groups[k])
    picked_set = set(picked)
    sub = OrderedDict((k, groups[k]) for k in groups if k in picked_set)
    def secondgap(g):
        vals = sorted(set(r["cost_to_go"] for r in g))
        return vals[1] - vals[0] if len(vals) >= 2 else None

    multi = [k for k, g in sub.items() if secondgap(g) is not None]
    neartie = [k for k in multi if secondgap(sub[k]) <= a.max_gap]
    print(f"source {n_all} recs / {len(groups)} groups -> subsample "
          f"{count} recs / {len(sub)} groups ({len(multi)} multi-value, "
          f"{len(neartie)} near-tie <= {a.max_gap})")

    manifest = {"source": a.data, "source_records": n_all,
                "source_groups": len(groups), "sub_records": count,
                "sub_groups": len(sub), "sub_multi_groups": len(multi),
                "sub_neartie_groups": len(neartie), "max_gap": a.max_gap,
                "seed": a.seed, "arms": {}}

    for dose in [float(d) for d in a.doses.split(",")]:
        # deep-ish copy: rebuild records fresh per arm
        arm = {k: [dict(r) for r in g] for k, g in sub.items()}
        n_corrupt = 0
        deltas = []
        if dose < 1.0:
            # overall agreement = 1 - corrupted/len(sub)
            n_target = round((1.0 - dose) * len(sub))
            if n_target > len(neartie):
                raise SystemExit(f"dose {dose} needs {n_target} near-tie "
                                 f"groups, only {len(neartie)} exist")
            crng = random.Random(a.seed * 1000 + int(dose * 1000))
            for k in crng.sample(neartie, n_target):
                moved, delta = corrupt_group(arm[k])
                deltas.append(delta - 1)   # planner-facing regret = s - m
                n_corrupt += 1
        achieved = 1.0 - n_corrupt / len(sub)
        tag = f"d{int(round(dose * 1000))}"
        out = f"{a.out_prefix}_{tag}.jsonl"
        with open(out + ".tmp", "w") as f:
            for g in arm.values():
                for r in g:
                    f.write(json.dumps(r) + "\n")
        import os
        os.replace(out + ".tmp", out)
        manifest["arms"][tag] = {
            "target_argmin_agreement": dose,
            "achieved_argmin_agreement": round(achieved, 4),
            "groups_corrupted": n_corrupt,
            "argmin_gap_mean": round(sum(deltas) / len(deltas), 3) if deltas else 0.0,
            "out": out}
        print(f"arm {tag}: target {dose} achieved {achieved:.4f} "
              f"({n_corrupt} groups corrupted, gap mean "
              f"{manifest['arms'][tag]['argmin_gap_mean']}) -> {out}")

    with open(f"{a.out_prefix}.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("MANIFEST", f"{a.out_prefix}.manifest.json")


if __name__ == "__main__":
    main()
