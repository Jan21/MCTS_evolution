"""Build eval/results/residual_failures_postfix.json from the raw records of
diag_residual_postfix.py (corrected classifier).

Classifier notes (learned from inspecting the raw records):

- walls-only reachability is the wrong geometry baseline for the bounce leg
  of a two-phased segment (the stopper is the whole point); the baseline used
  is "reachable with only the declared support on the board".
- A plan can contain TWO leaf nodes for the same physical robot (e.g. leaf_1
  and leaf_4 both Yellow at the same start cell). The DAG then schedules one
  robot in two places; detectable statically (>= 2 leaf-src segments of one
  color).
- A goal-hop stopper can be declared (parent_support_pos) without ANY plan
  node providing a robot for it (realize._support_node_for returns None);
  the plan completes and verifies abstractly but is missing a step.
- "swap" sub-pattern: a support-placement segment fails because the robot
  standing on the destination is the very robot that will later bounce off
  that cell; it must vacate (approach leg) before the placement, but the
  two-phase retry only ever splits the segment that failed, and a placement
  has no support to split on.
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get(
    "DIAG_RAW", os.path.join(HERE, "diag_residual_postfix_raw.json"))
REPO = "/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet"
OUT = os.path.join(REPO, "eval/results/residual_failures_postfix.json")


def dup_leaf_colors(segs):
    """Colors that the plan moves out of >= 2 distinct leaf nodes (one
    physical robot given two plan identities)."""
    n = Counter(s["color"] for s in segs if str(s["src"]).startswith("leaf"))
    return sorted(c for c, k in n.items() if k >= 2)


def swap_needed(segs, fail, fail_seg):
    """Failing segment's destination is occupied by a robot that some OTHER
    segment of that same robot bounces off (support == that cell): the robot
    must vacate and be replaced -- a swap the fixed schedule cannot do."""
    end = tuple(segs[fail_seg]["end"])
    occ = fail.get("end_occupied_by")
    if occ is None:
        return False
    return any(s["color"] == occ and s["support"] is not None
               and tuple(s["support"]) == end for i, s in enumerate(segs)
               if i != fail_seg)


def classify(rec):
    """-> (class, features) for one failing plain-mode record."""
    d = rec["diag"]
    ch = d.get("channel")
    if ch != "bfs_unreachable":
        return f"other_{ch}", {}
    f = d["fail"]
    segs = d["segs"]
    dup = dup_leaf_colors(segs)
    feats = {"terminal_kind": d["terminal_kind"], "dup_leaf_colors": dup,
             "mover_is_dup_leaf": f["mover"] in dup,
             "min_blocking_set": f["min_blocking_set"]}

    # 1. goal-hop stopper declared but never assigned to any robot
    if (f["support_cell"] and not f["support_present"]
            and not f["support_node_assigned"]
            and d["terminal_kind"] == "bounce"):
        return "goal_stopper_never_placed", feats

    # geometry baseline: can the mover get there at all from its CURRENT cell
    # (walls only, or with just its declared stopper present)?
    free_ok = f["walls_only_reachable"] or (
        f["support_present"] and bool(f["support_only_reachable"]))

    # 2. one robot scheduled as two plan robots: the single real robot is on
    #    the wrong side of its own stopper / was re-tasked mid-plan
    if not free_ok:
        if f["mover"] in dup or f["start_mismatch_at_fail"]:
            return "robot_reused_two_places", feats
        return "other_geometry", feats

    # blocking analysis on the minimal blocking sets
    opts = ([[b] for b in f["critical_blockers"]]
            if f["min_blocking_set"] == 1 else f["pair_blockers"])
    if not opts:
        return "other_blocking_set_3plus", feats

    # 3. the failing segment's own stopper robot stands on its stopper cell
    #    and blocks the mover's path to/through it. It may depart only after
    #    the bounce it exists for, and the bounce needs the mover's approach
    #    first -- circular. Only "step aside and return" plays it.
    sc = tuple(f["support_cell"]) if f["support_cell"] else None
    if sc is not None and any(tuple(b["pos"]) == sc for opt in opts
                              for b in opt):
        feats["blocker_is_own_stopper"] = True
        return "move_aside_and_return_needed", feats

    def needs_stay(b):
        return b["on_goal"] or (b["at_pending_support_cell"]
                                and not b["has_pending_own_segments"])

    def safely_movable(b):
        return (not b["at_pending_support_cell"] and not b["on_goal"]
                and not b["has_pending_own_segments"])

    kinds = []
    for opt in opts:
        if all(safely_movable(b) for b in opt):
            kinds.append("clearable")
        elif any(needs_stay(b) for b in opt):
            kinds.append("needs_return")
        else:
            kinds.append("ordering")
    feats["option_kinds"] = kinds
    if "clearable" in kinds:
        best = opts[kinds.index("clearable")]
        feats["all_bystanders"] = all(not b["plan_touches"] for b in best)
        return "pure_robot_blockage_clearable", feats
    if "ordering" in kinds:
        feats["swap_needed"] = swap_needed(segs, f, d["fail_seg"])
        return "blocked_by_plan_robot_moving_later", feats
    return "move_aside_and_return_needed", feats


EXPLAIN = {
    "blocked_by_plan_robot_moving_later": (
        "idx 1 (env 2400, d*=4): segment sp_2<-leaf_2 must place Yellow on "
        "(15,0) so Green can bounce off it (bn_2<-leaf_1: Green (15,0)->"
        "(14,0), stopper (15,0)) -- but Green itself is still standing on "
        "(15,0), its own start cell. Green only vacates by performing that "
        "very bounce, so the placement and the bounce must swap: Green's "
        "approach leg first, then Yellow's placement, then the bounce. The "
        "realizer's retry never reaches this order because the segment that "
        "fails (the placement) has no support to split on. The plan's robots "
        "would clear the cell; the fixed schedule cannot interleave them."),
    "pure_robot_blockage_clearable": (
        "idx 28 (env 2409, d*=7): final hop goal<-leaf_0, Red slides from "
        "(7,9) to the goal (7,3). A wall-legal route exists, but Blue -- a "
        "bystander the plan never moves -- stands at (10,3) in the corridor; "
        "removing Blue alone makes the goal reachable. A 'move the blocker "
        "aside' plan step would fix it outright."),
    "robot_reused_two_places": (
        "idx 42 (env 2414, d*=9): the plan contains TWO leaf nodes for "
        "Yellow (leaf_1 and leaf_4, both starting (0,5)). One Yellow "
        "identity is placed on (10,0) as a stopper (sp_4); the other is then "
        "supposed to run bn_3<-leaf_1 into (8,0) bouncing off Red at (9,0). "
        "The single real Yellow is now at (10,0), directly behind its own "
        "stopper, and no slide from there can stop on (8,0). One physical "
        "robot cannot fill two plan identities."),
    "goal_stopper_never_placed": (
        "idx 25 (env 2408, d*=7): the goal hop goal<-sg_1 needs target "
        "Green to slide (14,10)->(10,10) and stop on a robot at (9,10), but "
        "no plan node ever assigns or places a robot at (9,10) (support "
        "node unassigned). With a stopper present the bounce works "
        "(support-only reachable = true); the plan simply lacks that step, "
        "and plan completion/validation never notices."),
    "move_aside_and_return_needed": (
        "idx 107 (env 2435, d*=8): sp_1<-sg_2 wants Green to take over the "
        "stopper cell (4,0), but Yellow already stands there serving as the "
        "stopper for Red's pending hop bn_1<-leaf_0 into (3,0) -- and "
        "Yellow has no further plan moves, so it never leaves. Playing this "
        "needs 'Yellow steps aside after its bounce is consumed (or before, "
        "and returns)', a step the subgoal vocabulary cannot express. "
        "Variant (idx 23, 45): the failing segment's own stopper robot "
        "already stands on the stopper cell (idx 23: Red on (9,0), placed "
        "as a 0-move sp_3) and blocks the mover's approach; it may depart "
        "only after the bounce it exists for, and the bounce needs the "
        "approach first -- only step-aside-and-return breaks the circle."),
}

ANY_EXPLAIN = {
    "frontier_exhausted_all_completions_unplayable": (
        "idx 25 (env 2408): the search exhausts its whole reachable tree "
        "after 1 expansion; all 5 completions it can propose fail strict "
        "realization the same way (goal stopper at (9,10) declared but "
        "never placed). Re-run reproduces the stored counts exactly "
        "(11/11 instances). Per-instance defect of the rejected "
        "completions: goal-stopper-missing x4 instances (25, 52, 91, 111), "
        "bystander blockage x5 (28, 34, 43, 76, 113), "
        "robot-moving-later blockage x2 (29, 109)."),
    "budget_exhausted": (
        "idx 129 (env 2443): the anytime loop spends all 1200 expansions "
        "and rejects 825 complete plans, none strictly playable (the plain "
        "plan for this instance is a goal-stopper-never-placed failure; "
        "idx 105/121 are ordering failures, idx 107 move-aside-and-return). "
        "Rejected-completion counts for the five: 93, 35, 329, 165, 825."),
}


def main():
    raw = json.load(open(RAW))
    plain, anyrecs = raw["plain"], raw["anytime"]
    fails = [r for r in plain if not r["stored_solved"]]
    assert len(fails) == 38
    assert all(r.get("cost_matches_stored") for r in plain)
    assert all(r.get("regen_matches_stored_outcome") for r in plain)

    tax = {}
    feats_by_idx = {}
    for r in fails:
        cls, feats = classify(r)
        tax.setdefault(cls, []).append(r["idx"])
        feats_by_idx[r["idx"]] = feats

    order = ["blocked_by_plan_robot_moving_later",
             "pure_robot_blockage_clearable", "robot_reused_two_places",
             "goal_stopper_never_placed", "move_aside_and_return_needed"]
    taxonomy = {}
    for cls in order + sorted(set(tax) - set(order)):
        if cls not in tax:
            continue
        ids = sorted(tax[cls])
        entry = {"count": len(ids), "instance_ids": ids,
                 "example_explanation": EXPLAIN.get(cls, "")}
        if cls == "blocked_by_plan_robot_moving_later":
            entry["swap_subpattern_ids"] = sorted(
                i for i in ids if feats_by_idx[i].get("swap_needed"))
        if cls == "pure_robot_blockage_clearable":
            entry["bystander_ids"] = sorted(
                i for i in ids if feats_by_idx[i].get("all_bystanders"))
        taxonomy[cls] = entry

    any_tax = {"frontier_exhausted_all_completions_unplayable":
               sorted(a["idx"] for a in anyrecs
                      if a["exit"] == "frontier_exhausted"),
               "budget_exhausted":
               sorted(a["idx"] for a in anyrecs
                      if a["exit"] == "budget_exhausted")}
    anytime_failures = {
        cls: {"count": len(ids), "instance_ids": ids,
              "example_explanation": ANY_EXPLAIN[cls]}
        for cls, ids in any_tax.items()}

    n_blocker = taxonomy["pure_robot_blockage_clearable"]["count"]
    blocker_ids = set(taxonomy["pure_robot_blockage_clearable"]["instance_ids"])
    any_fail_ids = {a["idx"] for a in anyrecs}
    n_blocker_anytime = len(blocker_ids & any_fail_ids)

    summary = [
        "After the self-support proposal fix and the two-phase strict realizer, 38/150 bench instances still fail strict playability in plain mode (112/150) and 16/150 in anytime mode (134/150); all 150 plans regenerate bit-identically (cost and outcome match the stored run 150/150).",
        "Both pass-2 headline mechanisms are confirmed gone: 0 of the 150 regenerated plans contain a self-support subgoal, and no failure is the pass-2 atomic-order kind (every supported segment that could be two-phased was).",
        "Largest class, 16/38: a robot stands on or across a segment's destination while the plan only moves it later; the fixed dependency schedule cannot interleave them. In 4 of the 16 (idx 0, 1, 62, 105) the blocker must literally swap places with the robot that will later bounce off that same cell (e.g. idx 1: Green on (15,0) must vacate via the very bounce that needs Yellow placed on (15,0) first) -- a smarter realizer retry that splits the blocked-stopper's segment could rescue these without new vocabulary.",
        "7/38 are clearable blockages: a wall-legal route exists and a single safe-to-move robot blocks it -- 5 bystanders the plan never touches (idx 28, 34, 43, 76, 113) and 2 robots that finished all their plan moves and parked on the route (idx 48, 61).",
        "6/38 are plans that schedule one physical robot as two plan robots (duplicate leaf nodes with the same start cell, e.g. Yellow's leaf_1 and leaf_4 in idx 42); the single real robot ends up on the wrong side of its own stopper or stranded off-plan. This is a static plan property, the post-fix sibling of the self-support leak.",
        "6/38 declare a stopper cell for the final hop into the goal but never assign or place any robot there (support node unassigned, all on the goal<-sg_1 edge; e.g. idx 25 needs a robot at (9,10)); with a stopper present the bounce would work, so a whole plan step is missing and plan completion/validation never notices.",
        "3/38 need a robot to move aside and later return to (or be replaced at) a cell it must hold for a pending bounce (idx 23, 45, 107) -- a step the subgoal vocabulary cannot express.",
        "Anytime failures (16): 11 exhaust the entire search frontier within 0-1 expansions -- every completion the proposals can reach (1-5 plans) is unplayable, for the same defect as the plain plan (goal-stopper x4, bystander blockage x5, ordering x2); 5 burn the full 1200-expansion budget rejecting 35-825 unplayable completions. Anytime rescues most ordering failures but almost never the structural ones.",
        "Blocker-shaped residue (what a move-the-blocker-away candidate type would fix): 7/38 plain instances, of which 5 survive into the anytime residue = 3.3% of the bench, barely above the 3% do-not-build line.",
        "The bulk of the residue is plan-structure defects that are statically detectable on a finished plan (duplicate-leaf robot reuse, unassigned goal stopper, swap windows); closing them at the proposal/validation layer -- as the self-support fix did -- or pre-filtering completions with those static checks in anytime mode is the cheaper, larger-yield next step; a blocker-clearing candidate type is third in line behind them.",
    ]

    report = {
        "_summary": summary,
        "n_plain_failing": 38,
        "n_anytime_failing": 16,
        "protocol": {
            "instances": "eval/data/bench450_first150.jsonl",
            "plain_results": "eval/results/comparison_backward_postfix2.json",
            "anytime_results":
                "eval/results/comparison_backward_postfix2_anytime.json",
            "regeneration": "torch.manual_seed(0), k=5, 1200 expansions, CPU, "
                            "checkpoints_backward/{policy_v2,value_v2}.ckpt, "
                            "current working tree "
                            "(analysis/artifacts/diag_residual_postfix.py)",
            "determinism": "150/150 regenerated plans match stored abstract "
                           "cost and stored strict outcome; anytime re-runs "
                           "match stored expansion/rejection counts 11/11 "
                           "(budget-exhausted rows classified from the "
                           "stored run: expansions == 1200)",
            "instance_ids_are": "0-based row index in bench450_first150.jsonl",
        },
        "taxonomy": taxonomy,
        "anytime_failures": anytime_failures,
        "anytime_underlying_defect": {
            "goal_stopper_never_placed": [25, 52, 87, 91, 111, 129],
            "pure_robot_blockage_clearable": [28, 34, 43, 76, 113],
            "blocked_by_plan_robot_moving_later": [29, 105, 109, 121],
            "move_aside_and_return_needed": [107],
        },
        "build_decision_input": {
            "residue_pct_of_450_projection": {
                "plain": "38/150 = 25.3% -> ~114/450 projected",
                "anytime": "16/150 = 10.7% -> ~48/450 projected",
                "caveat": "the 150 instances are the head slice of "
                          "bench450.jsonl and are slightly harder than the "
                          "full set (9.3% vs 12.2% of instances at d*<=3), "
                          "so the full-450 rates should be marginally lower; "
                          "same boards family, no other known skew",
            },
            "blocker_shaped_count": n_blocker,
            "blocker_shaped_count_anytime": n_blocker_anytime,
            "recommendation":
                "residue is not blocker-shaped -- training/other",
            "justification":
                "only 7/38 plain failures (5/16 anytime, 3.3% of the bench) "
                "are clearable-blocker cases; 28/38 stem from plan-structure "
                "defects (blocker the plan itself moves later / swap "
                "windows: 16; robot duplicated across two leaves: 6; goal "
                "stopper never placed: 6) that are statically detectable "
                "and cheaper to close at the proposal, validation, or "
                "realizer-retry layer, as the self-support fix already "
                "demonstrated.",
        },
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", OUT)
    for cls, e in taxonomy.items():
        print(f"  {cls}: {e['count']}  {e['instance_ids']}")
    for cls, e in anytime_failures.items():
        print(f"  [anytime] {cls}: {e['count']}  {e['instance_ids']}")
    print("  blocker_shaped:", n_blocker, "plain /", n_blocker_anytime,
          "anytime")


if __name__ == "__main__":
    main()
