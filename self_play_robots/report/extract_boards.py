"""Extract verified board puzzles + realized move sequences for report/boards.html.

Reads three payloads that already exist on disk (no compute, no nets):

  supervised_valuenet/scaling/data/g24r4/bench.solved.jsonl   232 graded instances + d*
  results/variants/v07_hybrid_actions/bench_graded_hybrid_d2.json   flagship (dump_moves)
  results/variants/v07_hybrid_actions/bench_graded_stdmcts.json     matched control
  results/ceiling/g24r4_b2.json                              exhaustive B2 sub-goal probe

Every move sequence in both payloads is re-simulated from the board walls with
`simulate.slide` before anything is written; a sequence that does not put the
target robot on the target cell (or that contains a move which cannot move) is
dropped and reported.  The curated subset is then injected into boards.html
between the DATA-BEGIN / DATA-END markers, so the page stays self-contained
and regenerable:

  PYTHONPATH=supervised_valuenet:self_play_robots python3 self_play_robots/report/extract_boards.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "supervised_valuenet"))

from simulate import slide, wall_sets  # noqa: E402

CONFIG = "g24r4"
SIZE = 24
COLORS = ["Red", "Blue", "Green", "Yellow"]      # nn.gen_grids.PALETTE[:4] slot order
ENV_DIR = REPO / "supervised_valuenet" / f"environments_{CONFIG}"
INSTANCES = REPO / "supervised_valuenet" / "scaling" / "data" / CONFIG / "bench.solved.jsonl"
VARIANTS = REPO / "self_play_robots" / "results" / "variants" / "v07_hybrid_actions"
HYBRID = VARIANTS / "bench_graded_hybrid_d2.json"
PLAIN = VARIANTS / "bench_graded_stdmcts.json"
CEILING = REPO / "self_play_robots" / "results" / "ceiling" / f"{CONFIG}_b2.json"
PAGE = REPO / "self_play_robots" / "report" / "boards.html"
BEGIN, END = "/* DATA-BEGIN */", "/* DATA-END */"


# ---------------------------------------------------------------- loading

def one_system(path: Path):
    """(name, rows, protocol) from a bench payload holding exactly one system."""
    payload = json.loads(path.read_text())
    (name, sys_), = payload["systems"].items()
    return name, sys_["rows"], payload["protocol"]


_walls: dict[int, tuple] = {}


def walls(env_id: int):
    """(walls_right, walls_down) for a board, straight from its pickle."""
    if env_id not in _walls:
        with open(ENV_DIR / f"env_{env_id}.pkl", "rb") as fh:
            grid_data = pickle.load(fh)["grid_data"]
        assert len(grid_data) == SIZE * SIZE
        _walls[env_id] = wall_sets(grid_data, SIZE)
    return _walls[env_id]


# ---------------------------------------------------------------- replay

def replay(inst, moves):
    """(ok, reason) -- re-simulate a (colour, direction) sequence from the walls."""
    wr, wd = walls(inst["env_id"])
    pos = [tuple(p) for p in inst["positions"]]
    for step, (colour, direction) in enumerate(moves):
        if colour not in COLORS:
            return False, f"move {step}: unknown robot {colour!r}"
        slot = COLORS.index(colour)
        blockers = set(pos) - {pos[slot]}
        stop = slide(pos[slot], direction, blockers, wr, wd, SIZE)
        if stop == pos[slot]:
            return False, f"move {step}: {colour} {direction} cannot move"
        pos[slot] = stop
    if pos[inst["target_idx"]] != tuple(inst["target"]):
        return False, "target robot does not end on the target cell"
    return True, "ok"


# ---------------------------------------------------------------- assembly

def main():
    insts = [json.loads(l) for l in INSTANCES.read_text().splitlines() if l.strip()]
    h_name, h_rows, h_proto = one_system(HYBRID)
    p_name, p_rows, p_proto = one_system(PLAIN)
    ceiling = json.loads(CEILING.read_text())
    c_rows = ceiling["rows"]
    assert len(insts) == len(h_rows) == len(p_rows) == len(c_rows) == 232

    # rows are written in instance-file order; check the identity fields agree
    for i, (a, b, c, d) in enumerate(zip(insts, h_rows, p_rows, c_rows)):
        assert a["env_id"] == b["env_id"] == c["env_id"] == d["env_id"], i
        assert a["d_star"] == b["d_star"] == c["d_star"] == d["d_star"], i

    # ---- verify EVERY sequence in both payloads
    checked = failed = 0
    bad: list[str] = []
    for i, inst in enumerate(insts):
        for tag, row in (("hybrid", h_rows[i]), ("plain", p_rows[i])):
            mv = row.get("moves")
            if not mv:
                continue
            checked += 1
            ok, why = replay(inst, mv)
            if not ok:
                failed += 1
                bad.append(f"idx {i} env {inst['env_id']} {tag}: {why}")
                row["moves"] = None            # never displayed
            elif len(mv) != row["realized_strict"]:
                failed += 1
                bad.append(f"idx {i} env {inst['env_id']} {tag}: length "
                           f"{len(mv)} != realized_strict {row['realized_strict']}")
                row["moves"] = None

    # ---- per-instance facts
    facts = []
    for i, inst in enumerate(insts):
        h, p, c = h_rows[i], p_rows[i], c_rows[i]
        hm = h.get("moves")
        pm = p.get("moves")
        winner = (h.get("search") or {}).get("winner")
        opening = 0
        if winner and winner.startswith("slide2:"):
            opening = 2
        elif winner and winner.startswith("slide:"):
            opening = 1
        facts.append(dict(i=i, env=inst["env_id"], d=inst["d_star"],
                          hm=len(hm) if hm else None, pm=len(pm) if pm else None,
                          ceil=c["best_realizable_moves"], capped=c["capped"],
                          cat=c["category"], winner=winner, opening=opening))

    def f(i):
        return facts[i]

    # ---- the paired population of FINDINGS Sec.28(c)
    paired = [x["i"] for x in facts
              if x["cat"] == "REALIZABLE_EXISTS" and not x["capped"] and x["d"]
              and x["hm"] and x["pm"]]
    beats = [i for i in paired if f(i)["hm"] < f(i)["ceil"]]
    loses = [i for i in paired if f(i)["hm"] > f(i)["ceil"]]

    def mean(vals):
        return sum(vals) / len(vals)

    stats = dict(
        n=len(insts),
        n_paired=len(paired),
        solved_hybrid=sum(1 for x in facts if x["hm"]),
        solved_plain=sum(1 for x in facts if x["pm"]),
        seqs_checked=checked, seqs_failed=failed,
        beats_ceiling=len(beats), loses_to_ceiling=len(loses),
        regret_hybrid=round(mean([f(i)["hm"] - f(i)["d"] for i in paired]), 3),
        regret_plain=round(mean([f(i)["pm"] - f(i)["d"] for i in paired]), 3),
        regret_ceiling=round(mean([f(i)["ceil"] - f(i)["d"] for i in paired]), 3),
        shorter=sum(1 for x in facts if x["hm"] and x["pm"] and x["hm"] < x["pm"]),
        longer=sum(1 for x in facts if x["hm"] and x["pm"] and x["hm"] > x["pm"]),
        ties=sum(1 for x in facts if x["hm"] and x["pm"] and x["hm"] == x["pm"]),
        openings1=sum(1 for x in facts if x["opening"] == 1),
        openings2=sum(1 for x in facts if x["opening"] == 2),
        expansions_cap=h_proto["expansions"],
        hybrid_system=h_name, plain_system=p_name,
        hybrid_date=h_proto["date"], plain_date=p_proto["date"],
    )

    # ---- curated groups (each list is a filter over the same instances;
    #      a puzzle may appear in more than one group)
    gap = sorted((x for x in facts
                  if x["opening"] == 1 and x["hm"] and x["pm"]
                  and x["hm"] < x["pm"] and x["hm"] <= x["d"] + 3),
                 key=lambda x: (-(x["pm"] - x["hm"]), x["i"]))
    g_open = [x["i"] for x in gap[:10]]
    g_two = [x["i"] for x in facts if x["opening"] == 2]
    g_beat = beats
    tie_opt = [x["i"] for x in facts
               if x["hm"] and x["hm"] == x["pm"] == x["d"] and x["d"] >= 9][:3]
    tie_above = [x["i"] for x in facts
                 if x["hm"] and x["hm"] == x["pm"] and x["hm"] >= x["d"] + 2][:3]
    g_tie = sorted(tie_opt + tie_above)
    g_only = [x["i"] for x in facts if x["hm"] and not x["pm"]]
    g_worse = loses

    groups = [
        dict(key="opening", title="The opening slide pays for itself",
             blurb="The hybrid spends one ordinary robot move before it starts "
                   "planning. That move is not part of the sub-goal language, so "
                   "the plain search cannot try it. These are the ten puzzles where "
                   "the opening saved the most moves, among those where the hybrid "
                   "also finished within three moves of the known shortest solution.",
             members=g_open),
        dict(key="two", title="Two opening slides",
             blurb="Only two puzzles in the benchmark were won with a two-move "
                   "opening. Both are here.",
             members=g_two),
        dict(key="beat", title="Shorter than the sub-goal language can express",
             blurb="A separate exhaustive search enumerated sub-goal plans for "
                   "each puzzle until it ran out of ideas, giving the best answer "
                   "the sub-goal language can produce at all. On these puzzles the "
                   "hybrid's answer is shorter than that. No amount of extra "
                   "search in the old language would have found them.",
             members=g_beat),
        dict(key="tie", title="Both planners agree",
             blurb="The common case: the hybrid needs no opening move and returns "
                   "exactly what the plain search returns. Three where both are "
                   "already optimal, three where both are stuck above the optimum.",
             members=g_tie),
        dict(key="only", title="Only the hybrid finished",
             blurb="One puzzle the plain search left unsolved inside its budget.",
             members=g_only),
        dict(key="worse", title="Where the hybrid falls short",
             blurb="For honesty. On these puzzles the exhaustive sub-goal search "
                   "found a shorter answer than the hybrid did. Seven of the 225 "
                   "compared puzzles, against 27 the other way.",
             members=g_worse),
    ]

    keep = sorted({i for g in groups for i in g["members"]})

    # ---- board geometry, interior segments only (the border is implicit)
    boards = {}
    for i in keep:
        env_id = facts[i]["env"]
        if env_id in boards:
            continue
        wr, wd = walls(env_id)
        boards[env_id] = dict(
            size=SIZE,
            wr=sorted(y * SIZE + x for x, y in wr if x < SIZE - 1),
            wd=sorted(y * SIZE + x for x, y in wd if y < SIZE - 1),
        )

    out_instances = []
    for i in keep:
        x, inst = facts[i], insts[i]
        h, p, c = h_rows[i], p_rows[i], c_rows[i]
        out_instances.append(dict(
            id=i, env=x["env"], d_star=x["d"],
            positions=[list(map(int, q)) for q in inst["positions"]],
            target=list(map(int, inst["target"])), target_idx=inst["target_idx"],
            ceiling=dict(moves=x["ceil"], capped=bool(x["capped"]),
                         category=x["cat"]),
            hybrid=dict(moves=h.get("moves"), opening=x["opening"],
                        winner=x["winner"], expansions=h.get("expansions")),
            plain=dict(moves=p.get("moves"), expansions=p.get("expansions")),
        ))

    data = dict(
        config=CONFIG, size=SIZE, colors=COLORS,
        stats=stats, boards=boards, instances=out_instances,
        groups=[dict(g, members=[i for i in g["members"]]) for g in groups],
        sources=dict(
            instances=str(INSTANCES.relative_to(REPO)),
            hybrid=str(HYBRID.relative_to(REPO)),
            plain=str(PLAIN.relative_to(REPO)),
            ceiling=str(CEILING.relative_to(REPO)),
            boards=str(ENV_DIR.relative_to(REPO)),
        ),
    )
    data["stats"]["n_embedded"] = len(out_instances)
    data["stats"]["n_boards"] = len(boards)

    blob = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")          # cannot close the <script> block

    text = PAGE.read_text()
    a, b = text.index(BEGIN), text.index(END)
    PAGE.write_text(text[:a] + BEGIN + "\nconst DATA = " + blob + ";\n" + text[b:])

    print(f"sequences re-simulated: {checked}, rejected: {failed}")
    for line in bad:
        print("  REJECTED", line)
    print(f"instances embedded: {len(out_instances)} on {len(boards)} boards")
    for g in groups:
        print(f"  {g['key']:8s} {len(g['members']):3d}  {g['title']}")
    print(f"paired population {len(paired)}: hybrid shorter than the sub-goal "
          f"optimum on {len(beats)}, longer on {len(loses)}")
    print(f"wrote {PAGE}")


if __name__ == "__main__":
    main()
