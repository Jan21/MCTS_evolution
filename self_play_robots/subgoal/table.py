"""The one table of PLAN_SUBGOAL_DISCOVERY.md, recomputed from payload rows.

Headline metric: percent of the pinned 450-puzzle 16x16 four-robot benchmark
(`supervised_valuenet/eval/data/bench450.jsonl`) solved with a PROVABLY OPTIMAL
number of moves -- the move dump in the payload row is replayed here under the
real joint-game rules (`simulate.slide`, every other robot a blocker) and the
replayed length must equal that instance's `d_star`.

Nothing is copied from a payload aggregate or from any document: every number
below is recomputed from the per-instance `moves` / `moves_seq` dumps. The
recorded aggregates ARE read, but only to be diffed against the recomputation
(`recorded_*` columns of `score()`), which is how Stage 0's gate is checked.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.table
    ... --row "label=path.json"        add/override a row
    ... --json out.json                dump the full recomputation

Every later stage prints the same table with one more --row.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPR = HERE.parent                       # self_play_robots/
REPO = SPR.parent
SV = REPO / "supervised_valuenet"
sys.path.insert(0, str(SV))

from simulate import slide, wall_sets            # noqa: E402

BENCH = SV / "eval/data/bench450.jsonl"
ENV_DIR = SV / "environments"
SIZE = 16
COLOR_ORDER = ["Red", "Blue", "Green", "Yellow"]     # nn.gen_grids.COLORS

# The three rows of PLAN_SUBGOAL_DISCOVERY.md section 2. Budget is fixed at
# 1200 expansions / k=5 for every row; a row that has no payload at that
# budget on THIS bench is reported as missing, never substituted.
DEFAULT_ROWS = [
    ("forward (move-by-move, supervised)",
     SPR / "results/fwd_m0/astar_candidate_scored.json"),
    ("backward (subgoal, supervised)",
     SPR / "results/m0/g16r4_b1s21_b2flags.json"),
    # Produced 2026-08-28 by jobs/subgoal_stage0_selfplay16.slurm (job 4862862):
    # NO recorded 16x16 row for the self-play line existed. See SELFPLAY_NOTE.
    ("current self-play line (v14_stack nets, benched this session)",
     SPR / "results/subgoal/v14_stack_g16r4_bench450_astar.json"),
]
SELFPLAY_NOTE = (
    "Before 2026-08-28 no self-play checkpoint had EVER been benched on "
    "bench450: every payload under results/selfplay/ and results/variants/ "
    "runs on g24r4 / g24r8 / g32r4 bench.solved / bench.unsolved / "
    "g24r4_unseen. The size-free nets were therefore run unchanged at g16r4 "
    "under the same protocol as the backward supervised row (1200 expansions, "
    "k=5, B2 vocabulary, anytime), job 4862862. There is no recorded number to "
    "check this row against, so the Stage 0 gate can only confirm that the row "
    "reproduces its OWN payload aggregate from the move dumps. A second arm, "
    "the last main-line curriculum pair, is at "
    "results/subgoal/mix_b2mix_iter3_g16r4_bench450_astar.json."
)
MISSING_NOTE = "no payload at 1200 expansions / k=5 on bench450"

_GRIDS: dict[int, tuple] = {}


def board(env_id: int):
    """(walls_right, walls_down) for a bench board, from the raw env pickle."""
    if env_id not in _GRIDS:
        with open(ENV_DIR / f"env_{env_id}.pkl", "rb") as f:
            grid_data = pickle.load(f)["grid_data"]
        _GRIDS[env_id] = wall_sets(grid_data, SIZE)
    return _GRIDS[env_id]


def load_bench(path=BENCH):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def replay(inst, moves):
    """Replay `moves` ([color, direction], ...) under the real joint rules.

    Returns (n_moves, ok, reason). Certification is `eval/replay_validate.py`'s:
    every move must be a real full slide (a no-op is illegal) applied with every
    other robot on the board as a blocker -- the physics
    `eval/realize.py::strict_moves` also uses -- and the instance's target robot
    must end on the target cell.
    """
    wr, wd = board(inst["env_id"])
    pos = {COLOR_ORDER[i]: tuple(p) for i, p in enumerate(inst["positions"])}
    tcolor = COLOR_ORDER[inst["target_idx"]]
    target = tuple(inst["target"])
    for j, (color, direction) in enumerate(moves):
        blockers = {p for c, p in pos.items() if c != color}
        nxt = slide(pos[color], direction, blockers, wr, wd, SIZE)
        if nxt == pos[color]:
            return len(moves), False, f"move {j}: {color} {direction} is a no-op"
        pos[color] = nxt
    if pos[tcolor] != target:
        return len(moves), False, f"target robot ends at {pos[tcolor]}, not {target}"
    return len(moves), True, "ok"


def pick_system(payload, want=None):
    """The one scored system in a compare/bench payload."""
    cands = {k: v for k, v in payload["systems"].items()
             if isinstance(v, dict) and v.get("rows")}
    if want:
        return want, cands[want]
    if len(cands) != 1:
        raise SystemExit(f"payload has {len(cands)} scored systems: {list(cands)}")
    return next(iter(cands.items()))


def score(path, bench, system=None):
    """Recompute the three columns for one payload. No aggregate is trusted."""
    payload = json.loads(Path(path).read_text())
    proto = payload.get("protocol", {})
    name, sysd = pick_system(payload, system)
    rows = sysd["rows"]
    agg = sysd.get("aggregate", {})
    if len(rows) != len(bench):
        raise SystemExit(f"{path}: {len(rows)} rows vs {len(bench)} instances")

    solved = optimal = 0
    extras, bad_align, bad_replay, short_of_dstar = [], [], [], []
    disagree_len = []
    per_inst = []
    for i, (r, inst) in enumerate(zip(rows, bench)):
        if r.get("env_id") != inst["env_id"] or r.get("d_star") != inst["d_star"]:
            bad_align.append(i)
        d = inst["d_star"]
        seq = r.get("moves_seq")
        if seq is None and isinstance(r.get("moves"), list):
            seq = r["moves"]
        rec_len = r.get("realized_strict")
        if rec_len is None and isinstance(r.get("moves"), int):
            rec_len = r["moves"]
        ok = False
        n = None
        if seq:
            n, reached, why = replay(inst, seq)
            if not reached:
                bad_replay.append((i, why))
            else:
                ok = True
                solved += 1
                extras.append(n - d)
                if n == d:
                    optimal += 1
                if n < d:
                    short_of_dstar.append((i, n, d))
            if rec_len is not None and n != rec_len:
                disagree_len.append((i, n, rec_len))
        elif r.get("solved"):
            bad_replay.append(i)          # claimed solved with no move dump
        per_inst.append(dict(i=i, env_id=inst["env_id"], d_star=d,
                             replayed=n, ok=ok,
                             recorded_solved=bool(r.get("solved")),
                             recorded_len=rec_len))
    n_b = len(bench)
    return dict(
        path=str(path), system=name,
        expansions=proto.get("expansions"), k=proto.get("k"),
        instances_file=proto.get("instances_file"),
        n=n_b, solved=solved, optimal=optimal,
        pct_optimal=100.0 * optimal / n_b,              # the PLAN's headline
        pct_optimal_of_solved=(100.0 * optimal / solved) if solved else None,
        mean_extra=(sum(extras) / len(extras)) if extras else None,
        n_extra=len(extras),
        recorded_solved=agg.get("solved"),
        recorded_pct_optimal=agg.get("pct_optimal"),
        recorded_mean_regret=agg.get("mean_regret"),
        misaligned=bad_align, replay_failures=bad_replay,
        length_disagreements=disagree_len, below_d_star=short_of_dstar,
        per_instance=per_inst,
    )


def markdown(scored):
    out = ["| model | optimal % of 450 | solved / 450 | extra moves (on its solves) |",
           "|---|---|---|---|"]
    for label, s in scored:
        if s is None:
            out.append(f"| {label} | — | — | — |")
            continue
        me = "—" if s["mean_extra"] is None else f"{s['mean_extra']:.3f} (n={s['n_extra']})"
        out.append(f"| {label} | {s['pct_optimal']:.1f}% ({s['optimal']}/450) "
                   f"| {s['solved']}/450 | {me} |")
    return "\n".join(out)


def gate(scored, tol_pct=0.05, tol_solved=0):
    """Stage 0 gate: recomputation must reproduce each payload's own aggregate."""
    verdicts = []
    for label, s in scored:
        if s is None:
            verdicts.append((label, None, "no payload at 1200/k=5 on bench450"))
            continue
        probs = []
        if s["misaligned"]:
            probs.append(f"{len(s['misaligned'])} rows misaligned with bench450")
        if s["replay_failures"]:
            probs.append(f"{len(s['replay_failures'])} rows fail replay")
        if s["length_disagreements"]:
            probs.append(f"{len(s['length_disagreements'])} rows: replay length "
                         f"!= recorded realized_strict")
        if s["below_d_star"]:
            probs.append(f"{len(s['below_d_star'])} rows shorter than d_star")
        if s["recorded_solved"] is not None and \
                abs(s["solved"] - s["recorded_solved"]) > tol_solved:
            probs.append(f"solved {s['solved']} vs recorded {s['recorded_solved']}")
        # eval/compare.py::aggregate divides pct_optimal by len(solved), NOT by
        # n; the plan's headline divides by 450. Reproduce the payload under ITS
        # convention -- that is what "reproduces the recorded number" means.
        if s["recorded_pct_optimal"] is not None and \
                s["pct_optimal_of_solved"] is not None and \
                abs(s["pct_optimal_of_solved"] - s["recorded_pct_optimal"]) > tol_pct:
            probs.append(f"optimal%-of-solved {s['pct_optimal_of_solved']:.3f} vs "
                         f"recorded pct_optimal {s['recorded_pct_optimal']:.3f}")
        if s["recorded_mean_regret"] is not None and s["mean_extra"] is not None \
                and abs(s["mean_extra"] - s["recorded_mean_regret"]) > 1e-9:
            probs.append(f"mean extra {s['mean_extra']:.6f} vs recorded "
                         f"mean_regret {s['recorded_mean_regret']:.6f}")
        verdicts.append((label, not probs, "; ".join(probs) or "reproduces"))
    return verdicts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--row", action="append", default=[],
                   help="label=path[:system]; repeatable, appended after the defaults")
    p.add_argument("--only-rows", action="store_true", help="drop the default rows")
    p.add_argument("--bench", default=str(BENCH))
    p.add_argument("--json", dest="json_out")
    a = p.parse_args(argv)

    bench = load_bench(a.bench)
    rows = [] if a.only_rows else list(DEFAULT_ROWS)
    for spec in a.row:
        label, _, rest = spec.partition("=")
        rows.append((label, rest))

    scored = []
    for label, path in rows:
        scored.append((label, None if path is None else score(path, bench)))

    print(markdown(scored))
    print()
    for label, s in scored:
        if s is None:
            print(f"[{label}] {MISSING_NOTE}")
            continue
        print(f"[{label}] {s['path']}\n    system={s['system']!r} "
              f"exp={s['expansions']} k={s['k']}\n"
              f"    recomputed: {s['optimal']}/450 optimal "
              f"(= {s['pct_optimal_of_solved']:.3f}% of its {s['solved']} solves),"
              f" {s['solved']}/450 solved, mean extra {s['mean_extra']:.4f}\n"
              f"    recorded  : solved={s['recorded_solved']} "
              f"pct_optimal={s['recorded_pct_optimal']} "
              f"mean_regret={s['recorded_mean_regret']}")
    print("\nStage 0 gate:")
    for label, ok, why in gate(scored):
        print(f"  {'PASS' if ok else ('MISSING' if ok is None else 'FAIL')}  {label}: {why}")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            {"bench": a.bench,
             "rows": [{"label": l, **(s or {"missing": MISSING_NOTE})}
                      for l, s in scored]}, indent=1) + "\n")


if __name__ == "__main__":
    main()
