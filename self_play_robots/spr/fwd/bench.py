"""Arena driver for the PRIMITIVE-MOVE (forward) planner.

Emits the SAME payload layout and per-instance row schema as
`eval.compare.run_forward` (`{env_id, d_star, solved, moves, regret, expansions,
seconds, moves_seq, accounting}` under `{"protocol":..., "systems":{name:
{"kind":"forward","aggregate":aggregate(rows),"rows":rows}}}`), so
`eval/merge_compare_shards.py`, `eval/replay_validate.py` and the report tooling
consume it unchanged. Rows additionally carry `realized_strict = moves` -- the
key `spr.gate.paired` reads for the both-solved move comparison -- so the two
arms gate identically (see `spr/fwd/gate.py` for the `kind` shim).

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.fwd.bench \
        --instances chunk.jsonl --ckpt move_planner/checkpoints/candidate_scored.ckpt \
        --search astar --expansions 1200 --k 5 --dump-moves --out chunk.json

`--search`:
  astar          `move_planner.evaluate.nn_astar`, UNCHANGED -- the F-M0 parity arm;
  mcts           `spr.fwd.mcts.mcts` (PUCT over primitive moves);
  greedy_value   `move_planner.evaluate.nn_greedy_value`, UNCHANGED (the M2 "greed");
  greedy_policy  `move_planner.evaluate.nn_greedy_policy`, UNCHANGED.

Parity mode (F-M0) compares a produced payload against a recorded one row by row:

    python -m spr.fwd.bench parity --new out.json \
        --ref supervised_valuenet/eval/results/comparison_forward.json \
        --ref-system candidate_scored.ckpt

ONE BOARD CONFIG PER PROCESS: RR_GRID/RR_ROBOTS/RR_ENV_DIR are read at import by
the forward stack. For base 16x16 they must be UNSET.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def system_name(ckpt, search, opts):
    """Human-readable system key (mirrors eval.compare's forward naming so the
    report tooling and a human reading two files side by side can match them)."""
    name = f"spr.fwd {search} forward move planner ({ckpt})"
    if search == "mcts":
        name += f" [c={opts['mcts_c']} backup={opts['mcts_backup']}"
        if opts["best_at_budget"]:
            name += " best-at-budget"
        if opts["stop_after"]:
            name += f" stop={opts['stop_after']}"
        name += "]"
    return name


def run_bench(a):
    import torch
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, a.threads))

    from eval.compare import aggregate, load_instances, write_markdown, _CountingGuide
    from move_planner.evaluate import Guide, walls_for
    from nn.gen_grids import GRID
    from spr.fwd import mcts as fwd

    guide = _CountingGuide(Guide(a.ckpt, a.device))
    instances, sha, meta = load_instances(a.instances)
    placeholder = bool(instances) and all(i.get("d_star") in (0, None) for i in instances)
    opts = {"mcts_c": a.mcts_c, "mcts_backup": a.mcts_backup,
            "best_at_budget": a.best_at_budget, "stop_after": a.stop_after,
            "root_noise": a.root_noise, "max_depth": a.max_depth}
    kw = {}
    if a.search == "mcts":
        kw = dict(c_puct=a.mcts_c, backup=a.mcts_backup,
                  best_at_budget=a.best_at_budget, root_noise=a.root_noise,
                  stop_after_certified=a.stop_after, max_depth=a.max_depth)

    walls, rows = {}, []
    for i, inst in enumerate(instances):
        env_id = inst["env_id"]
        if env_id not in walls:
            walls[env_id] = walls_for(env_id)
        wr, wd = walls[env_id]
        positions = tuple(tuple(p) for p in inst["positions"])
        target = tuple(inst["target"])
        d_star = inst["d_star"]
        acct = {"nn_calls": 0, "nn_policy_calls": 0, "nn_value_calls": 0,
                "physics_calls_verify": 0, "expansions_from_calls": 0,
                "expansions_own": 0}
        c0 = guide.calls
        t0 = time.perf_counter()
        res = fwd.run(a.search, guide, env_id, positions, inst["target_idx"], target,
                      wr, wd, size=GRID, k=a.k, max_expansions=a.expansions,
                      acct=acct, **({**kw, "rng": random.Random(a.seed + i)}
                                    if a.search == "mcts" else kw))
        dt = time.perf_counter() - t0
        acct["nn_calls"] = guide.calls - c0
        acct["expansions_from_calls"] = fwd.expansions_from_calls(acct["nn_calls"])
        acct["expansions_own"] = res.expansions
        cost = res.cost
        # PROBLEM.md 4.6: nothing a net reports counts until replayed. Free here.
        if cost is not None:
            ok, why = fwd.verify_path(positions, res.path, inst["target_idx"], target,
                                      wr, wd, GRID)
            if not ok or len(res.path) != cost:
                raise SystemExit(f"[spr.fwd.bench] row {i} env {env_id}: the search "
                                 f"returned an unplayable path ({why}) -- refusing to "
                                 f"write an uncertified result")
        row = {
            "env_id": env_id,
            "d_star": None if placeholder else d_star,
            "solved": cost is not None,
            "moves": cost,
            "regret": None if cost is None or placeholder else cost - d_star,
            "expansions": res.expansions,
            "seconds": dt,
            # `spr.gate.paired` reads realized_strict; for the forward arm the
            # realized move count IS the plan cost (no realization gap).
            "realized_strict": cost,
        }
        if a.dump_moves and cost is not None:
            row["moves_seq"] = fwd.move_dump(res.path)
        row["accounting"] = acct
        if res.extra:
            row["search"] = res.extra
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  [spr.fwd.bench] {i + 1}/{len(instances)}", flush=True)

    name = system_name(a.ckpt, a.search, opts)
    protocol = {
        "expansions": a.expansions, "k": a.k, "instances_file": a.instances,
        "instances_sha256": sha, "n_instances": len(instances), "instances_meta": meta,
        "checkpoints": {a.ckpt: time.strftime("%Y-%m-%dT%H:%M:%S",
                                              time.localtime(os.path.getmtime(a.ckpt)))},
        "device": a.device, "d_star_placeholder": placeholder,
        "dump_moves": a.dump_moves, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results_file": a.out, "search": a.search, "arm": "forward",
        "search_options": opts, "seed": a.seed,
        "command": " ".join(["python -m spr.fwd.bench"] + sys.argv[1:]),
        "expansion_definition":
            "one expanded search node whose children are generated (= 1 policy pass "
            "+ 1 batched value pass over <= k children), identical to "
            "eval/compare.py's forward unit; greedy arms have no policy+value pair "
            "per step, so their `expansions` counts STEPS and accounting.nn_calls "
            "carries the true NN cost",
    }
    systems = {name: {"kind": "forward", "aggregate": aggregate(rows), "rows": rows}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps({"protocol": protocol, "systems": systems}, indent=2) + "\n")
    os.replace(tmp, out)
    write_markdown(a.md, protocol, systems, [name])
    ag = systems[name]["aggregate"]
    print(f"[spr.fwd.bench] {name}: solved {ag['solved']}/{ag['n']} "
          f"regret={ag['mean_regret']} opt%={ag['pct_optimal']} "
          f"moves={ag['mean_moves']} exp={ag['mean_expansions']}", flush=True)
    print(f"SPR FWD BENCH DONE {out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# parity (F-M0)
# ---------------------------------------------------------------------------

PARITY_KEYS = ("solved", "moves", "expansions", "d_star")


def _pick_system(payload, wanted=None):
    cands = [(n, s) for n, s in payload["systems"].items() if s.get("rows")]
    if wanted:
        cands = [(n, s) for n, s in cands if wanted in n]
    if not cands:
        raise SystemExit(f"no system with rows matching {wanted!r}")
    if len(cands) > 1:
        raise SystemExit(f"ambiguous system selection {wanted!r}: "
                         f"{[n for n, _ in cands]}")
    return cands[0]


def parity(new_path, ref_path, ref_system=None, new_system=None, offset=0, log=print):
    new = json.loads(Path(new_path).read_text())
    ref = json.loads(Path(ref_path).read_text())
    nn_, ns = _pick_system(new, new_system)
    rn_, rs = _pick_system(ref, ref_system)
    log(f"[fwd-parity] new: {nn_}  ({len(ns['rows'])} rows)")
    log(f"[fwd-parity] ref: {rn_}  ({len(rs['rows'])} rows)")
    nrows, rrows = ns["rows"], rs["rows"][offset:offset + len(ns["rows"])]
    if len(nrows) != len(rrows):
        log(f"[fwd-parity] row count {len(nrows)} vs {len(rrows)} after offset "
            f"{offset} -- cannot align")
        return False
    diff = []
    for i, (x, y) in enumerate(zip(nrows, rrows)):
        if x["env_id"] != y["env_id"]:
            diff.append((i, x["env_id"], {"env_id": (x["env_id"], y["env_id"])}))
            continue
        d = {kk: (x.get(kk), y.get(kk)) for kk in PARITY_KEYS if x.get(kk) != y.get(kk)}
        if d:
            diff.append((i, x["env_id"], d))
    log(f"[fwd-parity] per-row differences on {PARITY_KEYS}: {len(diff)}/{len(nrows)}")
    for i, e, d in diff[:20]:
        log(f"[fwd-parity]   row {i} env {e}: {d}")
    ok = not diff
    log(f"F-M0 PARITY {'PASS' if ok else 'FAIL'} {Path(new_path).name} "
        f"({len(nrows)} rows vs {Path(ref_path).name}[{offset}:])")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "parity":
        p = argparse.ArgumentParser(prog="spr.fwd.bench parity")
        p.add_argument("--new", required=True)
        p.add_argument("--ref", required=True)
        p.add_argument("--ref-system", default=None,
                       help="substring selecting the recorded system (e.g. a ckpt name)")
        p.add_argument("--new-system", default=None)
        p.add_argument("--offset", type=int, default=0,
                       help="index of the new file's first row in the reference")
        p.add_argument("--out", default=None)
        a = p.parse_args(argv[1:])
        lines = []
        ok = parity(a.new, a.ref, a.ref_system, a.new_system, a.offset,
                    log=lambda s: (lines.append(s), print(s, flush=True))[0])
        if a.out:
            Path(a.out).write_text(json.dumps(
                {"pass": ok, "new": a.new, "ref": a.ref, "offset": a.offset,
                 "log": lines}, indent=1) + "\n")
        return 0 if ok else 1

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--instances", required=True)
    p.add_argument("--ckpt", required=True, help="a MoveNet checkpoint")
    p.add_argument("--search", default="astar",
                   choices=["astar", "mcts", "greedy_value", "greedy_policy"])
    p.add_argument("--expansions", type=int, default=1200)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--best-at-budget", action="store_true",
                   help="mcts: return the SHORTEST goal path found in the budget "
                        "instead of the first")
    p.add_argument("--mcts-c", type=float, default=1.5)
    p.add_argument("--mcts-backup", choices=["min", "mean"], default="min")
    p.add_argument("--root-noise", type=float, default=0.0,
                   help="Dirichlet root noise (generation only; 0 for the gate)")
    p.add_argument("--stop-after", type=int, default=None,
                   help="mcts: stop when the best path has not improved for N "
                        "expansions")
    p.add_argument("--max-depth", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2, help="torch threads")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dump-moves", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--md", default="/dev/null")
    a = p.parse_args(argv)
    return run_bench(a)


if __name__ == "__main__":
    sys.exit(main())
