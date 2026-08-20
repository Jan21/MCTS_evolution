"""The unseen-board exam -- the variants lab's generalization instrument.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m variants.exam \
        --config g24r4 --boards 50 --per-board 4 --seed 777

Writes fresh lean boards (ids 20000..) to `results/variants/exam/boards_<cfg>/`
and a frontier-style instance file `results/variants/exam/<cfg>_unseen.jsonl`
(env_id, positions in COLOR_ORDER, target, target_idx, d_star=0 placeholder --
no exact labels exist and none are ever computed: solve rate, paired moves and
expansions are the metrics, exactly like the pinned frontier exams).

NOTHING may ever train on ids >= 20000 (variants/__init__.py's board-id budget);
generation uses 40000+. The exam is deterministic (seed): every arm, and the
supervised baselines, see the very same instances. Idempotent: existing boards/
instances are never overwritten (the exam is pinned the moment it first runs).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from variants import REPO

EXAM_DIR = REPO / "self_play_robots" / "results" / "variants" / "exam"
ID0 = 20000


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="g24r4")
    p.add_argument("--boards", type=int, default=50)
    p.add_argument("--per-board", type=int, default=4)
    p.add_argument("--seed", type=int, default=777)
    a = p.parse_args(argv)

    import sys
    sys.path.insert(0, str(REPO / "supervised_valuenet"))
    from scaling.configs import get, apply_env
    cfg = get(a.config)
    apply_env(cfg)
    from nn_labeler import leanboard
    from nn.generate import random_instance
    from move_planner.state import COLOR_ORDER

    boards_dir = EXAM_DIR / f"boards_{cfg.name}"
    boards_dir.mkdir(parents=True, exist_ok=True)
    out = EXAM_DIR / f"{cfg.name}_unseen.jsonl"
    ids = list(range(ID0, ID0 + a.boards))
    for i in ids:                      # boards are deterministic (seed): always
        if not (boards_dir / f"env_{i}.pkl").exists():   # restorable on a fresh clone
            leanboard.write_board(boards_dir, i, cfg.grid, walls=cfg.walls,
                                  robots=cfg.robots, seed=a.seed)
    if out.exists():
        print(f"[exam] {out} exists -- pinned, not regenerating instances")
        return
    rows = []
    for i in ids:
        env, s0 = leanboard.from_env(i, env_dir=boards_dir)
        colors = [s0.target_robot.color] + [h.color for h in s0.helpers]
        rng = random.Random(a.seed * 1000003 + i)
        for _ in range(a.per_board):
            st = random_instance(env, colors, rng)
            by_color = {st.target_robot.color: st.target_robot.position}
            by_color.update({h.color: h.position for h in st.helpers})
            order = [c for c in COLOR_ORDER if c in by_color]
            assert len(order) == len(by_color)
            rows.append({
                "env_id": i,
                "positions": [list(by_color[c]) for c in order],
                "target": list(st.target),
                "target_idx": order.index(st.target_robot.color),
                "d_star": 0,
            })
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out)
    print(f"[exam] {len(rows)} instances on {len(ids)} fresh boards -> {out}")
    print(f"EXAM DONE {out}")


if __name__ == "__main__":
    main()
