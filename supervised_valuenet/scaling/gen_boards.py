"""Generate a config's boards into its env dir (full stored pkl schema).

Boards come from `nn.gen_grids.make_board` seeded `random.Random(idx)` per
board id -- identical to `python -m nn.gen_grids --seed 0` for the same id --
so any shard or relaunch reproduces the same bytes. Existing pkls are NEVER
overwritten. The legacy config refuses generation (its boards pre-exist and
the stock 0-127 boards were shipped, not generated).

    python -m scaling.gen_boards --config g16r6 [--nshards 8 --shard 3] [--limit 3]
"""
from __future__ import annotations

import argparse
import pickle
import random
import time

from scaling.configs import apply_env, get


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N ids of this shard (smoke tests)")
    a = p.parse_args()
    cfg = get(a.config)
    if not 0 <= a.shard < a.nshards:
        raise SystemExit("--shard must be in [0, --nshards)")

    ids = cfg.all_ids()[a.shard::a.nshards]
    if a.limit is not None:
        ids = ids[:a.limit]
    out = cfg.env_dir_abs

    if cfg.legacy:
        missing = [i for i in ids if not (out / f"env_{i}.pkl").exists()]
        print(f"{cfg.name} is the legacy config: boards pre-exist in {out}; "
              f"nothing generated ({len(missing)} missing of {len(ids)} checked).")
        return

    apply_env(cfg)                       # must precede any repo import
    from nn.gen_grids import make_board  # reads RR_GRID/RR_WALLS/RR_ROBOTS now

    out.mkdir(parents=True, exist_ok=True)
    written = kept = 0
    t0 = time.time()
    for idx in ids:
        path = out / f"env_{idx}.pkl"
        if path.exists():
            kept += 1
            continue
        board = make_board(idx, random.Random(idx))
        with open(path, "wb") as f:
            pickle.dump(board, f)
        written += 1
        if written % 25 == 0:
            print(f"[{written + kept}/{len(ids)}] wrote env_{idx}.pkl "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {written} written, {kept} kept (existing) -> {out}")


if __name__ == "__main__":
    main()
