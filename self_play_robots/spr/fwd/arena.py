"""Chunked forward-arena runner: bench N-wide -> merge -> replay-certify -> parity.

The forward twin of `spr.arena.bench`, kept as a separate tiny wrapper because
`spr.arena`'s `Arm` carries a policy/value CHECKPOINT PAIR (the backward stack)
while the forward planner is a single two-headed `MoveNet`. Everything else is
the same machinery, called not forked: `spr.fwd.bench` per chunk,
`eval/merge_compare_shards.py` to concatenate the shards and recompute the
aggregates with `eval.compare.aggregate`, `eval.replay_validate` to certify
every dumped move sequence independently, and `spr.fwd.bench parity` against a
recorded comparison JSON.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.fwd.arena bench \
        --name m0_astar --ckpt move_planner/checkpoints/candidate_scored.ckpt \
        --search astar --width 8 --out runs/spr/fwd/m0_astar.json \
        --parity-ref eval/results/comparison_forward.json \
        --parity-system candidate_scored.ckpt

Idempotent + resumable: chunk outputs are keyed by a stamp over
(ckpt, instances, search, flags, budget) and skipped if already present, so a
walltime kill costs one chunk. Last line on success:
`SPR FWD ARENA DONE <name> <out>`.

ONE BOARD CONFIG PER PROCESS -- children get RR_* from `spr.arena.config_env`
(for the base `g16r4` config that means RR_GRID/RR_ROBOTS/RR_WALLS/RR_ENV_DIR
UNSET, which is what the forward stack's defaults expect).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from spr import REPO, SV
from spr.arena import _stamp, config_env, env_dir_of
from spr.fwd import RUNS


def bench(name, ckpt, instances, out, search="astar", flags=(), expansions=1200,
          k=5, width=8, threads=2, chunk_lines=8, limit=None, device="cpu",
          config="g16r4", run_root=None, log=print) -> Path:
    inst = Path(instances)
    if not inst.is_absolute():
        inst = SV / instances
    if not inst.is_file():
        raise SystemExit(f"missing instances {inst}")
    ck = Path(ckpt)
    if not ck.is_absolute():
        ck = SV / ckpt
    if not ck.is_file():
        raise SystemExit(f"missing checkpoint {ck}")
    flags = tuple(flags)
    stamp = _stamp(str(ck), str(inst), search, flags, expansions, k, limit, device)
    run = (Path(run_root) if run_root else RUNS) / f"{name}.{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints.txt").write_text(
        f"ckpt={ck}\ninstances={inst}\nsearch={search}\nflags={' '.join(flags)}\n"
        f"expansions={expansions} k={k} device={device}\n")

    lines = [l for l in inst.read_text().splitlines() if l.strip()]
    if limit:                                   # smoke runs only
        lines = lines[:limit]
        inst = run / "instances.limited.jsonl"
        inst.write_text("\n".join(lines) + "\n")
    chunks = []
    for i in range(0, len(lines), chunk_lines):
        cp = run / f"chunk.{i // chunk_lines:03d}.jsonl"
        if not cp.exists():
            cp.write_text("\n".join(lines[i:i + chunk_lines]) + "\n")
        chunks.append(cp)

    env = config_env(config)
    env["OMP_NUM_THREADS"] = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    log(f"[fwd-arena] {name}: {len(lines)} instances, {len(chunks)} chunks, "
        f"search={search} width={width} threads={threads} run={run}")

    def run_chunk(cp: Path):
        outp = cp.with_suffix(".json")
        if outp.is_file() and outp.stat().st_size > 0:
            return cp.name, "skip"
        tmp = outp.with_suffix(f".json.tmp.{os.getpid()}")
        cmd = [sys.executable, "-m", "spr.fwd.bench", "--instances", str(cp),
               "--ckpt", str(ck), "--search", search, "--expansions", str(expansions),
               "--k", str(k), "--threads", str(threads), "--device", device,
               "--dump-moves", *flags, "--out", str(tmp), "--md", "/dev/null"]
        with open(cp.with_suffix(".log"), "w") as lf:
            rc = subprocess.call(cmd, cwd=SV, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
        if rc != 0 or not tmp.is_file():
            return cp.name, f"FAIL rc={rc}"
        os.replace(tmp, outp)
        return cp.name, "ok"

    t0 = time.time()
    fails = 0
    with cf.ThreadPoolExecutor(max_workers=width) as ex:
        for nm, status in ex.map(run_chunk, chunks):
            if status.startswith("FAIL"):
                fails += 1
            log(f"[fwd-arena]   {nm}: {status} ({time.time() - t0:.0f}s)")
    if fails:
        raise SystemExit(f"[fwd-arena] {fails} chunk(s) failed (resubmit to resume)")

    out = Path(out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    shards = [str(c.with_suffix(".json")) for c in chunks]
    rc = subprocess.call([sys.executable, "eval/merge_compare_shards.py",
                          "--shards", *shards, "--instances", str(inst),
                          "--out", str(tmp), "--md", "/dev/null"],
                         cwd=SV, env=env)
    if rc != 0:
        raise SystemExit("[fwd-arena] merge failed")
    rc = subprocess.call([sys.executable, "-m", "eval.replay_validate",
                          "--compare", str(tmp), "--instances", str(inst),
                          "--env-dir", str(env_dir_of(config))],
                         cwd=SV, env=env,
                         stdout=open(run / "replay_validate.log", "w"),
                         stderr=subprocess.STDOUT)
    if rc != 0:
        os.replace(tmp, out.with_suffix(".json.uncertified"))
        raise SystemExit(f"[fwd-arena] REPLAY CERTIFICATION FAILED rc={rc} "
                         f"(quarantined; see {run / 'replay_validate.log'})")
    payload = json.loads(tmp.read_text())
    payload["spr"] = {
        "arm": name, "kind": "forward", "config": config, "search": search,
        "flags": list(flags), "expansions": expansions, "k": k, "ckpt": str(ck),
        "instances": str(inst), "run_dir": str(run),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": round(time.time() - t0, 1), "replay_certified": True,
        "width": width, "omp_threads": threads, "device": device,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, out)
    for nm, s in payload["systems"].items():
        a = s.get("aggregate") or {}
        log(f"[fwd-arena] {nm}: solved {a.get('solved')}/{a.get('n')} "
            f"regret={a.get('mean_regret')} opt%={a.get('pct_optimal')} "
            f"moves={a.get('mean_moves')} exp={a.get('mean_expansions')}")
    log(f"SPR FWD ARENA DONE {name} {out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--name", required=True)
    b.add_argument("--ckpt", required=True, help="MoveNet ckpt (abs or SV-relative)")
    b.add_argument("--instances", default="eval/data/bench450.jsonl")
    b.add_argument("--search", default="astar",
                   choices=["astar", "mcts", "greedy_value", "greedy_policy"])
    b.add_argument("--flags", default="", help="extra spr.fwd.bench flags, one string")
    b.add_argument("--expansions", type=int, default=1200)
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--width", type=int, default=8)
    b.add_argument("--threads", type=int, default=2)
    b.add_argument("--chunk-lines", type=int, default=None,
                   help="default: ceil(n/(8*width)) so every worker gets ~8 batches")
    b.add_argument("--limit", type=int, default=None, help="smoke: first N instances")
    b.add_argument("--device", default="cpu")
    b.add_argument("--config", default="g16r4")
    b.add_argument("--run-root", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--parity-ref", default=None, help="recorded comparison JSON")
    b.add_argument("--parity-system", default=None, help="substring picking its system")
    b.add_argument("--parity-out", default=None)
    a = p.parse_args(argv)

    chunk_lines = a.chunk_lines
    if chunk_lines is None:
        inst = Path(a.instances) if Path(a.instances).is_absolute() else SV / a.instances
        n = a.limit or sum(1 for l in inst.read_text().splitlines() if l.strip())
        chunk_lines = max(1, -(-n // max(1, 8 * a.width)))
    out = bench(a.name, a.ckpt, a.instances, a.out, search=a.search,
                flags=tuple(a.flags.split()), expansions=a.expansions, k=a.k,
                width=a.width, threads=a.threads, chunk_lines=chunk_lines,
                limit=a.limit, device=a.device, config=a.config, run_root=a.run_root)
    if a.parity_ref:
        from spr.fwd.bench import parity
        ref = Path(a.parity_ref)
        if not ref.is_absolute():
            ref = SV / a.parity_ref
        ok = parity(out, ref, a.parity_system)
        if a.parity_out:
            Path(a.parity_out).write_text(json.dumps({"pass": ok, "new": str(out),
                                                      "ref": str(ref)}, indent=1) + "\n")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
