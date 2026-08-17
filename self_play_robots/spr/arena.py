"""The gating arena: run the pinned benchmark protocol and check parity.

Thin wrapper around `supervised_valuenet/eval/compare.py` (called, never
forked): chunked 8-wide CPU evaluation, shard merge with
`eval/merge_compare_shards.py`, independent replay certification with
`eval/replay_validate.py`, and a parity check against a recorded comparison
JSON. This is PROBLEM.md M0 ("arena parity") and later the M1+ gate runner.

    # M0: reproduce a recorded supervised row from this folder
    python -m spr.arena bench --arm g16r4_b1s21_b2flags
    python -m spr.arena parity --arm g16r4_b1s21_b2flags

Arms are registered in `ARMS` below (checkpoints, instances, flags, the
recorded reference file). `bench` is idempotent + resumable (skip-if-exists per
chunk, keyed on the checkpoint identity); a walltime kill costs one chunk.
The last line of a successful bench is `SPR ARENA DONE <arm> <out>`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from spr import ASSETS, REPO, RESULTS, SV


@dataclass(frozen=True)
class Arm:
    name: str
    config: str                 # scaling config name; g16r4 = legacy (RR_* unset)
    instances: str              # relative to supervised_valuenet/
    policy: str                 # absolute or SV-relative checkpoint path
    value: str
    flags: tuple = ()           # extra eval.compare flags (search variant)
    ref: str | None = None      # recorded comparison JSON (SV-relative)
    expansions: int = 1200
    k: int = 5
    notes: str = ""
    driver: str = "compare"     # "compare" = eval.compare (per-size nets);
                                # "spr" = spr.bench (size-free nets / spr searches)


ARMS = {a.name: a for a in [
    Arm("g16r4_b1s21_b2flags", "g16r4", "eval/data/bench450.jsonl",
        str(ASSETS / "g16r4_backward_policy_b1s21.ckpt"),
        str(ASSETS / "g16r4_backward_value_b1s21.ckpt"),
        ("--backward-anytime", "--backward-b2"),
        ref="eval/results/final450_backward_b2_seed21.json",
        notes="FINDINGS 80 seed-21 replicate of the base B1 pair, benched on "
              "bench450 under the B2 inference flags (432/450 recorded)."),
    Arm("g24r4_exact_prefix", "g24r4", "scaling/data/g24r4/bench.solved.jsonl",
        str(ASSETS / "g24r4_backward_policy_exact.ckpt"),
        str(ASSETS / "g24r4_backward_value_exact.ckpt"),
        ("--backward-prefix-check",),
        ref="scaling/results/g24r4/comparison.json",
        notes="FINDINGS 67/74 g24r4 exact-taught headline pair, base vocabulary, "
              "prefix-check (205/232 recorded)."),
    Arm("g24r4_exact_prefix_frontier", "g24r4",
        "scaling/data/g24r4/bench.unsolved.jsonl",
        str(ASSETS / "g24r4_backward_policy_exact.ckpt"),
        str(ASSETS / "g24r4_backward_value_exact.ckpt"),
        ("--backward-prefix-check",),
        ref=None,
        notes="Frontier (beyond-oracle) set for the same pair; no exact optimum, "
              "solve rate + mean moves only."),
]}


# ---------------------------------------------------------------------------
# environment plumbing (one config per process; the legacy g16r4 config means
# RR_* UNSET, exactly as jobs/patterns/seed_headline_pair.slurm does)
# ---------------------------------------------------------------------------

def config_env(config: str) -> dict:
    """RR_* variables selecting `config` for a child process."""
    sys.path.insert(0, str(SV))
    from scaling.configs import get, env as cfg_env
    cfg = get(config)
    e = dict(os.environ)
    for k in ("RR_GRID", "RR_ROBOTS", "RR_WALLS", "RR_ENV_DIR"):
        e.pop(k, None)
    if not cfg.legacy:
        e.update(cfg_env(cfg))
    e["PYTHONPATH"] = f"{SV}:{SV.parent / 'self_play_robots'}"
    e["PYTHONUNBUFFERED"] = "1"
    return e


def env_dir_of(config: str) -> Path:
    sys.path.insert(0, str(SV))
    from scaling.configs import get
    return get(config).env_dir_abs


def _stamp(*parts) -> str:
    return hashlib.sha256("\n".join(str(p) for p in parts).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def bench(arm: Arm, out: Path, width=8, threads=2, chunk_lines=8,
          run_root: Path | None = None, limit: int | None = None,
          device="cpu", log=print) -> Path:
    inst = SV / arm.instances
    if not inst.is_file():
        raise SystemExit(f"missing instances {inst}")
    for c in (arm.policy, arm.value):
        if not Path(c).is_file():
            raise SystemExit(f"missing checkpoint {c}")
    stamp = _stamp(arm.policy, arm.value, arm.instances, arm.flags,
                   arm.expansions, arm.k, limit)
    run = (run_root or (REPO / "runs" / "spr")) / f"{arm.name}.{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints.txt").write_text(
        f"policy={arm.policy}\nvalue={arm.value}\ninstances={inst}\n"
        f"flags={' '.join(arm.flags)}\nexpansions={arm.expansions} k={arm.k}\n")
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
    env = config_env(arm.config)
    env["OMP_NUM_THREADS"] = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    log(f"[arena] {arm.name}: {len(lines)} instances, {len(chunks)} chunks, "
        f"width={width} threads={threads} run={run}")

    def run_chunk(cp: Path):
        outp = cp.with_suffix(".json")
        if outp.is_file() and outp.stat().st_size > 0:
            return cp.name, "skip"
        tmp = outp.with_suffix(f".json.tmp.{os.getpid()}")
        if arm.driver == "spr":
            cmd = [sys.executable, "-m", "spr.bench",
                   "--instances", str(cp), "--expansions", str(arm.expansions),
                   "--k", str(arm.k), "--policy", arm.policy,
                   "--value", arm.value, *arm.flags,
                   "--device", device, "--dump-moves",
                   "--out", str(tmp), "--md", "/dev/null"]
        else:
            cmd = [sys.executable, "-m", "eval.compare",
                   "--instances", str(cp), "--expansions", str(arm.expansions),
                   "--k", str(arm.k), "--backward-policy", arm.policy,
                   "--backward-value", arm.value, *arm.flags,
                   "--forward-ckpts", "", "--device", device, "--dump-moves",
                   "--out", str(tmp), "--md", "/dev/null"]
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
        for name, status in ex.map(run_chunk, chunks):
            if status.startswith("FAIL"):
                fails += 1
            log(f"[arena]   {name}: {status} ({time.time() - t0:.0f}s)")
    if fails:
        raise SystemExit(f"[arena] {fails} chunk(s) failed (resubmit to resume)")

    # merge (recomputes aggregates with eval.compare.aggregate) + certify
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    shards = [str(c.with_suffix(".json")) for c in chunks]
    rc = subprocess.call([sys.executable, "eval/merge_compare_shards.py",
                          "--shards", *shards, "--instances", str(inst),
                          "--out", str(tmp), "--md", "/dev/null"],
                         cwd=SV, env=env)
    if rc != 0:
        raise SystemExit("[arena] merge failed")
    rc = subprocess.call([sys.executable, "-m", "eval.replay_validate",
                          "--compare", str(tmp), "--env-dir",
                          str(env_dir_of(arm.config))],
                         cwd=SV, env=env,
                         stdout=open(run / "replay_validate.log", "w"),
                         stderr=subprocess.STDOUT)
    if rc != 0:
        os.replace(tmp, out.with_suffix(".json.uncertified"))
        raise SystemExit(f"[arena] REPLAY CERTIFICATION FAILED rc={rc} "
                         f"(quarantined; see {run / 'replay_validate.log'})")
    payload = json.loads(tmp.read_text())
    payload["spr"] = {
        "arm": arm.name, "config": arm.config, "flags": list(arm.flags),
        "driver": arm.driver, "expansions": arm.expansions, "k": arm.k,
        "policy": arm.policy, "value": arm.value,
        "instances": str(inst), "run_dir": str(run),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": round(time.time() - t0, 1),
        "replay_certified": True, "width": width, "omp_threads": threads,
        "device": device,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, out)
    log(f"SPR ARENA DONE {arm.name} {out}")
    return out


# ---------------------------------------------------------------------------
# parity / summary
# ---------------------------------------------------------------------------

def _backward_system(payload):
    for name, s in payload["systems"].items():
        if s.get("kind") == "backward" and s.get("rows"):
            return name, s
    raise SystemExit("no backward system with rows in payload")


def summarize(payload) -> dict:
    name, s = _backward_system(payload)
    a = s["aggregate"]
    return {"system": name, "n": a["n"], "solved": a["solved"],
            "solve_rate": a["solve_rate"], "mean_regret": a["mean_regret"],
            "pct_optimal": a["pct_optimal"], "mean_moves": a["mean_moves"],
            "mean_expansions": a["mean_expansions"],
            "mean_seconds": a["mean_seconds"]}


def parity(new_path: Path, ref_path: Path, log=print) -> bool:
    new = json.loads(new_path.read_text())
    ref = json.loads(ref_path.read_text())
    sn, sr = summarize(new), summarize(ref)
    log(f"[parity] new: {json.dumps(sn)}")
    log(f"[parity] ref: {json.dumps(sr)}")
    if (new["protocol"].get("instances_sha256")
            != ref["protocol"].get("instances_sha256")):
        log("[parity] instance sha DIFFERS -- not the same exam")
        return False
    _, ns = _backward_system(new)
    _, rs = _backward_system(ref)
    diff = []
    for i, (a, b) in enumerate(zip(ns["rows"], rs["rows"])):
        keys = ("solved", "realized_strict", "expansions", "plan_found")
        d = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
        if d:
            diff.append((i, a["env_id"], d))
    log(f"[parity] per-row differences on (solved, realized_strict, expansions,"
        f" plan_found): {len(diff)}/{len(ns['rows'])}")
    for i, e, d in diff[:20]:
        log(f"[parity]   row {i} env {e}: {d}")
    ok = (not diff and sn["solved"] == sr["solved"]
          and (sn["mean_regret"] == sr["mean_regret"]))
    log(f"M0 PARITY {'PASS' if ok else 'FAIL'} {new_path.name}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--arm", default=None, help="a registered arm name, or a NEW "
                   "name when --config/--instances/--policy/--value are given")
    b.add_argument("--config", default=None)
    b.add_argument("--instances", default=None, help="SV-relative")
    b.add_argument("--policy", default=None)
    b.add_argument("--value", default=None)
    b.add_argument("--driver", default="spr", choices=["compare", "spr"])
    b.add_argument("--flags", default="", help="extra driver flags, one string")
    b.add_argument("--expansions", type=int, default=1200)
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--out", default=None,
                   help="default self_play_robots/results/m0/<arm>.json")
    b.add_argument("--width", type=int, default=8)
    b.add_argument("--threads", type=int, default=2)
    b.add_argument("--chunk-lines", type=int, default=8)
    b.add_argument("--limit", type=int, default=None, help="smoke: first N instances")
    b.add_argument("--device", default="cpu", help="cpu (arena protocol) or cuda")
    q = sub.add_parser("parity")
    q.add_argument("--arm", required=True, choices=sorted(ARMS))
    q.add_argument("--new", default=None)
    q.add_argument("--ref", default=None)
    s = sub.add_parser("summary")
    s.add_argument("path")
    a = p.parse_args(argv)

    if a.cmd == "bench":
        if a.arm in ARMS and not a.policy:
            arm = ARMS[a.arm]
        else:
            if not (a.arm and a.config and a.instances and a.policy and a.value):
                raise SystemExit("ad-hoc arm needs --arm NAME --config --instances "
                                 "--policy --value")
            arm = Arm(a.arm, a.config, a.instances, str(Path(a.policy).resolve()),
                      str(Path(a.value).resolve()), tuple(a.flags.split()),
                      ref=None, expansions=a.expansions, k=a.k, driver=a.driver)
        out = Path(a.out) if a.out else RESULTS / "m0" / f"{arm.name}.json"
        bench(arm, out, width=a.width, threads=a.threads,
              chunk_lines=a.chunk_lines, limit=a.limit, device=a.device)
    elif a.cmd == "parity":
        arm = ARMS[a.arm]
        new = Path(a.new) if a.new else RESULTS / "m0" / f"{arm.name}.json"
        ref = Path(a.ref) if a.ref else (SV / arm.ref if arm.ref else None)
        if ref is None:
            raise SystemExit("arm has no recorded reference")
        ok = parity(new, ref)
        sys.exit(0 if ok else 1)
    else:
        print(json.dumps(summarize(json.loads(Path(a.path).read_text())),
                         indent=1))


if __name__ == "__main__":
    main()
