"""Self-test for nn_labeler.encode / nn_labeler.dataset -- real data, read-only.

    cd supervised_valuenet && PYTHONPATH=. python -m nn_labeler.tests.test_encode_dataset

Uses the first 2000 records of scaling/data/g16r6/backward.jsonl (config g16r6,
n=16) plus a small slice of g24r4 to exercise mixed sizes in ONE process.

The last section is the parity gate: the frozen 16x16 pipeline
(`train/encode.py::_node_features`, `train/looped_pc.py::_x257` / `_adj` / the
`ix` key construction) is run in a SUBPROCESS with RR_GRID/RR_ROBOTS/RR_WALLS/
RR_ENV_DIR set -- it can only exist there, since those modules freeze the grid
size at import -- and its arrays must match ours EXACTLY (np.array_equal).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from scaling import configs
from nn_labeler.encode import (BASE_CHANNELS, adjacency, channels, key_indices,
                               node_features, sin2d_pe)
from nn_labeler.dataset import (GroupDataset, SizeBucketBatchSampler, by_split,
                                group_by_decision, load_corpus)

REPO = configs.REPO
LIMIT = 2000
D_MODEL = 192


def _hdr(t):
    print(f"\n=== {t} ===", flush=True)


# --------------------------------------------------------------------------- #
# encode.py
# --------------------------------------------------------------------------- #

def test_node_features(recs, n):
    _hdr("node_features")
    size = n * n
    x = node_features(recs[0], n)
    assert x.shape == (size + 1, BASE_CHANNELS), x.shape
    assert x.dtype == np.float32
    assert channels() == 9 and channels(True) == 13
    print(f"shape {x.shape} dtype {x.dtype}")

    for r in recs[:200]:
        f = node_features(r, n)
        assert f.shape == (size + 1, 9)
        assert not f[size].any(), "global row must be all zeros"
        assert set(np.unique(f)) <= {0.0, 1.0}, "channels 0-8 are binary markers"
        # channel semantics (train/encode.py:240-255)
        assert f[r["seg_start"][1] * n + r["seg_start"][0], 0] == 1
        assert f[r["seg_end"][1] * n + r["seg_end"][0], 1] == 1
        assert f[r["cand_bottleneck"][1] * n + r["cand_bottleneck"][0], 2] == 1
        assert f[r["cand_support"][1] * n + r["cand_support"][0], 3] == 1
        h = r["cand_helper"][0]
        assert f[h[1] * n + h[0], 4] == 1
        assert f[:, 5].sum() == len({tuple(hh[0]) for hh in r["helpers"]})
        for p in r["ctx_open_endpoints"]:
            assert f[p[1] * n + p[0], 6] == 1
    print(f"200 records: shapes, zero global row, binary, channel semantics OK")

    c = node_features(recs[0], n, coord_channels=True)
    assert c.shape == (size + 1, 13), c.shape
    assert np.array_equal(c[:, :9], node_features(recs[0], n)), \
        "coord channels must not disturb the 9 marker channels"
    assert not c[size].any(), "global row zero incl. coord channels"
    coords = c[:size, 9:]
    assert coords.min() >= 0.0 and coords.max() <= 1.0
    # cell (x, y) at flat y*n + x
    for (xx, yy) in [(0, 0), (n - 1, 0), (3, 7), (n // 2, n - 1)]:
        row = c[yy * n + xx, 9:]
        want = [xx / (n - 1), yy / (n - 1),
                min(xx, n - 1 - xx) / (n - 1), min(yy, n - 1 - yy) / (n - 1)]
        assert np.allclose(row, want), (xx, yy, row, want)
    print(f"coord_channels: shape {c.shape}, values in [0,1], per-cell formula OK")


def test_key_indices(recs, n):
    _hdr("key_indices")
    for r in recs[:500]:
        k = key_indices(r, n)
        assert len(k) == 5
        assert all(0 <= i <= n * n for i in k), k
        assert k[0] == r["cand_bottleneck"][1] * n + r["cand_bottleneck"][0]
        assert k[3] == r["seg_start"][1] * n + r["seg_start"][0]
        assert k[4] == r["seg_end"][1] * n + r["seg_end"][0]
    # documented extension: missing cell -> global row
    r = dict(recs[0]); r["cand_support"] = None
    assert key_indices(r, n)[1] == n * n
    print("500 records: 5 indices, in [0, n*n], order matches the frozen key; "
          "None -> global row")


def test_adjacency(env_dir, n, env_ids):
    _hdr("adjacency")
    size = n * n
    for eid in env_ids:
        A_all, A_ind = adjacency(env_dir, eid, n)
        assert A_all.shape == (size + 1, size + 1) == A_ind.shape
        assert A_all.dtype == np.float32 and A_ind.dtype == np.float32
        assert set(np.unique(A_all)) <= {0.0, 1.0}, "binary mask, not normalised"
        # _adj:50-52 -- self-loops on every row incl. the global token
        assert np.array_equal(np.diag(A_all), np.ones(size + 1, np.float32))
        assert np.array_equal(np.diag(A_ind), np.ones(size + 1, np.float32))
        # _adj:41 -- global row/col otherwise zero
        assert A_all[size, :size].sum() == 0 and A_all[:size, size].sum() == 0
        assert A_ind[size, :size].sum() == 0 and A_ind[:size, size].sum() == 0
        # A_ind is the etype==0 subset of A_all (_adj:47-49)
        assert np.array_equal(np.minimum(A_all, A_ind), A_ind)
        # Measured property of the frozen matrices (nothing in _adj symmetrises
        # anything; this comes from the graph): the relaxed/all-slide graph is
        # symmetric -- every slide u->v has a reverse v->u -- while the
        # independent-slide graph is NOT (sliding back can overshoot).
        assert np.array_equal(A_all, A_all.T), "A_all symmetric"
        assert not np.array_equal(A_ind, A_ind.T), "A_ind directed"
        off = A_all.sum() - (size + 1)
        print(f"env {eid}: shape {A_all.shape} edges(all)={int(off)} "
              f"edges(ind)={int(A_ind.sum() - (size + 1))} "
              f"A_all symmetric, A_ind asymmetric "
              f"({int((A_ind != A_ind.T).sum())} one-way cells), diag=1, "
              f"global row/col clear")
    a1 = adjacency(env_dir, env_ids[0], n)
    a2 = adjacency(str(Path(env_dir)), env_ids[0], n)
    assert a1[0] is a2[0], "lru_cache must return the same arrays"
    print("cache: repeated calls return the identical arrays")


def test_sin2d_pe(n):
    _hdr("sin2d_pe")
    size = n * n
    pe = sin2d_pe(n, D_MODEL)
    assert pe.shape == (size + 1, D_MODEL), pe.shape
    assert pe.dtype == np.float32
    assert not pe[size].any(), "global row must be zero"
    uniq = np.unique(pe[:size], axis=0)
    assert uniq.shape[0] == size, f"only {uniq.shape[0]}/{size} distinct cell rows"
    assert pe is sin2d_pe(n, D_MODEL), "cached per (n, d_model)"
    other = sin2d_pe(24, D_MODEL)
    assert other.shape == (24 * 24 + 1, D_MODEL)
    # the x-half of a row depends only on x, the y-half only on y
    half = D_MODEL // 2
    assert np.array_equal(pe[0 * n + 5, :half], pe[9 * n + 5, :half])
    assert np.array_equal(pe[3 * n + 0, half:], pe[3 * n + 7, half:])
    print(f"shape {pe.shape}, zero global row, {size}/{size} distinct cell rows, "
          f"x/y halves separable, n=24 -> {other.shape}")


# --------------------------------------------------------------------------- #
# dataset.py
# --------------------------------------------------------------------------- #

def test_grouping(recs):
    _hdr("group_by_decision")
    groups = group_by_decision(recs)
    multi = [g for g in groups if len(g) > 1]
    sizes = [len(g) for g in groups]
    one_opt, opt_is_min, has_opt = 0, 0, 0
    for g in groups:
        opt = [r for r in g if r["is_optimal"]]
        if opt:
            has_opt += 1
            if len(opt) == 1:
                one_opt += 1
            if min(r["cost_to_go"] for r in opt) == min(r["cost_to_go"] for r in g):
                opt_is_min += 1
    print(f"{len(recs)} records -> {len(groups)} groups; "
          f"sizes min/mean/max = {min(sizes)}/{sum(sizes)/len(sizes):.2f}/{max(sizes)}")
    print(f"multi-record groups: {len(multi)}/{len(groups)} "
          f"({len(multi)/len(groups):.3f})")
    print(f"groups with >=1 is_optimal: {has_opt/len(groups):.3f}; "
          f"exactly one is_optimal: {one_opt/len(groups):.3f}; "
          f"optimal sits at min cost_to_go: {opt_is_min/max(has_opt,1):.3f}")
    assert len(multi) / len(groups) > 0.5, "most groups should hold >1 candidate"
    assert one_opt / len(groups) > 0.5, "most groups should have exactly one optimum"
    assert opt_is_min == has_opt, "is_optimal must coincide with min cost_to_go"
    # collision guard: the same records under a second config must not merge
    alias = [dict(r, _config="alias") for r in recs[:200]]
    assert len(group_by_decision(recs[:200] + alias)) == \
        2 * len(group_by_decision(recs[:200])), "_config must namespace env_id"
    print("cross-config alias check: identical records under 2 configs stay apart")
    return groups


def test_by_split(recs):
    _hdr("by_split")
    cfg = configs.get("g16r6")
    sets = {c: set(configs.get(c).ids("train")) for c in ("g16r6", "g24r4")}
    tr = by_split(recs, "train", {"g16r6": sets["g16r6"]})
    va = by_split(recs, "val", {"g16r6": set(cfg.ids("val"))})
    assert all(r["env_id"] in sets["g16r6"] for r in tr)
    assert not (set(r["env_id"] for r in tr) & set(cfg.ids("val")))
    assert by_split(recs, "train", {"other": {0}}) == [], "unknown config dropped"
    nested = by_split(recs, "train", {"g16r6": {"train": sets["g16r6"]}})
    assert len(nested) == len(tr), "nested {split: ids} form must agree"
    # the 2000-line head only covers low board ids, so route a copy into val
    moved = [dict(r, env_id=cfg.ids("val")[0]) for r in recs[:50]]
    assert len(by_split(recs + moved, "val", {"g16r6": set(cfg.ids("val"))})) == 50
    assert len(by_split(recs + moved, "train", {"g16r6": sets["g16r6"]})) == len(tr)
    print(f"train {len(tr)}/{len(recs)}, val {len(va)}/{len(recs)} "
          f"(g16r6 train ids 0-699, val 700-899; the 2000-line head is all-train, "
          f"so 50 records re-stamped to board {cfg.ids('val')[0]} route to val only); "
          f"unknown-config drop + nested form OK")
    return tr


def test_group_dataset(groups):
    _hdr("GroupDataset")
    ds = GroupDataset(groups, max_per_group=4, sample=True)
    mins = [min(r["cost_to_go"] for r in g) for g in ds.groups]
    assert mins == sorted(mins), "curriculum: groups sorted by min cost_to_go"
    assert len(ds) == len(groups)
    checked = 0
    for i in range(len(ds)):
        full = ds.groups[i]
        if len(full) <= 4:
            continue
        got = ds[i]
        opt_full = [r for r in full if r["is_optimal"]]
        opt_got = [r for r in got if r["is_optimal"]]
        assert len(opt_got) == len(opt_full), "every is_optimal record is kept"
        assert len(got) == max(4, len(opt_full)), (len(got), len(opt_full))
        assert all(any(r is rr for rr in full) for r in got)
        checked += 1
    det = GroupDataset(groups, max_per_group=4, sample=False)
    big = next(i for i in range(len(det)) if len(det.groups[i]) > 4)
    a, b = det[big], det[big]
    assert [id(r) for r in a] == [id(r) for r in b], "sample=False is deterministic"
    exp = [r for r in det.groups[big] if r["is_optimal"]] + \
          [r for r in det.groups[big] if not r["is_optimal"]][:4 - len(
              [r for r in det.groups[big] if r["is_optimal"]])]
    assert [id(r) for r in a] == [id(r) for r in exp], "deterministic head order"
    nocap = GroupDataset(groups, max_per_group=None, sample=True)
    assert len(nocap[big]) == len(nocap.groups[big])
    print(f"{len(ds)} groups sorted by min ctg; {checked} oversized groups "
          f"subsampled keeping all optima; sample=False head is deterministic; "
          f"max_per_group=None passes groups through")
    return ds


def test_size_sampler(groups16, groups24):
    _hdr("SizeBucketBatchSampler")
    mixed = groups16 + groups24
    ds = GroupDataset(mixed, max_per_group=4, sample=True)
    sam = SizeBucketBatchSampler(ds, batch_size=8, shuffle=True, seed=7)
    batches = list(sam)
    seen = []
    for b in batches:
        ns = {ds.group_n(i) for i in b}
        assert len(ns) == 1, f"batch mixes sizes: {ns}"
        assert 1 <= len(b) <= 8
        seen += b
    assert sorted(seen) == list(range(len(ds))), "every group exactly once"
    assert len(batches) == len(sam)
    b2 = list(SizeBucketBatchSampler(ds, 8, shuffle=True, seed=7))
    assert b2 == batches, "same seed -> same epoch-0 batches"
    b3 = list(SizeBucketBatchSampler(ds, 8, shuffle=True, seed=8))
    assert b3 != batches, "different seed -> different batches"
    s = SizeBucketBatchSampler(ds, 8, shuffle=True, seed=7)
    e0, e1 = list(s), list(s)
    assert e0 != e1, "successive epochs reshuffle"
    fixed = list(SizeBucketBatchSampler(ds, 8, shuffle=False))
    assert [i for b in fixed for i in b] == \
        sorted(range(len(ds)), key=lambda i: (ds.group_n(i), i)), \
        "shuffle=False: ascending size, dataset order within a size"
    counts = {n: len(v) for n, v in sam.buckets.items()}
    print(f"{len(ds)} groups over sizes {counts} -> {len(batches)} batches, "
          f"all size-homogeneous, nothing dropped (partial tails kept), "
          f"seeded reproducibly, epochs differ")


# --------------------------------------------------------------------------- #
# parity against the frozen 16x16 pipeline (subprocess)
# --------------------------------------------------------------------------- #

_CHILD = r'''
import json, os, sys
import numpy as np
from train.encode import GRID, ENV_DIR, _node_features
from train.looped_pc import _adj, _x257

recs_path, ids_json, out = sys.argv[1:4]
recs = [json.loads(l) for l in open(recs_path)]
print(f"[child] GRID={GRID} ENV_DIR={ENV_DIR} recs={len(recs)}", flush=True)
np.save(os.path.join(out, "feat.npy"),
        np.stack([_node_features(r).numpy() for r in recs]))
np.save(os.path.join(out, "x257.npy"),
        np.stack([_x257(r).numpy() for r in recs]))

def ix(p):
    return p[1] * GRID + p[0]

np.save(os.path.join(out, "key.npy"),
        np.array([[ix(r["cand_bottleneck"]), ix(r["cand_support"]),
                   ix(r["cand_helper"][0]), ix(r["seg_start"]),
                   ix(r["seg_end"])] for r in recs], dtype=np.int64))
ids = json.loads(ids_json)
np.save(os.path.join(out, "A_all.npy"), np.stack([_adj(i)[0].numpy() for i in ids]))
np.save(os.path.join(out, "A_ind.npy"), np.stack([_adj(i)[1].numpy() for i in ids]))
print("[child] done", flush=True)
'''


def test_parity(recs, n, cfg, env_ids, k=50):
    _hdr("PARITY vs frozen train.encode / train.looped_pc (subprocess)")
    rng = random.Random(1234)
    sel = rng.sample(recs, k)
    with tempfile.TemporaryDirectory() as tmp:
        rp = os.path.join(tmp, "recs.jsonl")
        with open(rp, "w") as fh:
            for r in sel:
                fh.write(json.dumps({kk: v for kk, v in r.items()
                                     if not kk.startswith("_")}) + "\n")
        child = os.path.join(tmp, "child.py")
        with open(child, "w") as fh:
            fh.write(_CHILD)
        env = dict(os.environ, PYTHONPATH=str(REPO), **configs.env(cfg))
        print(f"[parent] child env: " +
              " ".join(f"{k2}={v}" for k2, v in configs.env(cfg).items()))
        p = subprocess.run([sys.executable, child, rp, json.dumps(env_ids), tmp],
                           cwd=str(REPO), env=env, capture_output=True, text=True)
        print(p.stdout.strip())
        if p.returncode != 0:
            print(p.stderr[-4000:])
            raise SystemExit("parity child failed")
        o_feat = np.load(os.path.join(tmp, "feat.npy"))
        o_x257 = np.load(os.path.join(tmp, "x257.npy"))
        o_key = np.load(os.path.join(tmp, "key.npy"))
        o_aa = np.load(os.path.join(tmp, "A_all.npy"))
        o_ai = np.load(os.path.join(tmp, "A_ind.npy"))

    mine = np.stack([node_features(r, n) for r in sel])
    assert mine.shape == o_x257.shape, (mine.shape, o_x257.shape)
    assert np.array_equal(mine, o_x257), "node_features != _x257"
    assert np.array_equal(mine[:, :n * n], o_feat), "cells != _node_features"
    assert np.array_equal(mine[:, n * n], np.zeros((k, 9), np.float32))
    print(f"node_features: EXACT match on {k}/{k} records "
          f"(vs _node_features {o_feat.shape} and _x257 {o_x257.shape}); "
          f"zero global row identical")

    mkey = np.array([key_indices(r, n) for r in sel], dtype=np.int64)
    assert np.array_equal(mkey, o_key), "key_indices != looped_pc ix/key"
    print(f"key_indices: EXACT match on {k}/{k} records (shape {mkey.shape})")

    maa = np.stack([adjacency(cfg.env_dir_abs, i, n)[0] for i in env_ids])
    mai = np.stack([adjacency(cfg.env_dir_abs, i, n)[1] for i in env_ids])
    assert np.array_equal(maa, o_aa), "A_all != _adj"
    assert np.array_equal(mai, o_ai), "A_ind != _adj"
    print(f"adjacency: EXACT match on boards {env_ids} "
          f"(A_all {maa.shape}, A_ind {mai.shape})")


# --------------------------------------------------------------------------- #

def main():
    cfg = configs.get("g16r6")
    n, env_dir = cfg.grid, str(cfg.env_dir_abs)
    data = REPO / "scaling" / "data" / "g16r6" / "backward.jsonl"
    print(f"data   {data}\nboards {env_dir}\nconfig {cfg.name} n={n} "
          f"robots={cfg.robots} walls={cfg.walls}")

    recs = load_corpus(str(data), cfg.name, n, env_dir, limit=LIMIT)
    assert len(recs) == LIMIT
    assert all(r["_config"] == "g16r6" and r["_n"] == 16 and
               r["_env_dir"] == env_dir for r in recs)
    print(f"loaded {len(recs)} records, stamped _config/_n/_env_dir")

    env_ids = sorted({r["env_id"] for r in recs})[:3]

    test_node_features(recs, n)
    test_key_indices(recs, n)
    test_adjacency(env_dir, n, env_ids)
    test_sin2d_pe(n)

    groups = test_grouping(recs)
    tr = test_by_split(recs)
    ds = test_group_dataset(groups)

    # second size, real records: mixed-size handling in ONE process
    _hdr("mixed sizes in one process (g16r6 + g24r4)")
    c24 = configs.get("g24r4")
    r24 = load_corpus(str(REPO / "scaling" / "data" / "g24r4" / "backward.jsonl"),
                      c24.name, c24.grid, str(c24.env_dir_abs), limit=500)
    g24 = group_by_decision(r24)
    f24 = node_features(r24[0], 24)
    a24 = adjacency(c24.env_dir_abs, r24[0]["env_id"], 24)[0]
    assert f24.shape == (24 * 24 + 1, 9) and a24.shape == (577, 577)
    assert node_features(recs[0], 16).shape == (257, 9), "16x16 still works after 24"
    print(f"g24r4: {len(r24)} records -> {len(g24)} groups; "
          f"node_features {f24.shape}, adjacency {a24.shape}; "
          f"g16r6 features still {node_features(recs[0], 16).shape} in the same process")
    both = by_split(recs + r24, "train",
                    {"g16r6": set(cfg.ids("train")), "g24r4": set(c24.ids("train"))})
    assert {r["_config"] for r in both} == {"g16r6", "g24r4"}
    print(f"by_split over two configs at once: {len(both)} records kept")

    test_size_sampler(groups, g24)
    test_parity(recs, n, cfg, env_ids)

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
