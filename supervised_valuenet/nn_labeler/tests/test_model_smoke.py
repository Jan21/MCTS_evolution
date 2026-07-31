"""Smoke test for the size-free value net: SYNTHETIC inputs only.

Nothing here touches `nn_labeler.encode` / `nn_labeler.dataset`, board pkls or a
corpus: every record->tensor function is injected, so this stays green (and fast,
CPU-only, seconds) even if the encoder module is missing or changes. What it
proves is the property the fork exists for -- ONE model instance runs at several
board sizes -- plus the metric arithmetic and the batching contract.

    PYTHONPATH=. python -m nn_labeler.tests.test_model_smoke
"""
from __future__ import annotations

import math

import numpy as np
import torch

from nn_labeler.model import SizeFreeValueNet, collate_groups

RNG = np.random.default_rng(0)


# -- synthetic stand-ins for encode.node_features / adjacency / key_indices ----

def fake_features(rec, n, coord_channels=False):
    c = 13 if coord_channels else 9
    f = RNG.random((n * n + 1, c), dtype=np.float32)
    f[n * n] = 0.0                      # global/scratchpad row, as in the real one
    return f


def fake_adjacency(env_dir, env_id, n):
    """Eye + one deterministic off-diagonal ring, so the masks are not identical
    across boards but every row still has >=1 allowed key (an all-masked row would
    make softmax produce NaN)."""
    k = n * n + 1
    A_all = np.eye(k, dtype=np.float32)
    shift = 1 + (int(env_id) % 3)
    A_all[np.arange(k), (np.arange(k) + shift) % k] = 1.0
    A_ind = np.eye(k, dtype=np.float32)
    return A_all, A_ind


def fake_keys(rec, n):
    return [(i * 7 + int(rec["env_id"])) % (n * n + 1) for i in range(5)]


def fake_sin2d(n, d_model):
    pe = np.zeros((n * n + 1, d_model), np.float32)
    xs = np.tile(np.arange(n), n) / max(n - 1, 1)
    ys = np.repeat(np.arange(n), n) / max(n - 1, 1)
    half = d_model // 2
    pe[:n * n, :half] = np.sin(xs[:, None] * np.arange(1, half + 1)[None, :])
    pe[:n * n, half:] = np.cos(ys[:, None] * np.arange(1, d_model - half + 1)[None, :])
    return pe


def rec(n, env_id, ctg, opt, config="cfgA"):
    return {"env_id": env_id, "cost_to_go": ctg, "is_optimal": opt,
            "_n": n, "_config": config, "_env_dir": f"/nowhere/{config}"}


def batch(groups, coord=False, num_classes=16):
    return collate_groups(groups, featurize_fn=fake_features,
                          adjacency_fn=fake_adjacency, key_fn=fake_keys,
                          coord_channels=coord, num_classes=num_classes)


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# -- tests --------------------------------------------------------------------

def test_collate_shapes():
    g0 = [rec(8, 0, 3, True), rec(8, 0, 5, False), rec(8, 0, 4, False)]
    g1 = [rec(8, 1, 2, True), rec(8, 1, 9, False)]
    b = batch([g0, g1])
    R, K = 5, 8 * 8 + 1
    assert b["x"].shape == (R, K, 9), b["x"].shape
    assert b["A_all"].shape == (R, K, K) and b["A_ind"].shape == (R, K, K)
    assert b["key"].shape == (R, 5) and b["key"].dtype == torch.long
    assert b["ctg"].shape == (R,) and b["ctg"].dtype == torch.float32
    assert b["opt"].dtype == torch.bool and b["opt"].tolist() == [1, 0, 0, 1, 0]
    assert b["group"].tolist() == [0, 0, 0, 1, 1]      # group boundaries
    assert b["n"] == 8 and b["configs"] == ["cfgA", "cfgA"]
    assert int(b["key"].max()) <= K - 1 and int(b["key"].min()) >= 0
    print("ok  collate shapes/dtypes/group-boundaries")


def test_collate_clamp_and_dedupe():
    # num_classes=6 -> bins 0..5; 9 and 7 are clamped, 5 and 0 are not.
    b = batch([[rec(8, 0, 9, True), rec(8, 0, 5, False)],
               [rec(8, 2, 7, True), rec(8, 2, 0, False)]], num_classes=6)
    assert b["clamped"] == 2, b["clamped"]
    assert b["ctg"].tolist() == [5.0, 5.0, 5.0, 0.0]
    # two boards -> two distinct adjacency rows, each expanded to its records
    assert torch.equal(b["A_all"][0], b["A_all"][1])
    assert not torch.equal(b["A_all"][0], b["A_all"][2])
    print("ok  cost_to_go clamped to num_classes-1 (2 clamps counted), adjacency deduped")


def test_collate_rejects_mixed_sizes():
    try:
        batch([[rec(8, 0, 1, True)], [rec(12, 0, 1, True)]])
    except ValueError as e:
        assert "board sizes" in str(e), e
        print("ok  mixed-size batch rejected:", str(e).split(";")[0])
        return
    raise AssertionError("mixed _n in one batch must raise")


def test_same_instance_two_sizes(pe, inject_pe):
    """THE point of the fork: one instance, two board sizes, no reshaping."""
    m = SizeFreeValueNet(d_model=32, recurrence=2, heads=4, num_classes=16, pe=pe)
    if inject_pe:
        m.pe_fn = fake_sin2d
    assert not any("pos" in k for k in m.state_dict()), \
        "a size-shaped positional parameter came back"
    n_par = sum(p.numel() for p in m.parameters())

    outs = {}
    for n in (8, 12):
        b = batch([[rec(n, 0, 3, True), rec(n, 0, 6, False)],
                   [rec(n, 1, 2, True), rec(n, 1, 4, False)]])
        logits = m(b["x"], b["A_all"], b["A_ind"], b["n"], b["key"])
        assert logits.shape == (4, 16), (n, logits.shape)
        assert torch.isfinite(logits).all(), f"non-finite logits at n={n}"
        outs[n] = logits
    assert sum(p.numel() for p in m.parameters()) == n_par, \
        "parameter count changed with board size"
    if pe == "sin2d":
        assert len(m._pe_cache) == 2, m._pe_cache.keys()
    else:
        assert len(m._pe_cache) == 0
    # the checkpoint must stay size-free: a fresh net loads it unchanged
    SizeFreeValueNet(d_model=32, recurrence=2, heads=4, num_classes=16,
                     pe=pe).load_state_dict(m.state_dict())
    print(f"ok  pe={pe!r}: same instance forwards at n=8 and n=12 "
          f"({n_par} params, unchanged); state_dict reloads")
    return m


def test_backward(pe, inject_pe):
    m = SizeFreeValueNet(d_model=32, recurrence=2, heads=4, num_classes=16, pe=pe)
    if inject_pe:
        m.pe_fn = fake_sin2d
    total = None
    for n in (8, 12):
        b = batch([[rec(n, 0, 3, True), rec(n, 0, 6, False), rec(n, 0, 9, False)],
                   [rec(n, 2, 1, True), rec(n, 2, 5, False)]])
        loss = m.training_step(b, 0)
        assert loss.ndim == 0 and torch.isfinite(loss), (n, loss)
        total = loss if total is None else total + loss
    total.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) == len(list(m.parameters())), "some parameters got no gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    gn = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
    assert gn > 0, "zero gradient"
    print(f"ok  pe={pe!r}: training_step loss backward at both sizes "
          f"(|grad|={gn:.3f})")


def test_coord_mode():
    m = SizeFreeValueNet(d_model=32, recurrence=2, heads=4, num_classes=16,
                         pe="coord", in_channels=13)
    b = batch([[rec(8, 0, 3, True), rec(8, 0, 6, False)]], coord=True)
    assert b["x"].shape[-1] == 13
    logits = m(b["x"], b["A_all"], b["A_ind"], b["n"], b["key"])
    assert logits.shape == (2, 16) and torch.isfinite(logits).all()
    assert len(m._pe_cache) == 0, "coord mode must add no positional tensor"
    try:                                   # 9-channel input into a 13-channel net
        m(batch([[rec(8, 0, 3, True)]])["x"], b["A_all"][:1], b["A_ind"][:1], 8,
          b["key"][:1])
    except ValueError as e:
        assert "channels" in str(e), e
        print("ok  pe='coord': 13-channel forward; channel mismatch rejected")
        return
    raise AssertionError("channel mismatch must raise")


def test_val_metrics():
    """Hand-computed regret / top1 / MAE, incl. per-config split and the
    batch_idx namespacing of batch-local group ids."""
    m = SizeFreeValueNet(d_model=32, recurrence=2, heads=4, num_classes=16, pe="none")
    groups = [[rec(8, 0, 3, True, "cfgA"), rec(8, 0, 5, False, "cfgA")],
              [rec(8, 1, 2, True, "cfgB"), rec(8, 1, 7, False, "cfgB")]]
    b = batch(groups)                      # configs are per-GROUP, not per-batch
    assert b["configs"] == ["cfgA", "cfgB"]

    preds = [torch.tensor([1.0, 0.5, 0.1, 9.0]), torch.tensor([0.2, 5.0, 9.0, 0.3])]
    m._value = lambda logits: preds.pop(0)
    seen = {}
    m.log_dict = lambda d, **kw: seen.update({k: float(v) for k, v in d.items()})

    with torch.no_grad():
        m.validation_step(b, 0)            # A: picks ctg 5 (regret 2) | B: ctg 2 (0)
        m.validation_step(b, 1)            # A: picks ctg 3 (regret 0) | B: ctg 7 (5)
    m.on_validation_epoch_end()

    assert set(seen) == {"val_top1_optimal", "val_regret", "val_mae",
                         "val_regret/cfgA", "val_regret/cfgB"}, sorted(seen)
    assert close(seen["val_regret"], 7 / 4), seen          # (2+0+0+5)/4
    assert close(seen["val_top1_optimal"], 0.5), seen      # 2 of 4 decisions
    assert close(seen["val_mae"], 26.9 / 8, 1e-5), seen
    assert close(seen["val_regret/cfgA"], 1.0), seen       # (2+0)/2
    assert close(seen["val_regret/cfgB"], 2.5), seen       # (0+5)/2
    assert m._val == []
    print(f"ok  val metrics exact: regret={seen['val_regret']:.3f} "
          f"top1={seen['val_top1_optimal']:.3f} mae={seen['val_mae']:.4f} "
          f"per-config A={seen['val_regret/cfgA']:.3f} B={seen['val_regret/cfgB']:.3f}")


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)               # shared login node
    test_collate_shapes()
    test_collate_clamp_and_dedupe()
    test_collate_rejects_mixed_sizes()
    test_same_instance_two_sizes("none", inject_pe=False)
    test_same_instance_two_sizes("sin2d", inject_pe=True)
    test_backward("none", inject_pe=False)
    test_backward("sin2d", inject_pe=True)
    test_coord_mode()
    test_val_metrics()
    print("SMOKE OK  nn_labeler.model (synthetic, cpu)")


if __name__ == "__main__":
    main()
