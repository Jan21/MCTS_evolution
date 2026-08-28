"""Stage 3 of PLAN_SUBGOAL_DISCOVERY.md: the learned COST-TO-GO heuristic `h`.

Stage 2 learned the wrong object. Its target, `c(s, R, C)` = the slides robot R
needs to come to rest on cell C with the other robots frozen, is exactly the
edge weight of the macro expansion -- which physics computes in under a
millisecond and which the search must compute anyway to build the child state.
What Stage 2's beam measurement showed is that the binding constraint is the
RANKING, and that the term worth adding is the remaining distance to the final
goal (`STAGE2.md` section 6 and 8).

So Stage 3 keeps physics for the edges and learns only

    h(s, R, C) = the exact number of moves still needed to put the target robot
                 on the target cell, starting from the CHILD state
                 s' = s with robot R moved to cell C.

`h` is the classic cost-to-go / value function -- the same object the project's
existing value networks learn -- but over the state subgoal space rather than
the hand-written vocabulary. The label is the exact optimum from the project's
own engine (`move_planner/oracle.py::solve`, run through its verified Rust port
`rust_datagen` for speed), never a bound.

ARCHITECTURE. Deliberately the Stage 2 net with two changes, so the comparison
with Stage 2 is about the target and not about the model:

  * encoder: unchanged size-free `nn_labeler.model.LoopedLayer`, weight-tied,
    d_model 192, 4 heads, edge-masked attention over the board's slide graph,
    `pe="none"`. Recurrence defaults to 4: at the production 12 the Stage 2 net
    sat on the constant-value plateau for 30 epochs (`STAGE2.md` section 4);
  * FOUR input channels instead of two, because cost-to-go, unlike Stage 2's
    cost, depends on the puzzle:
        0  the query robot's current cell   (the robot the macro edge moves)
        1  every other robot's cell
        2  the TARGET robot's cell
        3  the TARGET cell
    walls still arrive only through the attention masks, so the net stays
    size-free;
  * the candidate cell stays OUT of the encoder and enters the readout, so one
    encoder pass over a (state, robot) pair scores all 256 children of that
    robot and four passes score a whole expansion's 1024 candidates. This is
    what makes a 1200-expansion search affordable: 4 network calls per
    expansion instead of one per child.
  * readout gathers FIVE tokens -- query robot cell, candidate cell, target
    robot cell, target cell, global scratchpad -- so the head's first Linear is
    5*d_model, the same width as the original value net's 5-token gather.

The 96-bin HL-Gauss distributional head is kept unchanged (sigma = 1,
softmax-expectation readout). No reachability head: physics filters the
candidates before the net is ever consulted, so an unreachable cell is never
scored.

LOSSES. Cross-entropy against the HL-Gauss target on every labelled child, plus
the listwise ranking loss of the reference net applied to the quantity the beam
actually orders by:  f = c(s,R,C) + h(s,R,C)  with the EXACT c (the search has
it for free) and the predicted h. Positives are the children that truly
minimise c + h.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

HERE = Path(__file__).resolve().parent
SPR = HERE.parent
REPO = SPR.parent
SV = REPO / "supervised_valuenet"
for _p in (str(SV), str(SPR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_labeler.model import LoopedLayer                    # noqa: E402
from nn_labeler import encode as nn_encode                  # noqa: E402
from subgoal.costnet import spearman                        # noqa: E402

IN_CHANNELS = 4
UNLABELED = -1          # int8 sentinel in the `ctg` / `cost` arrays
SIGMA = 1.0             # HL-Gauss width, the reference net's frozen default
BEAM_K = 5              # the arena beam, used by the val-time survival metric


# ---------------------------------------------------------------------------
# featurisation
# ---------------------------------------------------------------------------

def state_features(positions, robot, target_idx, target, n):
    """[n*n+1, 4] float32 for one (state, query robot) pair of one puzzle.

    Row n*n is the zero global token (the scratchpad row of the reference net).
    """
    f = np.zeros((n * n + 1, IN_CHANNELS), np.float32)
    for i, p in enumerate(positions):
        f[int(p[1]) * n + int(p[0]), 0 if i == robot else 1] = 1.0
    tp = positions[target_idx]
    f[int(tp[1]) * n + int(tp[0]), 2] = 1.0
    f[int(target[1]) * n + int(target[0]), 3] = 1.0
    return f


# ---------------------------------------------------------------------------
# the net
# ---------------------------------------------------------------------------

class CtgNet(pl.LightningModule):
    """h(state, query robot, every candidate cell) = moves still needed after
    that macro edge is taken.  forward -> logits [B, n*n, num_classes]."""

    def __init__(self, d_model=192, recurrence=4, heads=4, num_classes=96,
                 lr=3e-4, weight_decay=1e-4, pe="none", rank_weight=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.enc = nn.Linear(IN_CHANNELS, d_model)
        self.layer = LoopedLayer(d_model, heads, use_global=True)
        self.head = nn.Sequential(nn.Linear(5 * d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, num_classes))
        self.register_buffer("bins", torch.arange(num_classes, dtype=torch.float32))
        self._val = []
        self._pool = {}
        self._pe_cache = {}

    def _encode(self, x, A_all, A_ind, n):
        X = self.enc(x)
        if self.hparams.pe == "sin2d":
            key = (int(n), X.device, X.dtype)
            pe = self._pe_cache.get(key)
            if pe is None:
                pe = torch.as_tensor(np.asarray(nn_encode.sin2d_pe(int(n),
                                     self.hparams.d_model)),
                                     dtype=X.dtype, device=X.device)
                self._pe_cache[key] = pe
            X = X + pe
        for _ in range(self.hparams.recurrence):
            X = self.layer(X, A_all, A_ind)
        return X

    def forward(self, x, A_all, A_ind, n):
        n = int(n)
        N = n * n
        X = self._encode(x, A_all, A_ind, n)                  # [B, N+1, d]
        b = torch.arange(X.shape[0], device=X.device)
        qidx = x[:, :N, 0].argmax(1)                          # query robot cell
        tridx = x[:, :N, 2].argmax(1)                         # target robot cell
        gidx = x[:, :N, 3].argmax(1)                          # target cell
        qtok = X[b, qidx][:, None, :].expand(-1, N, -1)
        trtok = X[b, tridx][:, None, :].expand(-1, N, -1)
        gtok = X[b, gidx][:, None, :].expand(-1, N, -1)
        wtok = X[:, N][:, None, :].expand(-1, N, -1)          # global scratchpad
        cells = X[:, :N, :]
        h = torch.cat([qtok, cells, trtok, gtok, wtok], -1)   # [B, N, 5d]
        return self.head(h)

    def ctg_hat(self, logits):
        return (F.softmax(logits, -1) * self.bins).sum(-1)

    # -- losses -----------------------------------------------------------------
    def _hl_gauss(self, y):
        dd = (self.bins - y.clamp(0, self.hparams.num_classes - 1)[..., None]) / SIGMA
        w = torch.exp(-0.5 * dd * dd)
        return w / w.sum(-1, keepdim=True)

    def _losses(self, batch):
        x, A_all, A_ind, n = batch["x"], batch["A_all"], batch["A_ind"], batch["n"]
        y, c = batch["ctg"], batch["cost"]
        logits = self(x, A_all, A_ind, n)
        lab = y >= 0
        if lab.any():
            cls = -(self._hl_gauss(y[lab].float())
                    * F.log_softmax(logits[lab], -1)).sum(-1).mean()
            v = self.ctg_hat(logits)
            rank = self._rank(v, y, c, lab)
        else:
            cls = logits.sum() * 0
            rank = logits.sum() * 0
        return cls + self.hparams.rank_weight * rank, dict(cls=cls, rank=rank)

    def _rank(self, v, y, c, lab):
        """Listwise loss on the quantity the beam orders by: f = c + h, with the
        exact edge cost c and the predicted h. Positives are the children that
        truly minimise c + h."""
        ls = []
        for i in range(v.shape[0]):
            m = lab[i]
            if not bool(m.any()):
                continue
            ft = (c[i][m] + y[i][m]).float()
            fp = c[i][m].float() + v[i][m]
            o = (ft == ft.min()).float()
            ls.append(-(o / o.sum() * F.log_softmax(-fp, 0)).sum())
        return torch.stack(ls).mean() if ls else v.sum() * 0

    def training_step(self, batch, _):
        loss, parts = self._losses(batch)
        bs = batch["x"].shape[0]
        self.log("train_loss", loss, prog_bar=True, batch_size=bs)
        for k, val in parts.items():
            self.log(f"train_{k}", val, batch_size=bs)
        return loss

    def validation_step(self, batch, _):
        x, A_all, A_ind, n = batch["x"], batch["A_all"], batch["A_ind"], batch["n"]
        with torch.no_grad():
            v = self.ctg_hat(self(x, A_all, A_ind, n)).float().cpu().numpy()
        y = batch["ctg"].cpu().numpy()
        c = batch["cost"].cpu().numpy()
        sid = batch["sid"]
        for i in range(v.shape[0]):
            m = y[i] >= 0
            if m.sum() < 2:
                continue
            pred = v[i][m]
            true = y[i][m].astype(np.float64)
            cost = c[i][m].astype(np.float64)
            self._val.append(group_metrics(pred, true, cost))
            self._pool.setdefault(int(sid[i]), []).append((pred, true, cost))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        out = {}
        for k in ("mae", "spearman", "top1", "top5", "spread"):
            vals = [d[k] for d in self._val if d[k] == d[k]]
            out[f"val_{k}_group"] = float(np.mean(vals)) if vals else 0.0
        out["val_group_spread"] = out.pop("val_spread_group")   # collapse detector
        # The DECISION the beam actually makes pools all four robots of a state
        # into one candidate set, so the selection metric does too.
        t1, t5 = [], []
        for parts in self._pool.values():
            g = group_metrics(np.concatenate([x[0] for x in parts]),
                              np.concatenate([x[1] for x in parts]),
                              np.concatenate([x[2] for x in parts]))
            t1.append(g["top1"])
            t5.append(g["top5"])
        out["val_top1"] = float(np.mean(t1)) if t1 else 0.0
        out["val_top5"] = float(np.mean(t5)) if t5 else 0.0
        out["val_decisions"] = float(len(t5))
        self.log_dict(out, prog_bar=True)
        self._val.clear()
        self._pool.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


def group_metrics(pred, true, cost):
    """Metrics over ONE (state, robot) group's labelled children.

    `top5` is the quantity the beam lives on: does the 5 smallest predicted
    f = cost + h contain a child whose TRUE f is minimal for the group? Ties in
    the predicted score are broken by candidate order (the search's own
    deterministic rule), never randomly.
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    cost = np.asarray(cost, float)
    ft, fp = cost + true, cost + pred
    opt = ft == ft.min()
    order = np.argsort(fp, kind="stable")
    return dict(mae=float(np.abs(pred - true).mean()),
                spearman=spearman(pred, true),
                top1=float(opt[order[0]]),
                top5=float(opt[order[:BEAM_K]].any()),
                spread=float(pred.std()),
                n=int(len(pred)))


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class CtgDataset(torch.utils.data.Dataset):
    """One item = one (parent state, query robot) pair = up to 256 labelled
    children. `store` is the npz produced by `stage3.py gen`."""

    def __init__(self, store, env_dir, n):
        self.env_id = np.asarray(store["env_id"])
        self.pos = np.asarray(store["positions"])
        self.tidx = np.asarray(store["target_idx"])
        self.target = np.asarray(store["target"])
        self.cost = np.asarray(store["cost"])
        self.ctg = np.asarray(store["ctg"])
        self.env_dir = str(env_dir)
        self.n = int(n)
        S, R = self.ctg.shape[0], self.ctg.shape[1]
        # drop groups with no usable label at all
        self.index = [(s, r) for s in range(S) for r in range(R)
                      if (self.ctg[s, r] >= 0).sum() >= 2]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, r = self.index[i]
        return dict(x=state_features(self.pos[s], r, int(self.tidx[s]),
                                     self.target[s], self.n),
                    ctg=self.ctg[s, r], cost=self.cost[s, r], sid=s,
                    env_id=int(self.env_id[s]), n=self.n, env_dir=self.env_dir)


def collate(items):
    n = items[0]["n"]
    xs = torch.from_numpy(np.stack([it["x"] for it in items]))
    ctg = torch.from_numpy(np.stack([it["ctg"] for it in items]).astype(np.int64))
    cost = torch.from_numpy(np.stack([it["cost"] for it in items]).astype(np.int64))
    aa, ai = [], []
    for it in items:
        A_all, A_ind = nn_encode.adjacency(it["env_dir"], it["env_id"], n)
        aa.append(torch.as_tensor(np.asarray(A_all, np.float32)))
        ai.append(torch.as_tensor(np.asarray(A_ind, np.float32)))
    return dict(x=xs, A_all=torch.stack(aa), A_ind=torch.stack(ai),
                ctg=ctg, cost=cost, n=n, sid=[it["sid"] for it in items])
