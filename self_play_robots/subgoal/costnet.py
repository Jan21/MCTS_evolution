"""Stage 2 of PLAN_SUBGOAL_DISCOVERY.md: the goal-conditioned cost network.

THE COST TARGET (fixed here, and Stage 3 must use the same one)
---------------------------------------------------------------
    c(s, R, C) = the number of SLIDES robot R needs to come to rest on cell C
                 from joint state s, with EVERY OTHER ROBOT FROZEN where it
                 stands (they are blockers, they never move).

That is `subgoal/space.py::rest_cells` -- the exact joint-state BFS Stage 1
used, and the exact edge cost of the macro expansion Stage 3 will use. It is
chosen over the "other robots may move too" variant for three reasons:

  1. it is what the search's node expansion does. A macro edge in
     `space.py::search` moves ONE robot while the others stand still, so the
     edge weight IS this number. The free-robots variant is not an edge weight
     at all: it does not name a unique successor state (many joint states have
     R on C), so a search cannot take it as a child;
  2. because each edge moves one robot with the rest frozen, the concatenation
     of a path's edges is a legal primitive move sequence by construction --
     the property that makes every Stage 1 solution replay-certify;
  3. it is exactly computable and cheap (one BFS per (state, robot)), so every
     training label is ground truth, never a bound.

Cells R cannot come to rest on are UNREACHABLE (no cost). R's own cell is not a
candidate. Both are excluded from every cost metric; reachability is predicted
by a separate 1-logit head (below).

THE ARCHITECTURE CHANGE
-----------------------
Encoder: unchanged. `nn_labeler.model.LoopedLayer` stacked `recurrence` times
with weight tying, edge-masked attention over the board's slide graph
(`nn_labeler.encode.adjacency`), `pe="none"` -- the production size-free recipe
(FINDINGS 53). Board side enters only through the token count, so the net is
size-free exactly as before.

Conditioning: the input channels are the change. The old value net's 9 channels
describe a hand-written (bottleneck, support, helper) candidate. The physics of
c(s, R, C) depends on nothing but the query robot's cell, the other robots'
cells and the walls, so the input is TWO channels:

    0   the query robot's current cell
    1   every other robot's cell (shared channel; colour is irrelevant here)

and the walls arrive, as before, only through the attention masks.

The goal cell does NOT enter the encoder. It enters the readout, so ONE encoder
pass over a (state, robot) pair scores all 256 goal cells for that robot and
four passes score all 1024 candidates of a decision -- the "one batched network
pass" the plan asks for.

Head: the 96-bin distributional cost head is kept (same depth, same width, same
HL-Gauss target, same softmax-expectation readout, same ranking loss). The only
change is its gather: the old head reads 5 tokens (bottleneck, support, helper,
seg_start, seg_end); this one reads the 3 tokens a (state, R, C) query has --
the query robot's cell, the goal cell, and the global scratchpad token -- so the
first Linear is 3*d_model wide instead of 5*d_model.

Added: a 1-logit reachability head on the same 3 tokens. Cost is only defined
for reachable goals, so the cost head is trained masked to them; the binary head
carries "can R stop here at all". Ranking over all 1024 uses the expected cost
    score = p_reach * cost_hat + (1 - p_reach) * UNREACH_PENALTY
and ranking over the physics-filtered reachable set uses `cost_hat` alone.
"""
from __future__ import annotations

import sys
from collections import deque
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

from simulate import DIRECTIONS, slide                      # noqa: E402
from nn_labeler.model import LoopedLayer                    # noqa: E402
from nn_labeler import encode as nn_encode                  # noqa: E402

IN_CHANNELS = 2
UNREACHABLE = -1          # label in the int8 cost arrays
UNREACH_PENALTY = 50.0    # cost charged to an unreachable goal when ranking 1024


# ---------------------------------------------------------------------------
# exact labels
# ---------------------------------------------------------------------------

def cost_row(positions, robot, wr, wd, n):
    """[n*n] int8 of c(s, robot, C) for every cell C; -1 where unreachable.

    Same BFS as `subgoal/space.py::rest_cells`, written into a flat array. The
    query robot's own cell is -1: standing still is not a subgoal.
    """
    out = np.full(n * n, UNREACHABLE, np.int8)
    blockers = frozenset(tuple(p) for i, p in enumerate(positions) if i != robot)
    start = tuple(positions[robot])
    seen = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        g = seen[cur]
        for d in DIRECTIONS:
            nxt = slide(cur, d, blockers, wr, wd, n)
            if nxt == cur or nxt in seen:
                continue
            seen[nxt] = g + 1
            out[nxt[1] * n + nxt[0]] = g + 1
            q.append(nxt)
    return out


def relaxed_dist(goal, wr, wd, n, key):
    """Baseline B1: the "any-stop" relaxation of `subgoal/space.py::relaxed_h`.

    One move may end on ANY cell of the row/column segment it can see (walls
    only, other robots ignored). Every real slide is such a move, so this is an
    admissible lower bound on c(s, R, C); it reads the board only. Returned as
    [n*n] float with inf where unreachable-even-relaxed. Symmetric, so one BFS
    from `goal` gives the distance from every possible robot cell.
    """
    seg = _segments(wr, wd, n, key)
    dist = np.full(n * n, np.inf, np.float32)
    g = goal[1] * n + goal[0]
    dist[g] = 0.0
    q = deque([g])
    while q:
        c = q.popleft()
        for nb in seg[c]:
            if not np.isfinite(dist[nb]):
                dist[nb] = dist[c] + 1
                q.append(nb)
    return dist


def lone_robot_cost(positions, robot, wr, wd, n):
    """Baseline B2: the same BFS as `cost_row` but with the board EMPTIED of the
    other robots -- exact slide physics for a robot alone. Stronger than B1 (it
    obeys the real stopping rule) and still learning-free; not a bound in either
    direction, because other robots can both block and help."""
    return cost_row([positions[robot]], 0, wr, wd, n)


_SEG_CACHE: dict = {}


def _segments(wr, wd, n, key):
    """[n*n] lists of the cells each cell can see along its row/column.
    `key` must identify the board (env id); the wall sets are not hashable."""
    key = (key, n)
    hit = _SEG_CACHE.get(key)
    if hit is not None:
        return hit
    out = []
    for y in range(n):
        for x in range(n):
            nbr = []
            cx = x
            while not (cx == n - 1 or (cx, y) in wr):
                cx += 1
                nbr.append(cx + y * n)
            cx = x
            while not (cx == 0 or (cx - 1, y) in wr):
                cx -= 1
                nbr.append(cx + y * n)
            cy = y
            while not (cy == n - 1 or (x, cy) in wd):
                cy += 1
                nbr.append(x + cy * n)
            cy = y
            while not (cy == 0 or (x, cy - 1) in wd):
                cy -= 1
                nbr.append(x + cy * n)
            out.append(nbr)            # appended in flat-index order y*n + x
    _SEG_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# featurisation
# ---------------------------------------------------------------------------

def state_features(positions, robot, n):
    """[n*n+1, 2] float32 -- channel 0 the query robot, channel 1 the others.
    Row n*n is the zero global token (the scratchpad row of the reference net)."""
    f = np.zeros((n * n + 1, IN_CHANNELS), np.float32)
    for i, p in enumerate(positions):
        f[p[1] * n + p[0], 0 if i == robot else 1] = 1.0
    return f


# ---------------------------------------------------------------------------
# the net
# ---------------------------------------------------------------------------

SIGMA = 1.0            # HL-Gauss width, the reference net's frozen default
CLASS_WEIGHT = 1.0


class GoalCostNet(pl.LightningModule):
    """Goal-conditioned cost net: c(state, query robot, every goal cell).

    forward(x, A_all, A_ind, n) -> (cost_logits [B, n*n, num_classes],
                                    reach_logit [B, n*n]).
    """

    def __init__(self, d_model=192, recurrence=12, heads=4, num_classes=96,
                 lr=3e-4, weight_decay=1e-4, pe="none", rank_weight=1.0,
                 reach_weight=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.enc = nn.Linear(IN_CHANNELS, d_model)
        self.layer = LoopedLayer(d_model, heads, use_global=True)
        self.head = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, num_classes))
        self.reach = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(),
                                   nn.Linear(d_model, 1))
        self.register_buffer("bins", torch.arange(num_classes, dtype=torch.float32))
        self._val = []
        self._pe_cache = {}

    # -- encoder (identical maths to SizeFreeValueNet._encode) -----------------
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
        # gather: query-robot token (channel-0 cell), every goal cell, global row
        qidx = x[:, :N, 0].argmax(1)                          # [B]
        b = torch.arange(X.shape[0], device=X.device)
        qtok = X[b, qidx][:, None, :].expand(-1, N, -1)       # [B, N, d]
        gtok = X[:, N][:, None, :].expand(-1, N, -1)          # [B, N, d]
        cells = X[:, :N, :]                                   # [B, N, d]
        h = torch.cat([qtok, cells, gtok], -1)                # [B, N, 3d]
        return self.head(h), self.reach(h).squeeze(-1)

    # -- readouts ---------------------------------------------------------------
    def cost_hat(self, cost_logits):
        return (F.softmax(cost_logits, -1) * self.bins).sum(-1)

    def score(self, cost_logits, reach_logit, penalty=UNREACH_PENALTY):
        p = torch.sigmoid(reach_logit)
        return p * self.cost_hat(cost_logits) + (1 - p) * penalty

    # -- losses (HL-Gauss + ranking, ported from SizeFreeValueNet) --------------
    def _hl_gauss(self, ctg):
        dd = (self.bins - ctg.clamp(0, self.hparams.num_classes - 1)[..., None]) / SIGMA
        w = torch.exp(-0.5 * dd * dd)
        return w / w.sum(-1, keepdim=True)

    def _losses(self, batch):
        x, A_all, A_ind, n = batch["x"], batch["A_all"], batch["A_ind"], batch["n"]
        y = batch["cost"]                                     # [B, N] int8 (-1 unreach)
        logits, rlogit = self(x, A_all, A_ind, n)
        reach = y >= 0
        rl = F.binary_cross_entropy_with_logits(rlogit, reach.float())
        if reach.any():
            tgt = y[reach].float()
            lg = logits[reach]
            cls = -(self._hl_gauss(tgt) * F.log_softmax(lg, -1)).sum(-1).mean()
            v = self.cost_hat(logits)
            rank = self._rank(v, y, reach)
        else:
            cls = logits.sum() * 0
            rank = logits.sum() * 0
        loss = CLASS_WEIGHT * cls + self.hparams.rank_weight * rank \
            + self.hparams.reach_weight * rl
        return loss, dict(cls=cls, rank=rank, reach=rl)

    def _rank(self, v, y, reach):
        """Listwise ranking loss over each group's reachable candidates, with the
        cheapest ones as the positives -- `SizeFreeValueNet._rank`, one group per
        (state, robot) row instead of one per decision."""
        ls = []
        for i in range(v.shape[0]):
            m = reach[i]
            if not bool(m.any()):
                continue
            costs = y[i][m].float()
            o = (costs == costs.min()).float()
            ls.append(-(o / o.sum() * F.log_softmax(-v[i][m], 0)).sum())
        return torch.stack(ls).mean() if ls else v.sum() * 0

    # -- steps ------------------------------------------------------------------
    def training_step(self, batch, _):
        loss, parts = self._losses(batch)
        bs = batch["x"].shape[0]
        self.log("train_loss", loss, prog_bar=True, batch_size=bs)
        for k, val in parts.items():
            self.log(f"train_{k}", val, batch_size=bs)
        return loss

    def validation_step(self, batch, _):
        x, A_all, A_ind, n = batch["x"], batch["A_all"], batch["A_ind"], batch["n"]
        y = batch["cost"]
        with torch.no_grad():
            logits, rlogit = self(x, A_all, A_ind, n)
            v = self.cost_hat(logits).float().cpu().numpy()
            pr = torch.sigmoid(rlogit).float().cpu().numpy()
        yy = y.cpu().numpy()
        for i in range(v.shape[0]):
            m = yy[i] >= 0
            if m.sum() < 2:
                continue
            self._val.append(group_metrics(v[i][m], yy[i][m].astype(np.float64))
                             | dict(reach_acc=float(((pr[i] > .5) == m).mean())))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        keys = ("mae", "spearman", "top1")
        out = {f"val_{k}": float(np.mean([d[k] for d in self._val
                                          if d[k] == d[k]])) for k in keys}
        out["val_reach_acc"] = float(np.mean([d["reach_acc"] for d in self._val]))
        self.log_dict(out, prog_bar=True)
        self._val.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


# ---------------------------------------------------------------------------
# ranking metrics (shared by the net's val loop and by stage2.py's eval)
# ---------------------------------------------------------------------------

def _rankdata(a):
    """Average ranks, ties averaged (scipy.stats.rankdata 'average')."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(pred, true):
    """Spearman rho with tie-corrected ranks; nan if either side is constant."""
    rp, rt = _rankdata(pred), _rankdata(true)
    sp, st = rp.std(), rt.std()
    if sp == 0 or st == 0:
        return float("nan")
    return float(((rp - rp.mean()) * (rt - rt.mean())).mean() / (sp * st))


def group_metrics(pred, true):
    """pred/true over ONE group of candidates. mae, spearman, top1."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    best = true.min()
    return dict(mae=float(np.abs(pred - true).mean()),
                spearman=spearman(pred, true),
                top1=float(true[int(np.argmin(pred))] == best),
                n=int(len(pred)))


# ---------------------------------------------------------------------------
# dataset / collate
# ---------------------------------------------------------------------------

class GoalCostDataset(torch.utils.data.Dataset):
    """One item = one (state, query robot) pair, i.e. 256 labelled candidates.

    `store` is the npz produced by `stage2.py gen`: env_id [S], positions
    [S, R, 2], cost [S, R, n*n] int8.
    """

    def __init__(self, store, env_dir, n):
        self.env_id = np.asarray(store["env_id"])
        self.pos = np.asarray(store["positions"])
        self.cost = np.asarray(store["cost"])
        self.env_dir = str(env_dir)
        self.n = int(n)
        S, R = self.cost.shape[0], self.cost.shape[1]
        self.index = [(s, r) for s in range(S) for r in range(R)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, r = self.index[i]
        return dict(x=state_features(self.pos[s], r, self.n),
                    cost=self.cost[s, r], env_id=int(self.env_id[s]),
                    n=self.n, env_dir=self.env_dir)


def collate(items):
    n = items[0]["n"]
    xs = torch.from_numpy(np.stack([it["x"] for it in items]))
    cost = torch.from_numpy(np.stack([it["cost"] for it in items]).astype(np.int64))
    aa, ai = [], []
    for it in items:
        A_all, A_ind = nn_encode.adjacency(it["env_dir"], it["env_id"], n)
        aa.append(torch.as_tensor(np.asarray(A_all, np.float32)))
        ai.append(torch.as_tensor(np.asarray(A_ind, np.float32)))
    return dict(x=xs, A_all=torch.stack(aa), A_ind=torch.stack(ai),
                cost=cost, n=n)
