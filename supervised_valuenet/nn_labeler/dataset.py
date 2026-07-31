"""Size-parametric corpus loading, grouping and batching for the NN labeler.

Port of the frozen data path with per-record size/provenance plumbing (PLANS.md
S0.2):

  * `load_corpus`       <- `nn/benchmark.py::load` (30-31) + `_config`/`_n`/
                           `_env_dir` stamps
  * `group_by_decision` <- `nn/benchmark.py::group_by_decision` (39-51), with
                           `_config` prepended to the key
  * `by_split`          <- `nn/benchmark.py::by_split` (34-36), but the id sets
                           come from the caller per config instead of the frozen
                           global `SPLITS`
  * `GroupDataset`      <- `train/looped_pc.py::DenseDataset` (60-91), minus the
                           featurisation (which now needs `n` and belongs in the
                           model's collate) and minus `frac`

`SizeBucketBatchSampler` is new: the dense [N+1, N+1] attention masks make
cross-size padding wasteful, so every batch is drawn from one grid size.

`scaling.configs` may be imported here: it only reads the environment inside
`apply_env`/`env` (configs.py:70-88) and has no import-time env reads, so it
cannot freeze a board size the way `train.encode` / `GridEnv` do.
"""
from __future__ import annotations

import json
import os
import random

from torch.utils.data import Dataset, Sampler


def load_corpus(path: str, config: str, n: int, env_dir: str,
                limit: int | None = None) -> list[dict]:
    """Read a JSONL corpus and stamp every record with its board identity.

    The 18-field record schema carries no grid size and `env_id` restarts at 0 in
    every config, so merged corpora alias silently (PLANS.md fact 5). The three
    stamped fields are what makes a record self-describing:

        _config   config name, e.g. "g16r6" -- namespaces `env_id`
        _n        grid side
        _env_dir  absolute directory holding env_{id}.pkl for this config

    `limit` caps the number of LINES read (cheap smoke loads).
    """
    env_dir = os.path.abspath(os.fspath(env_dir))
    out: list[dict] = []
    with open(path) as fh:
        for line in fh:
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_config"], r["_n"], r["_env_dir"] = config, int(n), env_dir
            out.append(r)
    return out


def group_by_decision(records) -> list[list[dict]]:
    """Group candidate records belonging to the same decision state.

    Key is `nn/benchmark.py::group_by_decision` (46-49) verbatim -- (env_id,
    target, target_robot position, seg_start, seg_end, depth) -- with `_config`
    prepended, without which two configs' env_id 0 would merge into one group.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["_config"], r["env_id"], tuple(r["target"]),
               tuple(map(tuple, [r["target_robot"][0]])),
               tuple(r["seg_start"]), tuple(r["seg_end"]), r["depth"])
        groups.setdefault(key, []).append(r)
    return list(groups.values())


def by_split(records, split: str, ranges_by_config: dict) -> list[dict]:
    """Keep records whose `env_id` is in their own config's id set for `split`.

    `ranges_by_config` maps a config name to either
      * the id set for THIS split (build it with
        `set(scaling.configs.get(cfg).ids(split))`), or
      * a {split: ids} mapping, in which case `split` selects inside it.
    Records from a config absent from the mapping are dropped.
    """
    keep: list[dict] = []
    resolved: dict[str, set] = {}
    for cfg, spec in ranges_by_config.items():
        ids = spec.get(split, ()) if isinstance(spec, dict) else spec
        resolved[cfg] = set(ids)
    for r in records:
        ids = resolved.get(r["_config"])
        if ids is not None and r["env_id"] in ids:
            keep.append(r)
    return keep


class GroupDataset(Dataset):
    """Per-decision groups, curriculum-ordered, optionally subsampled.

    Semantics ported from `train/looped_pc.py::DenseDataset` (60-91):
      * groups sorted by their minimum `cost_to_go` (easy decisions first);
      * a group longer than `max_per_group` keeps EVERY `is_optimal` record and
        fills the remainder from the rest -- randomly when `sample=True`, else
        the deterministic head (looped_pc.py:74-79). Note the frozen quirk kept
        here: if the optimal records alone exceed `max_per_group`, the group is
        returned longer than the cap.

    `__getitem__` returns the list of (stamped) record dicts. Featurisation is
    the model's collate's job now, because it needs the per-record `_n`.
    """

    def __init__(self, groups, max_per_group: int | None = 32, sample: bool = True,
                 frac: float = 1.0):
        self.groups = sorted(groups, key=lambda g: min(r["cost_to_go"] for r in g))
        self.max_per_group, self.sample = max_per_group, sample
        self.frac = frac
        # one size per group (all records of a decision share a board)
        self.sizes = [g[0]["_n"] for g in self.groups]

    def set_frac(self, f: float) -> None:
        """Curriculum gate, ported from train/looped_pc.py::DenseDataset.set_frac
        (66-67): only the easiest `frac` of the min-ctg-sorted groups are
        trainable this epoch. Battery 1 (job 4606952) showed why it exists:
        without the ramp, 7/8 cold trainings never left the constant-value
        plateau (three architectures with byte-identical best val_regret
        1.9154), while the one arm whose data was a natural curriculum (8x8
        only, pe=none) trained fine (0.198)."""
        self.frac = max(0.05, min(1.0, f))

    def cutoff(self) -> int:
        return max(1, int(len(self.groups) * self.frac))

    def group_n(self, i: int) -> int:
        return self.sizes[i]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        g = self.groups[i]
        if self.max_per_group and len(g) > self.max_per_group:
            opt = [r for r in g if r["is_optimal"]]
            rest = [r for r in g if not r["is_optimal"]]
            k = max(0, self.max_per_group - len(opt))
            rest = random.sample(rest, min(k, len(rest))) if self.sample else rest[:k]
            g = opt + rest
        return g


class SizeBucketBatchSampler(Sampler):
    """Yield index batches whose groups all share one grid size `_n`.

    Use as `DataLoader(ds, batch_sampler=SizeBucketBatchSampler(...))`. Nothing
    is dropped: each size bucket's last partial batch is emitted too. With
    `shuffle=True` both the order WITHIN a bucket and the order of the emitted
    batches are shuffled, so sizes interleave across an epoch; with
    `shuffle=False` buckets come in ascending size, indices in dataset order
    (i.e. the curriculum order `GroupDataset` established).

    `seed=None` means "use the global RNG"; an int seeds a private RNG advanced
    once per epoch (epoch e uses seed+e), so runs are reproducible and epochs
    differ. `set_epoch(e)` overrides the counter for distributed-style control.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True,
                 seed: int | None = None):
        self.dataset, self.batch_size = dataset, int(batch_size)
        self.shuffle, self.seed, self._epoch = shuffle, seed, 0
        sizes = getattr(dataset, "sizes", None)
        if sizes is None:
            sizes = [dataset.groups[i][0]["_n"] for i in range(len(dataset))]
        self.buckets: dict[int, list[int]] = {}
        for i, n in enumerate(sizes):
            self.buckets.setdefault(int(n), []).append(i)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _batches(self, rng) -> list[list[int]]:
        # Respect the dataset's curriculum cutoff (indices are in the dataset's
        # min-ctg sort order, so "< cutoff" keeps exactly the easiest slice).
        cut = self.dataset.cutoff() if hasattr(self.dataset, "cutoff") \
            else len(self.dataset)
        out = []
        for n in sorted(self.buckets):
            idx = [i for i in self.buckets[n] if i < cut]
            if not idx:
                continue
            if rng is not None:
                rng.shuffle(idx)
            out += [idx[i:i + self.batch_size]
                    for i in range(0, len(idx), self.batch_size)]
        if rng is not None:
            rng.shuffle(out)
        return out

    def __iter__(self):
        rng = None
        if self.shuffle:
            rng = random if self.seed is None else random.Random(self.seed + self._epoch)
            self._epoch += 1
        yield from self._batches(rng)

    def __len__(self):
        return sum(-(-len(v) // self.batch_size) for v in self.buckets.values())
