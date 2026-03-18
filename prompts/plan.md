# Plan: Canonical Representation & Training Data for Neural Subgoal Prediction

## Goal

Create training data from solved MCTS plans for a transformer that predicts subgoal
decompositions. Architecture: bidirectional encoder (256 grid-cell tokens) + causal
decoder (3K tokens for K subgoals).

---

## Symmetries to Canonicalize

1. **Helper robot color permutation** — Red/Blue/Green are interchangeable.
   Sort helpers lexicographically by initial position → canonical IDs 0, 1, 2.

2. **DAG branch ordering** — Independent subtrees (e.g., bottleneck-branch vs
   support-branch of a subgoal) can execute in any order.
   Serialize via top-down DFS, bottleneck-branch first, tiebreak by
   lexicographic position.

Minor/deferred:
- Equivalent helper assignment (two helpers equidistant to support) — broken by
  canonical ID (MCTS picks one, canonical ID makes it deterministic).
- Multiple optimal plans (different decompositions, same cost) — training data
  curation issue, not representation. Train on whatever MCTS finds.
- Board rotation/reflection — rare, not worth handling in V1.

---

## Step 1: Per-Node Feature Vector

Each of the 256 grid cells (x, y) gets a feature vector:

| Feature       | Dim | Description                                    |
|---------------|-----|------------------------------------------------|
| position      | 2   | (x, y) as integers 0–15 (or normalized)        |
| walls         | 4   | Binary: wall on N, S, E, W side of this cell   |
| is_goal       | 1   | Binary: is this the goal cell?                 |
| is_target     | 1   | Binary: is the target robot here?              |
| is_helper_0   | 1   | Binary: canonical helper 0 here?               |
| is_helper_1   | 1   | Binary: canonical helper 1 here?               |
| is_helper_2   | 1   | Binary: canonical helper 2 here?               |
| **Total**     |**11**|                                               |

### Wall extraction

`grid_data` is stored in every env pkl file (and available as `grid_env.grid_data`).
It is a flat list of 256 strings — one per cell at index `y * 16 + x`.
Each string contains characters indicating walls on that cell's sides:
- `'NW'` → walls on north and west
- `'S'`  → wall on south only
- `'X'`  → no walls (interior cell)

Extraction per cell `(x, y)`:
```python
cell = grid_data[y * 16 + x]
wall_N = 'N' in cell or y == 0      # include board border
wall_S = 'S' in cell or y == 15
wall_W = 'W' in cell or x == 0
wall_E = 'E' in cell or x == 15
```

Border walls (row 0 = N, row 15 = S, col 0 = W, col 15 = E) are OR'd in,
matching the logic in `ricochet_robots_simple/board_utils.py:parse_walls`.

### Canonical robot IDs

```python
helpers_sorted = sorted(state.helpers, key=lambda r: r.position)
canonical_id = {helpers_sorted[i].color: i for i in range(3)}
```

---

## Step 2: Decoder Target Vocabulary

Each subgoal produces 3 tokens:

| Token       | Range  | Meaning                                 |
|-------------|--------|-----------------------------------------|
| bottleneck  | 0–255  | Grid cell index (row-major: y * 16 + x) |
| support     | 0–255  | Grid cell index                         |
| robot_id    | 256–259| Which robot serves as support            |

Robot ID mapping (4 values, not 3):
- 256 = target robot (yes, the target can serve as its own support
  in multi-step plans — e.g., env 18)
- 257 = canonical helper 0
- 258 = canonical helper 1
- 259 = canonical helper 2

Special tokens:
- 260 = BOS (start of sequence)
- 261 = EOS (end of sequence, no more subgoals)

Total vocabulary: 262

Note: the bottleneck robot (which robot passes through the bottleneck) is
NOT predicted — it is implicit from the DFS traversal context:
- In a bottleneck branch: same robot as the parent subgoal
- In a support branch: the support robot from the parent subgoal
In nested subgoals, helpers can be bottleneck robots (e.g., env 13 bn_9,
env 18 bn_11/bn_16).

Output sequence for K subgoals:
```
[BOS, bn_0, sp_0, h_0, bn_1, sp_1, h_1, ..., bn_{K-1}, sp_{K-1}, h_{K-1}, EOS]
```

For plans with 0 subgoals (direct path): `[BOS, EOS]`

---

## Step 3: Canonical DAG Serialization

Top-down DFS from goal, bottleneck-branch first.

```
serialize_plan(plan):
    subgoals = []
    visit(goal_node):
        for each child of current node (in DAG order):
            if child is a subgoal node:
                bn_child = get bottleneck child of subgoal
                sp_child = get support child of subgoal
                emit (bn_child.pos, sp_child.pos, sp_child.robot)
                recurse into bn_child's subtree first  (bottleneck-first)
                then recurse into sp_child's subtree    (support second)
    return subgoals
```

At any branching point where two children have the same structural role,
tiebreak by lexicographic (position) of the bottleneck.

---

## Step 4: Implementation

### 4a. `canonical.py` — core module

Functions:
- `assign_canonical_ids(state) -> dict[str, int]`
  Map original color → canonical ID (0, 1, 2).
- `extract_node_features(grid_env, state) -> ndarray (256, 10)`
  Build the per-cell feature matrix.
- `serialize_plan(plan, state) -> list[tuple[int, int, int]]`
  DAG → canonical sequence of (bn_pos, sp_pos, helper_id).
- `encode_targets(subgoal_sequence) -> list[int]`
  Convert to flat token sequence [BOS, bn, sp, h, ..., EOS].

### 4b. `generate_training_data.py` — pipeline

```
for env_idx in all_environments:
    grid_env, state = GridEnv.from_env(env_idx)
    features = extract_node_features(grid_env, state)       # (256, 10)

    result = A_star_V1().solve(grid_env, state)
    plan = result.best_plan

    subgoals = serialize_plan(plan, state)                   # [(bn, sp, h), ...]
    tokens = encode_targets(subgoals)                        # [BOS, ..., EOS]

    save(env_idx, features, tokens)
```

Output format: single `.npz` or `.pt` file per example, or batched HDF5.

---

## Step 5: Validation

1. **Color permutation invariance**: Permute helper colors in state, re-extract
   features and targets — verify identical output (up to canonical ID remapping).
2. **Branch ordering invariance**: Manually swap branch order in a plan DAG,
   re-serialize — verify identical token sequence.
3. **Spot checks**: Print a few examples, visually confirm the subgoal sequence
   matches the plan DAG structure.
4. **Round-trip**: Reconstruct partial plan DAG from token sequence, verify it
   matches the original (structurally).

---

## Step 6: Scale

- Current: 128 environments, 1 instance each.
- Future: many more environments. Pipeline should be embarrassingly parallel
  (each env is independent).
- Consider: multiple MCTS runs per env to get diverse optimal plans (data
  augmentation), if multiple optimal decompositions exist.

---

## Open Questions

1. **Position encoding**: Raw (x, y) integers, normalized floats, or learned
   embeddings? Depends on transformer architecture details.
2. **Variable-length output**: Max K in practice? From benchmarks: most plans
   have 1 subgoal, hard ones have 3–4. Max sequence length ~15 tokens.
3. **Multiple instances per environment**: Each env pkl may contain multiple
   puzzle instances (different robot placements on same board). Each is a
   separate training example sharing the same wall structure.
