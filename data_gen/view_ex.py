"""Load generated training data and display simple + complex examples.

Usage:  python -m data_gen.view_ex
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_gen.canonical import (
    FEATURE_DIM,
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_HELPER_0,
    TOKEN_TARGET_ROBOT,
    decode_targets,
    index_to_pos,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def robot_token_str(token: int) -> str:
    if token == TOKEN_TARGET_ROBOT:
        return "TARGET"
    return f"helper_{token - TOKEN_HELPER_0}"


def show_example(env_idx: int, features: np.ndarray, tokens: np.ndarray):
    subgoals = decode_targets(tokens.tolist())

    print(f"\n{'─' * 60}")
    print(f"  ENV {env_idx}  —  {len(subgoals)} subgoal(s),  "
          f"token seq length = {len(tokens)}")
    print(f"{'─' * 60}")

    # Feature summary
    print(f"\n  Features shape: {features.shape}  dtype: {features.dtype}")

    # Walls summary
    wall_counts = features[:, 2:6].sum(axis=0)
    print(f"  Wall counts  N={int(wall_counts[0])}  S={int(wall_counts[1])}  "
          f"W={int(wall_counts[2])}  E={int(wall_counts[3])}")

    # Goal cell
    goal_idx = int(np.argmax(features[:, 6]))
    goal_pos = index_to_pos(goal_idx)
    print(f"  Goal cell:    {goal_pos}  (index {goal_idx})")

    # Target robot
    target_idx = int(np.argmax(features[:, 7]))
    target_pos = index_to_pos(target_idx)
    print(f"  Target robot: {target_pos}  (index {target_idx})")

    # Helpers
    for h in range(3):
        h_idx = int(np.argmax(features[:, 8 + h]))
        h_pos = index_to_pos(h_idx)
        print(f"  Helper {h}:     {h_pos}  (index {h_idx})")

    # Subgoals
    print(f"\n  Subgoal sequence:")
    if not subgoals:
        print("    (direct path — no subgoals)")
    for i, (bn, sp, robot) in enumerate(subgoals):
        print(f"    [{i}]  bottleneck={index_to_pos(bn)}  "
              f"support={index_to_pos(sp)}  robot={robot_token_str(robot)}")

    # Raw tokens
    print(f"\n  Raw tokens: {tokens.tolist()}")


def main():
    files = sorted(OUTPUT_DIR.glob("example_*.npz"))
    if not files:
        print(f"No examples found in {OUTPUT_DIR}")
        return

    # Load all examples
    examples = []
    for f in files:
        data = np.load(f)
        env_idx = int(f.stem.split("_")[1])
        tokens = data["tokens"]
        n_subgoals = (len(tokens) - 2) // 3  # subtract BOS+EOS, divide by 3
        examples.append({
            "env_idx": env_idx,
            "features": data["features"],
            "tokens": tokens,
            "n_subgoals": n_subgoals,
        })

    # Distribution summary
    sg_counts = [e["n_subgoals"] for e in examples]
    print(f"Loaded {len(examples)} examples from {OUTPUT_DIR}")
    print(f"\nSubgoal distribution:")
    for k in sorted(set(sg_counts)):
        count = sg_counts.count(k)
        print(f"  {k} subgoal(s): {count} envs")

    seq_lens = [len(e["tokens"]) for e in examples]
    print(f"\nToken sequence lengths: min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={np.mean(seq_lens):.1f}")

    # Pick simplest (1 subgoal, lowest env index) and most complex (excluding env 18)
    one_sg = [e for e in examples if e["n_subgoals"] == 1]
    multi_sg = [e for e in examples
                if e["n_subgoals"] > 1 and e["env_idx"] != 18]

    if one_sg:
        simple = one_sg[0]
        print(f"\n{'=' * 60}")
        print("  SIMPLE EXAMPLE")
        print(f"{'=' * 60}")
        show_example(simple["env_idx"], simple["features"], simple["tokens"])

    if multi_sg:
        # Pick the one with most subgoals
        complex_ex = max(multi_sg, key=lambda e: e["n_subgoals"])
        print(f"\n{'=' * 60}")
        print("  COMPLEX EXAMPLE")
        print(f"{'=' * 60}")
        show_example(complex_ex["env_idx"], complex_ex["features"],
                     complex_ex["tokens"])


if __name__ == "__main__":
    main()
