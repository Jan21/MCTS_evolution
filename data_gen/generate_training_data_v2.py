"""Generate training data V2 — load saved plans and extract features.

Reads intermediate plans from data_gen/plans/ (produced by dump_plans.py),
which include pre-computed best subgoal proposals for each open edge.

For each intermediate plan, for each open edge with a valid proposal,
creates one training example:
  - Input:  extract_features(grid_env, state, plan, open_edge) → (N*N, 19)
  - Target: (bn_pos, sp_pos, robot_token) → 3 ints

Output: data_gen/output_v2/
  - example_{idx}.npz: 'features' (N*N,19) float32, 'target' (3,) int64
    (N is the board side length, inferred per environment)
  - metadata.pkl: list of per-example metadata dicts

Usage:
    python -m data_gen.generate_training_data_v2
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from GridEnv import GridEnv
from data_gen.canonical_v2 import (
    extract_features, pos_to_index, index_to_pos,
    GRID_SIZE, infer_grid_size, token_target,
)


def proposal_to_target(proposal: dict, state,
                       grid_size: int = GRID_SIZE) -> tuple[int, int, int]:
    """Convert a saved proposal dict to 3 target tokens.

    The support robot is identified by its INITIAL position on the board
    (from state), not its planned support position. ``grid_size`` is the board
    side length so the tokens are correct for any NxN board.
    """
    bn_idx = pos_to_index(*proposal['bn_pos'], grid_size)
    sp_idx = pos_to_index(*proposal['sp_pos'], grid_size)

    support_robot = proposal['support_robot']
    moving_robot = proposal['moving_robot']

    if support_robot.color == moving_robot.color:
        robot_token = token_target(grid_size)
    else:
        # Find the robot's initial position from state
        initial_pos = None
        for robot in [state.target_robot] + list(state.helpers):
            if robot.color == support_robot.color:
                initial_pos = robot.position
                break
        if initial_pos is None:
            raise ValueError(f"Robot {support_robot.color} not found in state")
        robot_token = pos_to_index(int(initial_pos[0]), int(initial_pos[1]),
                                   grid_size)

    return bn_idx, sp_idx, robot_token


def main():
    plans_dir = Path(os.path.dirname(__file__)) / 'plans'
    output_dir = Path(os.path.dirname(__file__)) / 'output_v2'
    output_dir.mkdir(exist_ok=True)

    env_dirs = sorted(
        [d for d in plans_dir.iterdir() if d.name.startswith('env_')],
        key=lambda x: int(x.name.split('_')[1])
    )

    global_idx = 0
    all_metadata = []
    errors = []

    for env_path in env_dirs:
        env_id = int(env_path.name.split('_')[1])
        intermediate_dir = env_path / 'intermediate'

        if not intermediate_dir.exists():
            continue

        plan_files = sorted(intermediate_dir.glob('plan_*.pkl'),
                            key=lambda x: int(x.stem.split('_')[1]))

        if not plan_files:
            continue

        # Load env
        try:
            grid_env, state = GridEnv.from_env(env_id)
        except Exception as e:
            errors.append((env_id, f'env load: {e}'))
            continue

        # Board side length for this env (supports any NxN board)
        grid_size = infer_grid_size(grid_env)

        # Robot colors for visualization
        robot_colors = {}
        for robot in [state.target_robot] + list(state.helpers):
            idx = pos_to_index(int(robot.position[0]),
                               int(robot.position[1]), grid_size)
            robot_colors[idx] = robot.color

        env_count = 0

        for plan_file in plan_files:
            with open(plan_file, 'rb') as f:
                plan_data = pickle.load(f)

            plan = plan_data['plan']
            proposals = plan_data.get('proposals', {})

            if not proposals:
                continue

            for (parent_id, child_id), proposal in proposals.items():
                if proposal is None:
                    continue  # edge fixable directly

                try:
                    features = extract_features(
                        grid_env, state, plan, (parent_id, child_id),
                        grid_size)
                    target = proposal_to_target(proposal, state, grid_size)
                except Exception as e:
                    errors.append((env_id, f'extract: {e}'))
                    continue

                np.savez_compressed(
                    str(output_dir / f'example_{global_idx}.npz'),
                    features=features.astype(np.float32),
                    target=np.array(target, dtype=np.int64),
                )

                all_metadata.append({
                    'global_idx': global_idx,
                    'env_id': env_id,
                    'grid_size': grid_size,
                    'open_edge': (parent_id, child_id),
                    'iteration': plan_data['iteration'],
                    'score': proposal['score'],
                    'target': target,
                    'target_bn_pos': proposal['bn_pos'],
                    'target_sp_pos': proposal['sp_pos'],
                    'robot_colors': robot_colors,
                })

                global_idx += 1
                env_count += 1

        print(f'env_{env_id}: {env_count} training examples')

    # Save metadata
    with open(str(output_dir / 'metadata.pkl'), 'wb') as f:
        pickle.dump(all_metadata, f)

    print(f'\nTotal: {global_idx} training examples from {len(env_dirs)} envs')
    if errors:
        print(f'Errors ({len(errors)}):')
        for env_id, msg in errors:
            print(f'  env_{env_id}: {msg}')

    # Stats — the target-robot token depends on each env's board size
    target_count = sum(1 for m in all_metadata
                       if m['target'][2] == token_target(m['grid_size']))
    print(f'Target-as-support: {target_count}, '
          f'Helper-as-support: {global_idx - target_count}')


if __name__ == '__main__':
    main()
