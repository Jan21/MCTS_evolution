"""Build viz_data.pkl — a single pickle with all envs + plans for the visualizer."""

import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from GridEnv import GridEnv


def main():
    plans_root = os.path.join(os.path.dirname(__file__), 'plans')
    n_envs = len([d for d in os.listdir(plans_root) if d.startswith('env_')])
    print(f'Found {n_envs} environments')

    data = {}
    total_plans = 0

    for env_id in range(n_envs):
        grid_env, state = GridEnv.from_env(env_id)
        plans_dir = os.path.join(plans_root, f'env_{env_id}')

        # Load intermediate plans
        intermediate_plans = []
        idir = os.path.join(plans_dir, 'intermediate')
        if os.path.exists(idir):
            for fname in sorted(os.listdir(idir)):
                with open(os.path.join(idir, fname), 'rb') as f:
                    intermediate_plans.append(pickle.load(f))

        # Load complete plans
        complete_plans = []
        cdir = os.path.join(plans_dir, 'complete')
        if os.path.exists(cdir):
            for fname in sorted(os.listdir(cdir)):
                with open(os.path.join(cdir, fname), 'rb') as f:
                    complete_plans.append(pickle.load(f))

        all_plans = intermediate_plans + complete_plans
        total_plans += len(all_plans)

        data[env_id] = {
            'grid_env': grid_env,
            'state': state,
            'plans': all_plans,
        }

        if env_id % 20 == 0:
            print(f'  env {env_id}: {len(intermediate_plans)} intermediate + '
                  f'{len(complete_plans)} complete')

    out_path = os.path.join(os.path.dirname(__file__), 'viz_data.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)

    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f'\nSaved {out_path}')
    print(f'  {len(data)} environments, {total_plans} plans, {sz:.1f} MB')


if __name__ == '__main__':
    main()
