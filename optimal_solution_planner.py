import json
import os
import pickle
import subprocess

SOLVER_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'solver/target/release/ricochet-solver-cli')
BOARD_SIZE = 16


def _parse_walls(grid_data):
    N = BOARD_SIZE
    h = [[False] * N for _ in range(N + 1)]
    v = [[False] * (N + 1) for _ in range(N)]
    for y in range(N):
        for x in range(N):
            cell = grid_data[y * N + x]
            if 'N' in cell: h[y][x] = True
            if 'S' in cell: h[y + 1][x] = True
            if 'W' in cell: v[y][x] = True
            if 'E' in cell: v[y][x + 1] = True
    for i in range(N):
        h[0][i] = True; h[N][i] = True
        v[i][0] = True; v[i][N] = True
    down_walls  = [[h[row + 1][col] for row in range(N)] for col in range(N)]
    right_walls = [[v[row][col + 1] for row in range(N)] for col in range(N)]
    return down_walls, right_walls


def solve(env, instance_idx=0, timeout=5.0):
    """Solve instance_idx from an env dict loaded from a .pkl file.

    Returns the solver result dict, or None on timeout/error.
    """
    down_walls, right_walls = _parse_walls(env['grid_data'])

    inst = env['instances'][instance_idx]
    tr = inst['target_robot']
    robots = [list(tr['position'])] + [list(h['position']) for h in inst['helper_robots']]
    target = list(inst['target'])

    payload = json.dumps([{
        'down_walls':  down_walls,
        'right_walls': right_walls,
        'robots':      robots,
        'target':      target,
        'target_robot': 0,
    }])

    try:
        proc = subprocess.run(
            [SOLVER_BIN],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None

    if proc.returncode != 0:
        return None

    results = json.loads(proc.stdout)
    return results[0] if results else None


if __name__ == '__main__':
    with open('env/env_0.pkl', 'rb') as f:
        env = pickle.load(f)

    for i, inst in enumerate(env['instances']):
        result = solve(env, instance_idx=i)
        if result and result['solved']:
            print(f"Instance {i}: {result['moves']} moves — {result['path']}")
        else:
            print(f"Instance {i}: no solution")
