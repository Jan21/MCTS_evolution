"""Measure strict-executability of the warm checkpoints' traced self-play search
on the TRAINING distribution (boards 1000-1799, sample_instance), i.e. the data
loss strict_filter=True would cause. Read-only w.r.t. the repo. GPU0 only.

CPU ONLY (project GPU quota is taken by the training run). Run from
/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet with
OMP_NUM_THREADS=8 PYTHONPATH=. python <this file> [n_instances]
"""
import random
import sys
import time
from collections import Counter

import torch

torch.set_num_threads(8)

from GridEnv import GridEnv
from skeleton.astar import AStar
from train.policy_tf import PolicyTF
from train.looped_pc import LoopedValueNet
from subgoal_selfplay.config import Config
from subgoal_selfplay.selfplay import nn_astar_traced
from subgoal_selfplay.start_states import sample_instance
from eval.realize import strict_moves

N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = 12345

cfg = Config(policy_ckpt="checkpoints_backward/policy_v2.ckpt",
             value_ckpt="checkpoints_backward/value_v2.ckpt",
             device="cpu")
dev = cfg.device
policy = PolicyTF.load_from_checkpoint(cfg.policy_ckpt, map_location="cpu").to(dev).eval()
value = LoopedValueNet.load_from_checkpoint(cfg.value_ckpt, map_location="cpu").to(dev).eval()
solver = AStar(max_iters=cfg.solver_max_iters, max_frontier=cfg.solver_max_frontier)
rng = random.Random(SEED)
boards = cfg.train_ids()
print(f"train boards available: {len(boards)}  budget: k_top={cfg.k_top} "
      f"astar_iters={cfg.astar_iters} eps={cfg.epsilon}", flush=True)

draw = Counter(rng.choice(boards) for _ in range(N_INST))
n_inst = n_solved = n_strict = 0
abs_costs, strict_costs, deltas, fail_costs = [], [], [], []
chain_pass = Counter()   # (chain_len_bucket) -> [pass, total] via two counters
chain_tot = Counter()
t0 = time.time()
with torch.no_grad():
    for gid in sorted(draw):
        env, s0 = GridEnv.from_env(gid)
        colors = [s0.target_robot.color] + [h.color for h in s0.helpers]
        for _ in range(draw[gid]):
            st = sample_instance(env, colors, rng, cfg)
            if st is None:
                continue
            n_inst += 1
            out = nn_astar_traced(env, st, solver, policy, value, gid, dev, cfg, rng)
            if out is None:
                continue
            win_plan, chain = out
            n_solved += 1
            ab = int(win_plan.cost())
            stx = strict_moves(env, st, win_plan, log=None)
            b = min(len(chain), 8)
            chain_tot[b] += 1
            if stx is not None:
                n_strict += 1
                abs_costs.append(ab)
                strict_costs.append(stx)
                deltas.append(stx - ab)
                chain_pass[b] += 1
            else:
                fail_costs.append(ab)
            if n_inst % 10 == 0:
                ma = sum(abs_costs)/len(abs_costs) if abs_costs else 0.0
                ms = sum(strict_costs)/len(strict_costs) if strict_costs else 0.0
                print(f"  {n_inst}/{N_INST} solved={n_solved} strict={n_strict} "
                      f"run_mean_abs={ma:.2f} run_mean_strict={ms:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

dt = time.time() - t0
print(f"\n=== training-distribution strict-executability "
      f"(warm v2 ckpts, GEN budget) ===")
print(f"instances sampled: {n_inst}  ({dt:.0f}s, {dt/max(n_inst,1):.2f}s/inst)")
print(f"solved (traced search): {n_solved}/{n_inst} = {n_solved/max(n_inst,1):.3f}")
print(f"strict-pass among winners: {n_strict}/{n_solved} = "
      f"{n_strict/max(n_solved,1):.3f}")
if abs_costs:
    print(f"mean abstract cost (strict-passers): {sum(abs_costs)/len(abs_costs):.2f}")
    print(f"mean strict cost   (strict-passers): {sum(strict_costs)/len(strict_costs):.2f}")
    print(f"delta strict-abstract: mean {sum(deltas)/len(deltas):.2f} "
          f"min {min(deltas)} max {max(deltas)}")
    print(f"strict cost max: {max(strict_costs)}  >=50 (bin cap): "
          f"{sum(1 for c in strict_costs if c >= 50)}")
if fail_costs:
    print(f"mean abstract cost (strict-FAILERS): {sum(fail_costs)/len(fail_costs):.2f}")
print("strict-pass by decision-chain length:",
      {k: f"{chain_pass[k]}/{chain_tot[k]}" for k in sorted(chain_tot)})
