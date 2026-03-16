# V7: K-th Best / Anti-Greedy A*

## Algorithm Description
Instead of always using the best-scored proposal, this variant systematically
explores proposals ranked 2nd, 3rd, ..., k-th. Runs multiple passes:
- Pass 0: use proposals ranked 0-2 (greedy neighborhood)
- Pass 1: use proposals ranked 1-3 (skip best)
- Pass k: use proposals ranked k to k+2

Keeps the best plan found across all passes.

## Heuristic Formula
Same as V1: `f(plan) = plan.cost()`. Only proposal selection changes.

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_iterations | 5000 | Per-pass PQ iteration limit |
| max_children | 50 | Max proposals per expansion |
| max_rank | 10 | Highest rank to try |
| time_limit | 30 | Wall-clock timeout (seconds) |

Tuned: max_rank ∈ {1,2,3,5,10}, max_children ∈ {50,100}. All configs equal or worse than V1.

## Per-Environment Behaviour
With default settings (max_rank=10): avg_cost=**9.30**, 20/20 solved.
- env_10: cost 12 (V1: 10) — 2nd-best proposal leads to worse plan
- env_11: cost 15 (V1: 9) — significant regression

Higher ranks fail completely on many environments (can't find complete plans).

## Comparison to Other Variants
**The only variant that differs from V1** — but in the wrong direction. Proves that
non-greedy proposal selection consistently leads to worse or incomplete plans.

## Intuition
This is a **negative result** that validates V1's greedy strategy: the `subgoal_score`
heuristic correctly ranks proposals for these environments. Forcing the algorithm to
use lower-ranked proposals always results in equal or higher total cost. The proposal
space is "well-behaved" — the heuristic ordering matches the true cost ordering.
