You are an expert software architect and engineer specializing in Python and algorithms.

I want you to implement an interface which will be used later (design of MCTS algorithms for Ricochet Robots).
You can glance at the problem_description.md and the ricochet_robots_and_or_mcts_chat.md to gain context for what will be done at a later time. There might be some definitions that you will use or explanation of certain things you might need.
You are free to look into /mnt/raid/data/Hyner_Petr/robots_mcts/ricochet_robots_simple to understand what the examples contain.

But for now - I need you to JUST create this interface:

There will be two main files: 
- benchmark.py, 
- partial_plan.py (already exists, but needs to be finished).

There can be any other files python files you might need (f.e. utils.py).

A method solve() in benchmark.py which accepts an instance of the environment (.pkl in environments folder).
This returns a list of dicts, where each dict has a key "plan" and key "stats".

We want, for now, just a mock algorithm which will create the plan/stats. The implementation should be modular; it should be simple to extend the class (multiple MCTS algorithms).

We need to be able to validate the plan (is the MCTS valid? How much does it cost?), this will be done on just N_small (=10) subset of the environment pickles. If the validation succeeds, then we evaluate on all examples (all 128 pickles).

You should obtain deep understanding of the Ricochet Robots environment that we are studying before implementing the validation part.

The validation must be implemented correctly, not mocked. We have partial_plan.py which already contains a DAG class structure. The validation must check whether the plan is possible. That means that we can walk between nodes (edges-nodes are valid). For example, node_1 and node_2 are connected by edge_1 (which the MCTS puts into plan), but this edge cannot be physically traversed, because node_2 contains an obstacle. You should make sure the validation is bulletproof.

The evaluation, for now, can be mock. Just prepare the skeleton. It can report (in the stats key) N values of AUC (N=num of examples in environments folder) - AUC is calculated from Y axis and X axis, where Y axis has the number COST (sum of distances between nodes from the plan, see problem_description.md) and X has the wallclock time. The stats key will contain all 3 things (AUC, wallclock time and cost).

You are allowed and encouraged to write a tests.py file, which you should use to validate your implementation, as well as the logic of the validate_plan method, etc.

Here are the two main methods:

- validate_plan(plan):
    - plan: the plan to validate
    - returns: True if the plan is valid, False otherwise

- run_benchmark(algorithm, num_instances):
    - algorithm: the algorithm to run
    - num_instance: the number of instances to run
    - returns: the average cost of the plans
