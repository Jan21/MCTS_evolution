You are an algorithm design expert and your goal is to propose A* algorithm which will find a promising plan for a given instance of the game called Ricochet Robots.

## Game description:
Ricochet Robots is a puzzle game played on a grid board with walls on some cell edges. There is one target robot that must reach a designated target cell, and N helper robots that serve as obstacles. All robots slide in a chosen cardinal direction until they hit a wall or another robot — they cannot stop mid-slide. The goal is to find a sequence of moves (of any robot) that brings the target robot to the target cell, ideally in as few moves as possible.


# Definitions:
Target robot - the robot which needs to be moved to the goal position.

Helper robots - the robots which serve as obstacles.

Graph of the instance - the graph of the instance is a graph which encodes which edges can be traversed with or without help of other robot. The edges connect cells in the grid. If the robot can slide from one cell to the other by stoping because of a wall then these two cells will be connected with an edge without any attribute. If the robot would require other robot stop at a given cell then this edge will have an attribute "dependent" with a value which says where the helper robot need to be.

Final Component - the final component is a set of all cells from which a given robot can get to the goal position without any other helper. In the graph of the instance it correspond to all ancestor vertices of the goal vertex if we ignore the "dependent" edges.

Bottleneck position - it is a position in the final component from which the robot can get to the goal without help. It is a position to which the robot can get from a state which is not in the final component through a "dependent" edge.

Bottleneck support - position of a helper robot for a given bottleneck.

Subgoal State - a tuple of a bottleneck position and its bottlenect support.

Exact Shortest Path Length - the length of a shortest path which is not using the dependent edges. It is infinity if there is no path

Relaxed Shortest Path Lenght - the length of a shortest path which is using the dependent edges. It is infinity if there is no path. The dependent edges are penalized with a cost of 1.

Partial Plan - is a DAG which says which subgoals need to be achieved in order to achieve the goal. The root of the DAG is the goal position and the leaves are the positions of helper robots used and the target robot.
The intermediated vertices of the DAG are subgoal states and the edges are dependencies between subgoal states.
Each intermediate vertex is a tuple of a bottleneck position and its bottlenect support and has two incoming edges which either go to the leaves or to other intermediate vertices.
The intermediate vertex can also have two outgoing edges if the helper robot in this subgoal state is later used as a support in another subgoal state.

Segment of a Partial Plan - is an edge in the DAG

Fixed Segment - is a segment for which the exact shortest path is known. It is created by proposing a subgoal state for an Open Segment because the path from the bottleneck position to the goal position does not require any helper robots so shortest path is easy to compute.

Open Segment - is a segment for which the exact shortest path is not yet known.

Cost of a segment - is eather the relaxed shortest path length for the Open Segment or the exact shortest path length for the Fixed Segment.

Complete Plan - is a Partial Plan which has no open segments.

Plan cost - is is a sum of costs of all segments in the plan.


## A* algorithm:
The A* algorithm will work by proposing subgoal states for open segments. It starts from the open segment which goes from the goal position to the position of the target robot. Each subgoal state creates two new open segments, one for the bottleneck position and one for the bottleneck support.
The subgoal state can also rewire the plan because if a subgoal requires a helper robot that is already used in another subgoal state then the plan needs to be rewired by introducing relocation edge. There could be multiple ways how this A* algorithm can be realized. In any case we will also use pruning to prune partial plans whose cost is higher than the best complete plan found so far.

Goal is to produce complete plan with the lowest cost. We do not care if the plan is realizable. The only thing which we need to check is whether all segments in the plan correspond to paths which do not traverse any "dependent" edges. The plan may not be realizable because some robots may block paths of other robots but we do not care about it. We will use the function validate in validate_plan.py

The algorithms should be implemented in A_star/Astar.py in the solve function

The algorithm will get the instance of the game and a state as an input and it can use the methods of that instance. Most importantly, it will use "propose_subgoal_states",  "subgoal_score", "compute_exact_shortest_path_length", "compute_relaxed_shortest_path_length".


# Benchmarking
There is already a benchmarking setup in benchmark.py
You can use it to debug the A* algorithm and compare different version of it. It test the algorithm on N instances, validates whether the plan is correct and evaluate its cost.



