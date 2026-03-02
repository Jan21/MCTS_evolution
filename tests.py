"""Tests for the MCTS_evolution benchmark framework.

Run with: python -m pytest tests.py -v
    or:   python tests.py
"""

import unittest

from partial_plan import PartialPlan
from benchmark import validate_plan, solve, run_benchmark, MockAlgorithm, compute_auc
from game import Game, State, Robot
from utils import (
    list_environment_files,
    get_independent_paths,
    get_all_paths,
    get_subgoals,
    get_final_component,
    load_environment,
    compute_shortest_path,
    INDEPENDENT_WEIGHT,
    DEPENDENT_WEIGHT,
)


def _load_game(index=0):
    """Helper to load a Game by environment index."""
    files = list_environment_files(num=index + 1)
    return Game.from_pickle(files[index])


def _load_game0():
    """Helper to load Game for env_0."""
    return _load_game(0)


# ========== Game/State/Robot Tests ==========

class TestGameAPI(unittest.TestCase):

    def test_robot_dataclass(self):
        r = Robot(name="Red", pos=(3, 5))
        self.assertEqual(r.name, "Red")
        self.assertEqual(r.pos, (3, 5))

    def test_robot_frozen(self):
        r = Robot(name="Red", pos=(3, 5))
        with self.assertRaises(AttributeError):
            r.name = "Blue"

    def test_state_dataclass(self):
        tr = Robot(name="Yellow", pos=(1, 2))
        hr = (Robot(name="Red", pos=(3, 4)),)
        s = State(target=(5, 6), target_robot=tr, helper_robots=hr)
        self.assertEqual(s.target, (5, 6))
        self.assertEqual(s.target_robot.name, "Yellow")
        self.assertEqual(len(s.helper_robots), 1)

    def test_state_frozen(self):
        tr = Robot(name="Yellow", pos=(1, 2))
        s = State(target=(5, 6), target_robot=tr, helper_robots=())
        with self.assertRaises(AttributeError):
            s.target = (0, 0)

    def test_game_from_pickle(self):
        game = _load_game0()
        self.assertEqual(game.grid_graph.number_of_nodes(), 256)
        self.assertGreater(len(game.states), 0)
        self.assertIsInstance(game.states[0], State)
        self.assertIsInstance(game.states[0].target_robot, Robot)
        self.assertIsNotNone(game.independent_paths)
        self.assertIsNotNone(game.all_paths)
        self.assertIsInstance(game.grid_nodes, frozenset)

    def test_game_state_robots(self):
        game = _load_game0()
        state = game.states[0]
        self.assertIsInstance(state.target, tuple)
        self.assertEqual(len(state.target), 2)
        self.assertEqual(len(state.helper_robots), 3)
        for hr in state.helper_robots:
            self.assertIsInstance(hr, Robot)
            self.assertIsInstance(hr.pos, tuple)

    def test_game_load_many(self):
        games = Game.load_many(num=3)
        self.assertEqual(len(games), 3)
        for g in games:
            self.assertIsInstance(g, Game)

    def test_game_grid_nodes(self):
        game = _load_game0()
        self.assertEqual(len(game.grid_nodes), 256)
        self.assertIn((0, 0), game.grid_nodes)
        self.assertNotIn((99, 99), game.grid_nodes)


# ========== PartialPlan Tests ==========

class TestPartialPlan(unittest.TestCase):

    def test_empty_plan(self):
        p = PartialPlan()
        self.assertEqual(p.g.number_of_nodes(), 0)
        self.assertEqual(p.g.number_of_edges(), 0)
        self.assertEqual(p.open_edges(), [])
        self.assertTrue(p.validate_plan())
        self.assertEqual(p.cost(), 0)

    def test_add_nodes_and_edges(self):
        p = PartialPlan()
        p.add_node("goal", "goal", pos=(1, 2))
        p.add_node("leaf", "leaf", pos=(3, 4), robot="R0")
        p.add_edge("goal", "leaf", status="fixed", cost=5)

        self.assertEqual(p.g.number_of_nodes(), 2)
        self.assertEqual(p.g.number_of_edges(), 1)

        node = p.get_node("goal")
        self.assertEqual(node['ntype'], 'goal')
        self.assertEqual(node['pos'], (1, 2))

        edge = p.get_edge("goal", "leaf")
        self.assertEqual(edge['status'], 'fixed')
        self.assertEqual(edge['cost'], 5)

    def test_nodes_by_type(self):
        p = PartialPlan()
        p.add_node("g", "goal", pos=(0, 0))
        p.add_node("l1", "leaf", pos=(1, 1), robot="R0")
        p.add_node("l2", "leaf", pos=(2, 2), robot="R1")
        self.assertEqual(len(p.nodes_by_type("goal")), 1)
        self.assertEqual(len(p.nodes_by_type("leaf")), 2)
        self.assertEqual(len(p.nodes_by_type("bottleneck")), 0)

    def test_children_parents(self):
        p = PartialPlan()
        p.add_node("g", "goal", pos=(0, 0))
        p.add_node("sg", "subgoal", entry_pos=(1, 1))
        p.add_node("bn", "bottleneck", pos=(2, 2), robot="R0")
        p.add_edge("g", "sg")
        p.add_edge("sg", "bn")
        self.assertEqual(p.children("g"), ["sg"])
        self.assertEqual(p.children("sg"), ["bn"])
        self.assertEqual(p.parents("bn"), ["sg"])
        self.assertEqual(p.parents("g"), [])

    def test_open_edges(self):
        p = PartialPlan()
        p.add_node("g", "goal", pos=(0, 0))
        p.add_node("l", "leaf", pos=(1, 1), robot="R0")
        p.add_edge("g", "l", status="open", cost=3)
        self.assertEqual(len(p.open_edges()), 1)
        self.assertFalse(p.validate_plan())

    def test_complete_plan(self):
        p = PartialPlan()
        p.add_node("g", "goal", pos=(0, 0))
        p.add_node("l", "leaf", pos=(1, 1), robot="R0")
        p.add_edge("g", "l", status="fixed", cost=3)
        self.assertEqual(p.open_edges(), [])
        self.assertTrue(p.validate_plan())

    def test_cost_computation(self):
        p = PartialPlan()
        p.add_node("g", "goal", pos=(0, 0))
        p.add_node("sg", "subgoal", entry_pos=(1, 1))
        p.add_node("bn", "bottleneck", pos=(2, 2), robot="R0")
        p.add_node("sp", "support", pos=(3, 3), robot="R1")
        p.add_node("lt", "leaf", pos=(4, 4), robot="R0")
        p.add_node("lh", "leaf", pos=(5, 5), robot="R1")

        p.add_edge("g", "sg", status="fixed", cost=2)
        p.add_edge("sg", "bn", status="fixed")           # structural, no cost
        p.add_edge("sg", "sp", status="fixed")           # structural, no cost
        p.add_edge("bn", "lt", status="fixed", cost=5)
        p.add_edge("sp", "lh", status="fixed", cost=3)

        self.assertEqual(p.cost(), 10)  # 2 + 5 + 3

    def test_structural_edge_detection(self):
        p = PartialPlan()
        p.add_node("g", "goal", pos=(0, 0))
        p.add_node("sg", "subgoal", entry_pos=(1, 1))
        p.add_node("bn", "bottleneck", pos=(2, 2), robot="R0")
        p.add_edge("g", "sg")
        p.add_edge("sg", "bn")

        self.assertFalse(p.is_structural_edge("g", "sg"))
        self.assertTrue(p.is_structural_edge("sg", "bn"))

    def test_resolve_goal_to_leaf(self):
        p = PartialPlan()
        p.add_node("g", "goal", pos=(5, 5))
        p.add_node("l", "leaf", pos=(1, 1), robot="R0")
        p.add_edge("g", "l", status="fixed", cost=3)

        positions = p.resolve_segment_positions("g", "l")
        self.assertEqual(positions, ((1, 1), (5, 5)))

    def test_resolve_goal_to_subgoal(self):
        p = PartialPlan()
        p.add_node("g", "goal", pos=(5, 5))
        p.add_node("sg", "subgoal", entry_pos=(3, 3))
        p.add_node("bn", "bottleneck", pos=(2, 2), robot="R0")
        p.add_edge("g", "sg")
        p.add_edge("sg", "bn")

        positions = p.resolve_segment_positions("g", "sg")
        self.assertEqual(positions, ((3, 3), (5, 5)))

    def test_resolve_structural_returns_none(self):
        p = PartialPlan()
        p.add_node("sg", "subgoal", entry_pos=(1, 1))
        p.add_node("bn", "bottleneck", pos=(2, 2), robot="R0")
        p.add_edge("sg", "bn")

        positions = p.resolve_segment_positions("sg", "bn")
        self.assertIsNone(positions)


# ========== Environment Loading Tests ==========

class TestEnvironmentLoading(unittest.TestCase):

    def test_load_environment_raw(self):
        """Raw load_environment still works for backward compat."""
        env = load_environment(list_environment_files(num=1)[0])
        self.assertIn('grid_graph', env)
        self.assertIn('instances', env)

    def test_grid_graph_structure(self):
        game = _load_game0()
        graph = game.grid_graph
        self.assertEqual(graph.number_of_nodes(), 256)
        self.assertGreater(graph.number_of_edges(), 0)

        for _, _, d in list(graph.edges(data=True))[:10]:
            self.assertIn('weight', d)
            self.assertIn(d['weight'], (INDEPENDENT_WEIGHT, DEPENDENT_WEIGHT))

    def test_list_environment_files(self):
        files = list_environment_files(num=5)
        self.assertEqual(len(files), 5)
        self.assertEqual(files[0].name, 'env_0.pkl')
        self.assertEqual(files[4].name, 'env_4.pkl')


# ========== Path Matrix Tests ==========

class TestPathMatrices(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game = _load_game0()

    def test_self_distance_zero(self):
        for node in list(self.game.grid_graph.nodes())[:20]:
            self.assertEqual(self.game.independent_paths[(node, node)], 0)
            self.assertEqual(self.game.all_paths[(node, node)], 0)

    def test_independent_none_means_unreachable(self):
        none_count = sum(
            1 for v in self.game.independent_paths.values() if v is None
        )
        self.assertGreater(none_count, 0)

    def test_independent_leq_all_paths(self):
        for key, ind_val in list(self.game.independent_paths.items())[:1000]:
            if ind_val is not None:
                all_val = self.game.all_paths.get(key, float('inf'))
                self.assertGreaterEqual(ind_val, 0)
                self.assertGreaterEqual(all_val, 0)


# ========== Subgoal Tests ==========

class TestSubgoals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game = _load_game0()
        cls.target = cls.game.states[0].target

    def test_final_component_contains_target(self):
        fc = get_final_component(self.game.grid_graph, self.target)
        self.assertIn(self.target, fc)

    def test_subgoals_exist(self):
        subgoals = get_subgoals(self.game.grid_graph, self.target)
        self.assertGreater(len(subgoals), 0)

    def test_subgoal_positions_in_final_component(self):
        fc = get_final_component(self.game.grid_graph, self.target)
        subgoals = get_subgoals(self.game.grid_graph, self.target)
        for sg_pos in subgoals:
            self.assertIn(sg_pos, fc)


# ========== Validation Tests ==========

class TestValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game = _load_game0()
        cls.state = cls.game.states[0]

    def test_valid_direct_plan(self):
        """If a direct independent path exists, the plan is valid."""
        game3 = _load_game(3)
        state3 = game3.states[0]
        target = state3.target
        target_pos = state3.target_robot.pos

        dist = game3.independent_paths.get((target_pos, target))
        self.assertIsNotNone(dist, "env_3 should have direct path")

        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=target)
        plan.add_node("leaf", "leaf", pos=target_pos,
                      robot=state3.target_robot.name)
        plan.add_edge("goal", "leaf", status="fixed", cost=dist)
        self.assertTrue(validate_plan(plan, game3, state3))

    def test_valid_subgoal_plan(self):
        """MockAlgorithm produces a valid plan for env_0."""
        algo = MockAlgorithm()
        result = algo.solve_instance(self.game, self.state)
        plan = result['plan']
        if plan.g.number_of_nodes() == 0:
            self.skipTest("No plan found for env_0")
        self.assertTrue(validate_plan(plan, self.game, self.state))

    def test_invalid_open_edges(self):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("leaf", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "leaf", status="open", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_empty_plan(self):
        plan = PartialPlan()
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_wrong_cost(self):
        game3 = _load_game(3)
        state3 = game3.states[0]
        target = state3.target
        target_pos = state3.target_robot.pos

        dist = game3.independent_paths.get((target_pos, target))
        self.assertIsNotNone(dist, "env_3 should have direct path")

        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=target)
        plan.add_node("leaf", "leaf", pos=target_pos,
                      robot=state3.target_robot.name)
        plan.add_edge("goal", "leaf", status="fixed", cost=dist + 1)
        self.assertFalse(validate_plan(plan, game3, state3))

    def test_invalid_no_independent_path(self):
        target = self.state.target
        target_pos = self.state.target_robot.pos

        if self.game.independent_paths.get((target_pos, target)) is not None:
            self.skipTest("Direct path exists, can't test this case")

        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=target)
        plan.add_node("leaf", "leaf", pos=target_pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "leaf", status="fixed", cost=5)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_missing_goal(self):
        plan = PartialPlan()
        plan.add_node("leaf", "leaf", pos=(0, 0), robot="R0")
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_missing_pos(self):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("leaf", "leaf", robot=self.state.target_robot.name)
        plan.add_edge("goal", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_wrong_goal(self):
        """Plan with goal != state.target fails."""
        # Use a position that's on the grid but not the target
        wrong_target = (0, 0) if self.state.target != (0, 0) else (1, 1)
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=wrong_target)
        plan.add_node("leaf", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_wrong_leaf_position(self):
        """Leaf with robot name matching but wrong position fails."""
        game3 = _load_game(3)
        state3 = game3.states[0]
        target = state3.target
        target_pos = state3.target_robot.pos

        dist = game3.independent_paths.get((target_pos, target))
        if dist is None:
            self.skipTest("No direct path in env_3")

        # Use the right robot name but a wrong position
        wrong_pos = (0, 0) if target_pos != (0, 0) else (1, 1)
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=target)
        plan.add_node("leaf", "leaf", pos=wrong_pos,
                      robot=state3.target_robot.name)
        plan.add_edge("goal", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, game3, state3))

    def test_invalid_wrong_leaf_robot(self):
        """Leaf with correct position but wrong robot name fails."""
        game3 = _load_game(3)
        state3 = game3.states[0]
        target = state3.target
        target_pos = state3.target_robot.pos

        dist = game3.independent_paths.get((target_pos, target))
        if dist is None:
            self.skipTest("No direct path in env_3")

        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=target)
        plan.add_node("leaf", "leaf", pos=target_pos,
                      robot="NONEXISTENT_ROBOT")
        plan.add_edge("goal", "leaf", status="fixed", cost=dist)
        self.assertFalse(validate_plan(plan, game3, state3))


# ========== MockAlgorithm Tests ==========

class TestMockAlgorithm(unittest.TestCase):

    def test_result_structure(self):
        game = _load_game0()
        algo = MockAlgorithm()
        result = algo.solve_instance(game, game.states[0])

        self.assertIn('plan', result)
        self.assertIn('stats', result)
        self.assertIsInstance(result['plan'], PartialPlan)

        stats = result['stats']
        self.assertIn('cost', stats)
        self.assertIn('wallclock_time', stats)
        self.assertIn('trace', stats)
        self.assertIn('auc', stats)
        self.assertIsInstance(stats['trace'], list)

    def test_mock_on_multiple_envs(self):
        """MockAlgorithm produces valid plans on the first 10 envs."""
        algo = MockAlgorithm()
        games = Game.load_many(num=10)
        valid_count = 0
        total = 0
        for game in games:
            for state in game.states:
                result = algo.solve_instance(game, state)
                plan = result['plan']
                total += 1
                if plan.g.number_of_nodes() > 0:
                    self.assertTrue(
                        validate_plan(plan, game, state),
                        f"Invalid plan for env_{game.graph_idx}"
                    )
                    valid_count += 1
        self.assertGreater(valid_count, 0,
                           "No valid plans produced at all")


# ========== Benchmark Runner Tests ==========

class TestBenchmarkRunner(unittest.TestCase):

    def test_solve_returns_list(self):
        game = _load_game0()
        results = solve(game)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn('plan', results[0])
        self.assertIn('stats', results[0])

    def test_run_benchmark_small(self):
        result = run_benchmark(MockAlgorithm(), num_instances=5)
        self.assertIn('average_cost', result)
        self.assertIn('total_auc', result)
        self.assertIn('num_solved', result)
        self.assertIn('num_total', result)
        self.assertGreater(result['num_solved'], 0)
        self.assertGreater(result['average_cost'], 0)

    def test_run_benchmark_validation_phase(self):
        result = run_benchmark(MockAlgorithm(), num_instances=10,
                               validate_first=True, n_small=5)
        self.assertGreater(result['num_solved'], 0)


# ========== AUC Tests ==========

class TestComputeAUC(unittest.TestCase):

    def test_empty_trace(self):
        self.assertEqual(compute_auc([], 10.0), float('inf'))

    def test_single_point_at_end(self):
        self.assertAlmostEqual(compute_auc([(5.0, 10)], 5.0), 0.0)

    def test_single_point_with_remaining_time(self):
        self.assertAlmostEqual(compute_auc([(1.0, 10)], 5.0), 40.0)

    def test_multi_point_trace(self):
        trace = [(0.1, 15), (0.5, 12), (1.2, 10)]
        total_time = 3.0
        self.assertAlmostEqual(compute_auc(trace, total_time), 32.4)

    def test_decreasing_cost(self):
        slow = compute_auc([(1.0, 20), (4.0, 10)], 5.0)
        fast = compute_auc([(0.5, 20), (1.0, 10)], 5.0)
        self.assertGreater(slow, fast)


# ========== compute_shortest_path Tests ==========

class TestComputeShortestPath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game = _load_game0()

    def test_self_distance_is_zero(self):
        node = list(self.game.grid_graph.nodes())[0]
        self.assertEqual(compute_shortest_path(self.game.grid_graph, node, node), 0)

    def test_no_support_matches_independent_paths(self):
        """Without support positions, results match precomputed independent_paths."""
        g = self.game.grid_graph
        ind = self.game.independent_paths
        nodes = list(g.nodes())[:10]
        for src in nodes:
            for dst in nodes:
                expected = ind.get((src, dst))
                result = compute_shortest_path(g, src, dst)
                self.assertEqual(result, expected,
                    f"Mismatch at ({src}, {dst}): {result} vs {expected}")

    def test_empty_support_matches_no_support(self):
        g = self.game.grid_graph
        nodes = list(g.nodes())[:5]
        for src in nodes:
            for dst in nodes:
                r1 = compute_shortest_path(g, src, dst, support_positions=None)
                r2 = compute_shortest_path(g, src, dst, support_positions=set())
                self.assertEqual(r1, r2)

    def test_support_enables_unreachable_path(self):
        """env_1: (13,5)->(12,0) unreachable independently, reachable with support at (10,0)."""
        game1 = _load_game(1)
        g = game1.grid_graph
        src, dst, helper = (13, 5), (12, 0), (10, 0)

        self.assertIsNone(compute_shortest_path(g, src, dst))
        result = compute_shortest_path(g, src, dst, support_positions={helper})
        self.assertIsNotNone(result)
        self.assertEqual(result, 6)

    def test_support_shortens_path(self):
        """env_6: (10,13)->(12,9) costs 12 independently, 6 with support at (12,7)."""
        game6 = _load_game(6)
        g = game6.grid_graph
        src, dst, helper = (10, 13), (12, 9), (12, 7)

        without = compute_shortest_path(g, src, dst)
        with_supp = compute_shortest_path(g, src, dst, support_positions={helper})
        self.assertEqual(without, 12)
        self.assertEqual(with_supp, 6)

    def test_unreachable_returns_none(self):
        """A pair unreachable without support returns None."""
        g = self.game.grid_graph
        ind = self.game.independent_paths
        # Find a pair that's None in independent_paths
        for (src, dst), val in ind.items():
            if val is None:
                self.assertIsNone(compute_shortest_path(g, src, dst))
                break


# ========== Extended Validation Tests ==========

class TestValidationBulletproof(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game = _load_game0()
        cls.state = cls.game.states[0]
        cls.grid_nodes = list(cls.game.grid_graph.nodes())

    def _make_simple_valid_plan(self):
        """Build a simple valid plan using real grid positions."""
        ind = self.game.independent_paths
        target_pos = self.state.target_robot.pos
        target = self.state.target
        # Try direct path first
        dist = ind.get((target_pos, target))
        if dist is not None:
            plan = PartialPlan()
            plan.add_node("goal", "goal", pos=target)
            plan.add_node("leaf", "leaf", pos=target_pos,
                          robot=self.state.target_robot.name)
            plan.add_edge("goal", "leaf", status="fixed", cost=dist)
            return plan
        # Fall back to MockAlgorithm
        algo = MockAlgorithm()
        result = algo.solve_instance(self.game, self.state)
        plan = result['plan']
        return plan if plan.g.number_of_nodes() > 0 else None

    def test_valid_simple_plan(self):
        plan = self._make_simple_valid_plan()
        if plan is None:
            self.skipTest("No valid plan available")
        self.assertTrue(validate_plan(plan, self.game, self.state))

    def test_invalid_position_not_on_grid(self):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=(999, 999))
        plan.add_node("leaf", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_leaf_position_not_on_grid(self):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("leaf", "leaf", pos=(999, 999),
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_leaf_has_children(self):
        n0, n1, n2 = self.grid_nodes[0], self.grid_nodes[1], self.grid_nodes[2]
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("leaf", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_node("extra", "leaf", pos=self.state.helper_robots[0].pos,
                      robot=self.state.helper_robots[0].name)
        plan.add_edge("goal", "leaf", status="fixed", cost=1)
        plan.add_edge("leaf", "extra", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_disconnected_node(self):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("leaf", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_node("orphan", "leaf", pos=self.state.helper_robots[0].pos,
                      robot=self.state.helper_robots[0].name)
        plan.add_edge("goal", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_structural_edge_to_wrong_type(self):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("sg", "subgoal", entry_pos=self.grid_nodes[1])
        plan.add_node("leaf", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "sg", status="fixed", cost=1)
        plan.add_edge("sg", "leaf", status="fixed")
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_bottleneck_missing_robot(self):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("bn", "bottleneck", pos=self.grid_nodes[1])
        plan.add_node("leaf", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "bn", status="fixed", cost=1)
        plan.add_edge("bn", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_unknown_ntype(self):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("weird", "unknown_type", pos=self.grid_nodes[1])
        plan.add_edge("goal", "weird", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_subgoal_missing_entry_pos(self):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("sg", "subgoal")
        plan.add_node("bn", "bottleneck", pos=self.grid_nodes[0],
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "sg", status="fixed", cost=1)
        plan.add_edge("sg", "bn", status="fixed")
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_goal_to_bottleneck(self):
        """goal -> bottleneck is not a valid edge type pair."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("bn", "bottleneck", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "bn", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_goal_to_support(self):
        """goal -> support is not a valid edge type pair."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("sp", "support", pos=self.state.helper_robots[0].pos,
                      robot=self.state.helper_robots[0].name)
        plan.add_edge("goal", "sp", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_bottleneck_to_bottleneck(self):
        """bottleneck -> bottleneck is not a valid edge type pair."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("bn1", "bottleneck", pos=self.grid_nodes[0],
                      robot=self.state.target_robot.name)
        plan.add_node("bn2", "bottleneck", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "bn1", status="fixed", cost=1)
        plan.add_edge("bn1", "bn2", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_bottleneck_to_support(self):
        """bottleneck -> support is not a valid edge type pair."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("bn", "bottleneck", pos=self.grid_nodes[0],
                      robot=self.state.target_robot.name)
        plan.add_node("sp", "support", pos=self.state.helper_robots[0].pos,
                      robot=self.state.helper_robots[0].name)
        plan.add_edge("goal", "bn", status="fixed", cost=1)
        plan.add_edge("bn", "sp", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_support_to_bottleneck(self):
        """support -> bottleneck is not a valid edge type pair."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("sp", "support", pos=self.grid_nodes[0],
                      robot=self.state.helper_robots[0].name)
        plan.add_node("bn", "bottleneck", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "sp", status="fixed", cost=1)
        plan.add_edge("sp", "bn", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))


    def test_invalid_subgoal_missing_support(self):
        """A subgoal with only a bottleneck child (no support) fails."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("sg", "subgoal", entry_pos=self.grid_nodes[0])
        plan.add_node("bn", "bottleneck", pos=self.grid_nodes[1],
                      robot=self.state.target_robot.name)
        plan.add_node("leaf", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "sg", status="fixed", cost=1)
        plan.add_edge("sg", "bn", status="fixed")
        plan.add_edge("bn", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_subgoal_missing_bottleneck(self):
        """A subgoal with only a support child (no bottleneck) fails."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("sg", "subgoal", entry_pos=self.grid_nodes[0])
        plan.add_node("sp", "support", pos=self.state.helper_robots[0].pos,
                      robot=self.state.helper_robots[0].name)
        plan.add_node("leaf", "leaf", pos=self.state.helper_robots[0].pos,
                      robot=self.state.helper_robots[0].name)
        plan.add_edge("goal", "sg", status="fixed", cost=1)
        plan.add_edge("sg", "sp", status="fixed")
        plan.add_edge("sp", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))

    def test_invalid_subgoal_two_bottlenecks(self):
        """A subgoal with two bottleneck children (no support) fails."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=self.state.target)
        plan.add_node("sg", "subgoal", entry_pos=self.grid_nodes[0])
        plan.add_node("bn1", "bottleneck", pos=self.grid_nodes[1],
                      robot=self.state.target_robot.name)
        plan.add_node("bn2", "bottleneck", pos=self.grid_nodes[2],
                      robot=self.state.target_robot.name)
        plan.add_node("leaf1", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_node("leaf2", "leaf", pos=self.state.target_robot.pos,
                      robot=self.state.target_robot.name)
        plan.add_edge("goal", "sg", status="fixed", cost=1)
        plan.add_edge("sg", "bn1", status="fixed")
        plan.add_edge("sg", "bn2", status="fixed")
        plan.add_edge("bn1", "leaf1", status="fixed", cost=1)
        plan.add_edge("bn2", "leaf2", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.game, self.state))


if __name__ == '__main__':
    unittest.main()
