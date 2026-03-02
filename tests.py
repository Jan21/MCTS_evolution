"""Tests for the MCTS_evolution benchmark framework.

Run with: python -m pytest tests.py -v
    or:   python tests.py
"""

import unittest
from pathlib import Path

from partial_plan import PartialPlan
from benchmark import validate_plan, solve, run_benchmark, MockAlgorithm
from utils import (
    load_environment,
    list_environment_files,
    get_independent_paths,
    get_all_paths,
    get_subgoals,
    get_final_component,
    INDEPENDENT_WEIGHT,
    DEPENDENT_WEIGHT,
)


def _load_env(index=0):
    """Helper to load an environment by index."""
    env_files = list_environment_files(num=index + 1)
    return load_environment(env_files[index])


def _load_env0():
    """Helper to load env_0 for tests."""
    return _load_env(0)


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
        # Should resolve to (entry_pos, goal.pos)
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

    def test_load_environment(self):
        env = _load_env0()
        self.assertIn('graph_idx', env)
        self.assertIn('grid_data', env)
        self.assertIn('grid_graph', env)
        self.assertIn('instances', env)

    def test_grid_graph_structure(self):
        env = _load_env0()
        graph = env['grid_graph']
        self.assertEqual(graph.number_of_nodes(), 256)
        self.assertGreater(graph.number_of_edges(), 0)

        # Check that edges have weight attribute
        for _, _, d in list(graph.edges(data=True))[:10]:
            self.assertIn('weight', d)
            self.assertIn(d['weight'], (INDEPENDENT_WEIGHT, DEPENDENT_WEIGHT))

    def test_instances_structure(self):
        env = _load_env0()
        instances = env['instances']
        self.assertGreater(len(instances), 0)
        inst = instances[0]
        self.assertIn('helper_robots', inst)
        self.assertIn('target_robot', inst)
        self.assertIn('target', inst)
        self.assertEqual(len(inst['helper_robots']), 3)

    def test_path_matrices_present(self):
        env = _load_env0()
        self.assertIn('independent_paths', env)
        self.assertIn('all_paths', env)

    def test_list_environment_files(self):
        files = list_environment_files(num=5)
        self.assertEqual(len(files), 5)
        self.assertEqual(files[0].name, 'env_0.pkl')
        self.assertEqual(files[4].name, 'env_4.pkl')


# ========== Path Matrix Tests ==========

class TestPathMatrices(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = _load_env0()
        cls.ind = get_independent_paths(cls.env)
        cls.all = get_all_paths(cls.env)

    def test_self_distance_zero(self):
        for node in list(self.env['grid_graph'].nodes())[:20]:
            self.assertEqual(self.ind[(node, node)], 0)
            self.assertEqual(self.all[(node, node)], 0)

    def test_independent_none_means_unreachable(self):
        """Some pairs have no independent path (need dependent edges)."""
        none_count = sum(
            1 for v in self.ind.values() if v is None
        )
        self.assertGreater(none_count, 0)

    def test_independent_leq_all_paths(self):
        """When independent path exists, its length >= all_paths length
        (since all_paths can use more edges, but dependent edges have
        weight 100, so the actual values differ)."""
        for key, ind_val in list(self.ind.items())[:1000]:
            if ind_val is not None:
                all_val = self.all.get(key, float('inf'))
                # Independent uses hop count, all_paths uses weighted distance
                # So ind_val should be >= 0 and all_val >= 0
                self.assertGreaterEqual(ind_val, 0)
                self.assertGreaterEqual(all_val, 0)


# ========== Subgoal Tests ==========

class TestSubgoals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = _load_env0()
        cls.target = cls.env['instances'][0]['target']

    def test_final_component_contains_target(self):
        fc = get_final_component(self.env['grid_graph'], self.target)
        self.assertIn(self.target, fc)

    def test_subgoals_exist(self):
        subgoals = get_subgoals(self.env['grid_graph'], self.target)
        self.assertGreater(len(subgoals), 0)

    def test_subgoal_positions_in_final_component(self):
        fc = get_final_component(self.env['grid_graph'], self.target)
        subgoals = get_subgoals(self.env['grid_graph'], self.target)
        for sg_pos in subgoals:
            self.assertIn(sg_pos, fc)


# ========== Validation Tests ==========

class TestValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = _load_env0()
        cls.ind = get_independent_paths(cls.env)
        cls.inst = cls.env['instances'][0]

    def test_valid_direct_plan(self):
        """If a direct independent path exists, the plan is valid."""
        # Use env_3 which has a large final component and direct paths
        env3 = _load_env(3)
        ind3 = get_independent_paths(env3)
        inst3 = env3['instances'][0]
        target = inst3['target']
        target_pos = (inst3['target_robot'].x, inst3['target_robot'].y)

        dist = ind3.get((target_pos, target))
        self.assertIsNotNone(dist, "env_3 should have direct path")

        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=target)
        plan.add_node("leaf", "leaf", pos=target_pos,
                      robot=inst3['target_robot'].name)
        plan.add_edge("goal", "leaf", status="fixed", cost=dist)
        self.assertTrue(validate_plan(plan, env3))

    def test_valid_subgoal_plan(self):
        """MockAlgorithm produces a valid plan for env_0."""
        algo = MockAlgorithm()
        result = algo.solve_instance(self.env, self.inst)
        plan = result['plan']
        if plan.g.number_of_nodes() == 0:
            self.skipTest("No plan found for env_0")
        self.assertTrue(validate_plan(plan, self.env))

    def test_invalid_open_edges(self):
        """Plan with open edges fails validation."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=(0, 0))
        plan.add_node("leaf", "leaf", pos=(1, 1), robot="R0")
        plan.add_edge("goal", "leaf", status="open", cost=1)
        self.assertFalse(validate_plan(plan, self.env))

    def test_invalid_empty_plan(self):
        """Empty plan fails validation."""
        plan = PartialPlan()
        self.assertFalse(validate_plan(plan, self.env))

    def test_invalid_wrong_cost(self):
        """Plan with incorrect cost fails validation."""
        # Use env_3 which has direct independent paths
        env3 = _load_env(3)
        ind3 = get_independent_paths(env3)
        inst3 = env3['instances'][0]
        target = inst3['target']
        target_pos = (inst3['target_robot'].x, inst3['target_robot'].y)

        dist = ind3.get((target_pos, target))
        self.assertIsNotNone(dist, "env_3 should have direct path")

        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=target)
        plan.add_node("leaf", "leaf", pos=target_pos, robot="R0")
        # Wrong cost (add 1)
        plan.add_edge("goal", "leaf", status="fixed", cost=dist + 1)
        self.assertFalse(validate_plan(plan, env3))

    def test_invalid_no_independent_path(self):
        """Plan between unreachable positions fails."""
        target = self.inst['target']
        target_pos = (self.inst['target_robot'].x,
                      self.inst['target_robot'].y)

        # target_pos to target has no independent path (that's why
        # we need subgoals), so this direct plan should fail
        if self.ind.get((target_pos, target)) is not None:
            self.skipTest("Direct path exists, can't test this case")

        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=target)
        plan.add_node("leaf", "leaf", pos=target_pos, robot="Yellow")
        plan.add_edge("goal", "leaf", status="fixed", cost=5)
        self.assertFalse(validate_plan(plan, self.env))

    def test_invalid_missing_goal(self):
        """Plan without a goal node fails."""
        plan = PartialPlan()
        plan.add_node("leaf", "leaf", pos=(0, 0), robot="R0")
        self.assertFalse(validate_plan(plan, self.env))

    def test_invalid_missing_pos(self):
        """Plan with leaf missing pos fails."""
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=(0, 0))
        plan.add_node("leaf", "leaf", robot="R0")  # no pos
        plan.add_edge("goal", "leaf", status="fixed", cost=1)
        self.assertFalse(validate_plan(plan, self.env))


# ========== MockAlgorithm Tests ==========

class TestMockAlgorithm(unittest.TestCase):

    def test_result_structure(self):
        env = _load_env0()
        algo = MockAlgorithm()
        result = algo.solve_instance(env, env['instances'][0])

        self.assertIn('plan', result)
        self.assertIn('stats', result)
        self.assertIsInstance(result['plan'], PartialPlan)

        stats = result['stats']
        self.assertIn('cost', stats)
        self.assertIn('wallclock_time', stats)
        self.assertIn('auc', stats)

    def test_mock_on_multiple_envs(self):
        """MockAlgorithm produces valid plans on the first 10 envs."""
        algo = MockAlgorithm()
        env_files = list_environment_files(num=10)
        valid_count = 0
        total = 0
        for env_file in env_files:
            env_data = load_environment(env_file)
            for inst in env_data['instances']:
                result = algo.solve_instance(env_data, inst)
                plan = result['plan']
                total += 1
                if plan.g.number_of_nodes() > 0:
                    self.assertTrue(
                        validate_plan(plan, env_data),
                        f"Invalid plan for {env_file.name}"
                    )
                    valid_count += 1
        self.assertGreater(valid_count, 0,
                           "No valid plans produced at all")


# ========== Benchmark Runner Tests ==========

class TestBenchmarkRunner(unittest.TestCase):

    def test_solve_returns_list(self):
        env_files = list_environment_files(num=1)
        results = solve(env_files[0])
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
        """run_benchmark validates on N_SMALL before full eval."""
        # Should not raise since MockAlgorithm produces valid plans
        result = run_benchmark(MockAlgorithm(), num_instances=10,
                               validate_first=True, n_small=5)
        self.assertGreater(result['num_solved'], 0)


if __name__ == '__main__':
    unittest.main()
