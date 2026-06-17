"""Tests for canonical_v2 — partial plan grid encoding."""

import sys
import os
import pickle
from types import SimpleNamespace

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from partial_plan import PartialPlan
from GridEnv import GridEnv, State, Robot_at
from data_gen.canonical_v2 import (
    FEATURE_DIM, NUM_CELLS, GRID_SIZE, DEFAULT_GRID_SIZE, TOKEN_TARGET,
    pos_to_index, index_to_pos, token_target, infer_grid_size,
    classify_temporal, compute_edge_encoding,
    extract_features, extract_target,
    generate_training_examples,
    _find_deepest_leaf_in_branch, _remove_subgoal,
)


# ---------------------------------------------------------------------------
# Helpers to build test plans
# ---------------------------------------------------------------------------

def make_simple_plan():
    """A simple plan: goal -> sg -> (bn -> leaf_0, sp -> leaf_1).

    goal(15,7) -> sg_1 -> bn_1(15,5) -> leaf_0(11,11)
                       -> sp_1(15,4) -> leaf_1(4,13)
    """
    p = PartialPlan()
    p.add_node("goal", "goal", pos=(15, 7))
    p.add_node("sg_1", "subgoal", parent_support_pos=(15, 6))
    p.add_node("bn_1", "bottleneck", pos=(15, 5),
               robot=Robot_at(position=(15, 5), color='Red'))
    p.add_node("sp_1", "support", pos=(15, 4),
               robot=Robot_at(position=(15, 4), color='Blue'))
    p.add_node("leaf_0", "leaf", pos=(11, 11),
               robot=Robot_at(position=(11, 11), color='Red'))
    p.add_node("leaf_1", "leaf", pos=(4, 13),
               robot=Robot_at(position=(4, 13), color='Blue'))

    p.add_edge("goal", "sg_1", status="fixed", cost=2)
    p.add_edge("sg_1", "bn_1", status="fixed", cost=0)
    p.add_edge("sg_1", "sp_1", status="fixed", cost=0)
    p.add_edge("bn_1", "leaf_0", status="fixed", cost=3)
    p.add_edge("sp_1", "leaf_1", status="fixed", cost=1)
    return p


def make_simple_state():
    """State matching the simple plan."""
    return State(
        target=(15, 7),
        target_robot=Robot_at(position=(11, 11), color='Red'),
        helpers=[
            Robot_at(position=(4, 13), color='Blue'),
            Robot_at(position=(7, 3), color='Green'),
            Robot_at(position=(1, 5), color='Yellow'),
        ],
    )


def make_two_level_plan():
    """Two-level plan with nested subgoal on bottleneck branch.

    goal(15,7) -> sg_1 -> bn_1(15,5) -> sg_2 -> bn_2(14,11) -> leaf_0(11,11)
                                              -> sp_2(15,11) -> leaf_2(11,11)
                       -> sp_1(15,4) -> leaf_1(4,13)
    """
    p = PartialPlan()
    p.add_node("goal", "goal", pos=(15, 7))
    p.add_node("sg_1", "subgoal", parent_support_pos=(15, 6))
    p.add_node("bn_1", "bottleneck", pos=(15, 5),
               robot=Robot_at(position=(15, 5), color='Red'))
    p.add_node("sp_1", "support", pos=(15, 4),
               robot=Robot_at(position=(15, 4), color='Blue'))
    p.add_node("leaf_1", "leaf", pos=(4, 13),
               robot=Robot_at(position=(4, 13), color='Blue'))

    p.add_node("sg_2", "subgoal", parent_support_pos=(15, 4))
    p.add_node("bn_2", "bottleneck", pos=(14, 11),
               robot=Robot_at(position=(14, 11), color='Red'))
    p.add_node("sp_2", "support", pos=(15, 11),
               robot=Robot_at(position=(15, 11), color='Red'))
    p.add_node("leaf_0", "leaf", pos=(11, 11),
               robot=Robot_at(position=(11, 11), color='Red'))
    p.add_node("leaf_2", "leaf", pos=(11, 11),
               robot=Robot_at(position=(11, 11), color='Red'))

    p.add_edge("goal", "sg_1", status="fixed", cost=2)
    p.add_edge("sg_1", "bn_1", status="fixed", cost=0)
    p.add_edge("sg_1", "sp_1", status="fixed", cost=0)
    p.add_edge("bn_1", "sg_2", status="fixed", cost=3)
    p.add_edge("sp_1", "leaf_1", status="fixed", cost=1)
    p.add_edge("sg_2", "bn_2", status="fixed", cost=0)
    p.add_edge("sg_2", "sp_2", status="fixed", cost=0)
    p.add_edge("bn_2", "leaf_0", status="fixed", cost=1)
    p.add_edge("sp_2", "leaf_2", status="fixed", cost=1)
    return p


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

class TestPositionHelpers:
    def test_pos_roundtrip(self):
        for x in range(16):
            for y in range(16):
                idx = pos_to_index(x, y)
                assert index_to_pos(idx) == (x, y)

    def test_known_values(self):
        assert pos_to_index(0, 0) == 0
        assert pos_to_index(15, 15) == 255
        assert pos_to_index(7, 3) == 55


# ---------------------------------------------------------------------------
# Temporal classification
# ---------------------------------------------------------------------------

class TestTemporalClassification:
    def test_simple_plan_open_at_bn(self):
        """Remove sg_1 → open edge goal->leaf_0.
        After removing sg_1: only goal and leaf_0 remain.
        The partial plan is just: goal -> leaf_0 (open).
        """
        p = PartialPlan()
        p.add_node("goal", "goal", pos=(15, 7))
        p.add_node("leaf_0", "leaf", pos=(11, 11),
                   robot=Robot_at(position=(11, 11), color='Red'))
        p.add_edge("goal", "leaf_0", status="open", cost=None)

        temporal = classify_temporal(p, ("goal", "leaf_0"))

        # goal is parent of open edge → after
        assert temporal["goal"] == "after"
        # leaf_0 is child of open edge → before
        assert temporal["leaf_0"] == "before"

    def test_two_level_plan_remove_inner(self):
        """In two-level plan, remove sg_2 → open edge bn_1->leaf_0.

        After removal: goal -> sg_1 -> bn_1 -> leaf_0 (open)
                                    -> sp_1 -> leaf_1

        Temporal relative to open edge (bn_1, leaf_0):
          - bn_1: parent of open edge → after
          - sg_1: ancestor of bn_1 → after
          - goal: ancestor → after
          - sp_1: support sibling at sg_1 → after
          - leaf_1: child of sp_1 (after node) → should follow support subtree
          - leaf_0: child of open edge → before
        """
        plan = make_two_level_plan()
        # Create partial by removing sg_2
        partial = _remove_subgoal(plan, "sg_2", "bn_1",
                                  "bn_2", "sp_2", "leaf_0")

        temporal = classify_temporal(partial, ("bn_1", "leaf_0"))

        assert temporal["bn_1"] == "after"
        assert temporal["sg_1"] == "after"
        assert temporal["goal"] == "after"
        assert temporal["sp_1"] == "after"
        assert temporal["leaf_0"] == "before"

    def test_two_level_plan_remove_outer(self):
        """In two-level plan, remove sg_1 → open edge goal->leaf_0.

        After removal: goal -> leaf_0 (open)

        Everything is just goal (after) and leaf_0 (before).
        """
        plan = make_two_level_plan()
        partial = _remove_subgoal(plan, "sg_1", "goal",
                                  "bn_1", "sp_1", "leaf_0")

        temporal = classify_temporal(partial, ("goal", "leaf_0"))
        assert temporal["goal"] == "after"
        assert temporal["leaf_0"] == "before"


# ---------------------------------------------------------------------------
# Edge encoding
# ---------------------------------------------------------------------------

class TestEdgeEncoding:
    def test_leaf_came_from(self):
        plan = make_simple_plan()
        enc = compute_edge_encoding(plan)

        # Leaves have came_from = (-1, -1)
        assert enc["leaf_0"]["came_from"] == (-1.0, -1.0)
        assert enc["leaf_1"]["came_from"] == (-1.0, -1.0)

    def test_leaf_goes_to(self):
        plan = make_simple_plan()
        enc = compute_edge_encoding(plan)

        # leaf_0 goes to bn_1 at (15, 5)
        assert enc["leaf_0"]["goes_to"] == (15.0, 5.0)
        # leaf_1 goes to sp_1 at (15, 4)
        assert enc["leaf_1"]["goes_to"] == (15.0, 4.0)

    def test_bottleneck_came_from(self):
        plan = make_simple_plan()
        enc = compute_edge_encoding(plan)

        # bn_1's child is leaf_0 at (11, 11)
        assert enc["bn_1"]["came_from"] == (11.0, 11.0)

    def test_bottleneck_goes_to(self):
        plan = make_simple_plan()
        enc = compute_edge_encoding(plan)

        # bn_1's parent is sg_1, sg_1's parent is goal at (15, 7)
        assert enc["bn_1"]["goes_to"] == (15.0, 7.0)

    def test_support_came_from(self):
        plan = make_simple_plan()
        enc = compute_edge_encoding(plan)

        # sp_1's child is leaf_1 at (4, 13)
        assert enc["sp_1"]["came_from"] == (4.0, 13.0)

    def test_goal_encoding(self):
        plan = make_simple_plan()
        enc = compute_edge_encoding(plan)

        assert enc["goal"]["came_from"] == (-1.0, -1.0)
        assert enc["goal"]["goes_to"] == (-1.0, -1.0)

    def test_nested_bottleneck_came_from(self):
        plan = make_two_level_plan()
        enc = compute_edge_encoding(plan)

        # bn_1 child is sg_2, whose bottleneck child is bn_2 at (14, 11)
        assert enc["bn_1"]["came_from"] == (14.0, 11.0)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_feature_shape(self):
        plan = make_simple_plan()
        state = make_simple_state()
        # Make a partial plan with open edge
        partial = PartialPlan()
        partial.add_node("goal", "goal", pos=(15, 7))
        partial.add_node("leaf_0", "leaf", pos=(11, 11),
                         robot=Robot_at(position=(11, 11), color='Red'))
        partial.add_edge("goal", "leaf_0", status="open", cost=None)

        grid_env, _ = GridEnv.from_env(44)
        features = extract_features(grid_env, state, partial,
                                    ("goal", "leaf_0"))

        assert features.shape == (NUM_CELLS, FEATURE_DIM)
        assert features.dtype == np.float32

    def test_board_features(self):
        grid_env, state = GridEnv.from_env(44)
        partial = PartialPlan()
        partial.add_node("goal", "goal", pos=(15, 7))
        partial.add_node("leaf_0", "leaf", pos=(11, 11),
                         robot=Robot_at(position=(11, 11), color='Red'))
        partial.add_edge("goal", "leaf_0", status="open", cost=None)

        features = extract_features(grid_env, state, partial,
                                    ("goal", "leaf_0"))

        # Check coordinates
        assert features[0, 0] == 0.0  # x=0
        assert features[0, 1] == 0.0  # y=0
        assert features[17, 0] == 1.0  # x=1, y=1 -> idx=17
        assert features[17, 1] == 1.0

        # Check borders have walls
        assert features[0, 2] == 1.0  # top-left has N wall (y=0)
        assert features[0, 4] == 1.0  # top-left has W wall (x=0)

    def test_goal_marker(self):
        grid_env, state = GridEnv.from_env(44)
        partial = PartialPlan()
        partial.add_node("goal", "goal", pos=(15, 7))
        partial.add_node("leaf_0", "leaf", pos=(11, 11),
                         robot=Robot_at(position=(11, 11), color='Red'))
        partial.add_edge("goal", "leaf_0", status="open", cost=None)

        features = extract_features(grid_env, state, partial,
                                    ("goal", "leaf_0"))

        goal_idx = pos_to_index(15, 7)
        assert features[goal_idx, 6] == 1.0
        # Non-goal cells should be 0
        assert features[0, 6] == 0.0

    def test_target_robot_marker(self):
        """For open edge goal->leaf_0, target robot is Red at (11,11)."""
        grid_env, state = GridEnv.from_env(44)
        partial = PartialPlan()
        partial.add_node("goal", "goal", pos=(15, 7))
        partial.add_node("leaf_0", "leaf", pos=(11, 11),
                         robot=Robot_at(position=(11, 11), color='Red'))
        partial.add_edge("goal", "leaf_0", status="open", cost=None)

        features = extract_features(grid_env, state, partial,
                                    ("goal", "leaf_0"))

        red_idx = pos_to_index(11, 11)
        assert features[red_idx, 7] == 1.0  # is_target_robot

    def test_helper_markers(self):
        """Helpers should be marked as is_helper."""
        grid_env, state = GridEnv.from_env(44)
        partial = PartialPlan()
        partial.add_node("goal", "goal", pos=(15, 7))
        partial.add_node("leaf_0", "leaf", pos=(11, 11),
                         robot=Robot_at(position=(11, 11), color='Red'))
        partial.add_edge("goal", "leaf_0", status="open", cost=None)

        features = extract_features(grid_env, state, partial,
                                    ("goal", "leaf_0"))

        # Helpers: Blue(4,13), Green(7,3), Yellow(1,5)
        assert features[pos_to_index(4, 13), 8] == 1.0
        assert features[pos_to_index(7, 3), 8] == 1.0
        assert features[pos_to_index(1, 5), 8] == 1.0

        # Target robot should NOT be marked as helper
        assert features[pos_to_index(11, 11), 8] == 0.0

    def test_open_edge_markers(self):
        grid_env, state = GridEnv.from_env(44)
        partial = PartialPlan()
        partial.add_node("goal", "goal", pos=(15, 7))
        partial.add_node("leaf_0", "leaf", pos=(11, 11),
                         robot=Robot_at(position=(11, 11), color='Red'))
        partial.add_edge("goal", "leaf_0", status="open", cost=None)

        features = extract_features(grid_env, state, partial,
                                    ("goal", "leaf_0"))

        # target_start = leaf_0 at (11, 11)
        assert features[pos_to_index(11, 11), 11] == 1.0
        # target_end = goal at (15, 7)
        assert features[pos_to_index(15, 7), 12] == 1.0

    def test_plan_node_features_with_plan(self):
        """Test that plan nodes get bottleneck/support markers."""
        grid_env, state = GridEnv.from_env(44)
        plan = make_simple_plan()
        # Remove sg_1 to create partial with open edge
        partial = _remove_subgoal(plan, "sg_1", "goal",
                                  "bn_1", "sp_1", "leaf_0")

        features = extract_features(grid_env, state, partial,
                                    ("goal", "leaf_0"))

        # In the partial plan (just goal->leaf_0), there are no
        # bottleneck or support nodes
        # All plan node type features should be 0 except the leaf
        leaf_idx = pos_to_index(11, 11)
        assert features[leaf_idx, 14] == 1.0  # is_support_or_leaf for leaf

    def test_temporal_features_in_partial_plan(self):
        """Test temporal encoding in a partial plan with context."""
        grid_env, state = GridEnv.from_env(44)

        # Two-level plan, remove inner sg_2 → open edge bn_1->leaf_0
        plan = make_two_level_plan()
        partial = _remove_subgoal(plan, "sg_2", "bn_1",
                                  "bn_2", "sp_2", "leaf_0")

        features = extract_features(grid_env, state, partial,
                                    ("bn_1", "leaf_0"))

        # bn_1 at (15, 5) should be 'after'
        bn1_idx = pos_to_index(15, 5)
        assert features[bn1_idx, 10] == 1.0  # temporal_after
        assert features[bn1_idx, 9] == 0.0   # not temporal_before

        # leaf_0 at (11, 11) should be 'before'
        leaf_idx = pos_to_index(11, 11)
        assert features[leaf_idx, 9] == 1.0  # temporal_before


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

class TestTargetExtraction:
    def test_simple_plan_target(self):
        """sg_1 with bn at (15,5), sp at (15,4), sp robot is Blue.
        Blue is a helper, initial pos is (4,13).
        """
        plan = make_simple_plan()
        state = make_simple_state()

        bn_idx, sp_idx, robot_token = extract_target(
            plan, state, "sg_1", ("goal", "leaf_0"))

        assert bn_idx == pos_to_index(15, 5)
        assert sp_idx == pos_to_index(15, 4)
        # Blue is a helper, identified by initial position (4, 13)
        assert robot_token == pos_to_index(4, 13)

    def test_target_robot_as_support(self):
        """When support robot is the same color as the moving robot → TARGET."""
        plan = make_two_level_plan()
        state = make_simple_state()

        # sg_2's support is Red (same as target robot)
        # open edge would be (bn_1, leaf_0)
        bn_idx, sp_idx, robot_token = extract_target(
            plan, state, "sg_2", ("bn_1", "leaf_0"))

        assert bn_idx == pos_to_index(14, 11)
        assert sp_idx == pos_to_index(15, 11)
        assert robot_token == TOKEN_TARGET


# ---------------------------------------------------------------------------
# Subgoal removal
# ---------------------------------------------------------------------------

class TestSubgoalRemoval:
    def test_remove_simple_subgoal(self):
        """Removing the only subgoal should leave goal->leaf_0 open edge."""
        plan = make_simple_plan()
        partial = _remove_subgoal(plan, "sg_1", "goal",
                                  "bn_1", "sp_1", "leaf_0")

        assert "goal" in partial.g.nodes
        assert "leaf_0" in partial.g.nodes
        assert "sg_1" not in partial.g.nodes
        assert "bn_1" not in partial.g.nodes
        assert "sp_1" not in partial.g.nodes
        assert "leaf_1" not in partial.g.nodes

        open_edges = partial.open_edges()
        assert len(open_edges) == 1
        assert open_edges[0] == ("goal", "leaf_0")

    def test_remove_inner_subgoal(self):
        """Removing inner sg_2 should keep sg_1 structure."""
        plan = make_two_level_plan()
        partial = _remove_subgoal(plan, "sg_2", "bn_1",
                                  "bn_2", "sp_2", "leaf_0")

        assert "sg_1" in partial.g.nodes
        assert "bn_1" in partial.g.nodes
        assert "sp_1" in partial.g.nodes
        assert "leaf_0" in partial.g.nodes  # reconnected
        assert "sg_2" not in partial.g.nodes
        assert "bn_2" not in partial.g.nodes
        assert "sp_2" not in partial.g.nodes

        # Open edge: bn_1 -> leaf_0
        open_edges = partial.open_edges()
        assert ("bn_1", "leaf_0") in open_edges

    def test_find_deepest_leaf(self):
        """Test finding the deepest leaf in bottleneck chain."""
        plan = make_two_level_plan()
        g = plan.g

        # From bn_1: chain goes bn_1 -> sg_2 -> bn_2 -> leaf_0
        leaf = _find_deepest_leaf_in_branch(g, "bn_1")
        assert leaf == "leaf_0"

        # From bn_2: direct to leaf_0
        leaf = _find_deepest_leaf_in_branch(g, "bn_2")
        assert leaf == "leaf_0"


# ---------------------------------------------------------------------------
# Full training data generation
# ---------------------------------------------------------------------------

class TestTrainingDataGeneration:
    def test_simple_plan_generates_one_example(self):
        """Simple plan with 1 subgoal → 1 training example."""
        grid_env, state = GridEnv.from_env(44)
        plan = make_simple_plan()

        examples = generate_training_examples(grid_env, state, plan)
        assert len(examples) == 1

        ex = examples[0]
        assert ex['features'].shape == (NUM_CELLS, FEATURE_DIM)
        assert len(ex['target']) == 3
        assert ex['subgoal_id'] == 'sg_1'

    def test_two_level_plan_generates_two_examples(self):
        """Two-level plan with 2 subgoals → 2 training examples."""
        grid_env, state = GridEnv.from_env(44)
        plan = make_two_level_plan()

        examples = generate_training_examples(grid_env, state, plan)
        assert len(examples) == 2

        sg_ids = {ex['subgoal_id'] for ex in examples}
        assert sg_ids == {'sg_1', 'sg_2'}

    def test_robot_colors_preserved(self):
        """Robot color metadata should be present for visualization."""
        grid_env, state = GridEnv.from_env(44)
        plan = make_simple_plan()

        examples = generate_training_examples(grid_env, state, plan)
        ex = examples[0]

        assert 'robot_colors' in ex
        red_idx = pos_to_index(11, 11)
        assert ex['robot_colors'][red_idx] == 'Red'

    def test_features_no_color_channels(self):
        """Feature vector should NOT have per-color channels."""
        grid_env, state = GridEnv.from_env(44)
        plan = make_simple_plan()

        examples = generate_training_examples(grid_env, state, plan)
        features = examples[0]['features']

        # Exactly FEATURE_DIM columns, no helper_0/1/2 channels
        assert features.shape[1] == FEATURE_DIM


# ---------------------------------------------------------------------------
# Integration test with real env 44
# ---------------------------------------------------------------------------

class TestIntegrationEnv44:
    @pytest.fixture
    def env44_data(self):
        env_dir = 'data_gen/plans/env_44'
        complete_path = os.path.join(env_dir, 'complete', 'plan_0.pkl')
        if not os.path.exists(complete_path):
            pytest.skip("Env 44 complete plan not available")
        with open(complete_path, 'rb') as f:
            d = pickle.load(f)
        grid_env, state = GridEnv.from_env(44)
        return grid_env, state, d['plan']

    def test_env44_generates_correct_number_of_examples(self, env44_data):
        grid_env, state, plan = env44_data
        # Env 44 complete plan has 5 subgoals
        subgoal_count = sum(1 for _, d in plan.g.nodes(data=True)
                            if d['ntype'] == 'subgoal')
        examples = generate_training_examples(grid_env, state, plan)
        assert len(examples) == subgoal_count

    def test_env44_feature_shapes(self, env44_data):
        grid_env, state, plan = env44_data
        examples = generate_training_examples(grid_env, state, plan)

        for ex in examples:
            assert ex['features'].shape == (NUM_CELLS, FEATURE_DIM)
            assert ex['features'].dtype == np.float32

    def test_env44_target_positions_valid(self, env44_data):
        grid_env, state, plan = env44_data
        examples = generate_training_examples(grid_env, state, plan)

        for ex in examples:
            bn_idx, sp_idx, robot_token = ex['target']
            assert 0 <= bn_idx < NUM_CELLS
            assert 0 <= sp_idx < NUM_CELLS
            assert robot_token == TOKEN_TARGET or 0 <= robot_token < NUM_CELLS

    def test_env44_open_edge_markers_set(self, env44_data):
        grid_env, state, plan = env44_data
        examples = generate_training_examples(grid_env, state, plan)

        for ex in examples:
            features = ex['features']
            # At least one cell should have target_start=1
            assert features[:, 11].sum() >= 1.0
            # At least one cell should have target_end=1
            assert features[:, 12].sum() >= 1.0

    def test_env44_temporal_consistency(self, env44_data):
        """No cell should be both before and after."""
        grid_env, state, plan = env44_data
        examples = generate_training_examples(grid_env, state, plan)

        for ex in examples:
            features = ex['features']
            # before and after are mutually exclusive
            overlap = (features[:, 9] > 0) & (features[:, 10] > 0)
            assert not overlap.any(), \
                "Some cells are marked both before and after"

    def test_env44_print_examples(self, env44_data):
        """Print examples for manual inspection."""
        grid_env, state, plan = env44_data
        examples = generate_training_examples(grid_env, state, plan)

        for i, ex in enumerate(examples):
            bn_idx, sp_idx, robot_token = ex['target']
            bn_pos = index_to_pos(bn_idx)
            sp_pos = index_to_pos(sp_idx)
            robot_str = "TARGET" if robot_token == TOKEN_TARGET else \
                f"helper@{index_to_pos(robot_token)}"

            print(f"\n--- Example {i} (subgoal {ex['subgoal_id']}) ---")
            print(f"  Open edge: {ex['open_edge']}")
            print(f"  Target: bn={bn_pos} sp={sp_pos} robot={robot_str}")

            features = ex['features']
            # Show which cells have plan features
            for idx in range(NUM_CELLS):
                parts = []
                if features[idx, 9] > 0:
                    parts.append("BEFORE")
                if features[idx, 10] > 0:
                    parts.append("AFTER")
                if features[idx, 11] > 0:
                    parts.append("START")
                if features[idx, 12] > 0:
                    parts.append("END")
                if features[idx, 13] > 0:
                    parts.append("BN")
                if features[idx, 14] > 0:
                    parts.append("SP/LEAF")
                if parts:
                    print(f"    cell {index_to_pos(idx)}: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Board-size generalization (boards larger than 16x16, any NxN)
# ---------------------------------------------------------------------------

def make_fake_grid_env(grid_size):
    """A minimal stand-in for GridEnv with just what the encoder reads.

    Provides ``grid_data`` (grid_size**2 empty cells, only border walls implied)
    and ``G`` (a graph with one node per cell) so both inference paths work.
    """
    import networkx as nx
    grid_data = ['' for _ in range(grid_size * grid_size)]
    g = nx.DiGraph()
    g.add_nodes_from((x, y) for y in range(grid_size) for x in range(grid_size))
    return SimpleNamespace(grid_data=grid_data, G=g)


class TestBoardSizeGeneralization:
    def test_pos_roundtrip_arbitrary_size(self):
        for n in (8, 16, 32, 9):
            for x in range(n):
                for y in range(n):
                    idx = pos_to_index(x, y, n)
                    assert index_to_pos(idx, n) == (x, y)
                    assert 0 <= idx < n * n

    def test_token_target_scales_with_size(self):
        assert token_target(16) == 256
        assert token_target(32) == 1024
        assert token_target(9) == 81
        # Default keeps the legacy 16x16 value.
        assert token_target() == TOKEN_TARGET == NUM_CELLS

    def test_defaults_are_legacy_16x16(self):
        assert DEFAULT_GRID_SIZE == 16
        assert GRID_SIZE == 16
        assert NUM_CELLS == 256
        assert pos_to_index(15, 15) == 255  # default arg path unchanged

    def test_infer_from_grid_data(self):
        assert infer_grid_size(make_fake_grid_env(32)) == 32
        assert infer_grid_size(make_fake_grid_env(9)) == 9

    def test_infer_from_graph_when_no_grid_data(self):
        import networkx as nx
        g = nx.DiGraph()
        g.add_nodes_from((x, y) for y in range(32) for x in range(32))
        env = SimpleNamespace(grid_data=None, G=g)
        assert infer_grid_size(env) == 32

    def test_extract_features_shape_32(self):
        grid_env = make_fake_grid_env(32)
        state = State(
            target=(31, 20),
            target_robot=Robot_at(position=(25, 30), color='Red'),
            helpers=[Robot_at(position=(4, 13), color='Blue')],
        )
        partial = PartialPlan()
        partial.add_node("goal", "goal", pos=(31, 20))
        partial.add_node("leaf_0", "leaf", pos=(25, 30),
                         robot=Robot_at(position=(25, 30), color='Red'))
        partial.add_edge("goal", "leaf_0", status="open", cost=None)

        features = extract_features(grid_env, state, partial, ("goal", "leaf_0"))

        # Matrix grows with the board; size is inferred, not hardcoded.
        assert features.shape == (32 * 32, FEATURE_DIM)
        # Goal marker lands at a cell index that only exists on a >16 board.
        goal_idx = pos_to_index(31, 20, 32)
        assert goal_idx >= NUM_CELLS  # beyond the 16x16 range
        assert features[goal_idx, 6] == 1.0
        # Target robot marked at its (high) coordinate.
        assert features[pos_to_index(25, 30, 32), 7] == 1.0
        # Far corner has S and E border walls.
        corner = pos_to_index(31, 31, 32)
        assert features[corner, 3] == 1.0  # S wall at y == N-1
        assert features[corner, 5] == 1.0  # E wall at x == N-1

    def test_generate_training_examples_indices_in_range_32(self):
        grid_env = make_fake_grid_env(32)
        state = State(
            target=(31, 20),
            target_robot=Robot_at(position=(25, 30), color='Red'),
            helpers=[Robot_at(position=(4, 13), color='Blue'),
                     Robot_at(position=(28, 28), color='Green')],
        )
        # Plan whose nodes use coordinates only valid on a 32x32 board.
        p = PartialPlan()
        p.add_node("goal", "goal", pos=(31, 20))
        p.add_node("sg_1", "subgoal", parent_support_pos=(31, 21))
        p.add_node("bn_1", "bottleneck", pos=(31, 22),
                   robot=Robot_at(position=(31, 22), color='Red'))
        p.add_node("sp_1", "support", pos=(31, 23),
                   robot=Robot_at(position=(31, 23), color='Green'))
        p.add_node("leaf_0", "leaf", pos=(25, 30),
                   robot=Robot_at(position=(25, 30), color='Red'))
        p.add_node("leaf_1", "leaf", pos=(28, 28),
                   robot=Robot_at(position=(28, 28), color='Green'))
        p.add_edge("goal", "sg_1", status="fixed", cost=2)
        p.add_edge("sg_1", "bn_1", status="fixed", cost=0)
        p.add_edge("sg_1", "sp_1", status="fixed", cost=0)
        p.add_edge("bn_1", "leaf_0", status="fixed", cost=3)
        p.add_edge("sp_1", "leaf_1", status="fixed", cost=1)

        examples = generate_training_examples(grid_env, state, p)
        assert len(examples) == 1
        ex = examples[0]
        assert ex['features'].shape == (32 * 32, FEATURE_DIM)
        bn_idx, sp_idx, robot_token = ex['target']
        assert 0 <= bn_idx < 32 * 32
        assert 0 <= sp_idx < 32 * 32
        assert robot_token == token_target(32) or 0 <= robot_token < 32 * 32
        # bottleneck (31,22) and support (31,23) exceed the 16x16 index range,
        # proving the encoding is not capped at 256.
        assert bn_idx == pos_to_index(31, 22, 32) >= NUM_CELLS
        assert sp_idx == pos_to_index(31, 23, 32) >= NUM_CELLS


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
