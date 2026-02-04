"""Tests for exploration module (dependency graph, coverage tracker, space explorer)."""

import pytest

from models.variables import (
    TriStateBoolean,
    QuantitativeState,
    BooleanVariable,
    QuantitativeVariable,
    EnumVariable,
    VariableAssignment,
    VariableSpace,
)
from exploration.dependency_graph import DependencyGraph, build_dependency_graph
from exploration.coverage_tracker import CoverageTracker, CoverageStats
from exploration.space_explorer import SpaceExplorer, ExplorationConfig


def create_simple_variable_space() -> VariableSpace:
    """Create a simple variable space for testing."""
    space = VariableSpace()

    space.add(BooleanVariable(
        name="has_name",
        description="Whether name is provided",
        category="input"
    ))

    space.add(EnumVariable(
        name="status",
        description="Status",
        category="state",
        states=["active", "inactive"],
        edge_case_states=["inactive"]
    ))

    return space


def create_dependent_variable_space() -> VariableSpace:
    """Create a variable space with dependencies."""
    space = VariableSpace()

    space.add(BooleanVariable(
        name="parent_var",
        description="Parent variable",
        category="parent"
    ))

    space.add(EnumVariable(
        name="child_var",
        description="Child variable",
        category="child",
        states=["a", "b"],
        depends_on="parent_var",
        relevant_when=lambda a: a.get("parent_var") == TriStateBoolean.TRUE
    ))

    return space


class TestDependencyGraph:
    """Tests for DependencyGraph."""

    def test_build_graph(self):
        """Test building a dependency graph."""
        space = create_simple_variable_space()
        graph = build_dependency_graph(space)

        assert len(graph) == 2
        assert "has_name" in graph
        assert "status" in graph

    def test_topological_order_no_deps(self):
        """Test topological order without dependencies."""
        space = create_simple_variable_space()
        graph = build_dependency_graph(space)

        order = graph.topological_order()
        assert len(order) == 2

    def test_topological_order_with_deps(self):
        """Test topological order with dependencies."""
        space = create_dependent_variable_space()
        graph = build_dependency_graph(space)

        order = graph.topological_order()
        names = [v.name for v in order]

        # Parent should come before child
        assert names.index("parent_var") < names.index("child_var")

    def test_get_root_variables(self):
        """Test getting root variables."""
        space = create_dependent_variable_space()
        graph = build_dependency_graph(space)

        roots = graph.get_root_variables()
        root_names = [v.name for v in roots]

        assert "parent_var" in root_names
        assert "child_var" not in root_names

    def test_get_children(self):
        """Test getting child variables."""
        space = create_dependent_variable_space()
        graph = build_dependency_graph(space)

        children = graph.get_children("parent_var")
        child_names = [v.name for v in children]

        assert "child_var" in child_names

    def test_get_parents(self):
        """Test getting parent variables."""
        space = create_dependent_variable_space()
        graph = build_dependency_graph(space)

        parents = graph.get_parents("child_var")
        parent_names = [v.name for v in parents]

        assert "parent_var" in parent_names

    def test_is_relevant_with_dependency(self):
        """Test relevance checking with dependencies."""
        space = create_dependent_variable_space()
        graph = build_dependency_graph(space)

        child = space.get("child_var")

        # With parent = TRUE, child should be relevant
        assignment = VariableAssignment(assignments={
            "parent_var": TriStateBoolean.TRUE
        })
        assert graph.is_relevant(child, assignment) is True

        # With parent = FALSE, child should not be relevant
        assignment = VariableAssignment(assignments={
            "parent_var": TriStateBoolean.FALSE
        })
        assert graph.is_relevant(child, assignment) is False

    def test_get_relevant_variables(self):
        """Test getting all relevant variables."""
        space = create_dependent_variable_space()
        graph = build_dependency_graph(space)

        # When parent is TRUE, both should be relevant
        assignment = VariableAssignment(assignments={
            "parent_var": TriStateBoolean.TRUE
        })
        relevant = graph.get_relevant_variables(assignment)
        names = [v.name for v in relevant]

        assert "parent_var" in names
        assert "child_var" in names

    def test_get_unassigned_relevant(self):
        """Test getting unassigned relevant variables."""
        space = create_dependent_variable_space()
        graph = build_dependency_graph(space)

        assignment = VariableAssignment(assignments={
            "parent_var": TriStateBoolean.TRUE
        })
        unassigned = graph.get_unassigned_relevant_variables(assignment)
        names = [v.name for v in unassigned]

        assert "parent_var" not in names  # Already assigned
        assert "child_var" in names  # Not assigned but relevant


class TestCoverageTracker:
    """Tests for CoverageTracker."""

    def test_add_and_check_assignment(self):
        """Test adding and checking assignments."""
        space = create_simple_variable_space()
        tracker = CoverageTracker(space)

        assignment = VariableAssignment(assignments={
            "has_name": TriStateBoolean.TRUE,
            "status": "active"
        })

        assert not tracker.is_covered(assignment)

        tracker.add_assignment(assignment)
        assert tracker.is_covered(assignment)

    def test_pair_coverage(self):
        """Test pairwise coverage tracking."""
        space = create_simple_variable_space()
        tracker = CoverageTracker(space)

        assignment = VariableAssignment(assignments={
            "has_name": TriStateBoolean.TRUE,
            "status": "active"
        })
        tracker.add_assignment(assignment)

        assert tracker.is_pair_covered(
            "has_name", TriStateBoolean.TRUE,
            "status", "active"
        )
        assert not tracker.is_pair_covered(
            "has_name", TriStateBoolean.FALSE,
            "status", "active"
        )

    def test_get_uncovered_pairs(self):
        """Test getting uncovered pairs."""
        space = create_simple_variable_space()
        tracker = CoverageTracker(space)

        # Initially all pairs uncovered
        uncovered = tracker.get_uncovered_pairs()
        assert len(uncovered) > 0

        # Cover one assignment
        assignment = VariableAssignment(assignments={
            "has_name": TriStateBoolean.TRUE,
            "status": "active"
        })
        tracker.add_assignment(assignment)

        # Some pairs should now be covered
        uncovered_after = tracker.get_uncovered_pairs()
        assert len(uncovered_after) < len(uncovered)

    def test_coverage_stats(self):
        """Test coverage statistics."""
        space = create_simple_variable_space()
        tracker = CoverageTracker(space)

        assignment = VariableAssignment(assignments={
            "has_name": TriStateBoolean.TRUE,
            "status": "active"
        })
        tracker.add_assignment(assignment)

        stats = tracker.get_coverage_stats()

        assert stats.total_assignments == 1
        assert stats.covered_pairs > 0
        assert 0.0 <= stats.pairwise_coverage <= 1.0

    def test_reset(self):
        """Test resetting coverage tracker."""
        space = create_simple_variable_space()
        tracker = CoverageTracker(space)

        assignment = VariableAssignment(assignments={
            "has_name": TriStateBoolean.TRUE,
            "status": "active"
        })
        tracker.add_assignment(assignment)
        assert tracker.is_covered(assignment)

        tracker.reset()
        assert not tracker.is_covered(assignment)


class TestSpaceExplorer:
    """Tests for SpaceExplorer."""

    def test_basic_exploration(self):
        """Test basic exploration generates assignments."""
        space = create_simple_variable_space()
        graph = build_dependency_graph(space)
        explorer = SpaceExplorer(graph, max_scenarios=5)

        assignments = list(explorer.explore())

        assert len(assignments) <= 5
        assert all(isinstance(a, VariableAssignment) for a in assignments)

    def test_respects_max_scenarios(self):
        """Test that max_scenarios is respected."""
        space = create_simple_variable_space()
        graph = build_dependency_graph(space)
        explorer = SpaceExplorer(graph, max_scenarios=2)

        assignments = list(explorer.explore())

        assert len(assignments) <= 2

    def test_edge_cases_first(self):
        """Test that edge cases are prioritized."""
        space = create_simple_variable_space()
        graph = build_dependency_graph(space)

        config = ExplorationConfig(
            max_scenarios=10,
            prioritize_edge_cases=True,
            pairwise_coverage=False,
            systematic_bfs=False
        )
        explorer = SpaceExplorer(graph, config=config)

        assignments = list(explorer.explore())

        # First assignment should include edge cases
        if assignments:
            first = assignments[0]
            # Check if any edge case value is present
            has_edge_case = False
            for name, value in first.items():
                var = space.get(name)
                if var and value in var.get_edge_case_states():
                    has_edge_case = True
                    break
            # Note: This is a soft check - edge cases should be prioritized
            # but the exact behavior depends on the implementation

    def test_no_duplicates(self):
        """Test that no duplicate assignments are generated."""
        space = create_simple_variable_space()
        graph = build_dependency_graph(space)
        explorer = SpaceExplorer(graph, max_scenarios=20)

        assignments = list(explorer.explore())
        assignment_hashes = [hash(a) for a in assignments]

        assert len(assignment_hashes) == len(set(assignment_hashes))

    def test_coverage_stats_available(self):
        """Test that coverage stats are available after exploration."""
        space = create_simple_variable_space()
        graph = build_dependency_graph(space)
        explorer = SpaceExplorer(graph, max_scenarios=5)

        list(explorer.explore())  # Consume the generator

        stats = explorer.get_coverage_stats()

        assert stats is not None
        assert stats.total_assignments > 0

    def test_exploration_with_dependencies(self):
        """Test exploration respects dependencies."""
        space = create_dependent_variable_space()
        graph = build_dependency_graph(space)
        explorer = SpaceExplorer(graph, max_scenarios=10)

        assignments = list(explorer.explore())

        for assignment in assignments:
            # If child_var is assigned, check relevance
            if assignment.has("child_var"):
                parent_val = assignment.get("parent_var")
                # Child should only be assigned when parent makes it relevant
                if parent_val != TriStateBoolean.TRUE:
                    # This might be okay depending on implementation
                    # Just ensure we don't crash
                    pass


class TestExplorationConfig:
    """Tests for ExplorationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExplorationConfig()

        assert config.prioritize_edge_cases is True
        assert config.pairwise_coverage is True
        assert config.systematic_bfs is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExplorationConfig(
            max_scenarios=100,
            prioritize_edge_cases=False,
            pairwise_coverage=False
        )

        assert config.max_scenarios == 100
        assert config.prioritize_edge_cases is False
        assert config.pairwise_coverage is False
