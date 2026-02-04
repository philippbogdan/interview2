"""Space explorer for systematic variable space traversal.

This module provides a generator that walks through the variable space,
yielding assignments in a strategic order: edge cases first, then
pairwise coverage, then systematic exploration.
"""

from dataclasses import dataclass, field
from typing import Any, Generator, Optional
from itertools import product, combinations

from models.variables import (
    VariableAssignment,
    VariableSpace,
    VariableDefinition,
)
from .dependency_graph import DependencyGraph
from .coverage_tracker import CoverageTracker, CoverageStats


@dataclass
class ExplorationConfig:
    """Configuration for space exploration."""

    max_scenarios: Optional[int] = None
    prioritize_edge_cases: bool = True
    pairwise_coverage: bool = True
    systematic_bfs: bool = True
    max_edge_case_combos: int = 100
    max_pairwise_scenarios: int = 500


class SpaceExplorer:
    """Generator that walks through the variable space systematically.

    Exploration phases:
    1. Edge cases first - test boundary conditions
    2. Pairwise coverage - ensure all variable pairs are tested together
    3. Systematic BFS - fill remaining gaps
    """

    def __init__(
        self,
        graph: DependencyGraph,
        config: Optional[ExplorationConfig] = None,
        max_scenarios: Optional[int] = None,
    ):
        """Initialize the space explorer.

        Args:
            graph: The dependency graph for the variable space.
            config: Optional exploration configuration.
            max_scenarios: Maximum number of scenarios (overrides config).
        """
        self.graph = graph
        self.variable_space = graph.variable_space
        self.config = config or ExplorationConfig()

        if max_scenarios is not None:
            self.config.max_scenarios = max_scenarios

        self._tracker = CoverageTracker(self.variable_space)
        self._yielded_count = 0

    def explore(self) -> Generator[VariableAssignment, None, None]:
        """Generate variable assignments in strategic order.

        Yields:
            VariableAssignment objects representing test scenarios.
        """
        self._yielded_count = 0

        # Phase 1: Edge cases
        if self.config.prioritize_edge_cases:
            for assignment in self._generate_edge_cases():
                if self._should_stop():
                    return
                yield assignment

        # Phase 2: Pairwise coverage
        if self.config.pairwise_coverage:
            for assignment in self._generate_pairwise():
                if self._should_stop():
                    return
                yield assignment

        # Phase 3: Systematic BFS exploration
        if self.config.systematic_bfs:
            for assignment in self._generate_systematic():
                if self._should_stop():
                    return
                yield assignment

    def _should_stop(self) -> bool:
        """Check if we should stop generating scenarios."""
        if self.config.max_scenarios is None:
            return False
        return self._yielded_count >= self.config.max_scenarios

    def _yield_if_new(
        self,
        assignment: VariableAssignment
    ) -> Generator[VariableAssignment, None, None]:
        """Yield an assignment if it hasn't been yielded before.

        Args:
            assignment: The assignment to potentially yield.

        Yields:
            The assignment if it's new.
        """
        if not self._tracker.is_covered(assignment):
            self._tracker.add_assignment(assignment)
            self._yielded_count += 1
            yield assignment

    def _generate_edge_cases(self) -> Generator[VariableAssignment, None, None]:
        """Generate scenarios that prioritize edge cases.

        Strategy:
        1. For each variable with edge cases, create a scenario with that edge case
        2. Combine multiple edge cases when possible
        3. Fill in non-edge-case variables with default/normal values

        Yields:
            VariableAssignment objects focusing on edge cases.
        """
        # Get variables sorted by priority (edge case priority first)
        variables = sorted(
            self.graph.topological_order(),
            key=lambda v: (not v.is_edge_case_priority, v.name)
        )

        # Collect all edge cases
        edge_case_vars: list[tuple[VariableDefinition, list[Any]]] = []
        for var in variables:
            edge_cases = var.get_edge_case_states()
            if edge_cases:
                edge_case_vars.append((var, edge_cases))

        # Generate single edge case scenarios
        for var, edge_cases in edge_case_vars:
            for edge_value in edge_cases:
                assignment = self._build_assignment_with_value(var, edge_value)
                if assignment:
                    yield from self._yield_if_new(assignment)
                    if self._should_stop():
                        return

        # Generate combinations of edge cases (2 at a time)
        combo_count = 0
        for (var1, edges1), (var2, edges2) in combinations(edge_case_vars, 2):
            # Skip if dependent
            if var1.depends_on == var2.name or var2.depends_on == var1.name:
                continue

            for edge1 in edges1:
                for edge2 in edges2:
                    assignment = self._build_assignment_with_values(
                        [(var1, edge1), (var2, edge2)]
                    )
                    if assignment:
                        yield from self._yield_if_new(assignment)
                        combo_count += 1
                        if self._should_stop():
                            return
                        if combo_count >= self.config.max_edge_case_combos:
                            return

    def _generate_pairwise(self) -> Generator[VariableAssignment, None, None]:
        """Generate scenarios for pairwise coverage.

        Strategy:
        Use a greedy algorithm to cover all pairs of variable values
        while minimizing the number of test cases.

        Yields:
            VariableAssignment objects for pairwise coverage.
        """
        uncovered_pairs = self._tracker.get_uncovered_pairs()
        scenario_count = 0

        while uncovered_pairs and scenario_count < self.config.max_pairwise_scenarios:
            if self._should_stop():
                return

            # Greedily select an assignment that covers the most uncovered pairs
            best_assignment = self._find_best_pairwise_assignment(uncovered_pairs)

            if best_assignment is None:
                break

            yield from self._yield_if_new(best_assignment)
            scenario_count += 1

            # Update uncovered pairs
            uncovered_pairs = self._tracker.get_uncovered_pairs()

    def _generate_systematic(self) -> Generator[VariableAssignment, None, None]:
        """Generate remaining scenarios via systematic BFS exploration.

        Strategy:
        Walk through the variable space breadth-first, prioritizing
        shorter assignments and then expanding.

        Yields:
            VariableAssignment objects for systematic coverage.
        """
        variables = self.graph.topological_order()

        # Start with root variables only
        root_vars = self.graph.get_root_variables()

        # Generate all combinations of root variables first
        root_states: list[list[tuple[VariableDefinition, Any]]] = []
        for var in root_vars:
            root_states.append([(var, state) for state in var.get_all_states()])

        for combo in product(*root_states):
            if self._should_stop():
                return

            assignment = VariableAssignment()
            for var, value in combo:
                assignment = assignment.set(var.name, value)

            # Expand to include dependent variables
            full_assignment = self._expand_assignment(assignment, variables)
            if full_assignment:
                yield from self._yield_if_new(full_assignment)

    def _build_assignment_with_value(
        self,
        target_var: VariableDefinition,
        target_value: Any
    ) -> Optional[VariableAssignment]:
        """Build a complete assignment with one variable set to a specific value.

        Args:
            target_var: The variable to set.
            target_value: The value to set.

        Returns:
            A complete assignment or None if impossible.
        """
        return self._build_assignment_with_values([(target_var, target_value)])

    def _build_assignment_with_values(
        self,
        targets: list[tuple[VariableDefinition, Any]]
    ) -> Optional[VariableAssignment]:
        """Build a complete assignment with multiple variables set.

        Args:
            targets: List of (variable, value) pairs to set.

        Returns:
            A complete assignment or None if impossible.
        """
        assignment = VariableAssignment()

        # Set target values
        for var, value in targets:
            assignment = assignment.set(var.name, value)

        # Set parent values if needed (to make targets relevant)
        for var, _ in targets:
            if var.depends_on:
                parent = self.variable_space.get(var.depends_on)
                if parent and not assignment.has(parent.name):
                    # Pick a value that makes the child relevant
                    parent_value = self._find_enabling_parent_value(var, parent)
                    if parent_value is not None:
                        assignment = assignment.set(parent.name, parent_value)

        # Fill in remaining variables with default values
        for var in self.graph.topological_order():
            if assignment.has(var.name):
                continue

            if self.graph.is_relevant(var, assignment):
                default = self._get_default_value(var)
                assignment = assignment.set(var.name, default)

        return assignment

    def _expand_assignment(
        self,
        partial: VariableAssignment,
        variables: list[VariableDefinition]
    ) -> Optional[VariableAssignment]:
        """Expand a partial assignment to include all relevant variables.

        Args:
            partial: The partial assignment to expand.
            variables: All variables in topological order.

        Returns:
            A complete assignment or None if impossible.
        """
        assignment = partial.copy()

        for var in variables:
            if assignment.has(var.name):
                continue

            if self.graph.is_relevant(var, assignment):
                # Pick a value, preferring non-edge-case defaults
                default = self._get_default_value(var)
                assignment = assignment.set(var.name, default)

        return assignment

    def _find_enabling_parent_value(
        self,
        child: VariableDefinition,
        parent: VariableDefinition
    ) -> Optional[Any]:
        """Find a parent value that makes the child variable relevant.

        Args:
            child: The child variable that needs to be enabled.
            parent: The parent variable.

        Returns:
            A parent value that enables the child, or None.
        """
        # Try each parent state
        for state in parent.get_all_states():
            test_assignment = VariableAssignment(assignments={parent.name: state})
            if child.is_relevant(test_assignment):
                return state
        return None

    def _get_default_value(self, var: VariableDefinition) -> Any:
        """Get a default (non-edge-case) value for a variable.

        Args:
            var: The variable.

        Returns:
            A default value.
        """
        all_states = var.get_all_states()
        edge_cases = set(str(e) for e in var.get_edge_case_states())

        # Prefer non-edge-case values
        for state in all_states:
            if str(state) not in edge_cases:
                return state

        # Fall back to first state
        return all_states[0] if all_states else None

    def _find_best_pairwise_assignment(
        self,
        uncovered_pairs: list[tuple[str, Any, str, Any]]
    ) -> Optional[VariableAssignment]:
        """Find an assignment that covers the most uncovered pairs.

        Args:
            uncovered_pairs: List of pairs that haven't been covered.

        Returns:
            An assignment that covers many pairs, or None.
        """
        if not uncovered_pairs:
            return None

        # Start with the first uncovered pair
        var1, val1, var2, val2 = uncovered_pairs[0]

        # Build an assignment covering this pair
        var1_def = self.variable_space.get(var1)
        var2_def = self.variable_space.get(var2)

        if not var1_def or not var2_def:
            return None

        # Convert string values back to actual values if needed
        val1_actual = self._parse_value(var1_def, val1)
        val2_actual = self._parse_value(var2_def, val2)

        return self._build_assignment_with_values([
            (var1_def, val1_actual),
            (var2_def, val2_actual)
        ])

    def _parse_value(self, var: VariableDefinition, value_str: str) -> Any:
        """Parse a string value back to the actual value type.

        Args:
            var: The variable definition.
            value_str: The string representation of the value.

        Returns:
            The actual value.
        """
        for state in var.get_all_states():
            if str(state) == value_str:
                return state
        return value_str

    def get_coverage_stats(self) -> CoverageStats:
        """Get current coverage statistics.

        Returns:
            CoverageStats object with coverage metrics.
        """
        return self._tracker.get_coverage_stats()

    def get_yielded_count(self) -> int:
        """Get the number of assignments yielded so far.

        Returns:
            Number of yielded assignments.
        """
        return self._yielded_count
