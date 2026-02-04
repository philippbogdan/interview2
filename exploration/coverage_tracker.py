"""Coverage tracking for combinatorial exploration.

This module tracks which variable combinations have been covered
and provides metrics about test coverage.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from itertools import combinations

from models.variables import VariableAssignment, VariableSpace


@dataclass
class CoverageStats:
    """Statistics about coverage of the variable space."""

    total_assignments: int = 0
    edge_cases_covered: int = 0
    total_edge_cases: int = 0
    pairwise_coverage: float = 0.0
    covered_pairs: int = 0
    total_pairs: int = 0
    categories_touched: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_assignments": self.total_assignments,
            "edge_cases_covered": self.edge_cases_covered,
            "total_edge_cases": self.total_edge_cases,
            "edge_case_coverage": (
                self.edge_cases_covered / self.total_edge_cases
                if self.total_edge_cases > 0 else 0.0
            ),
            "pairwise_coverage": self.pairwise_coverage,
            "covered_pairs": self.covered_pairs,
            "total_pairs": self.total_pairs,
            "categories_touched": self.categories_touched,
        }


class CoverageTracker:
    """Tracks coverage of variable assignments in the exploration space.

    Monitors:
    - Which specific assignments have been generated
    - Pairwise coverage (all pairs of variable values)
    - Edge case coverage
    - Category coverage
    """

    def __init__(self, variable_space: VariableSpace):
        """Initialize the coverage tracker.

        Args:
            variable_space: The variable space being explored.
        """
        self.variable_space = variable_space
        self._covered_assignments: set[int] = set()
        self._covered_pairs: set[tuple[str, Any, str, Any]] = set()
        self._covered_edge_cases: set[tuple[str, Any]] = set()
        self._category_counts: dict[str, int] = {}
        self._all_pairs: Optional[set[tuple[str, Any, str, Any]]] = None
        self._all_edge_cases: Optional[set[tuple[str, Any]]] = None

    def add_assignment(self, assignment: VariableAssignment) -> None:
        """Record a new assignment as covered.

        Args:
            assignment: The assignment to record.
        """
        # Track the full assignment
        self._covered_assignments.add(hash(assignment))

        # Track pairwise coverage
        items = list(assignment.items())
        for i, (var1, val1) in enumerate(items):
            for var2, val2 in items[i + 1:]:
                pair = self._normalize_pair(var1, val1, var2, val2)
                self._covered_pairs.add(pair)

        # Track edge case coverage
        for var_name, value in assignment.items():
            var = self.variable_space.get(var_name)
            if var:
                edge_cases = var.get_edge_case_states()
                if value in edge_cases or (
                    isinstance(value, tuple) and value in edge_cases
                ):
                    self._covered_edge_cases.add((var_name, str(value)))

        # Track category coverage
        for var_name, _ in assignment.items():
            var = self.variable_space.get(var_name)
            if var:
                if var.category not in self._category_counts:
                    self._category_counts[var.category] = 0
                self._category_counts[var.category] += 1

    def is_covered(self, assignment: VariableAssignment) -> bool:
        """Check if an assignment has already been covered.

        Args:
            assignment: The assignment to check.

        Returns:
            True if the assignment has been covered.
        """
        return hash(assignment) in self._covered_assignments

    def is_pair_covered(
        self,
        var1: str,
        val1: Any,
        var2: str,
        val2: Any
    ) -> bool:
        """Check if a specific variable pair has been covered.

        Args:
            var1: First variable name.
            val1: First variable value.
            var2: Second variable name.
            val2: Second variable value.

        Returns:
            True if the pair has been covered.
        """
        pair = self._normalize_pair(var1, val1, var2, val2)
        return pair in self._covered_pairs

    def get_uncovered_pairs(self) -> list[tuple[str, Any, str, Any]]:
        """Get all pairs that haven't been covered yet.

        Returns:
            List of uncovered pairs as (var1, val1, var2, val2) tuples.
        """
        all_pairs = self._get_all_pairs()
        return [p for p in all_pairs if p not in self._covered_pairs]

    def get_coverage_stats(self) -> CoverageStats:
        """Get current coverage statistics.

        Returns:
            CoverageStats object with coverage metrics.
        """
        all_pairs = self._get_all_pairs()
        all_edge_cases = self._get_all_edge_cases()

        stats = CoverageStats(
            total_assignments=len(self._covered_assignments),
            edge_cases_covered=len(self._covered_edge_cases),
            total_edge_cases=len(all_edge_cases),
            covered_pairs=len(self._covered_pairs),
            total_pairs=len(all_pairs),
            pairwise_coverage=(
                len(self._covered_pairs) / len(all_pairs)
                if all_pairs else 0.0
            ),
            categories_touched=self._category_counts.copy(),
        )

        return stats

    def mark_covered(self, assignment: VariableAssignment) -> None:
        """Alias for add_assignment for clarity.

        Args:
            assignment: The assignment to mark as covered.
        """
        self.add_assignment(assignment)

    def _normalize_pair(
        self,
        var1: str,
        val1: Any,
        var2: str,
        val2: Any
    ) -> tuple[str, Any, str, Any]:
        """Normalize a pair to ensure consistent ordering.

        Args:
            var1: First variable name.
            val1: First variable value.
            var2: Second variable name.
            val2: Second variable value.

        Returns:
            Normalized tuple with consistent ordering.
        """
        if var1 < var2:
            return (var1, str(val1), var2, str(val2))
        return (var2, str(val2), var1, str(val1))

    def _get_all_pairs(self) -> set[tuple[str, Any, str, Any]]:
        """Get all possible variable-value pairs.

        Returns:
            Set of all possible pairs.
        """
        if self._all_pairs is not None:
            return self._all_pairs

        self._all_pairs = set()
        variables = list(self.variable_space)

        for var1, var2 in combinations(variables, 2):
            # Skip if one depends on the other (not all combos are valid)
            if var1.depends_on == var2.name or var2.depends_on == var1.name:
                continue

            for val1 in var1.get_all_states():
                for val2 in var2.get_all_states():
                    pair = self._normalize_pair(
                        var1.name, val1, var2.name, val2
                    )
                    self._all_pairs.add(pair)

        return self._all_pairs

    def _get_all_edge_cases(self) -> set[tuple[str, Any]]:
        """Get all possible edge cases.

        Returns:
            Set of all edge cases as (variable_name, value) tuples.
        """
        if self._all_edge_cases is not None:
            return self._all_edge_cases

        self._all_edge_cases = set()

        for var in self.variable_space:
            for edge_case in var.get_edge_case_states():
                self._all_edge_cases.add((var.name, str(edge_case)))

        return self._all_edge_cases

    def reset(self) -> None:
        """Reset all coverage tracking."""
        self._covered_assignments.clear()
        self._covered_pairs.clear()
        self._covered_edge_cases.clear()
        self._category_counts.clear()
