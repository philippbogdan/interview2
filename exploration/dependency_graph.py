"""Dependency graph for variable relationships.

This module implements a DAG structure for tracking dependencies between variables,
ensuring proper ordering and relevance checking during exploration.
"""

from dataclasses import dataclass, field
from typing import Optional

from models.variables import (
    VariableDefinition,
    VariableSpace,
    VariableAssignment,
)


@dataclass
class DependencyNode:
    """A node in the dependency graph."""
    variable: VariableDefinition
    children: list["DependencyNode"] = field(default_factory=list)
    parents: list["DependencyNode"] = field(default_factory=list)


class DependencyGraph:
    """DAG structure for variable dependencies.

    Manages the relationships between variables where some variables
    are only relevant when parent variables have certain values.
    """

    def __init__(self, variable_space: VariableSpace):
        """Initialize the dependency graph from a variable space.

        Args:
            variable_space: The variable space to build the graph from.
        """
        self.variable_space = variable_space
        self.nodes: dict[str, DependencyNode] = {}
        self._root_nodes: list[DependencyNode] = []
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the dependency graph from the variable space."""
        # Create nodes for all variables
        for var in self.variable_space:
            self.nodes[var.name] = DependencyNode(variable=var)

        # Link dependencies
        for var in self.variable_space:
            node = self.nodes[var.name]

            if var.depends_on:
                parent_node = self.nodes.get(var.depends_on)
                if parent_node:
                    parent_node.children.append(node)
                    node.parents.append(parent_node)

        # Identify root nodes (no parents)
        self._root_nodes = [
            node for node in self.nodes.values()
            if not node.parents
        ]

    def topological_order(self) -> list[VariableDefinition]:
        """Return variables in topological order (parents before children).

        Returns:
            List of variables sorted so that parent variables come before
            their dependents.
        """
        visited: set[str] = set()
        result: list[VariableDefinition] = []

        def visit(node: DependencyNode) -> None:
            if node.variable.name in visited:
                return
            visited.add(node.variable.name)

            # Visit all parents first
            for parent in node.parents:
                visit(parent)

            result.append(node.variable)

            # Then visit children
            for child in node.children:
                visit(child)

        # Start from root nodes
        for root in self._root_nodes:
            visit(root)

        # Handle any disconnected nodes
        for node in self.nodes.values():
            if node.variable.name not in visited:
                visit(node)

        return result

    def get_root_variables(self) -> list[VariableDefinition]:
        """Get variables with no dependencies (root nodes).

        Returns:
            List of variables that don't depend on any other variables.
        """
        return [node.variable for node in self._root_nodes]

    def get_children(self, variable_name: str) -> list[VariableDefinition]:
        """Get variables that depend on the given variable.

        Args:
            variable_name: Name of the parent variable.

        Returns:
            List of variables that depend on the given variable.
        """
        node = self.nodes.get(variable_name)
        if not node:
            return []
        return [child.variable for child in node.children]

    def get_parents(self, variable_name: str) -> list[VariableDefinition]:
        """Get variables that the given variable depends on.

        Args:
            variable_name: Name of the child variable.

        Returns:
            List of variables that the given variable depends on.
        """
        node = self.nodes.get(variable_name)
        if not node:
            return []
        return [parent.variable for parent in node.parents]

    def is_relevant(
        self,
        variable: VariableDefinition,
        assignment: VariableAssignment
    ) -> bool:
        """Check if a variable is relevant given the current assignment.

        A variable is relevant if:
        1. It has no relevant_when condition, OR
        2. Its relevant_when condition returns True for the current assignment

        Additionally, if the variable depends on another variable,
        the parent must be assigned for relevance checking to be meaningful.

        Args:
            variable: The variable to check.
            assignment: The current variable assignment.

        Returns:
            True if the variable is relevant, False otherwise.
        """
        # Check if parent variable is assigned (if dependency exists)
        if variable.depends_on:
            if not assignment.has(variable.depends_on):
                # Parent not assigned yet - variable may become relevant
                return True

        # Check the relevant_when condition
        return variable.is_relevant(assignment)

    def get_relevant_variables(
        self,
        assignment: VariableAssignment
    ) -> list[VariableDefinition]:
        """Get all variables that are relevant given the current assignment.

        Args:
            assignment: The current variable assignment.

        Returns:
            List of relevant variables in topological order.
        """
        return [
            var for var in self.topological_order()
            if self.is_relevant(var, assignment)
        ]

    def get_unassigned_relevant_variables(
        self,
        assignment: VariableAssignment
    ) -> list[VariableDefinition]:
        """Get relevant variables that haven't been assigned yet.

        Args:
            assignment: The current variable assignment.

        Returns:
            List of unassigned relevant variables in topological order.
        """
        return [
            var for var in self.get_relevant_variables(assignment)
            if not assignment.has(var.name)
        ]

    def validate_assignment(
        self,
        assignment: VariableAssignment
    ) -> tuple[bool, list[str]]:
        """Validate that an assignment respects dependency constraints.

        Args:
            assignment: The assignment to validate.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: list[str] = []

        for var in self.variable_space:
            if not assignment.has(var.name):
                continue

            # If variable is assigned, check if it's relevant
            if not self.is_relevant(var, assignment):
                errors.append(
                    f"Variable '{var.name}' is assigned but not relevant "
                    f"given current assignment"
                )

            # If variable has dependency, parent should be assigned
            if var.depends_on and not assignment.has(var.depends_on):
                errors.append(
                    f"Variable '{var.name}' depends on '{var.depends_on}' "
                    f"which is not assigned"
                )

        return len(errors) == 0, errors

    def __len__(self) -> int:
        """Return the number of variables in the graph."""
        return len(self.nodes)

    def __contains__(self, variable_name: str) -> bool:
        """Check if a variable is in the graph."""
        return variable_name in self.nodes


def build_dependency_graph(variable_space: VariableSpace) -> DependencyGraph:
    """Build a dependency graph from a variable space.

    Args:
        variable_space: The variable space to build from.

    Returns:
        A DependencyGraph instance.
    """
    return DependencyGraph(variable_space)
