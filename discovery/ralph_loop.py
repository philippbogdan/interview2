"""Ralph Loop for iterative variable expansion.

This module implements the "Ralph Loop" - an iterative LLM-based
expansion process that grows the variable space by ~50% each iteration.
"""

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI  # xAI uses OpenAI-compatible API

from models.variables import (
    VariableSpace,
    VariableDefinition,
    BooleanVariable,
    QuantitativeVariable,
    EnumVariable,
)


@dataclass
class RalphExpansion:
    """Result of a single Ralph Loop iteration."""

    iteration: int
    variables_before: int
    variables_after: int
    new_variables: list[VariableDefinition]
    expansion_rate: float


class RalphLoop:
    """Iterative LLM-based variable expansion.

    The Ralph Loop takes the current variable list and asks an LLM
    to expand it by ~50% each iteration, discovering new dimensions
    and edge cases that weren't initially considered.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-4-1-fast-non-reasoning",
        target_expansion_rate: float = 0.5
    ):
        """Initialize the Ralph Loop.

        Args:
            api_key: xAI API key. If not provided, reads from XAI_API_KEY.
            model: xAI model to use for expansion.
            target_expansion_rate: Target expansion rate per iteration (0.5 = 50%).
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.model = model
        self.target_expansion_rate = target_expansion_rate
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy-initialize the xAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "xAI API key required. Set XAI_API_KEY or pass api_key."
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
        return self._client

    def expand(
        self,
        variables: VariableSpace,
        description: str,
        iterations: int = 3
    ) -> tuple[VariableSpace, list[RalphExpansion]]:
        """Run the Ralph Loop expansion.

        Args:
            variables: Initial variable space to expand.
            description: Agent description for context.
            iterations: Number of expansion iterations.

        Returns:
            Tuple of (expanded VariableSpace, list of expansion results).
        """
        expansions: list[RalphExpansion] = []

        for i in range(iterations):
            before_count = len(variables)

            # Calculate target number of new variables
            target_new = max(1, int(before_count * self.target_expansion_rate))

            # Run expansion
            new_vars = self._expand_iteration(variables, description, target_new)

            # Add new variables (deduplicating)
            for var in new_vars:
                if not variables.get(var.name):
                    variables.add(var)

            after_count = len(variables)
            actual_new = after_count - before_count

            expansion = RalphExpansion(
                iteration=i + 1,
                variables_before=before_count,
                variables_after=after_count,
                new_variables=new_vars[:actual_new] if new_vars else [],
                expansion_rate=(
                    actual_new / before_count if before_count > 0 else 0.0
                )
            )
            expansions.append(expansion)

        return variables, expansions

    def _expand_iteration(
        self,
        variables: VariableSpace,
        description: str,
        target_count: int
    ) -> list[VariableDefinition]:
        """Run a single expansion iteration.

        Args:
            variables: Current variable space.
            description: Agent description.
            target_count: Target number of new variables.

        Returns:
            List of new variables to add.
        """
        # Format current variables for the prompt
        current_vars = self._format_variables(variables)

        prompt = f"""You are an expert test engineer expanding a test variable space.

## Agent Description
{description}

## Current Variables
{current_vars}

## Task
Suggest {target_count} NEW test variables that are NOT already in the list.
Think about:
1. Edge cases that aren't covered
2. Environmental factors (network, time of day, load)
3. User state variations (emotional, cognitive, situational)
4. System state variations (capacity, errors, degraded service)
5. Data variations (format, completeness, validity)
6. Interaction patterns (interruptions, corrections, abandonment)

For each variable, provide:
- name: snake_case variable name
- type: "boolean", "quantitative", or "enum"
- description: what this variable represents
- states: for enum type, list the possible states
- edge_case_states: which states are edge cases (for prioritization)
- category: logical grouping (e.g., "user_state", "system_state", "environment")

Format as JSON:
{{"variables": [
  {{"name": "...", "type": "boolean|quantitative|enum", "description": "...",
    "states": [...], "edge_case_states": [...], "category": "..."}}
]}}

Ensure variables are:
1. NOT duplicates of existing variables
2. Relevant to the agent's domain
3. Testable (can be controlled or observed in tests)
4. Specific enough to be actionable"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            import json
            content = response.choices[0].message.content
            data = json.loads(content)

            new_vars: list[VariableDefinition] = []

            for var_spec in data.get("variables", []):
                var = self._create_variable(var_spec)
                if var and not variables.get(var.name):
                    new_vars.append(var)

            return new_vars

        except Exception:
            return []

    def _format_variables(self, variables: VariableSpace) -> str:
        """Format variables for the prompt.

        Args:
            variables: Variable space to format.

        Returns:
            Formatted string representation.
        """
        lines = []
        for var in variables:
            type_name = type(var).__name__.replace("Variable", "").lower()
            states = var.get_all_states()
            lines.append(
                f"- {var.name} ({type_name}): {var.description} "
                f"[{var.category}] states={states[:5]}{'...' if len(states) > 5 else ''}"
            )
        return "\n".join(lines) if lines else "(no variables yet)"

    def _create_variable(self, spec: dict) -> Optional[VariableDefinition]:
        """Create a variable from a specification dict.

        Args:
            spec: Variable specification.

        Returns:
            VariableDefinition or None if invalid.
        """
        name = spec.get("name", "")
        var_type = spec.get("type", "boolean")
        description = spec.get("description", "")
        category = spec.get("category", "generated")
        states = spec.get("states", [])
        edge_states = spec.get("edge_case_states", [])

        if not name or not description:
            return None

        # Clean the name
        name = name.lower().replace(" ", "_").replace("-", "_")

        if var_type == "boolean":
            return BooleanVariable(
                name=name,
                description=description,
                category=category,
                is_edge_case_priority=bool(edge_states)
            )
        elif var_type == "quantitative":
            return QuantitativeVariable(
                name=name,
                description=description,
                category=category,
                single_variants=states[:2] if len(states) <= 3 else [],
                is_edge_case_priority=bool(edge_states)
            )
        elif var_type == "enum" and states:
            return EnumVariable(
                name=name,
                description=description,
                category=category,
                states=states,
                edge_case_states=edge_states,
                is_edge_case_priority=bool(edge_states)
            )

        return None


def ralph_loop(
    variables: VariableSpace,
    description: str,
    iterations: int = 3,
    api_key: Optional[str] = None
) -> VariableSpace:
    """Run the Ralph Loop variable expansion.

    Convenience function that creates a RalphLoop and runs expansion.

    Args:
        variables: Initial variable space.
        description: Agent description.
        iterations: Number of iterations (default 3).
        api_key: Optional xAI API key.

    Returns:
        Expanded variable space.
    """
    loop = RalphLoop(api_key=api_key)
    expanded, _ = loop.expand(variables, description, iterations)
    return expanded
