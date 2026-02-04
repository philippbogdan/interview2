"""Scenario constructor for building scenarios from variable assignments.

This module transforms variable assignments into complete, coherent
test scenarios using LLM-based construction.
"""

import os
from typing import Any, Optional

from openai import OpenAI  # xAI uses OpenAI-compatible API

from parsers.input_parser import NormalizedAgentConfig
from parsers.domain_analyzer import DomainAnalysis
from models.variables import (
    VariableAssignment,
    TriStateBoolean,
    QuantitativeState,
)


class ScenarioConstructor:
    """Constructs complete test scenarios from variable assignments."""

    def __init__(
        self,
        config: NormalizedAgentConfig,
        domain_analysis: DomainAnalysis,
        api_key: Optional[str] = None,
        model: str = "grok-4-1-fast-non-reasoning"
    ):
        """Initialize the scenario constructor.

        Args:
            config: Agent configuration.
            domain_analysis: Domain analysis results.
            api_key: xAI API key. If not provided, reads from XAI_API_KEY.
            model: xAI model to use for scenario construction.
        """
        self.config = config
        self.domain_analysis = domain_analysis
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.model = model
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

    def construct(self, assignment: VariableAssignment) -> dict[str, Any]:
        """Construct a complete scenario from a variable assignment.

        Args:
            assignment: The variable assignment to build from.

        Returns:
            Complete scenario dictionary with all required fields.
        """
        # Format the assignment for the prompt
        assignment_text = self._format_assignment(assignment)

        prompt = self._build_prompt(assignment_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )

            import json
            content = response.choices[0].message.content
            scenario = json.loads(content)

            # Ensure required fields exist
            scenario = self._ensure_required_fields(scenario, assignment)

            # Add assignment metadata
            scenario["_variable_assignment"] = {
                k: self._serialize_value(v)
                for k, v in assignment.items()
            }

            return scenario

        except Exception as e:
            # Return a basic scenario on error
            return self._build_fallback_scenario(assignment, str(e))

    def _get_system_prompt(self) -> str:
        """Get the system prompt for scenario construction."""
        return """You are an expert test scenario designer. Your task is to create
a coherent, realistic test scenario based on specific variable assignments.

The scenario should:
1. Be internally consistent with all variable values
2. Feel natural and realistic
3. Include specific, testable criteria
4. Cover the exact conditions specified by the variables

Always respond with valid JSON containing:
- scenarioName: A concise, descriptive name
- scenarioDescription: Detailed description of the scenario
- criteria: Array of specific, measurable success criteria
- Any domain-specific fields as appropriate"""

    def _build_prompt(self, assignment_text: str) -> str:
        """Build the construction prompt.

        Args:
            assignment_text: Formatted variable assignment.

        Returns:
            Complete prompt for scenario construction.
        """
        parts = [
            "Create a test scenario with the following conditions:",
            "",
            "## Variable Assignment",
            assignment_text,
            "",
        ]

        # Add agent context
        if self.config.description:
            parts.append("## Agent Description")
            parts.append(self.config.description)
            parts.append("")

        # Add domain context
        parts.append("## Domain Context")
        parts.append(f"Domain: {self.domain_analysis.detected_domain}")
        parts.append("")

        # Add available actions
        if self.config.actions:
            parts.append("## Available Actions")
            for action in self.config.actions:
                parts.append(f"- {action}")
            parts.append("")

        # Add field requirements
        if self.domain_analysis.field_model:
            parts.append("## Required Fields")
            fields = self.domain_analysis.field_model.model_fields
            for field_name, field_info in fields.items():
                desc = field_info.description or ""
                parts.append(f"- {field_name}: {desc}")
            parts.append("")

        parts.append("## Requirements")
        parts.append("1. Create a realistic scenario that matches ALL variable conditions")
        parts.append("2. Include 3-5 specific, measurable success criteria")
        parts.append("3. Make the scenario feel natural, not contrived")
        parts.append("4. Include domain-specific data (names, dates, etc.) as appropriate")

        return "\n".join(parts)

    def _format_assignment(self, assignment: VariableAssignment) -> str:
        """Format a variable assignment for the prompt.

        Args:
            assignment: The assignment to format.

        Returns:
            Human-readable assignment description.
        """
        lines = []

        for name, value in sorted(assignment.items()):
            readable_value = self._format_value(value)
            # Convert snake_case to readable
            readable_name = name.replace("_", " ").title()
            lines.append(f"- {readable_name}: {readable_value}")

        return "\n".join(lines) if lines else "(no specific conditions)"

    def _format_value(self, value: Any) -> str:
        """Format a value for human readability.

        Args:
            value: The value to format.

        Returns:
            Human-readable string.
        """
        if isinstance(value, TriStateBoolean):
            return {
                TriStateBoolean.TRUE: "yes",
                TriStateBoolean.FALSE: "no",
                TriStateBoolean.UNKNOWN: "unknown",
            }.get(value, str(value))

        if isinstance(value, QuantitativeState):
            return {
                QuantitativeState.NONE: "none",
                QuantitativeState.SINGLE: "single",
                QuantitativeState.MANY: "many",
            }.get(value, str(value))

        if isinstance(value, tuple) and len(value) == 2:
            state, variant = value
            return f"{state}:{variant}"

        return str(value)

    def _serialize_value(self, value: Any) -> str:
        """Serialize a value for storage.

        Args:
            value: The value to serialize.

        Returns:
            Serialized string representation.
        """
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, tuple):
            return f"{value[0]}:{value[1]}"
        return str(value)

    def _ensure_required_fields(
        self,
        scenario: dict,
        assignment: VariableAssignment
    ) -> dict:
        """Ensure the scenario has all required fields.

        Args:
            scenario: The constructed scenario.
            assignment: The source assignment.

        Returns:
            Scenario with all required fields.
        """
        # Ensure base fields
        if "scenarioName" not in scenario:
            scenario["scenarioName"] = self._generate_name(assignment)

        if "scenarioDescription" not in scenario:
            scenario["scenarioDescription"] = self._generate_description(assignment)

        if "criteria" not in scenario or not scenario["criteria"]:
            scenario["criteria"] = self._generate_criteria(assignment)

        return scenario

    def _generate_name(self, assignment: VariableAssignment) -> str:
        """Generate a scenario name from assignment.

        Args:
            assignment: The variable assignment.

        Returns:
            Generated scenario name.
        """
        # Find the most interesting variable
        edge_vars = []
        for name, value in assignment.items():
            if isinstance(value, TriStateBoolean) and value != TriStateBoolean.TRUE:
                edge_vars.append((name, value))
            elif isinstance(value, QuantitativeState) and value != QuantitativeState.SINGLE:
                edge_vars.append((name, value))

        if edge_vars:
            name, value = edge_vars[0]
            return f"Test: {name.replace('_', ' ').title()} = {self._format_value(value)}"

        return f"Scenario with {len(assignment.keys())} variables"

    def _generate_description(self, assignment: VariableAssignment) -> str:
        """Generate a scenario description from assignment.

        Args:
            assignment: The variable assignment.

        Returns:
            Generated scenario description.
        """
        conditions = [
            f"{k.replace('_', ' ')} is {self._format_value(v)}"
            for k, v in assignment.items()
        ]
        return f"Test scenario where: {'; '.join(conditions[:5])}"

    def _generate_criteria(self, assignment: VariableAssignment) -> list[str]:
        """Generate basic criteria from assignment.

        Args:
            assignment: The variable assignment.

        Returns:
            List of generated criteria.
        """
        criteria = []

        for name, value in assignment.items():
            if isinstance(value, TriStateBoolean):
                if value == TriStateBoolean.FALSE:
                    criteria.append(
                        f"Agent handles missing {name.replace('_', ' ')} gracefully"
                    )
                elif value == TriStateBoolean.UNKNOWN:
                    criteria.append(
                        f"Agent handles uncertain {name.replace('_', ' ')} appropriately"
                    )
            elif isinstance(value, QuantitativeState):
                if value == QuantitativeState.NONE:
                    criteria.append(
                        f"Agent handles zero {name.replace('_', ' ')} correctly"
                    )
                elif value == QuantitativeState.MANY:
                    criteria.append(
                        f"Agent handles multiple {name.replace('_', ' ')} correctly"
                    )

        if not criteria:
            criteria.append("Agent completes the interaction successfully")
            criteria.append("Agent follows expected conversation flow")

        return criteria[:5]

    def _build_fallback_scenario(
        self,
        assignment: VariableAssignment,
        error: str
    ) -> dict[str, Any]:
        """Build a fallback scenario when LLM fails.

        Args:
            assignment: The variable assignment.
            error: Error message.

        Returns:
            Basic fallback scenario.
        """
        return {
            "scenarioName": self._generate_name(assignment),
            "scenarioDescription": self._generate_description(assignment),
            "criteria": self._generate_criteria(assignment),
            "_error": error,
            "_variable_assignment": {
                k: self._serialize_value(v)
                for k, v in assignment.items()
            }
        }
