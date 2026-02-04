"""Scenario generator using OpenAI's Structured Outputs."""

import os
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel

from parsers.input_parser import NormalizedAgentConfig
from parsers.domain_analyzer import DomainAnalysis
from models.base import BaseScenario


class ScenarioGenerator:
    """Generate test scenarios using OpenAI's Structured Outputs."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-2024-08-06"):
        """
        Initialize the scenario generator.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model: Model to use for generation. Must support structured outputs.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        config: NormalizedAgentConfig,
        domain_analysis: DomainAnalysis,
        num_scenarios: int,
    ) -> list[dict]:
        """
        Generate test scenarios based on the agent configuration.

        Args:
            config: Normalized agent configuration
            domain_analysis: Results of domain analysis
            num_scenarios: Number of scenarios to generate

        Returns:
            List of scenario dictionaries
        """
        prompt = self._build_scenario_generation_prompt(
            config, domain_analysis, num_scenarios
        )
        response_schema = self._build_response_schema(domain_analysis)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "scenarios_response",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        )

        import json
        result = json.loads(response.choices[0].message.content)
        return result["scenarios"]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for scenario generation."""
        return """You are an expert test scenario designer for voice agents. Your task is to create comprehensive, realistic test scenarios that thoroughly evaluate a voice agent's capabilities.

When generating scenarios, ensure diversity by including:
1. Happy paths - Standard successful interactions
2. Edge cases - Unusual but valid inputs
3. Error handling - Invalid inputs, misunderstandings, corrections
4. Multi-turn conversations - Complex interactions requiring context tracking
5. Boundary conditions - Limits of the agent's capabilities

Each scenario should be realistic, specific, and actionable. Include concrete data values that would be used during testing."""

    def _build_scenario_generation_prompt(
        self,
        config: NormalizedAgentConfig,
        domain_analysis: DomainAnalysis,
        num_scenarios: int,
    ) -> str:
        """Build the prompt for scenario generation."""
        parts = [
            f"Generate {num_scenarios} diverse test scenarios for the following voice agent:",
            "",
        ]

        # Add agent description
        if config.description:
            parts.append("## Agent Description")
            parts.append(config.description)
            parts.append("")

        # Add detected domain info
        parts.append("## Detected Domain")
        parts.append(f"Domain: {domain_analysis.detected_domain}")
        parts.append(f"Confidence: {domain_analysis.confidence:.2f}")
        if domain_analysis.keywords_found:
            parts.append(f"Keywords found: {', '.join(domain_analysis.keywords_found)}")
        parts.append("")

        # Add available actions if present
        if config.actions:
            parts.append("## Available Actions")
            for action in config.actions:
                parts.append(f"- {action}")
            parts.append("")

        # Add states if present
        if config.states:
            parts.append("## Agent States")
            for state in config.states:
                parts.append(f"- {state}")
            parts.append("")

        # Add entities if present
        if config.entities:
            parts.append("## Entities/Slots")
            for entity in config.entities:
                parts.append(f"- {entity}")
            parts.append("")

        # Add suggested edge cases
        if domain_analysis.suggested_edge_cases:
            parts.append("## Suggested Edge Cases to Consider")
            for edge_case in domain_analysis.suggested_edge_cases:
                parts.append(f"- {edge_case}")
            parts.append("")

        # Add field requirements
        if domain_analysis.field_model:
            parts.append("## Required Domain Fields")
            parts.append(
                f"Each scenario must include realistic values for: "
                f"{', '.join(domain_analysis.field_model.model_fields.keys())}"
            )
            parts.append("")

        parts.append("## Requirements")
        parts.append(f"- Generate exactly {num_scenarios} scenarios")
        parts.append("- Each scenario must have a unique name and test different aspects")
        parts.append("- Include a mix of happy paths, edge cases, and error scenarios")
        parts.append("- Criteria should be specific and measurable")
        parts.append("- All field values should be realistic and internally consistent")

        return "\n".join(parts)

    def _build_response_schema(self, domain_analysis: DomainAnalysis) -> dict[str, Any]:
        """
        Build the JSON schema for structured output.

        Dynamically merges base scenario fields with domain-specific fields.
        OpenAI strict mode requires ALL properties in the required array.
        Optional fields are handled by making them nullable types.
        """
        # Start with base scenario fields
        base_properties = {
            "scenarioName": {
                "type": "string",
                "description": "A concise, descriptive name for the test scenario",
            },
            "scenarioDescription": {
                "type": "string",
                "description": "Detailed description of what this scenario tests",
            },
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of success criteria for this scenario",
            },
        }
        required_fields = ["scenarioName", "scenarioDescription", "criteria"]

        # Add domain-specific fields if we have a field model
        if domain_analysis.field_model:
            domain_schema = domain_analysis.field_model.model_json_schema()
            pydantic_required = set(domain_schema.get("required", []))

            for field_name, field_info in domain_schema.get("properties", {}).items():
                is_optional = field_name not in pydantic_required
                field_schema = self._convert_field_schema(field_info, is_optional)
                base_properties[field_name] = field_schema
                # OpenAI strict mode requires ALL fields in required array
                required_fields.append(field_name)

        # Build the complete schema
        scenario_schema = {
            "type": "object",
            "properties": base_properties,
            "required": required_fields,
            "additionalProperties": False,
        }

        return {
            "type": "object",
            "properties": {
                "scenarios": {
                    "type": "array",
                    "items": scenario_schema,
                    "description": "List of generated test scenarios",
                }
            },
            "required": ["scenarios"],
            "additionalProperties": False,
        }

    def _convert_field_schema(self, field_info: dict, make_nullable: bool = False) -> dict:
        """Convert a Pydantic field schema to JSON Schema format for OpenAI.

        Args:
            field_info: The Pydantic field schema
            make_nullable: If True, make the field nullable (for optional fields)
        """
        result = {}

        # Handle anyOf for Optional fields (Pydantic already marks these as nullable)
        if "anyOf" in field_info:
            # Find the non-null type
            for option in field_info["anyOf"]:
                if option.get("type") != "null":
                    result = dict(option)
                    break
            # Already optional in Pydantic, make nullable for OpenAI
            if result.get("type"):
                result["type"] = [result["type"], "null"]
        else:
            result = dict(field_info)
            # Make nullable if this is an optional field
            if make_nullable and result.get("type"):
                result["type"] = [result["type"], "null"]

        # Copy description if present
        if "description" in field_info:
            result["description"] = field_info["description"]

        # Remove default - OpenAI strict mode doesn't support defaults
        result.pop("default", None)

        return result
