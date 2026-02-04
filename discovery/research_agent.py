"""Deep research agent for discovering edge cases and variables.

This module uses LLM queries to discover real-world edge cases,
regulatory requirements, and accessibility considerations
that should be tested.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI  # xAI uses OpenAI-compatible API

from models.variables import (
    VariableSpace,
    VariableDefinition,
    BooleanVariable,
    EnumVariable,
)


@dataclass
class ResearchFinding:
    """A finding from the research agent."""

    category: str
    description: str
    source_type: str  # "incident", "regulatory", "accessibility", "best_practice"
    implied_variables: list[dict] = field(default_factory=list)
    priority: str = "medium"  # "low", "medium", "high", "critical"


class ResearchAgent:
    """Deep research agent for discovering edge cases via LLM."""

    def __init__(self, api_key: Optional[str] = None, model: str = "grok-4-1-fast-non-reasoning"):
        """Initialize the research agent.

        Args:
            api_key: xAI API key. If not provided, reads from XAI_API_KEY.
            model: xAI model to use for research queries.
        """
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

    def research(
        self,
        domain: str,
        description: str,
        existing_variables: Optional[VariableSpace] = None
    ) -> list[ResearchFinding]:
        """Conduct deep research on the domain for edge cases.

        Args:
            domain: The detected domain (e.g., "healthcare", "restaurant").
            description: The agent description.
            existing_variables: Already extracted variables to avoid duplication.

        Returns:
            List of research findings with implied variables.
        """
        findings: list[ResearchFinding] = []

        # Research real-world incidents
        incident_findings = self._research_incidents(domain, description)
        findings.extend(incident_findings)

        # Research regulatory/compliance requirements
        regulatory_findings = self._research_regulatory(domain, description)
        findings.extend(regulatory_findings)

        # Research accessibility considerations
        accessibility_findings = self._research_accessibility(domain, description)
        findings.extend(accessibility_findings)

        return findings

    def _research_incidents(
        self,
        domain: str,
        description: str
    ) -> list[ResearchFinding]:
        """Research real-world incidents and failures in the domain.

        Args:
            domain: The domain to research.
            description: Agent description for context.

        Returns:
            List of findings about incidents.
        """
        prompt = f"""You are a test engineer researching edge cases for a voice agent.

Domain: {domain}
Agent Description: {description}

Think about real-world incidents, failures, and problematic scenarios that have occurred
in this domain. Consider:
1. What can go wrong during interactions?
2. What unusual user behaviors cause problems?
3. What system states lead to failures?
4. What edge cases are commonly missed in testing?

For each finding, provide:
- A brief description of the incident/edge case
- What variables would need to be tested (e.g., "user_is_minor: boolean")
- Priority (low/medium/high/critical)

Format your response as a JSON array with objects containing:
{{"description": "...", "variables": [{{"name": "...", "type": "boolean|enum|quantitative", "states": [...] if enum}}], "priority": "..."}}

Provide 3-5 findings. Focus on non-obvious edge cases that are often missed."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            import json
            content = response.choices[0].message.content
            data = json.loads(content)

            findings = []
            items = data.get("findings", data) if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = [items]

            for item in items:
                if isinstance(item, dict) and "description" in item:
                    findings.append(ResearchFinding(
                        category="incident",
                        description=item["description"],
                        source_type="incident",
                        implied_variables=item.get("variables", []),
                        priority=item.get("priority", "medium")
                    ))

            return findings

        except Exception as e:
            # Return empty on error - research is supplementary
            return []

    def _research_regulatory(
        self,
        domain: str,
        description: str
    ) -> list[ResearchFinding]:
        """Research regulatory and compliance requirements.

        Args:
            domain: The domain to research.
            description: Agent description for context.

        Returns:
            List of findings about regulatory requirements.
        """
        prompt = f"""You are a compliance expert reviewing test requirements for a voice agent.

Domain: {domain}
Agent Description: {description}

Identify regulatory and compliance requirements that affect testing:
1. What legal requirements apply to this domain?
2. What data protection rules must be followed?
3. What consent requirements exist?
4. What documentation/audit requirements apply?

For each requirement, specify:
- The requirement description
- What variables/states need testing
- Priority based on legal risk

Format as JSON array:
{{"description": "...", "variables": [{{"name": "...", "type": "boolean|enum", "states": [...]}}], "priority": "..."}}

Provide 2-4 findings focusing on commonly overlooked compliance issues."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            import json
            content = response.choices[0].message.content
            data = json.loads(content)

            findings = []
            items = data.get("findings", data) if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = [items]

            for item in items:
                if isinstance(item, dict) and "description" in item:
                    findings.append(ResearchFinding(
                        category="regulatory",
                        description=item["description"],
                        source_type="regulatory",
                        implied_variables=item.get("variables", []),
                        priority=item.get("priority", "high")
                    ))

            return findings

        except Exception:
            return []

    def _research_accessibility(
        self,
        domain: str,
        description: str
    ) -> list[ResearchFinding]:
        """Research accessibility requirements and considerations.

        Args:
            domain: The domain to research.
            description: Agent description for context.

        Returns:
            List of findings about accessibility.
        """
        prompt = f"""You are an accessibility expert reviewing a voice agent for inclusive design.

Domain: {domain}
Agent Description: {description}

Identify accessibility scenarios that need testing:
1. Users with hearing impairments
2. Users with speech impairments
3. Users with cognitive impairments
4. Users with motor impairments
5. Non-native speakers
6. Users in noisy environments
7. Users using assistive technology

For each scenario:
- Describe the accessibility challenge
- What variables need to be tested
- Priority for inclusive design

Format as JSON array:
{{"description": "...", "variables": [{{"name": "...", "type": "boolean|enum", "states": [...]}}], "priority": "..."}}

Provide 3-5 accessibility findings."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            import json
            content = response.choices[0].message.content
            data = json.loads(content)

            findings = []
            items = data.get("findings", data) if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = [items]

            for item in items:
                if isinstance(item, dict) and "description" in item:
                    findings.append(ResearchFinding(
                        category="accessibility",
                        description=item["description"],
                        source_type="accessibility",
                        implied_variables=item.get("variables", []),
                        priority=item.get("priority", "high")
                    ))

            return findings

        except Exception:
            return []


def incorporate_research(
    variables: VariableSpace,
    findings: list[ResearchFinding]
) -> VariableSpace:
    """Incorporate research findings into the variable space.

    Args:
        variables: Existing variable space.
        findings: Research findings to incorporate.

    Returns:
        Updated variable space with new variables.
    """
    for finding in findings:
        for var_spec in finding.implied_variables:
            name = var_spec.get("name", "")
            if not name or variables.get(name):
                continue

            var_type = var_spec.get("type", "boolean")
            states = var_spec.get("states", [])

            is_high_priority = finding.priority in ("high", "critical")

            if var_type == "boolean":
                variables.add(BooleanVariable(
                    name=name,
                    description=f"From research: {finding.description[:100]}",
                    category=f"research_{finding.category}",
                    is_edge_case_priority=is_high_priority
                ))
            elif var_type == "enum" and states:
                variables.add(EnumVariable(
                    name=name,
                    description=f"From research: {finding.description[:100]}",
                    category=f"research_{finding.category}",
                    states=states,
                    edge_case_states=states[-1:] if states else [],
                    is_edge_case_priority=is_high_priority
                ))

    return variables


def deep_research(
    domain: str,
    description: str,
    api_key: Optional[str] = None
) -> list[ResearchFinding]:
    """Conduct deep research on the domain.

    Convenience function that creates an agent and runs research.

    Args:
        domain: The detected domain.
        description: Agent description.
        api_key: Optional xAI API key.

    Returns:
        List of research findings.
    """
    agent = ResearchAgent(api_key=api_key)
    return agent.research(domain, description)
