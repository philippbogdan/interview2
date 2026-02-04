"""Algorithmic variable extraction from agent configuration.

This module extracts test variables from the parsed agent configuration
using pattern matching and heuristic analysis.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from parsers.input_parser import NormalizedAgentConfig
from parsers.domain_analyzer import DomainAnalysis
from models.variables import (
    VariableSpace,
    VariableDefinition,
    BooleanVariable,
    QuantitativeVariable,
    EnumVariable,
    TriStateBoolean,
)


@dataclass
class ExtractionContext:
    """Context for variable extraction."""
    config: NormalizedAgentConfig
    domain_analysis: DomainAnalysis
    extracted_patterns: dict[str, list[str]] = field(default_factory=dict)


class VariableExtractor:
    """Extracts test variables from agent configuration."""

    # Patterns for identifying variable types from text
    AVAILABILITY_PATTERNS = [
        r"(?:provide|collect|gather|get|ask for|request)\s+(\w+(?:\s+\w+)?)",
        r"(\w+)\s+(?:is|are)\s+(?:available|provided|given|present)",
        r"(?:check|verify|validate)\s+(\w+(?:\s+\w+)?)",
    ]

    SEARCH_ACTION_PATTERNS = [
        r"(?:search|find|look up|lookup|query)\s+(?:for\s+)?(\w+(?:\s+\w+)?)",
        r"(\w+)\s+(?:search|lookup|query)",
    ]

    OUTCOME_PATTERNS = [
        r"(?:result|outcome|response|status)\s+(?:of|from)\s+(\w+)",
        r"(\w+)\s+(?:result|outcome|response|status)",
    ]

    ACCESSIBILITY_KEYWORDS = [
        "speak", "hear", "see", "read", "write", "walk", "stand",
        "mobility", "vision", "hearing", "speech", "cognitive",
        "disability", "impairment", "assistance", "accessible",
    ]

    SYSTEM_STATE_KEYWORDS = [
        "available", "unavailable", "online", "offline", "busy",
        "maintenance", "capacity", "stock", "inventory", "supply",
        "staffing", "schedule", "hours", "open", "closed",
    ]

    def __init__(self):
        """Initialize the variable extractor."""
        pass

    def extract(
        self,
        config: NormalizedAgentConfig,
        domain_analysis: DomainAnalysis
    ) -> VariableSpace:
        """Extract variables from agent configuration.

        Args:
            config: Normalized agent configuration.
            domain_analysis: Domain analysis results.

        Returns:
            VariableSpace with extracted variables.
        """
        context = ExtractionContext(config=config, domain_analysis=domain_analysis)
        space = VariableSpace()

        # Extract from entities
        self._extract_from_entities(context, space)

        # Extract from actions
        self._extract_from_actions(context, space)

        # Extract from description patterns
        self._extract_from_description(context, space)

        # Add domain-specific standard variables
        self._add_domain_variables(context, space)

        return space

    def _extract_from_entities(
        self,
        context: ExtractionContext,
        space: VariableSpace
    ) -> None:
        """Extract variables from entity definitions.

        Each entity typically maps to an availability boolean variable.

        Args:
            context: Extraction context.
            space: Variable space to populate.
        """
        for entity in context.config.entities:
            name = self._normalize_name(entity)

            # Create availability variable for each entity
            space.add(BooleanVariable(
                name=f"{name}_available",
                description=f"Whether {entity} is available/provided",
                category="entity_availability",
                is_edge_case_priority=True
            ))

            # If entity looks like a searchable field, add search result variable
            if self._is_searchable_entity(entity):
                space.add(QuantitativeVariable(
                    name=f"{name}_search_results",
                    description=f"Number of matches when searching by {entity}",
                    category="search",
                    single_variants=["correct", "incorrect"],
                    depends_on=f"{name}_available",
                    relevant_when=lambda a, n=f"{name}_available": (
                        a.get(n) == TriStateBoolean.TRUE
                    )
                ))

    def _extract_from_actions(
        self,
        context: ExtractionContext,
        space: VariableSpace
    ) -> None:
        """Extract variables from action definitions.

        Actions can imply outcome variables (success/failure states).

        Args:
            context: Extraction context.
            space: Variable space to populate.
        """
        for action in context.config.actions:
            name = self._normalize_name(action)

            # Actions with search/lookup imply result quantity
            if self._is_search_action(action):
                if not space.get(f"{name}_results"):
                    space.add(QuantitativeVariable(
                        name=f"{name}_results",
                        description=f"Results from {action} action",
                        category="action_outcomes",
                        single_variants=["match", "partial_match", "no_match"]
                    ))

            # Actions can have availability (system can perform action)
            space.add(BooleanVariable(
                name=f"{name}_possible",
                description=f"Whether {action} action is possible in current state",
                category="action_availability"
            ))

    def _extract_from_description(
        self,
        context: ExtractionContext,
        space: VariableSpace
    ) -> None:
        """Extract variables from description text using pattern matching.

        Args:
            context: Extraction context.
            space: Variable space to populate.
        """
        text = context.config.description.lower()

        # Look for accessibility-related mentions
        for keyword in self.ACCESSIBILITY_KEYWORDS:
            if keyword in text:
                if not space.get(f"can_{keyword}"):
                    space.add(BooleanVariable(
                        name=f"can_{keyword}",
                        description=f"User accessibility: can {keyword}",
                        category="accessibility",
                        is_edge_case_priority=True
                    ))

        # Look for system state mentions
        for keyword in self.SYSTEM_STATE_KEYWORDS:
            if keyword in text:
                # Extract context around the keyword
                match = re.search(rf"(\w+)\s+{keyword}", text)
                if match:
                    subject = match.group(1)
                    var_name = f"{subject}_{keyword}"
                    if not space.get(var_name):
                        space.add(BooleanVariable(
                            name=var_name,
                            description=f"Whether {subject} is {keyword}",
                            category="system_state",
                            is_edge_case_priority=True
                        ))

    def _add_domain_variables(
        self,
        context: ExtractionContext,
        space: VariableSpace
    ) -> None:
        """Add standard variables based on detected domain.

        Args:
            context: Extraction context.
            space: Variable space to populate.
        """
        domain = context.domain_analysis.detected_domain

        if domain == "healthcare":
            self._add_healthcare_variables(space)
        elif domain == "restaurant":
            self._add_restaurant_variables(space)
        elif domain == "customer_service":
            self._add_customer_service_variables(space)

    def _add_healthcare_variables(self, space: VariableSpace) -> None:
        """Add standard healthcare domain variables."""
        # Only add if not already present

        if not space.get("patient_urgency"):
            space.add(EnumVariable(
                name="patient_urgency",
                description="Urgency of patient's condition",
                category="patient_state",
                states=["routine", "urgent", "emergency"],
                edge_case_states=["emergency"]
            ))

        if not space.get("patient_age_category"):
            space.add(EnumVariable(
                name="patient_age_category",
                description="Patient's age category",
                category="patient_state",
                states=["minor", "adult", "elderly"],
                edge_case_states=["minor"]
            ))

        if not space.get("insurance_status"):
            space.add(EnumVariable(
                name="insurance_status",
                description="Patient's insurance status",
                category="patient_info",
                states=["none", "expired", "active", "pending"],
                edge_case_states=["none", "expired"]
            ))

    def _add_restaurant_variables(self, space: VariableSpace) -> None:
        """Add standard restaurant domain variables."""

        if not space.get("item_availability"):
            space.add(EnumVariable(
                name="item_availability",
                description="Availability of ordered item",
                category="inventory",
                states=["available", "limited", "out_of_stock"],
                edge_case_states=["out_of_stock"]
            ))

        if not space.get("order_complexity"):
            space.add(EnumVariable(
                name="order_complexity",
                description="Complexity of the order",
                category="order",
                states=["simple", "moderate", "complex"],
                edge_case_states=["complex"]
            ))

        if not space.get("payment_status"):
            space.add(EnumVariable(
                name="payment_status",
                description="Payment processing status",
                category="payment",
                states=["success", "declined", "pending", "error"],
                edge_case_states=["declined", "error"]
            ))

    def _add_customer_service_variables(self, space: VariableSpace) -> None:
        """Add standard customer service domain variables."""

        if not space.get("customer_sentiment"):
            space.add(EnumVariable(
                name="customer_sentiment",
                description="Customer's emotional state",
                category="customer",
                states=["satisfied", "neutral", "frustrated", "angry"],
                edge_case_states=["frustrated", "angry"]
            ))

        if not space.get("issue_complexity"):
            space.add(EnumVariable(
                name="issue_complexity",
                description="Complexity of the customer's issue",
                category="issue",
                states=["simple", "moderate", "complex", "requires_escalation"],
                edge_case_states=["complex", "requires_escalation"]
            ))

        if not space.get("customer_history"):
            space.add(EnumVariable(
                name="customer_history",
                description="Customer's history with the company",
                category="customer",
                states=["new", "returning", "vip", "problematic"],
                edge_case_states=["vip", "problematic"]
            ))

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for use as a variable name.

        Args:
            name: The name to normalize.

        Returns:
            Normalized variable name.
        """
        # Convert to lowercase, replace spaces/special chars with underscore
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
        # Remove leading/trailing underscores
        normalized = normalized.strip("_")
        return normalized

    def _is_searchable_entity(self, entity: str) -> bool:
        """Check if an entity is typically searchable.

        Args:
            entity: Entity name.

        Returns:
            True if entity is searchable.
        """
        searchable_keywords = [
            "name", "id", "number", "email", "phone", "account",
            "patient", "customer", "user", "order", "reference"
        ]
        entity_lower = entity.lower()
        return any(keyword in entity_lower for keyword in searchable_keywords)

    def _is_search_action(self, action: str) -> bool:
        """Check if an action is a search-type action.

        Args:
            action: Action name.

        Returns:
            True if action is a search action.
        """
        search_keywords = [
            "search", "find", "look", "query", "fetch", "get",
            "retrieve", "check", "verify", "lookup"
        ]
        action_lower = action.lower()
        return any(keyword in action_lower for keyword in search_keywords)


def extract_variables(
    config: NormalizedAgentConfig,
    domain_analysis: DomainAnalysis
) -> VariableSpace:
    """Extract variables from agent configuration.

    Convenience function that creates an extractor and runs extraction.

    Args:
        config: Normalized agent configuration.
        domain_analysis: Domain analysis results.

    Returns:
        VariableSpace with extracted variables.
    """
    extractor = VariableExtractor()
    return extractor.extract(config, domain_analysis)
