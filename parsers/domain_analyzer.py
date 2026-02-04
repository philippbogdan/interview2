"""Domain analyzer for detecting agent domain and extracting relevant information."""

from dataclasses import dataclass, field
from typing import Optional, Type
from pydantic import BaseModel

from .input_parser import NormalizedAgentConfig
from models.base import DOMAIN_REGISTRY, BaseScenario


@dataclass
class DomainAnalysis:
    """Results of domain analysis."""

    detected_domain: str
    confidence: float
    field_model: Optional[Type[BaseModel]]
    extracted_entities: list[str] = field(default_factory=list)
    suggested_edge_cases: list[str] = field(default_factory=list)
    keywords_found: list[str] = field(default_factory=list)


def analyze_domain(config: NormalizedAgentConfig) -> DomainAnalysis:
    """
    Analyze the agent configuration to detect its domain.

    Args:
        config: Normalized agent configuration

    Returns:
        DomainAnalysis with detected domain and suggested field model
    """
    # Combine all text for analysis
    text_to_analyze = _get_analyzable_text(config)
    text_lower = text_to_analyze.lower()

    # Count keyword matches for each domain
    domain_scores: dict[str, tuple[float, list[str]]] = {}

    for keyword, model in DOMAIN_REGISTRY.items():
        domain_name = _get_domain_name(model)

        if keyword in text_lower:
            if domain_name not in domain_scores:
                domain_scores[domain_name] = (0.0, [])

            score, keywords = domain_scores[domain_name]
            # Weight multi-word keywords higher
            weight = 1.5 if " " in keyword or "-" in keyword else 1.0
            domain_scores[domain_name] = (score + weight, keywords + [keyword])

    # Find the best matching domain
    if domain_scores:
        best_domain = max(domain_scores.items(), key=lambda x: x[1][0])
        domain_name = best_domain[0]
        score, keywords = best_domain[1]

        # Calculate confidence based on score
        confidence = min(score / 3.0, 1.0)  # Normalize to 0-1

        # Get the field model for this domain
        field_model = _get_field_model_for_domain(domain_name)

        return DomainAnalysis(
            detected_domain=domain_name,
            confidence=confidence,
            field_model=field_model,
            extracted_entities=_extract_entities(config),
            suggested_edge_cases=_suggest_edge_cases(domain_name),
            keywords_found=keywords,
        )

    # Default to general domain
    return DomainAnalysis(
        detected_domain="general",
        confidence=0.5,
        field_model=None,
        extracted_entities=_extract_entities(config),
        suggested_edge_cases=_suggest_edge_cases("general"),
        keywords_found=[],
    )


def _get_analyzable_text(config: NormalizedAgentConfig) -> str:
    """Combine all relevant text from the config for analysis."""
    parts = [config.description]

    if config.actions:
        parts.extend(config.actions)

    if config.states:
        parts.extend(config.states)

    if config.entities:
        parts.extend(config.entities)

    # Include raw input if it's a string
    if isinstance(config.raw_input, str):
        parts.append(config.raw_input)
    elif isinstance(config.raw_input, dict):
        # Recursively extract string values
        parts.extend(_extract_strings_from_dict(config.raw_input))

    return " ".join(parts)


def _extract_strings_from_dict(d: dict, max_depth: int = 3) -> list[str]:
    """Extract all string values from a nested dictionary."""
    if max_depth <= 0:
        return []

    strings = []
    for value in d.values():
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            strings.extend(_extract_strings_from_dict(value, max_depth - 1))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    strings.append(item)
                elif isinstance(item, dict):
                    strings.extend(_extract_strings_from_dict(item, max_depth - 1))

    return strings


def _get_domain_name(model: Type[BaseModel]) -> str:
    """Get a canonical domain name from a field model."""
    name = model.__name__
    if "Clinic" in name or "Patient" in name:
        return "healthcare"
    elif "DriveThru" in name or "Order" in name:
        return "restaurant"
    elif "CustomerService" in name:
        return "customer_service"
    return "general"


def _get_field_model_for_domain(domain: str) -> Optional[Type[BaseModel]]:
    """Get the field model for a given domain name."""
    from models.base import ClinicPatientFields, DriveThruOrderFields, CustomerServiceFields

    domain_to_model = {
        "healthcare": ClinicPatientFields,
        "restaurant": DriveThruOrderFields,
        "customer_service": CustomerServiceFields,
    }
    return domain_to_model.get(domain)


def _extract_entities(config: NormalizedAgentConfig) -> list[str]:
    """Extract entities from the configuration."""
    entities = list(config.entities) if config.entities else []

    # Try to extract additional entities from description
    # This is a simple implementation - could use NLP for better extraction
    return entities


def _suggest_edge_cases(domain: str) -> list[str]:
    """Suggest edge cases based on the domain."""
    edge_cases = {
        "healthcare": [
            "Patient provides invalid date of birth",
            "Insurance information is missing",
            "Requested appointment time is unavailable",
            "Patient needs to cancel or reschedule",
            "Multiple patients with similar names",
            "Patient is a minor requiring guardian consent",
        ],
        "restaurant": [
            "Item is out of stock",
            "Complex customization request",
            "Order modification after confirmation",
            "Payment declined",
            "Large group order",
            "Dietary restriction inquiry",
        ],
        "customer_service": [
            "Customer is extremely frustrated",
            "Issue requires escalation",
            "Customer provides incomplete information",
            "Multiple issues in one call",
            "Request outside agent capabilities",
            "Language barrier",
        ],
        "general": [
            "User provides incomplete information",
            "Request is ambiguous",
            "User changes mind mid-conversation",
            "Technical issues during interaction",
            "User requests something outside scope",
        ],
    }
    return edge_cases.get(domain, edge_cases["general"])
