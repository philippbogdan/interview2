"""Input parser for normalizing various input formats."""

import json
from typing import Union, Any
from dataclasses import dataclass, field


@dataclass
class NormalizedAgentConfig:
    """Unified configuration representation for voice agents."""

    raw_input: Union[str, dict]
    description: str = ""
    actions: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_structured(self) -> bool:
        """Check if the config has structured data."""
        return bool(self.actions or self.states or self.transitions)


def parse_input(input_data: Union[str, dict]) -> NormalizedAgentConfig:
    """
    Parse and normalize input from various formats.

    Supports:
    - JSON dict (already parsed)
    - JSON string
    - Plain text description

    Args:
        input_data: The input to parse (dict, JSON string, or plain text)

    Returns:
        NormalizedAgentConfig with extracted information
    """
    if isinstance(input_data, dict):
        return _parse_dict(input_data)

    if isinstance(input_data, str):
        # Try to parse as JSON first
        try:
            parsed = json.loads(input_data)
            if isinstance(parsed, dict):
                return _parse_dict(parsed)
        except json.JSONDecodeError:
            pass

        # Treat as plain text description
        return _parse_text(input_data)

    raise ValueError(f"Unsupported input type: {type(input_data)}")


def _parse_dict(data: dict) -> NormalizedAgentConfig:
    """Parse a dictionary configuration."""
    config = NormalizedAgentConfig(raw_input=data)

    # Extract description from various possible keys
    description_keys = ["description", "prompt", "system_prompt", "instructions", "agent_description"]
    for key in description_keys:
        if key in data and data[key]:
            config.description = str(data[key])
            break

    # Extract actions
    action_keys = ["actions", "intents", "capabilities", "functions"]
    for key in action_keys:
        if key in data and isinstance(data[key], list):
            config.actions = [str(a) if isinstance(a, str) else a.get("name", str(a)) for a in data[key]]
            break

    # Extract states
    state_keys = ["states", "nodes", "steps"]
    for key in state_keys:
        if key in data and isinstance(data[key], list):
            config.states = [str(s) if isinstance(s, str) else s.get("name", str(s)) for s in data[key]]
            break

    # Extract transitions
    transition_keys = ["transitions", "edges", "flows"]
    for key in transition_keys:
        if key in data and isinstance(data[key], list):
            config.transitions = data[key]
            break

    # Extract entities
    entity_keys = ["entities", "slots", "variables", "fields"]
    for key in entity_keys:
        if key in data and isinstance(data[key], list):
            config.entities = [str(e) if isinstance(e, str) else e.get("name", str(e)) for e in data[key]]
            break

    # Store any additional metadata
    known_keys = set(description_keys + action_keys + state_keys + transition_keys + entity_keys)
    config.metadata = {k: v for k, v in data.items() if k not in known_keys}

    return config


def _parse_text(text: str) -> NormalizedAgentConfig:
    """Parse a plain text description."""
    config = NormalizedAgentConfig(
        raw_input=text,
        description=text.strip()
    )

    # Try to extract some structure from the text
    text_lower = text.lower()

    # Look for action-like phrases
    action_indicators = ["can ", "will ", "should ", "able to ", "handles "]
    for indicator in action_indicators:
        if indicator in text_lower:
            # Simple extraction - could be made more sophisticated
            pass

    return config
