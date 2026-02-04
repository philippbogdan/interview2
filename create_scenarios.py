"""
Voice Agent Test Scenario Generator

This module provides functionality to automatically generate test cases for voice agents
using an LLM-based approach with OpenAI's Structured Outputs.

Example usage:
    from create_scenarios import create_scenarios

    # With a JSON configuration
    config = {
        "description": "A clinic appointment scheduling assistant",
        "actions": ["schedule_appointment", "cancel_appointment", "check_availability"],
    }
    scenarios = create_scenarios(config, num_scenarios=3)

    # With a plain text description
    scenarios = create_scenarios(
        "A drive-through order taking assistant for a fast food restaurant",
        num_scenarios=5
    )
"""

from typing import Union

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from parsers.input_parser import parse_input
from parsers.domain_analyzer import analyze_domain
from generators.scenario_generator import ScenarioGenerator


def create_scenarios(
    input: Union[str, dict],
    num_scenarios: int,
    api_key: str = None,
    model: str = "gpt-4o-2024-08-06",
) -> list[dict]:
    """
    Generate test scenarios for a voice agent.

    This function takes an agent configuration (as JSON dict, JSON string, or plain text)
    and generates realistic test scenarios using an LLM-based approach. The function
    automatically detects the agent's domain (healthcare, restaurant, etc.) and includes
    domain-specific fields in the generated scenarios.

    Args:
        input: Agent configuration. Can be:
            - dict: JSON configuration with description, actions, states, etc.
            - str: Either a JSON string or plain text description
        num_scenarios: Number of test scenarios to generate
        api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        model: OpenAI model to use. Must support structured outputs (default: gpt-4o-2024-08-06)

    Returns:
        List of scenario dictionaries, each containing:
            - scenarioName: Name of the test scenario
            - scenarioDescription: Detailed description of what the scenario tests
            - criteria: List of success criteria
            - Domain-specific fields (e.g., name, dob for healthcare; order_items for restaurant)

    Raises:
        ValueError: If input format is not supported or API key is missing

    Example:
        >>> scenarios = create_scenarios(
        ...     {"description": "Clinic appointment scheduler", "actions": ["book", "cancel"]},
        ...     num_scenarios=3
        ... )
        >>> len(scenarios)
        3
        >>> "scenarioName" in scenarios[0]
        True
    """
    # Step 1: Parse and normalize the input
    config = parse_input(input)

    # Step 2: Analyze the domain
    domain_analysis = analyze_domain(config)

    # Step 3: Generate scenarios using the LLM
    generator = ScenarioGenerator(api_key=api_key, model=model)
    scenarios = generator.generate(config, domain_analysis, num_scenarios)

    return scenarios


# For convenience, allow running as a script
if __name__ == "__main__":
    import json
    import sys

    # Example: Run with a sample configuration
    sample_config = {
        "description": """You are a friendly clinic receptionist assistant.
        Help patients schedule, reschedule, or cancel appointments.
        Collect patient information including name, date of birth, phone number,
        and insurance details. Check doctor availability and confirm appointments.""",
        "actions": [
            "schedule_appointment",
            "reschedule_appointment",
            "cancel_appointment",
            "check_availability",
            "collect_patient_info",
        ],
        "entities": ["patient_name", "date_of_birth", "phone_number", "insurance", "appointment_date", "doctor_name"],
    }

    num = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print(f"Generating {num} scenarios for clinic assistant...")
    scenarios = create_scenarios(sample_config, num)

    print(json.dumps(scenarios, indent=2))
