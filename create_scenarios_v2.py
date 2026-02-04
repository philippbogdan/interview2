"""
Combinatorial Voice Agent Test Scenario Generator (v2)

This module provides systematic combinatorial exploration for generating test scenarios,
rather than relying solely on ad-hoc LLM generation. It explores a multi-dimensional
variable space to ensure comprehensive coverage of edge cases and boundary conditions.

Example usage:
    from create_scenarios_v2 import create_scenarios_combinatorial

    # With a JSON configuration
    config = {
        "description": "A clinic appointment scheduling assistant",
        "actions": ["schedule_appointment", "cancel_appointment", "check_availability"],
    }

    # Returns a generator that yields scenarios
    for scenario in create_scenarios_combinatorial(config, max_scenarios=20):
        print(scenario["scenarioName"])

    # Get all scenarios as a list
    scenarios = list(create_scenarios_combinatorial(config, max_scenarios=20))
"""

from typing import Any, Generator, Optional, Union

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from parsers.input_parser import parse_input, NormalizedAgentConfig
from parsers.domain_analyzer import analyze_domain, DomainAnalysis
from discovery.variable_extractor import extract_variables
from discovery.research_agent import deep_research, incorporate_research
from discovery.ralph_loop import ralph_loop, RalphLoop
from exploration.dependency_graph import build_dependency_graph, DependencyGraph
from exploration.coverage_tracker import CoverageStats
from exploration.space_explorer import SpaceExplorer, ExplorationConfig
from generators.scenario_constructor import ScenarioConstructor
from models.variables import VariableSpace


class CombinatorialScenarioGenerator:
    """Generates test scenarios through systematic combinatorial exploration.

    This generator:
    1. Extracts variables from the agent configuration
    2. Optionally expands variables via research and Ralph Loop
    3. Builds a dependency graph
    4. Systematically explores the variable space
    5. Constructs coherent scenarios from assignments
    """

    def __init__(
        self,
        config: NormalizedAgentConfig,
        domain_analysis: DomainAnalysis,
        api_key: Optional[str] = None,
        model: str = "grok-4-1-fast-non-reasoning",
        ralph_iterations: int = 3,
        enable_research: bool = True,
        enable_ralph_loop: bool = True,
    ):
        """Initialize the combinatorial generator.

        Args:
            config: Normalized agent configuration.
            domain_analysis: Domain analysis results.
            api_key: xAI API key.
            model: xAI model to use for LLM operations.
            ralph_iterations: Number of Ralph Loop iterations.
            enable_research: Whether to run deep research.
            enable_ralph_loop: Whether to run Ralph Loop expansion.
        """
        self.config = config
        self.domain_analysis = domain_analysis
        self.api_key = api_key
        self.model = model
        self.ralph_iterations = ralph_iterations
        self.enable_research = enable_research
        self.enable_ralph_loop = enable_ralph_loop

        self._variables: Optional[VariableSpace] = None
        self._graph: Optional[DependencyGraph] = None
        self._explorer: Optional[SpaceExplorer] = None
        self._constructor: Optional[ScenarioConstructor] = None
        self._coverage_stats: Optional[CoverageStats] = None

    def _initialize(self) -> None:
        """Initialize the discovery pipeline and exploration components."""
        # Step 1: Extract variables from configuration
        self._variables = extract_variables(self.config, self.domain_analysis)

        # Step 2: Deep research (optional)
        if self.enable_research and self.api_key:
            try:
                findings = deep_research(
                    self.domain_analysis.detected_domain,
                    self.config.description,
                    api_key=self.api_key
                )
                self._variables = incorporate_research(self._variables, findings)
            except Exception:
                # Research is supplementary, continue without it
                pass

        # Step 3: Ralph Loop expansion (optional)
        if self.enable_ralph_loop and self.api_key and self.ralph_iterations > 0:
            try:
                loop = RalphLoop(api_key=self.api_key, model=self.model)
                self._variables, _ = loop.expand(
                    self._variables,
                    self.config.description,
                    iterations=self.ralph_iterations
                )
            except Exception:
                # Ralph Loop is supplementary, continue without it
                pass

        # Step 4: Build dependency graph
        self._graph = build_dependency_graph(self._variables)

        # Step 5: Initialize constructor
        self._constructor = ScenarioConstructor(
            self.config,
            self.domain_analysis,
            api_key=self.api_key,
            model=self.model
        )

    def generate(
        self,
        max_scenarios: Optional[int] = None,
        exploration_config: Optional[ExplorationConfig] = None
    ) -> Generator[dict[str, Any], None, None]:
        """Generate test scenarios.

        Args:
            max_scenarios: Maximum number of scenarios to generate.
            exploration_config: Optional exploration configuration.

        Yields:
            Scenario dictionaries.
        """
        # Initialize if not done
        if self._variables is None:
            self._initialize()

        # Create explorer
        config = exploration_config or ExplorationConfig(max_scenarios=max_scenarios)
        if max_scenarios is not None:
            config.max_scenarios = max_scenarios

        self._explorer = SpaceExplorer(self._graph, config=config)

        # Explore and construct scenarios
        for assignment in self._explorer.explore():
            scenario = self._constructor.construct(assignment)
            yield scenario

        # Store final coverage stats
        self._coverage_stats = self._explorer.get_coverage_stats()

    def get_coverage_stats(self) -> Optional[CoverageStats]:
        """Get coverage statistics after generation.

        Returns:
            CoverageStats if generation has run, None otherwise.
        """
        if self._explorer:
            return self._explorer.get_coverage_stats()
        return self._coverage_stats

    def get_variables(self) -> Optional[VariableSpace]:
        """Get the discovered variable space.

        Returns:
            VariableSpace if initialized, None otherwise.
        """
        return self._variables

    def get_graph(self) -> Optional[DependencyGraph]:
        """Get the dependency graph.

        Returns:
            DependencyGraph if initialized, None otherwise.
        """
        return self._graph


def create_scenarios_combinatorial(
    input: Union[str, dict],
    max_scenarios: Optional[int] = None,
    ralph_iterations: int = 3,
    api_key: Optional[str] = None,
    model: str = "grok-4-1-fast-non-reasoning",
    enable_research: bool = True,
    enable_ralph_loop: bool = True,
) -> Generator[dict, None, None]:
    """
    Generate test scenarios using combinatorial exploration.

    This function takes an agent configuration and systematically explores
    a multi-dimensional variable space to generate comprehensive test scenarios.

    Args:
        input: Agent configuration. Can be:
            - dict: JSON configuration with description, actions, states, etc.
            - str: Either a JSON string or plain text description
        max_scenarios: Maximum number of scenarios to generate. If None,
            generates until the variable space is fully explored.
        ralph_iterations: Number of Ralph Loop iterations for variable expansion.
            Set to 0 to disable Ralph Loop.
        api_key: xAI API key. If not provided, reads from XAI_API_KEY env var.
        model: xAI model to use for LLM operations.
        enable_research: Whether to run deep research for edge case discovery.
        enable_ralph_loop: Whether to run Ralph Loop for variable expansion.

    Yields:
        Scenario dictionaries, each containing:
            - scenarioName: Name of the test scenario
            - scenarioDescription: Detailed description
            - criteria: List of success criteria
            - _variable_assignment: The underlying variable values
            - Domain-specific fields as appropriate

    Example:
        >>> scenarios = list(create_scenarios_combinatorial(
        ...     {"description": "Clinic scheduler", "actions": ["book", "cancel"]},
        ...     max_scenarios=10
        ... ))
        >>> len(scenarios) <= 10
        True
        >>> "_variable_assignment" in scenarios[0]
        True
    """
    # Step 1: Parse and normalize the input
    config = parse_input(input)

    # Step 2: Analyze the domain
    domain_analysis = analyze_domain(config)

    # Step 3: Create and run the generator
    generator = CombinatorialScenarioGenerator(
        config=config,
        domain_analysis=domain_analysis,
        api_key=api_key,
        model=model,
        ralph_iterations=ralph_iterations,
        enable_research=enable_research,
        enable_ralph_loop=enable_ralph_loop,
    )

    yield from generator.generate(max_scenarios=max_scenarios)


# Convenience function to get scenarios as a list with stats
def create_scenarios_with_stats(
    input: Union[str, dict],
    max_scenarios: Optional[int] = None,
    ralph_iterations: int = 3,
    api_key: Optional[str] = None,
    **kwargs
) -> tuple[list[dict], dict]:
    """
    Generate scenarios and return with coverage statistics.

    Args:
        input: Agent configuration.
        max_scenarios: Maximum scenarios to generate.
        ralph_iterations: Ralph Loop iterations.
        api_key: xAI API key.
        **kwargs: Additional arguments passed to CombinatorialScenarioGenerator.

    Returns:
        Tuple of (list of scenarios, coverage stats dict).
    """
    config = parse_input(input)
    domain_analysis = analyze_domain(config)

    generator = CombinatorialScenarioGenerator(
        config=config,
        domain_analysis=domain_analysis,
        api_key=api_key,
        ralph_iterations=ralph_iterations,
        **kwargs
    )

    scenarios = list(generator.generate(max_scenarios=max_scenarios))
    stats = generator.get_coverage_stats()

    return scenarios, stats.to_dict() if stats else {}


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
        "entities": [
            "patient_name",
            "date_of_birth",
            "phone_number",
            "insurance",
            "appointment_date",
            "doctor_name"
        ],
    }

    max_scenarios = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ralph_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # Default to 0 for quick testing

    print(f"Generating up to {max_scenarios} scenarios (combinatorial exploration)...")
    print(f"Ralph Loop iterations: {ralph_iterations}")
    print()

    scenarios, stats = create_scenarios_with_stats(
        sample_config,
        max_scenarios=max_scenarios,
        ralph_iterations=ralph_iterations,
        enable_research=False,  # Disable for quick testing
        enable_ralph_loop=ralph_iterations > 0,
    )

    print(f"Generated {len(scenarios)} scenarios")
    print(f"\nCoverage stats: {json.dumps(stats, indent=2)}")
    print("\nScenarios:")
    print(json.dumps(scenarios, indent=2))
