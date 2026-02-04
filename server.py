"""FastAPI backend for the Combinatorial Test Scenario Visualizer.

Uses the full CombinatorialScenarioGenerator pipeline including
research and Ralph Loop for rich, concrete scenario generation.
"""

import os
import uuid
from typing import Any, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parsers.input_parser import parse_input
from parsers.domain_analyzer import analyze_domain
from create_scenarios_v2 import CombinatorialScenarioGenerator
from models.variables import VariableSpace


# Session storage
sessions: dict[str, "ExplorationSession"] = {}


class ExplorationSession:
    """Holds state for an active exploration session."""

    def __init__(self, generator: CombinatorialScenarioGenerator):
        self.generator = generator
        self._scenario_generator = None
        self.scenarios: list[dict] = []
        self.initialized = False

    def initialize(self, max_scenarios: int = 100):
        """Initialize the generator and start exploration."""
        self._scenario_generator = self.generator.generate(max_scenarios=max_scenarios)
        self.initialized = True

    def next_scenario(self) -> Optional[dict]:
        """Generate the next scenario."""
        if not self.initialized:
            return None
        try:
            scenario = next(self._scenario_generator)
            self.scenarios.append(scenario)
            return scenario
        except StopIteration:
            return None

    @property
    def variables(self) -> Optional[VariableSpace]:
        return self.generator.get_variables()

    @property
    def coverage_stats(self):
        return self.generator.get_coverage_stats()


# Request/Response Models

class InitRequest(BaseModel):
    config: dict[str, Any]
    max_scenarios: Optional[int] = 100
    enable_research: bool = False  # Disabled by default for speed
    enable_ralph_loop: bool = False  # Disabled by default for speed
    ralph_iterations: int = 2


class InitResponse(BaseModel):
    session_id: str
    variables: list[dict[str, Any]]
    domain: str
    total_combinations: int


class GenerateResponse(BaseModel):
    scenario: Optional[dict[str, Any]]
    coverage: dict[str, Any]
    scenario_count: int
    complete: bool


# FastAPI app

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    sessions.clear()


app = FastAPI(
    title="Combinatorial Test Scenario Visualizer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def state_to_str(s) -> str:
    """Convert a state value to a simple string."""
    if hasattr(s, 'value'):
        return s.value
    if isinstance(s, tuple):
        return f"{s[0]}:{s[1]}"
    return str(s)


def variable_to_dict(var) -> dict[str, Any]:
    """Convert a VariableDefinition to a dictionary."""
    return {
        "name": var.name,
        "description": var.description,
        "category": var.category,
        "states": [state_to_str(s) for s in var.get_all_states()],
        "edge_cases": [state_to_str(s) for s in var.get_edge_case_states()],
        "depends_on": var.depends_on,
        "is_edge_case_priority": var.is_edge_case_priority,
    }


def calculate_total_combinations(variable_space: VariableSpace) -> int:
    """Calculate total number of variable combinations."""
    total = 1
    for var in variable_space:
        total *= len(var.get_all_states())
    return total


@app.post("/init", response_model=InitResponse)
async def init_exploration(request: InitRequest) -> InitResponse:
    """Initialize a new exploration session with full pipeline."""
    try:
        # Parse and analyze
        config = parse_input(request.config)
        domain_analysis = analyze_domain(config)

        # Get API key from environment
        api_key = os.getenv("XAI_API_KEY")

        # Create the full generator
        generator = CombinatorialScenarioGenerator(
            config=config,
            domain_analysis=domain_analysis,
            api_key=api_key,
            model="grok-4-1-fast-non-reasoning",  # Same model as CLI
            ralph_iterations=request.ralph_iterations,
            enable_research=request.enable_research,
            enable_ralph_loop=request.enable_ralph_loop,
        )

        # Initialize (this runs variable extraction, research, ralph loop)
        generator._initialize()

        # Create session
        session_id = str(uuid.uuid4())
        session = ExplorationSession(generator)
        session.initialize(max_scenarios=request.max_scenarios)
        sessions[session_id] = session

        # Get variables
        variables = [variable_to_dict(var) for var in generator.get_variables()]

        return InitResponse(
            session_id=session_id,
            variables=variables,
            domain=domain_analysis.detected_domain,
            total_combinations=calculate_total_combinations(generator.get_variables()),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/generate/{session_id}", response_model=GenerateResponse)
async def generate_scenario(session_id: str) -> GenerateResponse:
    """Generate the next scenario."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    try:
        scenario = session.next_scenario()

        stats = session.coverage_stats
        coverage = stats.to_dict() if stats else {}

        return GenerateResponse(
            scenario=scenario,
            coverage=coverage,
            scenario_count=len(session.scenarios),
            complete=scenario is None,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check with API key status."""
    api_key = os.getenv("XAI_API_KEY")
    return {
        "status": "healthy",
        "api_key_configured": bool(api_key),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
