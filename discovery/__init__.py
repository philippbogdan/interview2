"""Discovery module for variable extraction and expansion."""

from .variable_extractor import extract_variables, VariableExtractor
from .research_agent import deep_research, ResearchFinding, ResearchAgent
from .ralph_loop import ralph_loop, RalphLoop

__all__ = [
    "extract_variables",
    "VariableExtractor",
    "deep_research",
    "ResearchFinding",
    "ResearchAgent",
    "ralph_loop",
    "RalphLoop",
]
