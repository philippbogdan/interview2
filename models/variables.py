"""Variable definitions for combinatorial test scenario generation.

This module defines the variable types and structures used to represent
the multi-dimensional variable space for systematic test exploration.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass, field


class TriStateBoolean(Enum):
    """Three-state boolean: no → unknown → yes."""
    FALSE = "no"
    UNKNOWN = "unknown"
    TRUE = "yes"

    @classmethod
    def all_states(cls) -> list["TriStateBoolean"]:
        """Return all states in order."""
        return [cls.FALSE, cls.UNKNOWN, cls.TRUE]

    @classmethod
    def edge_cases(cls) -> list["TriStateBoolean"]:
        """Return edge case states (False and Unknown are typically edge cases)."""
        return [cls.FALSE, cls.UNKNOWN]


class QuantitativeState(Enum):
    """Quantitative states: none → single → many."""
    NONE = "none"
    SINGLE = "single"
    MANY = "many"

    @classmethod
    def all_states(cls) -> list["QuantitativeState"]:
        """Return all states in order."""
        return [cls.NONE, cls.SINGLE, cls.MANY]

    @classmethod
    def edge_cases(cls) -> list["QuantitativeState"]:
        """Return edge case states."""
        return [cls.NONE, cls.MANY]


@dataclass
class VariableDefinition(ABC):
    """Base class for variable definitions in the exploration space."""

    name: str
    description: str
    category: str
    depends_on: Optional[str] = None
    relevant_when: Optional[Callable[["VariableAssignment"], bool]] = None
    is_edge_case_priority: bool = False

    @abstractmethod
    def get_all_states(self) -> list[Any]:
        """Return all possible states for this variable."""
        pass

    @abstractmethod
    def get_edge_case_states(self) -> list[Any]:
        """Return states that represent edge cases."""
        pass

    def is_relevant(self, assignment: "VariableAssignment") -> bool:
        """Check if this variable is relevant given the current assignment."""
        if self.relevant_when is None:
            return True
        return self.relevant_when(assignment)


@dataclass
class BooleanVariable(VariableDefinition):
    """A tri-state boolean variable (False/Unknown/True)."""

    def get_all_states(self) -> list[TriStateBoolean]:
        return TriStateBoolean.all_states()

    def get_edge_case_states(self) -> list[TriStateBoolean]:
        return TriStateBoolean.edge_cases()


@dataclass
class QuantitativeVariable(VariableDefinition):
    """A quantitative variable (none/single/many) with optional single variants."""

    single_variants: list[str] = field(default_factory=list)

    def get_all_states(self) -> list[Any]:
        """Return all states, expanding single with its variants."""
        states: list[Any] = [QuantitativeState.NONE]

        if self.single_variants:
            for variant in self.single_variants:
                states.append(("single", variant))
        else:
            states.append(QuantitativeState.SINGLE)

        states.append(QuantitativeState.MANY)
        return states

    def get_edge_case_states(self) -> list[Any]:
        """Return edge case states."""
        edge_cases: list[Any] = [QuantitativeState.NONE, QuantitativeState.MANY]

        # If there are variants, "incorrect" variants are edge cases
        if self.single_variants:
            for variant in self.single_variants:
                if variant.lower() in ("incorrect", "invalid", "error", "wrong"):
                    edge_cases.append(("single", variant))

        return edge_cases


@dataclass
class EnumVariable(VariableDefinition):
    """A variable with custom discrete states."""

    states: list[str] = field(default_factory=list)
    edge_case_states: list[str] = field(default_factory=list)

    def get_all_states(self) -> list[str]:
        return self.states.copy()

    def get_edge_case_states(self) -> list[str]:
        return self.edge_case_states.copy() if self.edge_case_states else []


@dataclass
class VariableAssignment:
    """Represents a specific assignment of values to variables."""

    assignments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        """Get the value assigned to a variable."""
        return self.assignments.get(name, default)

    def set(self, name: str, value: Any) -> "VariableAssignment":
        """Create a new assignment with the given variable set."""
        new_assignments = self.assignments.copy()
        new_assignments[name] = value
        return VariableAssignment(
            assignments=new_assignments,
            metadata=self.metadata.copy()
        )

    def copy(self) -> "VariableAssignment":
        """Create a copy of this assignment."""
        return VariableAssignment(
            assignments=self.assignments.copy(),
            metadata=self.metadata.copy()
        )

    def has(self, name: str) -> bool:
        """Check if a variable has been assigned."""
        return name in self.assignments

    def keys(self) -> list[str]:
        """Return all assigned variable names."""
        return list(self.assignments.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Return all variable-value pairs."""
        return list(self.assignments.items())

    def __hash__(self) -> int:
        """Make assignments hashable for tracking coverage."""
        items = tuple(sorted((k, str(v)) for k, v in self.assignments.items()))
        return hash(items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VariableAssignment):
            return False
        return self.assignments == other.assignments

    def __repr__(self) -> str:
        parts = [f"{k}={v}" for k, v in sorted(self.assignments.items())]
        return f"VariableAssignment({', '.join(parts)})"


@dataclass
class VariableSpace:
    """Collection of all variables defining the exploration space."""

    variables: list[VariableDefinition] = field(default_factory=list)

    def add(self, variable: VariableDefinition) -> None:
        """Add a variable to the space."""
        self.variables.append(variable)

    def get(self, name: str) -> Optional[VariableDefinition]:
        """Get a variable by name."""
        for var in self.variables:
            if var.name == name:
                return var
        return None

    def get_by_category(self, category: str) -> list[VariableDefinition]:
        """Get all variables in a category."""
        return [v for v in self.variables if v.category == category]

    def get_categories(self) -> list[str]:
        """Get all unique categories."""
        return list(set(v.category for v in self.variables))

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)


# Pre-defined healthcare domain variables for reference
def create_healthcare_variables() -> VariableSpace:
    """Create a standard set of healthcare domain variables."""
    space = VariableSpace()

    # Patient info variables
    space.add(BooleanVariable(
        name="name_available",
        description="Whether patient's name is available/provided",
        category="patient_info",
        is_edge_case_priority=True
    ))

    space.add(BooleanVariable(
        name="dob_available",
        description="Whether patient's date of birth is available/provided",
        category="patient_info",
        is_edge_case_priority=True
    ))

    space.add(EnumVariable(
        name="insurance_status",
        description="Patient's insurance status",
        category="patient_info",
        states=["none", "expired", "active", "pending"],
        edge_case_states=["none", "expired"]
    ))

    # Search/lookup variables
    space.add(QuantitativeVariable(
        name="search_results",
        description="Number of patient matches found in search",
        category="search",
        single_variants=["correct", "incorrect"],
        depends_on="name_available",
        relevant_when=lambda a: a.get("name_available") == TriStateBoolean.TRUE
    ))

    # Accessibility variables
    space.add(BooleanVariable(
        name="can_speak",
        description="Whether patient can speak",
        category="accessibility",
        is_edge_case_priority=True
    ))

    space.add(BooleanVariable(
        name="can_hear",
        description="Whether patient can hear",
        category="accessibility",
        is_edge_case_priority=True
    ))

    space.add(EnumVariable(
        name="mobility",
        description="Patient's mobility level",
        category="accessibility",
        states=["full", "limited", "wheelchair", "bedridden"],
        edge_case_states=["bedridden"]
    ))

    # System state variables
    space.add(BooleanVariable(
        name="equipment_available",
        description="Whether required equipment is available",
        category="system_state",
        is_edge_case_priority=True
    ))

    space.add(EnumVariable(
        name="supplies_status",
        description="Medical supplies availability status",
        category="system_state",
        states=["adequate", "low", "out_of_stock"],
        edge_case_states=["out_of_stock"],
        depends_on="equipment_available",
        relevant_when=lambda a: a.get("equipment_available") == TriStateBoolean.TRUE
    ))

    space.add(EnumVariable(
        name="staffing_level",
        description="Current staffing level",
        category="system_state",
        states=["normal", "understaffed", "critical"],
        edge_case_states=["critical"]
    ))

    # Patient state variables
    space.add(EnumVariable(
        name="urgency",
        description="Urgency level of patient's condition",
        category="patient_state",
        states=["routine", "urgent", "emergency"],
        edge_case_states=["emergency"]
    ))

    space.add(EnumVariable(
        name="mental_capacity",
        description="Patient's mental capacity for decision-making",
        category="patient_state",
        states=["full", "limited", "incapacitated"],
        edge_case_states=["incapacitated"],
        depends_on="age_category",
        relevant_when=lambda a: a.get("age_category") != "minor"
    ))

    space.add(EnumVariable(
        name="age_category",
        description="Patient's age category",
        category="patient_state",
        states=["minor", "adult", "elderly"],
        edge_case_states=["minor"]
    ))

    return space
