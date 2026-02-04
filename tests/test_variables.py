"""Tests for variable models and definitions."""

import pytest

from models.variables import (
    TriStateBoolean,
    QuantitativeState,
    BooleanVariable,
    QuantitativeVariable,
    EnumVariable,
    VariableAssignment,
    VariableSpace,
    create_healthcare_variables,
)


class TestTriStateBoolean:
    """Tests for TriStateBoolean enum."""

    def test_all_states(self):
        """Test that all_states returns all three states."""
        states = TriStateBoolean.all_states()
        assert len(states) == 3
        assert TriStateBoolean.FALSE in states
        assert TriStateBoolean.UNKNOWN in states
        assert TriStateBoolean.TRUE in states

    def test_edge_cases(self):
        """Test that edge_cases returns FALSE and UNKNOWN."""
        edge_cases = TriStateBoolean.edge_cases()
        assert len(edge_cases) == 2
        assert TriStateBoolean.FALSE in edge_cases
        assert TriStateBoolean.UNKNOWN in edge_cases
        assert TriStateBoolean.TRUE not in edge_cases

    def test_values(self):
        """Test enum values."""
        assert TriStateBoolean.FALSE.value == "false"
        assert TriStateBoolean.UNKNOWN.value == "unknown"
        assert TriStateBoolean.TRUE.value == "true"


class TestQuantitativeState:
    """Tests for QuantitativeState enum."""

    def test_all_states(self):
        """Test that all_states returns all three states."""
        states = QuantitativeState.all_states()
        assert len(states) == 3
        assert QuantitativeState.NONE in states
        assert QuantitativeState.SINGLE in states
        assert QuantitativeState.MANY in states

    def test_edge_cases(self):
        """Test that edge_cases returns NONE and MANY."""
        edge_cases = QuantitativeState.edge_cases()
        assert len(edge_cases) == 2
        assert QuantitativeState.NONE in edge_cases
        assert QuantitativeState.MANY in edge_cases
        assert QuantitativeState.SINGLE not in edge_cases


class TestBooleanVariable:
    """Tests for BooleanVariable."""

    def test_get_all_states(self):
        """Test that boolean variable returns tri-state values."""
        var = BooleanVariable(
            name="test_var",
            description="Test variable",
            category="test"
        )
        states = var.get_all_states()
        assert states == TriStateBoolean.all_states()

    def test_get_edge_case_states(self):
        """Test edge case states for boolean variable."""
        var = BooleanVariable(
            name="test_var",
            description="Test variable",
            category="test"
        )
        edge_cases = var.get_edge_case_states()
        assert edge_cases == TriStateBoolean.edge_cases()

    def test_is_relevant_no_condition(self):
        """Test relevance without condition."""
        var = BooleanVariable(
            name="test_var",
            description="Test variable",
            category="test"
        )
        assignment = VariableAssignment()
        assert var.is_relevant(assignment) is True

    def test_is_relevant_with_condition(self):
        """Test relevance with condition."""
        var = BooleanVariable(
            name="child_var",
            description="Child variable",
            category="test",
            relevant_when=lambda a: a.get("parent_var") == "active"
        )

        # Without parent set
        assignment = VariableAssignment()
        assert var.is_relevant(assignment) is False

        # With parent set to wrong value
        assignment = VariableAssignment(assignments={"parent_var": "inactive"})
        assert var.is_relevant(assignment) is False

        # With parent set correctly
        assignment = VariableAssignment(assignments={"parent_var": "active"})
        assert var.is_relevant(assignment) is True


class TestQuantitativeVariable:
    """Tests for QuantitativeVariable."""

    def test_get_all_states_no_variants(self):
        """Test states without single variants."""
        var = QuantitativeVariable(
            name="results",
            description="Search results",
            category="search"
        )
        states = var.get_all_states()
        assert len(states) == 3
        assert QuantitativeState.NONE in states
        assert QuantitativeState.SINGLE in states
        assert QuantitativeState.MANY in states

    def test_get_all_states_with_variants(self):
        """Test states with single variants."""
        var = QuantitativeVariable(
            name="results",
            description="Search results",
            category="search",
            single_variants=["correct", "incorrect"]
        )
        states = var.get_all_states()
        assert len(states) == 4
        assert QuantitativeState.NONE in states
        assert ("single", "correct") in states
        assert ("single", "incorrect") in states
        assert QuantitativeState.MANY in states

    def test_edge_cases_include_incorrect_variant(self):
        """Test that 'incorrect' variants are edge cases."""
        var = QuantitativeVariable(
            name="results",
            description="Search results",
            category="search",
            single_variants=["correct", "incorrect"]
        )
        edge_cases = var.get_edge_case_states()
        assert QuantitativeState.NONE in edge_cases
        assert QuantitativeState.MANY in edge_cases
        assert ("single", "incorrect") in edge_cases


class TestEnumVariable:
    """Tests for EnumVariable."""

    def test_get_all_states(self):
        """Test getting all enum states."""
        var = EnumVariable(
            name="urgency",
            description="Urgency level",
            category="patient",
            states=["routine", "urgent", "emergency"]
        )
        states = var.get_all_states()
        assert states == ["routine", "urgent", "emergency"]

    def test_get_edge_case_states(self):
        """Test getting edge case states."""
        var = EnumVariable(
            name="urgency",
            description="Urgency level",
            category="patient",
            states=["routine", "urgent", "emergency"],
            edge_case_states=["emergency"]
        )
        edge_cases = var.get_edge_case_states()
        assert edge_cases == ["emergency"]


class TestVariableAssignment:
    """Tests for VariableAssignment."""

    def test_empty_assignment(self):
        """Test empty assignment."""
        assignment = VariableAssignment()
        assert len(assignment.keys()) == 0
        assert assignment.get("nonexistent") is None

    def test_set_and_get(self):
        """Test setting and getting values."""
        assignment = VariableAssignment()
        new_assignment = assignment.set("var1", "value1")

        # Original unchanged
        assert not assignment.has("var1")

        # New assignment has value
        assert new_assignment.has("var1")
        assert new_assignment.get("var1") == "value1"

    def test_immutability(self):
        """Test that set returns a new instance."""
        assignment = VariableAssignment(assignments={"a": 1})
        new_assignment = assignment.set("b", 2)

        assert assignment.get("b") is None
        assert new_assignment.get("a") == 1
        assert new_assignment.get("b") == 2

    def test_copy(self):
        """Test copying assignments."""
        assignment = VariableAssignment(assignments={"a": 1}, metadata={"key": "value"})
        copied = assignment.copy()

        assert copied.get("a") == 1
        assert copied.metadata["key"] == "value"

        # Modify copy doesn't affect original
        copied.assignments["b"] = 2
        assert assignment.get("b") is None

    def test_hash_and_equality(self):
        """Test hashing and equality."""
        a1 = VariableAssignment(assignments={"x": 1, "y": 2})
        a2 = VariableAssignment(assignments={"y": 2, "x": 1})  # Same but different order
        a3 = VariableAssignment(assignments={"x": 1, "y": 3})  # Different

        assert a1 == a2
        assert hash(a1) == hash(a2)
        assert a1 != a3

    def test_items(self):
        """Test items iteration."""
        assignment = VariableAssignment(assignments={"a": 1, "b": 2})
        items = assignment.items()
        assert ("a", 1) in items
        assert ("b", 2) in items


class TestVariableSpace:
    """Tests for VariableSpace."""

    def test_add_and_get(self):
        """Test adding and getting variables."""
        space = VariableSpace()
        var = BooleanVariable(name="test", description="Test", category="cat")

        space.add(var)
        assert space.get("test") == var
        assert space.get("nonexistent") is None

    def test_len(self):
        """Test length."""
        space = VariableSpace()
        assert len(space) == 0

        space.add(BooleanVariable(name="v1", description="D1", category="c"))
        space.add(BooleanVariable(name="v2", description="D2", category="c"))
        assert len(space) == 2

    def test_get_by_category(self):
        """Test filtering by category."""
        space = VariableSpace()
        space.add(BooleanVariable(name="v1", description="D", category="cat1"))
        space.add(BooleanVariable(name="v2", description="D", category="cat2"))
        space.add(BooleanVariable(name="v3", description="D", category="cat1"))

        cat1_vars = space.get_by_category("cat1")
        assert len(cat1_vars) == 2
        assert all(v.category == "cat1" for v in cat1_vars)

    def test_get_categories(self):
        """Test getting unique categories."""
        space = VariableSpace()
        space.add(BooleanVariable(name="v1", description="D", category="cat1"))
        space.add(BooleanVariable(name="v2", description="D", category="cat2"))
        space.add(BooleanVariable(name="v3", description="D", category="cat1"))

        categories = space.get_categories()
        assert set(categories) == {"cat1", "cat2"}

    def test_iteration(self):
        """Test iteration over variables."""
        space = VariableSpace()
        space.add(BooleanVariable(name="v1", description="D", category="c"))
        space.add(BooleanVariable(name="v2", description="D", category="c"))

        names = [v.name for v in space]
        assert names == ["v1", "v2"]


class TestCreateHealthcareVariables:
    """Tests for the healthcare variables factory."""

    def test_creates_expected_variables(self):
        """Test that healthcare variables are created correctly."""
        space = create_healthcare_variables()

        # Check that key variables exist
        assert space.get("name_available") is not None
        assert space.get("dob_available") is not None
        assert space.get("insurance_status") is not None
        assert space.get("search_results") is not None
        assert space.get("can_speak") is not None
        assert space.get("urgency") is not None
        assert space.get("age_category") is not None

    def test_variable_types(self):
        """Test that variables have correct types."""
        space = create_healthcare_variables()

        # Boolean variables
        assert isinstance(space.get("name_available"), BooleanVariable)
        assert isinstance(space.get("can_speak"), BooleanVariable)

        # Quantitative variables
        assert isinstance(space.get("search_results"), QuantitativeVariable)

        # Enum variables
        assert isinstance(space.get("insurance_status"), EnumVariable)
        assert isinstance(space.get("urgency"), EnumVariable)

    def test_dependencies_set(self):
        """Test that dependencies are set correctly."""
        space = create_healthcare_variables()

        search_results = space.get("search_results")
        assert search_results.depends_on == "name_available"

        supplies = space.get("supplies_status")
        assert supplies.depends_on == "equipment_available"
