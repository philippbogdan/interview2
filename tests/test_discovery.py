"""Tests for discovery module (variable extractor, research agent, ralph loop)."""

import pytest
from unittest.mock import Mock, patch

from models.variables import (
    VariableSpace,
    BooleanVariable,
    QuantitativeVariable,
    EnumVariable,
)
from parsers.input_parser import NormalizedAgentConfig
from parsers.domain_analyzer import DomainAnalysis
from discovery.variable_extractor import VariableExtractor, extract_variables
from discovery.research_agent import (
    ResearchAgent,
    ResearchFinding,
    incorporate_research,
    deep_research,
)
from discovery.ralph_loop import RalphLoop, ralph_loop


class TestVariableExtractor:
    """Tests for VariableExtractor."""

    def create_config(self, **kwargs) -> NormalizedAgentConfig:
        """Create a test config."""
        return NormalizedAgentConfig(
            raw_input=kwargs.get("raw_input", "test"),
            description=kwargs.get("description", "Test agent"),
            actions=kwargs.get("actions", []),
            states=kwargs.get("states", []),
            transitions=kwargs.get("transitions", []),
            entities=kwargs.get("entities", []),
        )

    def create_domain_analysis(self, domain: str = "general") -> DomainAnalysis:
        """Create a test domain analysis."""
        return DomainAnalysis(
            detected_domain=domain,
            confidence=0.8,
            field_model=None,
        )

    def test_extract_empty_config(self):
        """Test extraction from empty config."""
        config = self.create_config()
        analysis = self.create_domain_analysis()

        extractor = VariableExtractor()
        space = extractor.extract(config, analysis)

        # Should still create some variables (possibly domain-based)
        assert isinstance(space, VariableSpace)

    def test_extract_from_entities(self):
        """Test extraction from entities."""
        config = self.create_config(
            entities=["patient_name", "date_of_birth", "phone_number"]
        )
        analysis = self.create_domain_analysis()

        extractor = VariableExtractor()
        space = extractor.extract(config, analysis)

        # Should create availability variables for entities
        assert space.get("patient_name_available") is not None
        assert space.get("date_of_birth_available") is not None
        assert space.get("phone_number_available") is not None

    def test_extract_from_searchable_entities(self):
        """Test that searchable entities get search result variables."""
        config = self.create_config(
            entities=["patient_name", "customer_id"]
        )
        analysis = self.create_domain_analysis()

        extractor = VariableExtractor()
        space = extractor.extract(config, analysis)

        # Should create search result variables for searchable entities
        assert space.get("patient_name_search_results") is not None
        assert space.get("customer_id_search_results") is not None

    def test_extract_from_actions(self):
        """Test extraction from actions."""
        config = self.create_config(
            actions=["search_patient", "schedule_appointment", "cancel_booking"]
        )
        analysis = self.create_domain_analysis()

        extractor = VariableExtractor()
        space = extractor.extract(config, analysis)

        # Should create possibility variables for actions
        assert space.get("search_patient_possible") is not None
        assert space.get("schedule_appointment_possible") is not None

        # Should create result variables for search actions
        assert space.get("search_patient_results") is not None

    def test_extract_healthcare_domain_variables(self):
        """Test that healthcare domain adds specific variables."""
        config = self.create_config(
            description="A clinic appointment scheduler"
        )
        analysis = self.create_domain_analysis(domain="healthcare")

        extractor = VariableExtractor()
        space = extractor.extract(config, analysis)

        # Should include healthcare-specific variables
        assert space.get("patient_urgency") is not None
        assert space.get("patient_age_category") is not None
        assert space.get("insurance_status") is not None

    def test_extract_restaurant_domain_variables(self):
        """Test that restaurant domain adds specific variables."""
        config = self.create_config(
            description="A drive-through order system"
        )
        analysis = self.create_domain_analysis(domain="restaurant")

        extractor = VariableExtractor()
        space = extractor.extract(config, analysis)

        # Should include restaurant-specific variables
        assert space.get("item_availability") is not None
        assert space.get("order_complexity") is not None
        assert space.get("payment_status") is not None

    def test_extract_customer_service_domain_variables(self):
        """Test that customer service domain adds specific variables."""
        config = self.create_config(
            description="A customer support helpdesk"
        )
        analysis = self.create_domain_analysis(domain="customer_service")

        extractor = VariableExtractor()
        space = extractor.extract(config, analysis)

        # Should include customer service-specific variables
        assert space.get("customer_sentiment") is not None
        assert space.get("issue_complexity") is not None


class TestExtractVariablesFunction:
    """Tests for the extract_variables convenience function."""

    def test_convenience_function(self):
        """Test that convenience function works."""
        config = NormalizedAgentConfig(
            raw_input="test",
            description="Test agent",
            entities=["name"]
        )
        analysis = DomainAnalysis(
            detected_domain="general",
            confidence=0.8,
            field_model=None
        )

        space = extract_variables(config, analysis)
        assert isinstance(space, VariableSpace)


class TestResearchFinding:
    """Tests for ResearchFinding dataclass."""

    def test_create_finding(self):
        """Test creating a research finding."""
        finding = ResearchFinding(
            category="incident",
            description="Patient lookup failed due to name mismatch",
            source_type="incident",
            implied_variables=[
                {"name": "name_mismatch", "type": "boolean"}
            ],
            priority="high"
        )

        assert finding.category == "incident"
        assert finding.priority == "high"
        assert len(finding.implied_variables) == 1


class TestIncorporateResearch:
    """Tests for incorporate_research function."""

    def test_incorporate_boolean_variable(self):
        """Test incorporating boolean variable from research."""
        space = VariableSpace()
        findings = [
            ResearchFinding(
                category="incident",
                description="Test finding",
                source_type="incident",
                implied_variables=[
                    {"name": "test_condition", "type": "boolean"}
                ],
                priority="high"
            )
        ]

        result = incorporate_research(space, findings)

        var = result.get("test_condition")
        assert var is not None
        assert isinstance(var, BooleanVariable)
        assert var.is_edge_case_priority is True  # high priority

    def test_incorporate_enum_variable(self):
        """Test incorporating enum variable from research."""
        space = VariableSpace()
        findings = [
            ResearchFinding(
                category="regulatory",
                description="Test finding",
                source_type="regulatory",
                implied_variables=[
                    {
                        "name": "compliance_level",
                        "type": "enum",
                        "states": ["compliant", "partial", "non_compliant"]
                    }
                ],
                priority="medium"
            )
        ]

        result = incorporate_research(space, findings)

        var = result.get("compliance_level")
        assert var is not None
        assert isinstance(var, EnumVariable)
        assert var.states == ["compliant", "partial", "non_compliant"]

    def test_no_duplicate_variables(self):
        """Test that existing variables aren't duplicated."""
        space = VariableSpace()
        space.add(BooleanVariable(
            name="existing_var",
            description="Already exists",
            category="test"
        ))

        findings = [
            ResearchFinding(
                category="incident",
                description="Test",
                source_type="incident",
                implied_variables=[
                    {"name": "existing_var", "type": "boolean"}
                ],
                priority="high"
            )
        ]

        result = incorporate_research(space, findings)
        assert len(result) == 1  # Still just one variable


class TestRalphLoop:
    """Tests for RalphLoop."""

    def test_create_ralph_loop(self):
        """Test creating a Ralph Loop instance."""
        loop = RalphLoop(api_key="test_key")
        assert loop.target_expansion_rate == 0.5
        assert loop.model == "grok-4-1-fast-non-reasoning"

    def test_format_variables(self):
        """Test variable formatting for prompt."""
        loop = RalphLoop(api_key="test_key")
        space = VariableSpace()
        space.add(BooleanVariable(
            name="test_var",
            description="Test variable",
            category="test"
        ))

        formatted = loop._format_variables(space)
        assert "test_var" in formatted
        assert "boolean" in formatted.lower()

    def test_format_empty_variables(self):
        """Test formatting empty variable space."""
        loop = RalphLoop(api_key="test_key")
        space = VariableSpace()

        formatted = loop._format_variables(space)
        assert "no variables" in formatted.lower()

    def test_create_variable_from_spec(self):
        """Test creating variables from specification."""
        loop = RalphLoop(api_key="test_key")

        # Boolean variable
        bool_spec = {
            "name": "test_bool",
            "type": "boolean",
            "description": "A test boolean",
            "category": "test"
        }
        var = loop._create_variable(bool_spec)
        assert isinstance(var, BooleanVariable)
        assert var.name == "test_bool"

        # Enum variable
        enum_spec = {
            "name": "test_enum",
            "type": "enum",
            "description": "A test enum",
            "category": "test",
            "states": ["a", "b", "c"],
            "edge_case_states": ["c"]
        }
        var = loop._create_variable(enum_spec)
        assert isinstance(var, EnumVariable)
        assert var.states == ["a", "b", "c"]

        # Quantitative variable
        quant_spec = {
            "name": "test_quant",
            "type": "quantitative",
            "description": "A test quantitative",
            "category": "test"
        }
        var = loop._create_variable(quant_spec)
        assert isinstance(var, QuantitativeVariable)

    def test_create_variable_invalid_spec(self):
        """Test that invalid specs return None."""
        loop = RalphLoop(api_key="test_key")

        # Missing name
        var = loop._create_variable({"type": "boolean", "description": "test"})
        assert var is None

        # Missing description
        var = loop._create_variable({"name": "test", "type": "boolean"})
        assert var is None


class TestRalphLoopConvenienceFunction:
    """Tests for ralph_loop convenience function."""

    @patch.object(RalphLoop, 'expand')
    def test_ralph_loop_function(self, mock_expand):
        """Test the ralph_loop convenience function."""
        space = VariableSpace()
        mock_expand.return_value = (space, [])

        result = ralph_loop(
            space,
            "Test description",
            iterations=2,
            api_key="test_key"
        )

        assert result is space
        mock_expand.assert_called_once_with(space, "Test description", 2)


class TestResearchAgentWithMock:
    """Tests for ResearchAgent with mocked API calls."""

    @patch('discovery.research_agent.OpenAI')
    def test_research_returns_findings(self, mock_openai_class):
        """Test that research returns findings."""
        # Set up mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "findings": [
                {
                    "description": "Test incident",
                    "variables": [{"name": "test_var", "type": "boolean"}],
                    "priority": "high"
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_response

        agent = ResearchAgent(api_key="test_key")
        findings = agent._research_incidents("healthcare", "Test agent")

        assert len(findings) == 1
        assert findings[0].category == "incident"
        assert findings[0].priority == "high"
