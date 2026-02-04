"""Tests for the Voice Agent Test Scenario Generator."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from parsers.input_parser import parse_input, NormalizedAgentConfig
from parsers.domain_analyzer import analyze_domain, DomainAnalysis
from models.base import (
    BaseScenario,
    ClinicPatientFields,
    DriveThruOrderFields,
    CustomerServiceFields,
)


# ============================================================================
# Input Parser Tests
# ============================================================================

class TestParseInput:
    """Tests for the input parser."""

    def test_parse_dict_with_description(self):
        """Test parsing a dictionary with a description."""
        input_data = {"description": "A clinic appointment assistant"}
        config = parse_input(input_data)

        assert isinstance(config, NormalizedAgentConfig)
        assert config.description == "A clinic appointment assistant"
        assert config.raw_input == input_data

    def test_parse_dict_with_prompt(self):
        """Test parsing a dictionary with 'prompt' key instead of 'description'."""
        input_data = {"prompt": "A restaurant order assistant"}
        config = parse_input(input_data)

        assert config.description == "A restaurant order assistant"

    def test_parse_dict_with_actions(self):
        """Test parsing a dictionary with actions."""
        input_data = {
            "description": "Test agent",
            "actions": ["action1", "action2", "action3"],
        }
        config = parse_input(input_data)

        assert config.actions == ["action1", "action2", "action3"]

    def test_parse_dict_with_action_objects(self):
        """Test parsing actions that are objects with 'name' field."""
        input_data = {
            "description": "Test agent",
            "actions": [
                {"name": "book_appointment", "params": ["date", "time"]},
                {"name": "cancel_appointment"},
            ],
        }
        config = parse_input(input_data)

        assert config.actions == ["book_appointment", "cancel_appointment"]

    def test_parse_dict_with_states(self):
        """Test parsing a dictionary with states."""
        input_data = {
            "description": "Test agent",
            "states": ["greeting", "collecting_info", "confirming", "done"],
        }
        config = parse_input(input_data)

        assert config.states == ["greeting", "collecting_info", "confirming", "done"]

    def test_parse_dict_with_transitions(self):
        """Test parsing a dictionary with transitions."""
        transitions = [
            {"from": "greeting", "to": "collecting_info", "trigger": "start"},
            {"from": "collecting_info", "to": "confirming", "trigger": "info_complete"},
        ]
        input_data = {"description": "Test agent", "transitions": transitions}
        config = parse_input(input_data)

        assert config.transitions == transitions

    def test_parse_dict_with_entities(self):
        """Test parsing a dictionary with entities."""
        input_data = {
            "description": "Test agent",
            "entities": ["patient_name", "appointment_date", "phone"],
        }
        config = parse_input(input_data)

        assert config.entities == ["patient_name", "appointment_date", "phone"]

    def test_parse_json_string(self):
        """Test parsing a JSON string."""
        input_data = json.dumps({"description": "A JSON string agent"})
        config = parse_input(input_data)

        assert config.description == "A JSON string agent"

    def test_parse_plain_text(self):
        """Test parsing plain text description."""
        input_data = "This is a voice agent for scheduling doctor appointments"
        config = parse_input(input_data)

        assert config.description == input_data
        assert config.raw_input == input_data

    def test_parse_dict_stores_metadata(self):
        """Test that unknown keys are stored in metadata."""
        input_data = {
            "description": "Test agent",
            "custom_field": "custom_value",
            "another_field": 123,
        }
        config = parse_input(input_data)

        assert config.metadata["custom_field"] == "custom_value"
        assert config.metadata["another_field"] == 123

    def test_is_structured_true(self):
        """Test is_structured property returns True when structured."""
        config = parse_input({"description": "Test", "actions": ["a1"]})
        assert config.is_structured is True

    def test_is_structured_false(self):
        """Test is_structured property returns False for plain text."""
        config = parse_input("Just a plain text description")
        assert config.is_structured is False

    def test_parse_invalid_type_raises_error(self):
        """Test that invalid input types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported input type"):
            parse_input(12345)


# ============================================================================
# Domain Analyzer Tests
# ============================================================================

class TestDomainAnalyzer:
    """Tests for the domain analyzer."""

    def test_detect_healthcare_domain(self):
        """Test detection of healthcare/clinic domain."""
        config = parse_input({
            "description": "A clinic appointment scheduling system for patients"
        })
        analysis = analyze_domain(config)

        assert analysis.detected_domain == "healthcare"
        assert analysis.field_model == ClinicPatientFields
        assert analysis.confidence > 0

    def test_detect_healthcare_from_keywords(self):
        """Test healthcare detection from various keywords."""
        keywords = ["clinic", "patient", "doctor", "appointment", "medical", "hospital"]

        for keyword in keywords:
            config = parse_input({"description": f"A {keyword} management system"})
            analysis = analyze_domain(config)
            assert analysis.detected_domain == "healthcare", f"Failed for keyword: {keyword}"

    def test_detect_restaurant_domain(self):
        """Test detection of restaurant/drive-through domain."""
        config = parse_input({
            "description": "A drive-through order taking system for a fast food restaurant"
        })
        analysis = analyze_domain(config)

        assert analysis.detected_domain == "restaurant"
        assert analysis.field_model == DriveThruOrderFields

    def test_detect_restaurant_from_keywords(self):
        """Test restaurant detection from various keywords."""
        keywords = ["restaurant", "drive-through", "drive-thru", "food", "order", "menu"]

        for keyword in keywords:
            config = parse_input({"description": f"A {keyword} system"})
            analysis = analyze_domain(config)
            assert analysis.detected_domain == "restaurant", f"Failed for keyword: {keyword}"

    def test_detect_customer_service_domain(self):
        """Test detection of customer service domain."""
        config = parse_input({
            "description": "A customer service support helpdesk for handling complaints"
        })
        analysis = analyze_domain(config)

        assert analysis.detected_domain == "customer_service"
        assert analysis.field_model == CustomerServiceFields

    def test_detect_general_domain(self):
        """Test fallback to general domain."""
        config = parse_input({
            "description": "A generic assistant that helps with various tasks"
        })
        analysis = analyze_domain(config)

        assert analysis.detected_domain == "general"
        assert analysis.field_model is None

    def test_keywords_found_populated(self):
        """Test that keywords_found is populated correctly."""
        config = parse_input({
            "description": "A clinic for patient appointment scheduling"
        })
        analysis = analyze_domain(config)

        assert len(analysis.keywords_found) > 0
        assert any(k in ["clinic", "patient", "appointment"] for k in analysis.keywords_found)

    def test_suggested_edge_cases_for_healthcare(self):
        """Test that edge cases are suggested for healthcare domain."""
        config = parse_input({"description": "A clinic appointment system"})
        analysis = analyze_domain(config)

        assert len(analysis.suggested_edge_cases) > 0
        # Check for healthcare-specific edge cases
        edge_case_text = " ".join(analysis.suggested_edge_cases).lower()
        assert any(term in edge_case_text for term in ["insurance", "appointment", "patient"])

    def test_suggested_edge_cases_for_restaurant(self):
        """Test that edge cases are suggested for restaurant domain."""
        config = parse_input({"description": "A drive-through ordering system"})
        analysis = analyze_domain(config)

        assert len(analysis.suggested_edge_cases) > 0
        edge_case_text = " ".join(analysis.suggested_edge_cases).lower()
        assert any(term in edge_case_text for term in ["stock", "order", "payment", "customization"])

    def test_multi_word_keywords_weighted_higher(self):
        """Test that multi-word keywords like 'drive-through' are weighted higher."""
        # This test ensures that specific compound terms are prioritized
        config = parse_input({
            "description": "A drive-through service"
        })
        analysis = analyze_domain(config)

        assert analysis.detected_domain == "restaurant"

    def test_analyzes_actions_for_domain(self):
        """Test that actions are analyzed for domain detection."""
        config = parse_input({
            "description": "An assistant",
            "actions": ["schedule_appointment", "check_patient_records", "verify_insurance"]
        })
        analysis = analyze_domain(config)

        # Should detect healthcare from action names
        assert analysis.detected_domain == "healthcare"


# ============================================================================
# Pydantic Models Tests
# ============================================================================

class TestPydanticModels:
    """Tests for Pydantic models."""

    def test_base_scenario_fields(self):
        """Test BaseScenario has required fields."""
        scenario = BaseScenario(
            scenarioName="Test Scenario",
            scenarioDescription="A test description",
            criteria=["Criterion 1", "Criterion 2"]
        )

        assert scenario.scenarioName == "Test Scenario"
        assert scenario.scenarioDescription == "A test description"
        assert len(scenario.criteria) == 2

    def test_clinic_patient_fields(self):
        """Test ClinicPatientFields model."""
        fields = ClinicPatientFields(
            name="John Doe",
            dob="1990-01-15",
            phone="555-1234",
            email="john@example.com",
            insurance="BlueCross",
            appointment_type="checkup"
        )

        assert fields.name == "John Doe"
        assert fields.appointment_type == "checkup"

    def test_clinic_patient_fields_optional(self):
        """Test ClinicPatientFields with optional fields omitted."""
        fields = ClinicPatientFields(
            name="Jane Doe",
            dob="1985-05-20",
            phone="555-5678",
            appointment_type="consultation"
        )

        assert fields.email is None
        assert fields.insurance is None

    def test_drive_thru_order_fields(self):
        """Test DriveThruOrderFields model."""
        fields = DriveThruOrderFields(
            order_items=["burger", "fries", "drink"],
            customizations=["no onions", "extra ketchup"],
            total_items=3,
            payment_method="credit"
        )

        assert len(fields.order_items) == 3
        assert fields.total_items == 3
        assert fields.payment_method == "credit"

    def test_customer_service_fields(self):
        """Test CustomerServiceFields model."""
        fields = CustomerServiceFields(
            customer_id="CUST-12345",
            issue_category="billing",
            sentiment="frustrated",
            priority="high"
        )

        assert fields.issue_category == "billing"
        assert fields.sentiment == "frustrated"


# ============================================================================
# Scenario Generator Tests (Mocked)
# ============================================================================

class TestScenarioGenerator:
    """Tests for the scenario generator with mocked OpenAI API."""

    @patch("generators.scenario_generator.OpenAI")
    def test_generator_initialization(self, mock_openai_class):
        """Test generator initializes with API key."""
        from generators.scenario_generator import ScenarioGenerator

        generator = ScenarioGenerator(api_key="test-key")
        mock_openai_class.assert_called_once_with(api_key="test-key")

    def test_generator_raises_without_api_key(self):
        """Test generator raises error without API key."""
        from generators.scenario_generator import ScenarioGenerator

        with patch.dict("os.environ", {}, clear=True):
            # Remove OPENAI_API_KEY if present
            import os
            env_backup = os.environ.get("OPENAI_API_KEY")
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

            try:
                with pytest.raises(ValueError, match="OpenAI API key is required"):
                    ScenarioGenerator()
            finally:
                if env_backup:
                    os.environ["OPENAI_API_KEY"] = env_backup

    @patch("generators.scenario_generator.OpenAI")
    def test_generate_returns_scenarios(self, mock_openai_class):
        """Test generate method returns scenarios from API."""
        from generators.scenario_generator import ScenarioGenerator

        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "scenarios": [
                {
                    "scenarioName": "Test Scenario 1",
                    "scenarioDescription": "Description 1",
                    "criteria": ["Criterion 1"]
                },
                {
                    "scenarioName": "Test Scenario 2",
                    "scenarioDescription": "Description 2",
                    "criteria": ["Criterion 2"]
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        generator = ScenarioGenerator(api_key="test-key")
        config = parse_input({"description": "Test agent"})
        analysis = analyze_domain(config)

        scenarios = generator.generate(config, analysis, num_scenarios=2)

        assert len(scenarios) == 2
        assert scenarios[0]["scenarioName"] == "Test Scenario 1"
        assert scenarios[1]["scenarioName"] == "Test Scenario 2"

    @patch("generators.scenario_generator.OpenAI")
    def test_generate_uses_structured_output(self, mock_openai_class):
        """Test that generate uses structured output format."""
        from generators.scenario_generator import ScenarioGenerator

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"scenarios": []})

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        generator = ScenarioGenerator(api_key="test-key")
        config = parse_input({"description": "Test agent"})
        analysis = analyze_domain(config)

        generator.generate(config, analysis, num_scenarios=1)

        # Verify structured output was requested
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"

    @patch("generators.scenario_generator.OpenAI")
    def test_generate_includes_domain_fields_in_schema(self, mock_openai_class):
        """Test that domain-specific fields are included in the schema."""
        from generators.scenario_generator import ScenarioGenerator

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"scenarios": []})

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        generator = ScenarioGenerator(api_key="test-key")
        config = parse_input({"description": "A clinic appointment system for patients"})
        analysis = analyze_domain(config)

        generator.generate(config, analysis, num_scenarios=1)

        # Check the schema includes domain fields
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        schema = call_kwargs["response_format"]["json_schema"]["schema"]
        scenario_props = schema["properties"]["scenarios"]["items"]["properties"]

        # Should have healthcare domain fields
        assert "name" in scenario_props
        assert "dob" in scenario_props
        assert "appointment_type" in scenario_props


# ============================================================================
# Integration Tests (Mocked)
# ============================================================================

class TestCreateScenariosIntegration:
    """Integration tests for create_scenarios function."""

    @patch("generators.scenario_generator.OpenAI")
    def test_create_scenarios_with_dict(self, mock_openai_class):
        """Test create_scenarios with dictionary input."""
        from create_scenarios import create_scenarios

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "scenarios": [
                {
                    "scenarioName": "Happy Path",
                    "scenarioDescription": "Standard booking",
                    "criteria": ["Success"],
                    "name": "John Doe",
                    "dob": "1990-01-01",
                    "phone": "555-1234",
                    "appointment_type": "checkup"
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = {
            "description": "A clinic appointment scheduler",
            "actions": ["book", "cancel"]
        }

        scenarios = create_scenarios(config, num_scenarios=1, api_key="test-key")

        assert len(scenarios) == 1
        assert "name" in scenarios[0]  # Healthcare domain field

    @patch("generators.scenario_generator.OpenAI")
    def test_create_scenarios_with_text(self, mock_openai_class):
        """Test create_scenarios with plain text input."""
        from create_scenarios import create_scenarios

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "scenarios": [
                {
                    "scenarioName": "Order Scenario",
                    "scenarioDescription": "Taking an order",
                    "criteria": ["Order confirmed"],
                    "order_items": ["burger"],
                    "customizations": [],
                    "total_items": 1,
                    "payment_method": "cash"
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        scenarios = create_scenarios(
            "A drive-through order taking system for a restaurant",
            num_scenarios=1,
            api_key="test-key"
        )

        assert len(scenarios) == 1
        assert "order_items" in scenarios[0]  # Restaurant domain field

    @patch("generators.scenario_generator.OpenAI")
    def test_create_scenarios_multiple(self, mock_openai_class):
        """Test create_scenarios generates requested number of scenarios."""
        from create_scenarios import create_scenarios

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "scenarios": [
                {"scenarioName": f"Scenario {i}", "scenarioDescription": f"Desc {i}", "criteria": []}
                for i in range(5)
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        scenarios = create_scenarios(
            "A general assistant",
            num_scenarios=5,
            api_key="test-key"
        )

        assert len(scenarios) == 5


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_description(self):
        """Test handling of empty description."""
        config = parse_input({"description": ""})
        analysis = analyze_domain(config)

        assert analysis.detected_domain == "general"

    def test_very_long_description(self):
        """Test handling of very long description."""
        long_text = "clinic patient " * 1000
        config = parse_input(long_text)
        analysis = analyze_domain(config)

        assert analysis.detected_domain == "healthcare"

    def test_mixed_domain_keywords(self):
        """Test handling when multiple domains are mentioned."""
        config = parse_input({
            "description": "A clinic that also has a restaurant for patients"
        })
        analysis = analyze_domain(config)

        # Should pick the more prominent domain
        assert analysis.detected_domain in ["healthcare", "restaurant"]

    def test_unicode_in_description(self):
        """Test handling of unicode characters."""
        config = parse_input({
            "description": "A clínic for patiénts with appointments 日本語"
        })
        analysis = analyze_domain(config)

        # Should still detect healthcare
        assert analysis.detected_domain == "healthcare"

    def test_nested_dict_input(self):
        """Test parsing of deeply nested dictionary."""
        config = parse_input({
            "agent": {
                "description": "Nested clinic agent",
                "config": {
                    "patient_handling": True
                }
            }
        })

        # Should extract text from nested structure
        analysis = analyze_domain(config)
        assert analysis.detected_domain == "healthcare"
