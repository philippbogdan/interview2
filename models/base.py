"""Pydantic models for scenario generation."""

from typing import Optional
from pydantic import BaseModel, Field


class BaseScenario(BaseModel):
    """Base scenario fields required for all test scenarios."""

    scenarioName: str = Field(
        description="A concise, descriptive name for the test scenario"
    )
    scenarioDescription: str = Field(
        description="Detailed description of what this scenario tests, including the user's goal and expected agent behavior"
    )
    criteria: list[str] = Field(
        description="List of success criteria that define when this scenario passes"
    )


class ClinicPatientFields(BaseModel):
    """Domain-specific fields for clinic/healthcare scenarios."""

    name: str = Field(
        description="Patient's full name"
    )
    dob: str = Field(
        description="Patient's date of birth (format: YYYY-MM-DD or MM/DD/YYYY)"
    )
    phone: str = Field(
        description="Patient's phone number"
    )
    email: Optional[str] = Field(
        default=None,
        description="Patient's email address"
    )
    insurance: Optional[str] = Field(
        default=None,
        description="Insurance provider name or policy number"
    )
    appointment_type: str = Field(
        description="Type of appointment being scheduled (e.g., checkup, consultation, follow-up)"
    )


class DriveThruOrderFields(BaseModel):
    """Domain-specific fields for restaurant/drive-through scenarios."""

    order_items: list[str] = Field(
        description="List of items being ordered"
    )
    customizations: list[str] = Field(
        default_factory=list,
        description="List of customizations or modifications to items (e.g., no onions, extra cheese)"
    )
    total_items: int = Field(
        description="Total number of items in the order"
    )
    payment_method: str = Field(
        description="Payment method (e.g., cash, credit, debit, app)"
    )


class CustomerServiceFields(BaseModel):
    """Domain-specific fields for customer service scenarios."""

    customer_id: Optional[str] = Field(
        default=None,
        description="Customer ID or account number"
    )
    issue_category: str = Field(
        description="Category of the customer's issue (e.g., billing, technical, returns)"
    )
    sentiment: str = Field(
        description="Customer's emotional state (e.g., frustrated, neutral, satisfied)"
    )
    priority: str = Field(
        description="Issue priority level (e.g., low, medium, high, urgent)"
    )


# Domain registry mapping keywords to field models
DOMAIN_REGISTRY: dict[str, type[BaseModel]] = {
    "clinic": ClinicPatientFields,
    "healthcare": ClinicPatientFields,
    "medical": ClinicPatientFields,
    "hospital": ClinicPatientFields,
    "patient": ClinicPatientFields,
    "appointment": ClinicPatientFields,
    "doctor": ClinicPatientFields,
    "restaurant": DriveThruOrderFields,
    "drive-through": DriveThruOrderFields,
    "drive-thru": DriveThruOrderFields,
    "food": DriveThruOrderFields,
    "order": DriveThruOrderFields,
    "menu": DriveThruOrderFields,
    "customer service": CustomerServiceFields,
    "support": CustomerServiceFields,
    "helpdesk": CustomerServiceFields,
    "complaint": CustomerServiceFields,
    "issue": CustomerServiceFields,
}
