# Combinatorial Test Scenario Generator

Automatically generates comprehensive test scenarios for voice agents using combinatorial exploration of variable spaces.

## The Problem

Testing voice agents is hard. You need to cover:
- Happy paths (everything works)
- Edge cases (missing info, invalid input)
- Error conditions (system failures, timeouts)
- Accessibility scenarios (hearing/speech impaired)
- Regulatory requirements (HIPAA, PCI-DSS)

Manually writing test cases misses combinations. Pure LLM generation lacks systematic coverage.

## The Solution

This system combines **algorithmic exploration** with **LLM-powered scenario construction**:

```
Input (voice agent config)
        ↓
   Parse & Analyze Domain
        ↓
   Extract Test Variables      ← pattern matching
        ↓
   Research Edge Cases         ← LLM finds real-world failures
        ↓
   Expand Variable Space       ← Ralph Loop grows dimensions
        ↓
   Explore Combinations        ← systematic coverage
        ↓
   Construct Scenarios         ← LLM writes coherent tests
        ↓
Output (concrete test cases)
```

## How It Works

### 1. Variable Extraction

From a voice agent config like:
```json
{
  "description": "AI receptionist for a clinic",
  "actions": ["find_patient", "schedule_appointment"],
  "entities": ["patient_name", "date_of_birth", "insurance"]
}
```

The system extracts test variables:

| Variable | Type | States |
|----------|------|--------|
| `name_available` | Boolean | no, unknown, yes |
| `name_search_results` | Quantitative | none, single:correct, single:incorrect, many |
| `insurance_status` | Enum | none, expired, active, pending |
| `patient_urgency` | Enum | routine, urgent, emergency |

### 2. Research & Expansion (Optional)

**Research Agent** finds real-world edge cases:
- "Patient is a minor requiring guardian consent"
- "HIPAA requires identity verification"
- "Hearing impaired patients need text alternatives"

**Ralph Loop** expands the variable space iteratively:
- Iteration 1: 15 vars → 22 vars
- Iteration 2: 22 vars → 33 vars
- Iteration 3: 33 vars → 50 vars

### 3. Systematic Exploration

Three-phase exploration ensures coverage:

**Phase 1: Edge Cases First**
```
name_available=no, dob_available=no, urgency=emergency
```

**Phase 2: Pairwise Coverage**
- Every pair of (variable, value) appears in at least one scenario
- Catches interaction bugs

**Phase 3: Systematic BFS**
- Fill remaining gaps
- Respect variable dependencies

### 4. Scenario Construction

Each variable assignment becomes a concrete scenario via LLM:

```json
{
  "scenarioName": "Emergency Patient Missing Critical Information",
  "scenarioDescription": "Sarah Chen calls frantically about severe chest pain. She's disoriented and cannot remember her date of birth. Her insurance card is at home. The system must handle this emergency while working with incomplete information.",
  "name": "Sarah Chen",
  "dob": null,
  "phone": "555-0147",
  "insurance": null,
  "appointment_type": "emergency",
  "criteria": [
    "Agent immediately recognizes emergency keywords",
    "Agent does not block on missing DOB",
    "Agent offers to verify identity via phone number",
    "Agent escalates to human within 30 seconds",
    "Agent provides emergency instructions while transferring"
  ]
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT                                │
│  JSON config, plain text, or structured agent definition    │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      PARSERS                                │
│  input_parser.py      → NormalizedAgentConfig               │
│  domain_analyzer.py   → DomainAnalysis (healthcare, etc.)   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     DISCOVERY                               │
│  variable_extractor.py  → Extract from config               │
│  research_agent.py      → Find edge cases via LLM           │
│  ralph_loop.py          → Expand variable space             │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    EXPLORATION                              │
│  dependency_graph.py    → Build variable DAG                │
│  space_explorer.py      → Systematic traversal              │
│  coverage_tracker.py    → Track what's been tested          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    GENERATION                               │
│  scenario_constructor.py → LLM builds concrete scenarios    │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                       OUTPUT                                │
│  Complete test scenarios with names, data, criteria         │
│  Coverage statistics (edge cases, pairwise, categories)     │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### CLI

```python
from create_scenarios_v2 import create_scenarios_combinatorial

config = {
    "description": "AI receptionist for a clinic...",
    "actions": ["find_patient", "schedule_appointment"],
    "entities": ["patient_name", "date_of_birth"]
}

for scenario in create_scenarios_combinatorial(config, max_scenarios=50):
    print(scenario)
```

### Web UI

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "XAI_API_KEY=your-key" > .env

# Run server
python server.py

# Open browser
open frontend/index.html
```

### Vercel Deployment

The app deploys to Vercel as serverless functions:

```bash
# Set environment variable
vercel env add XAI_API_KEY

# Deploy
vercel --prod
```

## Variable Types

### BooleanVariable (Tri-state)
```
no → unknown → yes
```
Used for: availability, capability, presence

### QuantitativeVariable
```
none → single → many
```
With optional variants: `single:correct`, `single:incorrect`

Used for: search results, matches, quantities

### EnumVariable
```
Custom states: ["routine", "urgent", "emergency"]
Edge cases marked: ["emergency"]
```
Used for: categories, statuses, levels

## Coverage Metrics

The system tracks:

| Metric | Description |
|--------|-------------|
| **Edge Case Coverage** | % of edge case values tested |
| **Pairwise Coverage** | % of variable pairs covered |
| **Category Coverage** | Which dimensions tested |
| **Total Scenarios** | Unique combinations generated |

Example output:
```
Coverage: 87% pairwise, 12/15 edge cases
Categories: entity_availability(3), patient_state(5), accessibility(2)
```

## Supported Domains

Auto-detected from keywords:

| Domain | Keywords | Field Model |
|--------|----------|-------------|
| Healthcare | clinic, patient, appointment, medical | name, dob, insurance, appointment_type |
| Restaurant | order, menu, drive-thru, food | order_items, customizations, payment |
| Customer Service | support, refund, complaint | customer_id, issue_type, priority |

## Configuration

```python
CombinatorialScenarioGenerator(
    config=config,
    domain_analysis=domain_analysis,
    api_key="xai-...",
    model="grok-4-1-fast-non-reasoning",
    ralph_iterations=3,        # Variable expansion rounds
    enable_research=True,      # LLM edge case research
    enable_ralph_loop=True,    # Variable space expansion
)
```

## Requirements

```
openai>=1.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
```

## Environment Variables

```bash
XAI_API_KEY=your-xai-api-key  # Required for LLM calls
```

## Example Output

Input:
```json
{
  "description": "Drive-through order assistant",
  "actions": ["take_order", "modify_order", "process_payment"],
  "entities": ["order_items", "customizations", "payment_method"]
}
```

Output:
```json
{
  "scenarioName": "Complex Order with Allergen Restrictions",
  "scenarioDescription": "Customer orders multiple items with specific allergen requirements. They need gluten-free buns, no dairy, and ask about cross-contamination procedures. Mid-order, they change their drink size and add an item.",
  "order_items": ["cheeseburger", "fries", "milkshake"],
  "customizations": ["gluten-free bun", "no cheese", "almond milk"],
  "payment_method": "card",
  "criteria": [
    "Agent confirms each allergen restriction explicitly",
    "Agent warns about cross-contamination risks",
    "Agent handles mid-order modifications smoothly",
    "Agent reads back complete order with all customizations",
    "Agent confirms total before payment"
  ]
}
```

## Why This Approach?

| Approach | Coverage | Coherence | Scalability |
|----------|----------|-----------|-------------|
| Manual test writing | Low | High | Poor |
| Pure LLM generation | Random | High | Good |
| **This system** | **Systematic** | **High** | **Good** |

The key insight: **use algorithms for coverage, LLMs for coherence**.

## License

MIT
