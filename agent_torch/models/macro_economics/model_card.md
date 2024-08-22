# Labor Market Model Card for AgentTorch

## Model Details
- **Name:** Labor Market Model for AgentTorch
- **Version:** 0.4.0
- **Type:** Agent-based model (ABM) environment
- **Framework:** AgentTorch v0.4.0
- **Execution Mode:** Archetype (Large Language Model Archetype)
- **Location:** New York City

## Intended Use
- **Primary Use:** Simulate labor market dynamics, including employment decisions, wage changes, and unemployment rates in a large urban population
- **Intended Users:** Economists, policymakers, and researchers studying labor markets and macroeconomic trends

## Model Architecture
- **Agent Population:** 8.4 million synthetic agents representing New York City
- **Simulation Duration:** 80 weeks (configurable)
- **Time Steps:** 20 steps per episode, 4 substeps per step
- **Model Components:** Agents (consumers), Jobs, Environment (macro indicators)

### Agent Properties
- **Static:** ID, age, gender, ethnicity, region
- **Dynamic:** assets, monthly income, work propensity, will_work status

### Environment Variables
- Unemployment rates (city-wide and borough-specific)
- Interest rate
- Inflation rate
- Price of essential goods
- Labor force participation

### Jobs
- Two job types (JobA and JobB) with properties:
  - Productivity
  - Wage

## Input Data
- Age distribution file
- Gender distribution file
- Ethnicity distribution file
- Region (borough) distribution file
- Initial unemployment claims data
- COVID-19 case data

## Model Parameters
- **Hourly Wage:** 1.0 (initial value, adjustable)
- **Natural Unemployment Rate:** 5%
- **Natural Interest Rate:** 5%
- **Target Inflation Rate:** 2%
- **Tax Brackets and Rates:** Multiple brackets with rates from 10% to 35%

## Simulation Substeps
1. **Agents Earning:**
   - Determine work and consumption propensity
   - Update assets and monthly income
2. **Agents Consumption:**
   - Update assets based on consumption decisions
3. **Labor Market:**
   - Update unemployment rates and wages
4. **Financial Market:**
   - Update interest rates and inflation

## Key Features
- Integration with Large Language Models for agent decision-making
- Borough-specific unemployment tracking
- Dynamic wage and price adjustments
- Consideration of COVID-19 cases in decision-making

## Output Data
- Monthly unemployment rates (city-wide and borough-specific)
- Wage trends
- Labor force participation rates
- Individual agent income and consumption patterns

## Calibration
- **Calibration Mode:** Enabled
- **Learning Parameters:**
  - Learning rate: 0.002
  - Beta values: [0.5, 0.5]
  - LR gamma: 0.9999

## Technical Specifications
- **Programming Language:** Python
- **Dependencies:** AgentTorch v0.4.0 framework, PyTorch, OpenAI API (for LLM integration)
- **Compute Requirements:** CPU (configurable to GPU for large-scale simulations)

## Limitations
- Simplified representation of job market (only two job types)
- Assumes uniform productivity across job types

## Ethical Considerations
- Privacy considerations when using demographic data for agent initialization
- Potential for biases in LLM-generated agent behaviors
- Responsible use required when informing policy decisions based on model outputs

## References
- AgentTorch GitHub repository: [github.com/AgentTorch/AgentTorch/models/macro_economics](https://github.com/AgentTorch/AgentTorch/models/macro_economics)
