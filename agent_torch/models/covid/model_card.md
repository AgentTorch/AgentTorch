# COVID-19 Environment Model Card for AgentTorch

## Model Details
- **Name:** COVID-19 Environment for AgentTorch
- **Version:** 0.4
- **Type:** Agent-based model (ABM) environment
- **Framework:** AgentTorch
- **Execution Mode:** Heuristic
- **Location:** New York City

## Intended Use
- **Primary Use:** Simulate the spread of COVID-19 and its impact on health outcomes in a large urban population
- **Intended Users:** Researchers, policymakers, and public health officials

## Model Architecture
- **Agent Population:** 8.4 million synthetic agents representing New York City
- **Simulation Duration:** Configurable (e.g., 3 weeks in the sample configuration)
- **Time Steps:** 21 steps per episode, 2 substeps per step (configurable)
- **Disease Model:** SEIRM (Susceptible, Exposed, Infected, Recovered, Mortality)

### Agent Properties
- **Static:** age, id
- **Dynamic:** disease stage, infected time, quarantine status, test status

### Environment Variables
- **Disease Dynamics:** 
  - Infectiousness by disease stage
  - Age-based susceptibility
  - Transition times between stages (e.g., exposed to infected: 5 days)
- **Intervention Measures:**
  - Quarantine duration: 12 days
  - Test ineligibility period: 2 days
  - Test result delay: 3 days

### Network
- Infection network based on mobility data across the city

## Calibration
- **Calibration Mode:** Enabled
- **Learning Parameters:**
  - Learning rate: 5e-3 (adjustable)
  - Beta values: [0.5, 0.5]
  - LR gamma: 0.9999

## Input Data
- Age distribution file for NYC population
- Initial disease stages file
- Mobility network file for NYC

## Model Parameters
- **R0:** Learnable parameter, initialized to 4.75
- **Mortality Rate:** Learnable parameter, initialized to 0.12
- **Quarantine Compliance:** 70% initial compliance, 10% break probability
- **Testing:**
  - Compliance probability: 95%
  - False positive rate: 30%
  - True positive rate: 80%

## Simulation Substeps
1. **Transmission:** 
   - New infections based on network interactions
   - Isolation decisions with LLM alignment capability
2. **Disease Progression:** 
   - SEIRM state transitions
   - Mortality calculations

## Output Data
- Daily infected cases
- Daily deaths
- Disease stage distribution

## Technical Specifications
- **Programming Language:** Python
- **Dependencies:** AgentTorch framework, PyTorch
- **Compute Requirements:** Scalable from CPU to GPU for large-scale simulations

## Limitations
- While the model is designed for NYC, parameters may need adjustment for different urban environments
- Behavioral dynamics are simplified and may not capture all nuances of real-world decision-making
- The model assumes some constant parameters for interventions (e.g., quarantine duration) which may vary in reality

## Ethical Considerations
- Privacy considerations when using demographic and health data for a large population
- Potential for biases in infection networks and testing probabilities across diverse NYC neighborhoods
- Responsible use required when informing city-wide policy decisions

## References
- AgentTorch GitHub repository: [github.com/AgentTorch/AgentTorch/models/covid](https://github.com/AgentTorch/AgentTorch/models/covid)
