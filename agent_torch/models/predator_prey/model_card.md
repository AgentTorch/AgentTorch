# Predator-Prey Model Card for AgentTorch

## Model Details
- **Name:** Predator-Prey Model for AgentTorch
- **Version:** 0.4.0
- **Type:** Agent-based model (ABM) environment
- **Framework:** AgentTorch v0.4.0
- **Execution Mode:** Heuristic

## Intended Use
- **Primary Use:** Simulate ecological dynamics in a predator-prey ecosystem with grass as a food source
- **Intended Users:** Ecologists, biologists, environmental scientists, and researchers studying population dynamics

## Model Architecture
- **Environment Size:** 18 x 25 grid
- **Agent Population:** 
  - 40 Predators
  - 80 Prey
- **Objects:** 450 Grass patches
- **Simulation Duration:** 3 episodes, 20 steps per episode, 4 substeps per step

## Components

### Agents
1. **Predators:**
   - Properties: coordinates, energy, stride_work
   - Behaviors: move, hunt prey

2. **Prey:**
   - Properties: coordinates, energy, stride_work, nutritional_value
   - Behaviors: move, eat grass

### Objects
- **Grass:**
  - Properties: coordinates, growth_stage, regrowth_time, growth_countdown, nutritional_value

### Environment Variables
- Bounds (18 x 25 grid)

## Simulation Substeps
1. **Move:** Both predators and prey move
2. **Eat:** Prey consume grass
3. **Hunt:** Predators hunt prey
4. **Grow:** Grass regrows

## Input Data
- Predator initial coordinates
- Prey initial coordinates
- Grass patch coordinates
- Grass growth stage
- Grass growth countdown

## Model Parameters
- **Predator:**
  - Initial energy: Random between 30-100
  - Stride work (energy lost per step): 1 (learnable)

- **Prey:**
  - Initial energy: Random between 40-100
  - Stride work (energy lost per step): 5 (learnable)
  - Nutritional value: 20 (learnable)

- **Grass:**
  - Regrowth time: 100 steps (learnable)
  - Nutritional value: 7 (learnable)

## Key Features
- Dynamic energy levels for predators and prey
- Grass growth and consumption mechanics
- Spatial interactions on a 2D grid
- Learnable parameters for ecological balance

## Output Data
- Agent positions over time
- Population dynamics of predators and prey
- Grass growth patterns
- Energy levels of agents

## Technical Specifications
- **Programming Language:** Python
- **Dependencies:** AgentTorch v0.4.0 framework, PyTorch
- **Compute Requirements:** CPU (as specified in config)

## Limitations
- Simplified 2D grid environment
- Fixed number of agents and grass patches
- Does not account for factors like reproduction, aging, or environmental changes

## Ethical Considerations
- Model simplifications may not accurately represent real-world ecosystems
- Results should be interpreted cautiously when applied to conservation or environmental policy

## References
- AgentTorch GitHub repository: [github.com/AgentTorch/AgentTorch/models/predator_prey](https://github.com/AgentTorch/AgentTorch/models/predator_prey)
