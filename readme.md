<h1 align="center">
  <a href="https://lpm.media.mit.edu/" target="_blank">
    Large Population Models
  </a>
</h1>

<p align="center">
  <strong>making complexity simple</strong><br>
  differentiable learning over millions of autonomous agents
</p>

<p align="center">
  <a href="https://github.com/AgentTorch/AgentTorch/blob/HEAD/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="Released under the MIT license." />
  </a>

  <a href="https://web.media.mit.edu/~ayushc/motivation.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Quick%20Introduction-green" alt="Quick Introduction" />
  </a>
  <a href="https://twitter.com/intent/follow?screen_name=ayushchopra96" target="_blank">
    <img src="https://img.shields.io/twitter/follow/ayushchopra96?style=social&label=Get%20in%20Touch" alt="Get in Touch" />
  </a>
</p>

Large Population Models (LPMs) are grounded in state-of-the-art AI research, a
summary of which can be found
[here](https://web.media.mit.edu/~ayushc/motivation.pdf).

AgentTorch LPMs have four key features:

- **Scalability**: AgentTorch models can simulate country-size populations in
  seconds on commodity hardware.
- **Differentiability**: AgentTorch models can differentiate through simulations
  with stochastic dynamics and conditional interventions, enabling
  gradient-based learning.
- **Composition**: AgentTorch models can compose with deep neural networks (eg:
  LLMs), mechanistic simulators (eg: mitsuba) or other LPMs. This helps describe
  agent behavior, calibrate simulation parameters and specify expressive
  interaction rules.
- **Generalization**: AgentTorch helps simulate diverse ecosystems - humans in
  geospatial worlds, cells in anatomical worlds, autonomous avatars in digital
  worlds.

AgentTorch is building the future of decision engines - inside the body, around
us and beyond!

## Installation

Install the framework using `pip`, like so:

```sh
> pip install git+https://github.com/agenttorch/agenttorch
```

> You may also want to install
> [`torch`](https://pytorch.org/get-started/locally/) and
> [`torch_geometric`](https://github.com/pyg-team/pytorch_geometric#pytorch-20).

## Getting Started

The following section depicts the usage of existing models and population data
to run simulations on your machine. It also acts as a showcase of the Agent
Torch API.

A Jupyter Notebook containing the below examples can be found
[here](docs/tutorials/using-models/walkthrough.ipynb).

### Executing a Simulation

```py
# re-use existing models and population data easily
from AgentTorch.models import disease
from AgentTorch.populations import new_zealand

# use the executor to plug-n-play
from AgentTorch.execute import Executor

simulation = Executor(disease, new_zealand)
simulation.execute()
```

### Using Gradient-Based Learning

```py
# agent_"torch" works seamlessly with the pytorch API
from torch.optim import SGD

# create the simulation
# ...

# create an optimizer for the learnable parameters
# in the simulation
optimizer = SGD(simulation.parameters())

# learn from each "episode" and run the next one
# with optimized parameters
for i in range(episodes):
	optimizer.zero_grad()

	simulation.execute()
	optimizer.step()
	simulation.reset()
```

### Talking to the Simulation

```py
from AgentTorch.LLM.qa import SimulationAnalysisAgent, load_state_trace

# create the simulation
# ...

state_trace = load_state_trace(simulation)
analyzer = SimulationAnalysisAgent(simulation, state_trace)

# ask questions regarding the simulation
analyzer.query("How are stimulus payments affecting disease?")
analyzer.query("Which age group has the lowest median income, and how much is it?")
```

## Guides and Tutorials

### Understanding the Framework

A detailed explanation of the architecture of the Agent Torch framework can be
found [here](docs/architecture.md).

### Creating a Model

A tutorial on how to create a simple predator-prey model can be found in the
[`tutorials/`](docs/tutorials/) folder.

### Contributing to Agent Torch

Thank you for your interest in contributing! You can contribute by reporting and
fixing bugs in the framework or models, working on new features for the
framework, creating new models, or by writing documentation for the project.

Take a look at the [contributing guide](contributing.md) for instructions on how
to setup your environment, make changes to the codebase, and contribute them
back to the project.

## Citation

If you use this project or code in your work, please cite it using the following
BibTex entry, which can also be found in [`citation.bib`](citation.bib).

```bib
@inproceedings{chopra2024framework,
  title = {A Framework for Learning in Agent-Based Models},
  author = {Chopra, Ayush and Subramanian, Jayakumar and Krishnamurthy, Balaji and Raskar, Ramesh},
  booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multi-agent Systems},
  year = {2024},
  organization = {International Foundation for Autonomous Agents and Multiagent Systems},
}
```
