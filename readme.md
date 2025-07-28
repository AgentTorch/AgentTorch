<h1 align="center">
  <a href="https://media.mit.edu/projects/ai-lpm" target="_blank">
    Large Population Models
  </a>
</h1>

<p align="center">
  <strong>making complexity simple</strong><br>
  differentiable learning over millions of autonomous agents
</p>

<p align="center">

  <a href="https://agenttorch.github.io/AgentTorch/" target="_blank">
    <img src="https://img.shields.io/badge/Quick%20Introduction-green" alt="Documentation" />
  </a>
  <a href="https://twitter.com/intent/follow?screen_name=ayushchopra96" target="_blank">
    <img src="https://img.shields.io/twitter/follow/ayushchopra96?style=social&label=Get%20in%20Touch" alt="Get in Touch" />
  </a>
  <a href="https://join.slack.com/t/largepopulationmodels/shared_invite/zt-2jalzf9ki-n9nXG5FryVSMaPmEL7Wm2w" target="_blank">
     <img src="https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white" alt="Join Us"/>
  </a>
</p>

## Overview

Many grand challenges like climate change and pandemics emerge from complex interactions of millions of individual decisions. While LLMs and agent simulations excel at individual behavior, they can't model these intricate societal dynamics. Enter Large Population Models (LPMs): a new paradigm to simulate millions of interacting entities, capturing behaviors at collective scale. It   exponentially to understand the ripple effects of countless decisions. 

AgentTorch, our open-source platform, makes building and running massive LPMs accessible. It's optimized for GPUs, allowing efficient simulation of large-scale systems. Think PyTorch, but for large-scale agent-based simulations. AgentTorch LPMs have four design principles:

- **Scalability**: AgentTorch models can simulate large-scale populations in
  seconds on commodity hardware.
- **Differentiability**: AgentTorch models can differentiate through simulations
  with stochastic dynamics and conditional interventions, enabling
  gradient-based optimization.
- **Composition**: AgentTorch models can compose with deep neural networks (eg:
  LLMs), mechanistic simulators (eg: mitsuba) or other LPMs. This helps describe
  agent behavior using LLMs, calibrate simulation parameters and specify
  expressive interaction rules.
- **Generalization**: AgentTorch helps simulate diverse ecosystems - humans in
  geospatial worlds, cells in anatomical worlds, autonomous avatars in digital
  worlds.

Our research is making an impact - winning awards at AI conferences and being used in real-world applications.
Learn more [here](https://media.mit.edu/projects/ai-lpm).

https://github.com/AgentTorch/AgentTorch/assets/13482350/4c3f9fa9-8bce-4ddb-907c-3ee4d62e7148

## Installation
Install the most recent version from source using `pip`:

```sh
> pip install git+https://github.com/agenttorch/agenttorch
```

> Some models require extra dependencies that have to be installed separately.
> For more information regarding this, as well as the hardware the project has
> been run on, please see [`docs/install.md`](docs/install.md).

Alternately, the easiest way to install AgentTorch (v0.6.0) is from pypi:
```
> pip install agent-torch
```

> AgentTorch is meant to be used in a Python >=3.9 environment. If you have not
> installed Python 3.9, please do so first from
> [python.org/downloads](https://www.python.org/downloads/).

## Getting Started

The following section depicts the usage of existing models and population data
to run simulations on your machine. It also acts as a showcase of the Agent
Torch API.

### Executing a Simulation

```py
# re-use existing models and population data easily
from agent_torch.examples.models import movement
from agent_torch.populations import astoria
from agent_torch.core.environment import envs

runner = envs.create(model=covid, population=astoria) # create simulation and init runner

sim_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
num_episodes = runner.config["simulation_metadata"]["num_episodes"]

for epi in range(num_episodes):
  runner.step(sim_steps)

  runner.reset() # re-initializes the sim parameters for new episode

```

## License
Copyright (c) 2023-2025 Ayush Chopra

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This means:
- You can freely use, modify, and distribute this software
- If you use this software to provide services over a network, you must make your source code available to users
- Any modifications or derivative works must also be licensed under AGPL-3.0
- You must give appropriate credit and indicate any changes made
- For full terms, see [LICENSE.md](LICENSE.md) file in this repository

For commercial licensing options or inquiries about using this software in a proprietary product, please reach out to request a license wavier.

## Guides and Tutorials

### Understanding the Framework

A detailed explanation of the architecture of the Agent Torch framework can be
found [here](docs/architecture.md).

### Creating a Model

A tutorial on how to create a simple predator-prey model can be found in the
[`tutorials/`](docs/tutorials/) folder.

### Prompting Agent Behavior with LLM Archetypes
```py
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.core.llm.backend import LangchainLLM
from agent_torch.populations import NYC
user_prompt_template = "Your age is {age} {gender},{unemployment_rate} the number of COVID cases is {covid_cases}."
# Using Langchain to build LLM Agents
agent_profile = "You are a person living in NYC. Given some info about you and your surroundings, decide your willingness to work. Give answer as a single number between 0 and 1, only."
llm_langchain = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo"
)
# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype = Archetype(n_arch=7)
# Create an object of the Behavior class
# You have options to pass any of the above created llm objects to the behavior class
# Specify the region for which the behavior is to be generated. This should be the name of any of the regions available in the populations folder.
earning_behavior = Behavior(
    archetype=archetype.llm(llm=llm_langchain, user_prompt=user_prompt_template), region=NYC
)
kwargs = {
    "month": "January",
    "year": "2020",
    "covid_cases": 1200,
    "device": "cpu",
    "current_memory_dir": "/path-to-save-memory",
    "unemployment_rate": 0.05,
}
output = earning_behavior.sample(kwargs)
```

### Contributing to Agent Torch

Thank you for your interest in contributing! You can contribute by reporting and
fixing bugs in the framework or models, working on new features, creating new models, or by writing documentation for the project.

Take a look at the [contributing guide](docs/contributing.md) for instructions
on how to setup your environment, make changes to the codebase, and contribute
them back to the project.
