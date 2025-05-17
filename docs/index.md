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

Many grand challenges like climate change and pandemics emerge from complex interactions of millions of individual decisions. While LLMs and AI agents excel at individual behavior, they can't model these intricate societal dynamics. Enter Large Population Models LPMs: a new AI paradigm simulating millions of interacting agents simultaneously, capturing collective behaviors at societal scale. It's like scaling up AI agents exponentially to understand the ripple effects of countless decisions.

AgentTorch, our open-source platform, makes building and running these massive simulations accessible. It's optimized for GPUs, allowing efficient simulation of entire cities or countries. Think PyTorch, but for large-scale agent-based simulations. AgentTorch LPMs have four design principles:

- **Scalability**: AgentTorch models can simulate country-size populations in
  seconds on commodity hardware.
- **Differentiability**: AgentTorch models can differentiate through simulations
  with stochastic dynamics and conditional interventions, enabling
  gradient-based learning.
- **Composition**: AgentTorch models can compose with deep neural networks (eg:
  LLMs), mechanistic simulators (eg: mitsuba) or other LPMs. This helps describe
  agent behavior using LLMs, calibrate simulation parameters and specify
  expressive interaction rules.
- **Generalization**: AgentTorch helps simulate diverse ecosystems - humans in
  geospatial worlds, cells in anatomical worlds, autonomous avatars in digital
  worlds.

LPMs are already making real-world impact. They're being used to help immunize millions of people by optimizing vaccine distribution strategies, and to track billions of dollars in global supply chains, improving efficiency and reducing waste. Our long-term goal is to "re-invent the census": built entirely in simulation, captured passively and used to protect country-scale populations. Our research is early but actively making an impact - winning awards at AI conferences and being deployed across the world. Learn more about LPMs [here](https://lpm.media.mit.edu/research.pdf).

AgentTorch is building the future of decision engines - inside the body, around us and beyond!

https://github.com/AgentTorch/AgentTorch/assets/13482350/4c3f9fa9-8bce-4ddb-907c-3ee4d62e7148

## Installation

The easiest way to install AgentTorch (v0.4.0) is from pypi:
```
> pip install agent-torch
```

> AgentTorch is meant to be used in a Python 3.9 environment. If you have not
> installed Python 3.9, please do so first from
> [python.org/downloads](https://www.python.org/downloads/).

Install the most recent version from source using `pip`:

```sh
> pip install git+https://github.com/agenttorch/agenttorch
```

> Some models require extra dependencies that have to be installed separately.
> For more information regarding this, as well as the hardware the project has
> been run on, please see [`docs/install.md`](docs/install.md).

## Getting Started

The following section depicts the usage of existing models and population data
to run simulations on your machine. It also acts as a showcase of the Agent
Torch API.

A Jupyter Notebook containing the below examples can be found
[here](docs/tutorials/using-models/walkthrough.ipynb).

### Executing a Simulation

```py
# re-use existing models and population data easily
from agent_torch.models import covid
from agent_torch.populations import astoria

# use the executor to plug-n-play
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

# agent_"torch" works seamlessly with the pytorch API
from torch.optim import SGD

loader = LoadPopulation(astoria)
simulation = Executor(model=covid, pop_loader=loader)

simulation.init(SGD)
simulation.execute()
```

## Guides and Tutorials

### Understanding the Framework

A detailed explanation of the architecture of the Agent Torch framework can be
found [here](architecture.md).

### Building Simulations with the Configuration API

Learn how to create and customize agent-based simulations using AgentTorch's powerful Configuration API. This tutorial walks you through:
- Creating agents with custom properties
- Defining environment variables and networks
- Building simulation substeps with policies and transitions
- Best practices for organizing your simulation

[Get started with the Config API tutorial →](tutorials/config_api/index.md)

### Optimizing Performance with Vectorized Operations

Learn how to leverage AgentTorch's vectorized operations for high-performance simulations:
- Understanding vectorized vs standard operations
- Converting standard functions to vectorized implementations
- Using batched processing for large populations
- Performance optimization techniques

[Learn about vectorized operations →](tutorials/vectorized_operations/index.md)

### Creating a Model

A tutorial on how to create a simple predator-prey model can be found in the
[`tutorials/`](tutorials/) folder.

### Prompting Collective Behavior with LLM Archetypes

```py
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.core.llm.backend import LangchainLLM
from agent_torch.populations import NYC

user_prompt_template = "Your age is {age} {gender},{unemployment_rate} the number of COVID cases is {covid_cases}."

# Using Langchain to build LLM Agents
agent_profile = "You are a person living in NYC. Given some info about you and your surroundings, decide your willingness to work. Give answer as a single number between 0 and 1, only."
llm_langchian = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo"
)

# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype = Archetype(n_arch=7)

# Create an object of the Behavior class
# You have options to pass any of the above created llm objects to the behavior class
# Specify the region for which the behavior is to be generated. This should be the name of any of the regions available in the populations folder.
earning_behavior = Behavior(
    archetype=archetype.llm(llm=llm_langchian, user_prompt=user_prompt_template), region=NYC
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
fixing bugs in the framework or models, working on new features for the
framework, creating new models, or by writing documentation for the project.

Take a look at the [contributing guide](contributing.md) for instructions
on how to setup your environment, make changes to the codebase, and contribute
them back to the project.

## Impact

> **AgentTorch models are being deployed across the globe.**

![Impact](media/impact.png)
