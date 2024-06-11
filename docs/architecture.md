# Framework Architecture

This document details the architecture of the AgentTorch project, explains all
the building blocks involved and points to relevant code implementation and
examples.

---

A high-level overview of the AgentTorch Python API is provided by the following
block diagram:

![Agent Torch Block Diagram](https://github.com/agenttorch/agenttorch/assets/34235681/3ecadf85-d949-44a6-92b3-5dbfa337fb20)

The AgentTorch Python API provides developers with the ability to
programmatically create and configure LPMs. This functionality is detailed
further in the following sections.

#### Runtime

The AgentTorch runtime is composed of three essential blocks: the configuration,
the registry, and the runner.

The configuration holds information about the environment, initial and current
state, agents, objects, network metadata, as well as substep definitions. The
'configurator' is defined in
[`config.py`](https://github.com/agenttorch/agenttorch/agent_torch/config.py).

The registry stores all registered substeps, and helper functions, to be called
by the runner. It is defined in
[`registry.py`](https://github.com/agenttorch/agenttorch/agent_torch/registry.py).

The runner accepts a registry and configuration, and exposes an API to execute
all, single or multiple episodes/steps in a simulation. It also maintains the
state and trajectory of the simulation across these episodes. It is defined in
[`runner.py`](https://github.com/agenttorch/agenttorch/agent_torch/runner.py),
and the substep execution and optimization logic is part of
[`controller.py`](https://github.com/agenttorch/agenttorch/agent_torch/controller.py).

#### Data

The data layer is composed of any raw, domain-specific data used by the model
(such as agent or object initialization data, environment variables, etc.) as
well as the files (YAML or Python code) used to configure the model. An example
of domain-specific data for a LPM can be found in the
[`models/covid/data`](/models/covid/data) folder. The configuration for the same
model can be found in [`config.yaml`](/models/covid/config.yaml).

#### Base Classes

The base classes of `Agent`, `Object` and `Substep` form the foundation of the
simulation. The agents defined in the configuration learn and interact with
either their environment, other agents, or objects through substeps. Substeps
are executed in the order of their definition in the configuration, and are
split into three parts:
[`SubstepObservation`](https://github.com/agenttorch/agenttorch/agent_torch/substep.py#L10),
[`SubstepAction`](https://github.com/agenttorch/agenttorch/agent_torch/substep.py#L27)
and
[`SubstepTransition`](https://github.com/agenttorch/agenttorch/agent_torch/substep.py#45).

- A `SubstepObservation` is defined to observe the state, and pick out those
  variables that are of use to the current substep.
- A `SubstepAction`, sometimes called a `SubstepPolicy`, decides the course of
  action based on the observations made, and then simulates that action.
- A `SubstepTransition` outputs the updates to be made to state variables based
  on the action taken in the substep.

An example of a substep will all three parts defined can be found
[here](/models/covid/substeps/quarantine).

#### Domain Extended Classes

These classes are defined by the developer/user configuring the model, in
accordance with the domain of the model. For example,
[in the COVID model](/models/covid), citizens of the populace
[are defined as `Agents`](/models/covid/config.yaml#L189), and `Transmission`
and `Quarantine` [as substeps](/models/covid/substeps).
