# AgentTorch-Beckn Solar Model

## Overview

AgentTorch is a differentiable learning framework that enables you to run simulations with
over millions of autonomous agents. [Beckn](https://becknprotocol.io) is a protocol that
enables the creation of open, peer-to-peer decentralized networks for pan-sector economic
transactions.

This model integrates Beckn with AgentTorch, to simulate a solar energy network in which
households in a locality can decide to either buy solar panels and act as providers of
solar energy, or decide to use the energy provided by other households instead of
installing solar panels themselves.

> ![A visualization of increase in net solar energy used per street](./visualization.gif)
>
> A visualization of increase in net solar energy used per street.

## Mapping Beckn Protocol to AgentTorch

### 1. Network

The participants in the Beckn network (providers, customers and gateways) are considered
agents that interact with each other.

### 2. Operations

The following operations are simulated as substeps:

1. a customer will `search` and `select` a provider
2. the customer will `order` from the provider
3. the provider will `fulfill` the order
4. the customer will `pay` for the work done
5. the provider will `restock` their solar energy

Each of the substeps' code (apart from #5) is taken as-is from the
[AgentTorch Beckn integration](https://github.com/AgentTorch/agent-torch-beckn).

> Note that while Beckn's API calls are asynchronous, the simulation assumes they are
> synchronous for simplicity.

### 3. Data

The data for this example model is currently sourced from various websites, mostly from
[data.boston.gov](http://data.boston.gov/). However, the data should actually come from
the Beckn Protocol's implementation of a solar network.

## Running the Model

To run the model, clone the github repository first:

```python
# git clone --depth 1 --branch solar https://github.com/AgentTorch/agent-torch-beckn solar-netowkr
```

Then, setup a virtual environment and install all dependencies:

```python
# cd solar-network/
# python -m venv .venv/bin/activate
# . .venv/bin/activate
# pip install -r requirements.txt
```

Once that is done, you can edit the configuration ([`config.yaml`](../config.yaml)), and
change the data used in the simulation by editing the simulation's data files
([`data/simulator/{agent}/{property}.csv`](../data/simulator/)).

Then, open Jupyter Lab and open the `main.ipynb` notebook, and run all the cells.

```python
# pip install jupyterlab
# jupyter lab
```

## Todos

- Add more visualizations (plots/graphs/heatmaps/etc.)
- Improve the data used for the simulation, reduce the number of random values.
- Include and run a sample beckn instance to pull fake data from.
