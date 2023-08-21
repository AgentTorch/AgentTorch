This repository contains code corresponding to the paper: "AgentTorch: a framework for agent-based modeling with automatic differentiation". The basic framework is given in the figure below:

![Agent Torch Framework](https://github.com/AgentTorch/AgentTorch/blob/eb567e3aa2ffcf6f74aafc1c4801267f3b8eada1/Figure2_Framework_final.pdf "AgentTorch Framework")

# 1. Installation

The easiest way to install AgentTorch is to install it from the PyPI repository
```
pip install torch_agent
```

# 2. Creating an ABM
Creating a new simulator using AgentTorch involves the following steps:
1. Defining the configuration: Here we define the variables and functions to be used in the simulator. In this module a Configurator object is to be created to which the variables and functions to be used in the simulator are added as properties. These are then used to instantiate a Runner object. An example for this can be found in "models/nca/simulator.py". 
2. Defining the trainer: This module loads the configuration, the various variables and functions that form the substeps and executes the main simulation and learning loop. Any learning related loss and optimization function need to be defined here. An example for this can be found in "models/nca/trainer_with_config.py"
3. Defining substeps: As described in the figure above, each simulation comprises of multiple substeps. Each substep comprises of the following four functions: observation, action, transition and reward. Each of these need to be defined in a separate module, using the base classes provided in "AgentTorch/substep.py". Since these functions need to be differentiable, we provide several differentiable utilities in AgentTorch/helpers/soft.py. These can be used to create differentiable variants of operations such as maximum, logical comparison etc. An example for substep definition can be found in "models/nca/substeps"
4. Using helpers: AgentTorch has several useful functions defined in "AgentTorch/helpers" that can be used in defining the various functions. These include utilities ranging from accessing different components of variables to commonly used initialization functions. 

# 3. Running examples
You can use the following commands for running each of the three examples presented in the paper:

```
python 
```
```
python 
```
```
python 
```
