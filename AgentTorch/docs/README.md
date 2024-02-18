# Starter Guide: Build your own AgentTorch simulation project

## Step 0: Make sure you have correctly installed `AgentTorch` and dependencies. 
Please follow the [Installation guide here](../README.md)

## Step 1: Create a project for your simulator `my_new_model`
```
mkdir models/my_new_model
cd models/my_new_model
touch __init__.py
 
touch simulator.py # will contain the `config` for your simulator

touch trainer.py # # will contain the `runner` and execution loop
```

## Step 2: Create a config using `Configurator`

* Config is used to describe the simulator state comprising agents, objects, interaction environments, execution metadata and register substeps that execute in each step. 
* The config is generated using the [Configurator](../AgentTorch/config.py) API and maintained in `models/my_new_model/simulator.py`. 
* A step-by-step guide on how to generate and intepret a config for your project given in [example_config_notebook](examples/config/config_nca.ipynb). Further, you may see an sample config generator for `models/nca` in [configure_nca](../models/nca/simulator.py).

## Step 3: Create substeps

Substeps are maintained in the `my_new_model/substeps`. The directory structure for a substep `substep_1` (implemented in `my_new_model/substeps/my_substep_1`) is oprgamized as below:

```
mkdir my_new_model/substeps
cd my_new_model/substeps_1
touch transition.py
touch observation.py
touch action.py
```

* Each substep (eg: `my_new_model/substeps/my_substep_1`) has three types of modules `observation`, `action` and `transition`. Each `observation` module is implemented in `my_substep_1/observation.py` and extends [SubstepObservation](../AgentTorch/substep.py). 
* Each `action` module is implemented in `my_substep_1/action.py` and extends [SubstepAction](../AgentTorch/substep.py). Each `transition` module is implemented in `my_substep_1/transition.py` and extends [SubstepTransition](../AgentTorch/substep.py). 
* Each substep model is invoked from the `forward` method for the corresponding substep and has complete flexibility when organizing code there. 
* AgentTorch provides multiple helpers in [AgentTorch/helpers](AgentTorch/helpers/) to streamline execution of the substeps. Further, [AgentTorch/helpers/soft.py](../AgentTorch/helpers/soft.py) provide several differentiable utilities to ensure gradient propogation through substeps. For example, some of these helpers can be invoked as following:

```
from AgentTorch.helpers import discrete_sample, logical_and, get_by_path, read_config, compare

# discrete_sample helps to do a **differentiable** sample from a categorical distribution which is otherwise a non-differentiable operation
# read_config helps load a `config.yaml` file
# get_by_path helps read a property inside the `state` of the simulator
# logical_and helps to do a **differentiable** boolean operation which is otherwise non-differentiable in Pytorch
# compare helps to do a **differentiable** comparison of two integers which is otherwise non-differentiable in Pytorch
```

* Some example substeps are implemented in [evolve_cell](../models/nca/substeps/evolve_cell/) from NCA (case-study 1)
[quarantine](../models/covid/substeps/quarantine/transition.py) from COVID (case-study 2) and [purchase_product](../models/opinion/substeps/purchase_product/) from opinion dynamics (case-study 3).


## Step 4: Create runner

* AgentTorch simulations are executed using an instance of the [Runner](../AgentTorch/runner.py) class. For each project, this is maintained in `models/my_new_model/trainer.py`.
* Below is an example pseudocode to use the runner for executing the simulation logic described in `models/my_new_model/trainer.py`. An implemented runner for NCA is given in [models/nca/simulator.py](../models/nca/simulator.py). 

```
## This is pseudocode for defining a runner

from my_new_model.simulator import my_config
from AgentTorch import Runner
import torch

runner = Runner(my_config)

# initializes the runner
runner.init() 

# runner stores the trainable simulation parameters
optimizer = torch.optim.Adam(runner.parameters()) 

num_episodes = my_config.get('simulation_metadata.num_episodes') # reads properties from config (see example in part 2 above)
num_steps_per_episode = my_config.get('simulation_metadata.num_steps_per_episode')

all_outputs = []

for epi in range(config.get('simulation_metadata.num_episodes')):

    # reinitializes the runner at the start of each new episode
    runner.reset() 
    
    # execute one episode of simulation
    runner.step(num_steps_per_episode)

    # runner trajectory stores the compute graph and tracks simulation output.
    all_outputs.append(runner.trajectory)
```  


## Step 5: Reach out if any issues or would like to contribute.
We are glad you considered AgentTorch to build your simulation project. If you encounter any challenge or would like to contribute, please do [create an issue here](https://github.com/AgentTorch/AgentTorch/issues) and we would reach out as soon as possible.