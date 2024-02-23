Differentiable agent-based learning for million-scale populations - inside the body, around us and beyond.

https://github.com/AgentTorch/AgentTorch/assets/13482350/4c3f9fa9-8bce-4ddb-907c-3ee4d62e7148

# 1. Installation

## Download
The simplest way to install AgentTorch is from PyPi at:
```
pip install agent-torch
```

To get the latest version of AgentTorch, you can install it directly from git at:
```
pip install git+https://github.com/AgentTorch/AgentTorch
```

# 2. Setup

## Hardware
The code has been tested for macOS Catalina 10.1.7 and Ubuntu 22.04.2 LTS. Large-scale experiments are run using NVIDIA TITANX GPU and V100 GPU.

## Dependencies

> Step 1: Create a virtual environment `agent_torch_env`. We recommend using python 3.8 and pip as the install.
```
python3.8 -m venv agent_torch_env
source agent_torch_env/bin/activate
```
To install python3.8, follow these tutorials for [Mac](https://www.laptopmag.com/how-to/install-python-on-macos) and [Ubuntu](https://linux.how2shout.com/install-python-3-9-or-3-8-on-ubuntu-22-04-lts-jammy-jellyfish/) respectively. To install pip, follow these tutorials for [Mac](https://phoenixnap.com/kb/install-pip-mac) and [Ubuntu](https://linuxize.com/post/how-to-install-pip-on-ubuntu-20.04/) respectively. 


> Step 2: Install pytorch and pytorch geometric. We recommend using Pytorch 2.0 and corresponding Pytorch geometric bindings. We recommend following the guides for [offical pytorch install](https://pytorch.org/get-started/locally/) and [official pytorch-geometric install](https://github.com/pyg-team/pytorch_geometric#pytorch-20). We summarize the commands below:

Mac:
```
# CUDA is not available on MacOS, please use default package
pip install torch torchvision torchaudio
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
```

Ubuntu:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA}
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
```
where ${CUDA} is the CUDA version. We have tested our code on cu118. 


> Step 3: Install AgentTorch specific dependencies as below:
```
cd AgentTorch
pip3 install -r requirements.txt
```

# 3. AgentTorch overview
Creating a new simulator using AgentTorch involves the following steps:
1. Defining the configuration: Here we define the variables and functions to be used in the simulator. In this module a `Configurator` object is to be created to which the variables and functions to be used in the simulator are added as properties. These are then used to instantiate a `Runner` object. An example for this can be found in [nca_simulator.py](models/nca/simulator.py"). 
2. Defining the trainer: This module loads the configuration, the various variables and functions that form the substeps and executes the main simulation and learning loop. Any learning related loss and optimization function need to be defined here. An example for this can be found in "models/nca/trainer.py"
3. Defining substeps: As described in the figure above, each simulation comprises of multiple substeps. Each substep comprises of the following four functions: observation, action, transition and reward. Each of these need to be defined in a separate module, using the base classes for `SubstepObservation`, `SubstepTransition`, `SubstepPolicy` provided in [substep.py](AgentTorch/substep.py). Since these functions need to be differentiable, we provide several differentiable utilities in [helpers_soft.py](AgentTorch/helpers/soft.py). These can be used to create differentiable variants of operations such as maximum, logical comparison etc. An example for substep definition can be found in [nca_evolve.py](models/nca/substeps/evolve_cell/transition.py), [covid_quarantine.py](models/covid/substeps/quarantine/transition.py)
4. Using helpers: AgentTorch has several useful functions defined in [helpers](AgentTorch/helpers) that can be used in defining the various functions. These include library of utilities to support differentiability of substeps, loading public data sources such as from US census and, initialization of state properties and environment networks. For instance, [helpers_soft.py](AgentTorch/helpers/soft.py) include differentiable utilities and [helpers_general.py](AgentTorch/helpers/general.py) includes uitilies for data reading and writing.

A detailed code specific documentation is provided in [create model docs](docs/create.md)

# 4. Running examples
You can run a sample experiment with the following command:
```
cd models/nca
python trainer.py --c config.yaml
```
```
cd models/opinion
python trainer.py --c config.yaml
```

# 5. Starter Guide

## Generate and Interpret `config.yaml` file
An interactive notebook with step-by-step guide to define and understand a `config.yaml` is given in [config_example_docs](docs/examples/config/config_nca.ipynb).

## Build your own AgentTorch model
A step-by-step guide to start a new AgentTorch project is given in [starter documentation](docs/create.md)

# 6. Issues
The AgentTorch project is under active development and are continually fixing issues. Please feel free to leave a comment at [Troubleshooting issues](https://github.com/AgentTorch/AgentTorch/issues/1)

## Citation
If you use this project or code in your work, please cite it using the following BibTeX entry:

@inproceedings{chopra2024framework,
  title = {A Framework for Learning in Agent-Based Models},
  author = {Chopra, Ayush and Subramanian, Jayakumar and Krishnamurthy, Balaji and Raskar, Ramesh},
  booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multi-agent Systems},
  year = {2024},
  organization = {International Foundation for Autonomous Agents and Multiagent Systems},
}
