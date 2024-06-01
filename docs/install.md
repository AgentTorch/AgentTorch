# Installation Guide

To install the project, run:

```sh
> pip install git+https://github.com/agenttorch/agenttorch
```

To run some models, you may need to separately install their dependencies. These
usually include [`torch`](https://pytorch.org/get-started/locally/),
[`torch_geometric`](https://github.com/pyg-team/pytorch_geometric#pytorch-20),
and [`osmnx`](https://osmnx.readthedocs.io/en/stable/installation.html).

For the sake of completeness, a summary of the commands required is given below:

```sh
# on macos, cuda is not available:
> pip install torch torchvision torchaudio
> pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
> pip install osmnx

# on ubuntu, where ${CUDA} is the cuda version:
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA}
> pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
> pip install osmnx
```

## Hardware

The code has been tested on macOS Catalina 10.1.7 and Ubuntu 22.04.2 LTS.
Large-scale experiments are run using Nvidia's TitanX and V100 GPUs.
