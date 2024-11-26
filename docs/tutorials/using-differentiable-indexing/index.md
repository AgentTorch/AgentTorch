# MaskedTensor Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [API Reference](#api-reference)
5. [Implementation Details](#implementation-details)
6. [Examples](#examples)

## Introduction

MaskedTensor is a component of the AgentTorch library that implements differentiable indexing for one-dimensional tensors. It allows gradients to flow through indexing operations, enabling more flexible and powerful neural network architectures that involve learnable indices.

Key features:
- Differentiable indexing
- Support for basic arithmetic operations (add, multiply)
- Compatible with PyTorch's autograd system
- Designed for use with one-dimensional tensors

## Installation

To use MaskedTensor, you need to install the AgentTorch library. You can do this using pip:

```bash
pip install agent-torch
```

Ensure you have PyTorch installed in your environment as well.

## Usage

Here's a basic example of how to use MaskedTensor:

```python
import torch
from agent_torch.utils.masked_tensor import MaskedTensor

# Create a MaskedTensor
data = torch.arange(10, dtype=torch.float32)
masked_array = MaskedTensor(data)

# Use learnable index
a = torch.tensor([3.0], requires_grad=True)
index = (a * 2).long()
result = masked_array[index]

# Compute gradients
result.backward()

print(f"Gradient of a: {a.grad}")
```

## API Reference

### MaskedTensor

```python
1. `__getitem__(index)`: Allows indexing with integers, slices, tuples, or tensors.
2. `__len__()`: Returns the length of the underlying tensor.
3. `size()`: Returns the size of the underlying tensor.
4. `dim()`: Returns the number of dimensions of the underlying tensor.
5. `add(other)`: Element-wise addition with another tensor or MaskedTensor.
6. `mul(other)`: Element-wise multiplication with another tensor or MaskedTensor.
7. `sum(dim=None, keepdim=False)`: Computes the sum along the specified dimensions.
8. `mean(dim=None, keepdim=False)`: Computes the mean along the specified dimensions.
```

## Implementation Details

MaskedTensor is implemented using custom autograd functions to enable differentiable indexing and arithmetic operations:

- `IndexingFunction`: Handles differentiable indexing

These custom functions define both forward and backward passes, allowing gradients to flow through all operations.

## Examples

### Differentiable Indexing

```python
from agent_torch.utils.masked_tensor import MaskedTensor
import torch

data = torch.arange(10, dtype=torch.float32)
masked_array = MaskedTensor(data)

a = torch.tensor([2.0], requires_grad=True)
index = (a * 3).long()
result = masked_array[index]
result.backward()

print(f"Gradient of a: {a.grad}")
```

### Arithmetic Operations

```python
from agent_torch.utils.masked_tensor import MaskedTensor
import torch

data = torch.arange(10, dtype=torch.float32)
masked_array = MaskedTensor(data)

# Addition
result_add = masked_array.add(5)

# Multiplication
result_mul = masked_array.mul(2)
```

### Operations between MaskedTensors

```python
from agent_torch.utils.masked_tensor import MaskedTensor
import torch

# Create two MaskedTensors
data1 = torch.arange(10, dtype=torch.float32)
data2 = torch.randn(10)
masked_array1 = MaskedTensor(data1)
masked_array2 = MaskedTensor(data2)

# Addition of two MaskedTensors
result_add = masked_array1.add(masked_array2)

# Multiplication of two MaskedTensors
result_mul = masked_array1.mul(masked_array2)```
