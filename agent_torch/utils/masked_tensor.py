import torch
import torch.nn as nn


class IndexConverter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, index):
        return index.long()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class IndexingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, index):
        index = IndexConverter.apply(index)
        ctx.save_for_backward(input, index)
        return input[index]

    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_input[index] = grad_output
        grad_index = torch.ones_like(index, dtype=torch.float)
        return grad_input, grad_index


class MaskedTensor(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.data = nn.Parameter(data[0] if isinstance(data, list) else data)

    def __getitem__(self, index):
        if isinstance(index, (int, slice, tuple)):
            return self.data[index]
        elif torch.is_tensor(index):
            return IndexingFunction.apply(self.data, index)
        else:
            raise TypeError("Index must be an integer, slice, tuple, or tensor")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"MaskedTensor({self.data})"

    def size(self):
        return self.data.size()

    def dim(self):
        return self.data.dim()

    def add(self, other):
        if isinstance(other, MaskedTensor):
            return MaskedTensor(self.data + other.data)
        return MaskedTensor(self.data + other)

    def mul(self, other):
        if isinstance(other, MaskedTensor):
            return MaskedTensor(self.data * other.data)
        return MaskedTensor(self.data * other)

    def sum(self, dim=None, keepdim=False):
        return MaskedTensor(self.data.sum(dim=dim, keepdim=keepdim))

    def mean(self, dim=None, keepdim=False):
        return MaskedTensor(self.data.mean(dim=dim, keepdim=keepdim))

    @staticmethod
    def from_tensor(tensor):
        return MaskedTensor(tensor)
