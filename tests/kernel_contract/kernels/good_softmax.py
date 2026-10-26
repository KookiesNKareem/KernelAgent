import torch


def kernel_function(x, dim):
    return torch.softmax(x.float(), dim).to(x.dtype)
