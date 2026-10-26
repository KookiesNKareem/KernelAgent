import torch


def kernel_function(x, dim):
    assert x.is_contiguous(), "input must be contiguous"
    return torch.softmax(x.float(), dim).to(x.dtype)
