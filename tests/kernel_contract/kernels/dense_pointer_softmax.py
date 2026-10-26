import torch


def kernel_function(x, dim):
    # reads storage as if the tensor were dense, the archive's raw-pointer bug
    dense = torch.as_strided(x, x.shape, torch.empty(x.shape).stride(), x.storage_offset())
    return torch.softmax(dense.float(), dim).to(x.dtype)
