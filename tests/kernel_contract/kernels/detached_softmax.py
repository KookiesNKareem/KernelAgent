import torch


class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        return torch.softmax(x, dim)

    @staticmethod
    def backward(ctx, g):
        return g, None  # wrong: identity instead of the softmax Jacobian


def kernel_function(x, dim):
    return _F.apply(x, dim)
