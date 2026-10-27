import torch

import muse_kernel


@muse_kernel.fast
def decayed_scan(x, a):
    """y_t = a_t * y_{t-1} + x_t over the time axis (dim 1)."""
    y = torch.zeros_like(x[:, 0])
    out = []
    for t in range(x.shape[1]):
        y = a[:, t] * y + x[:, t]
        out.append(y)
    return torch.stack(out, dim=1)
