from __future__ import annotations
import torch as torch

def reference(q, k, v, e, g, beta, gamma, scale):
    """Erase memory, then write the prediction error. Try reversing these two updates."""
    B, H, T, K = q.shape
    tensors = (q, k, v, e, g, beta, gamma)
    q, k, v, e, g, beta, gamma = (x.to(torch.promote_types(x.dtype, torch.float32)) for x in tensors)
    state = q.new_zeros(B, H, K, v.shape[-1])
    outputs = []
    for t in range(T):
        key, value, erase = (k[:, :, t], v[:, :, t], e[:, :, t])
        state = state * g[:, :, t].exp()[..., None]
        state = state - gamma[:, :, t, None, None] * erase[..., None] * (state * erase[..., None]).sum(-2)[..., None, :]
        prediction = (state * key[..., None]).sum(-2)
        state = state + key[..., None] * (beta[:, :, t, None] * (value - prediction))[..., None, :]
        outputs.append((state * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outputs, dim=2)
