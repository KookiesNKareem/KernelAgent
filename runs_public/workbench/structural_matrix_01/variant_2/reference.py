import torch
CHANNEL_DECAY = True

def reference(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    q, k, v, e, g, beta, gamma = (x.to(torch.promote_types(x.dtype, torch.float32)) for x in (q, k, v, e, g, beta, gamma))
    S = q.new_zeros(B, H, K, V)
    outs = []
    for t in range(T):
        kt, vt, et, bt = (k[:, :, t], v[:, :, t], e[:, :, t], beta[:, :, t])
        ct = gamma[:, :, t]
        S = S * g[:, :, t].exp()[..., None]
        pred = (S * kt[..., None]).sum(-2)
        S = S + (bt[..., None] * (vt - pred))[..., None, :] * kt[..., None]
        S = S - ct[..., None, None] * et[..., None] * (S * et[..., None]).sum(-2)[..., None, :]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)
