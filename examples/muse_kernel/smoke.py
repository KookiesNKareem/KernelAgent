"""End-to-end check of @muse_kernel.fast: observe, build in the background, swap in the verified kernel."""

import time

import torch
import torch.nn.functional as F

import muse_kernel


@muse_kernel.fast
def gated_delta(q, k, v, g, beta, scale):
    B, H, T, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1]).float()
    outs = []
    for t in range(T):
        S = S * g[:, :, t].float().exp()[..., None, None]
        kt = k[:, :, t].float()
        pred = (S * kt[..., None]).sum(-2)
        S = S + (beta[:, :, t].float()[..., None] * (v[:, :, t].float() - pred))[..., None, :] * kt[..., None]
        outs.append((S * (q[:, :, t].float() * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def batch(seed):
    torch.manual_seed(seed)
    B, H, T, D = 2, 4, 512, 128
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = F.normalize(torch.randn(B, H, T, D, device="cuda"), dim=-1).bfloat16().requires_grad_(True)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = F.logsigmoid(torch.randn(B, H, T, device="cuda")).requires_grad_(True)
    beta = torch.rand(B, H, T, device="cuda").sigmoid().requires_grad_(True)
    return q, k, v, g, beta, D ** -0.5


step = 0
while True:
    x = batch(step)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gated_delta(*x).float().sum().backward()
    torch.cuda.synchronize()
    st = gated_delta.status()
    print(f"step {step}: {1000 * (time.perf_counter() - t0):.1f} ms  state={st['state']}  calls={st['calls']}", flush=True)
    if st["state"] == "failed" or st["calls"].get("kernel", 0) >= 5:
        break
    step += 1
    time.sleep(0 if st["state"] == "ready" else 20)
