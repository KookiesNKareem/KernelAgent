"""@muse_kernel.fast on a two-argument recurrence (examples/muse_kernel/slow_layer.py): train until the kernel lands."""

import time

import torch

from slow_layer import decayed_scan


def batch(seed):
    torch.manual_seed(seed)
    x = torch.randn(8, 1024, 256, device="cuda", requires_grad=True)
    a = torch.rand(8, 1024, 256, device="cuda").mul(0.2).add(0.79).requires_grad_(True)
    return x, a


step = 0
while True:
    x = batch(step)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decayed_scan(*x).sum().backward()
    torch.cuda.synchronize()
    st = decayed_scan.status()
    print(f"step {step}: {1000 * (time.perf_counter() - t0):.1f} ms  state={st['state']}  "
          f"model={st.get('model')}  calls={st['calls']}", flush=True)
    if st["state"] == "failed" or st["calls"].get("kernel", 0) >= 5:
        break
    step += 1
    time.sleep(0 if st["state"] == "ready" else 20)
