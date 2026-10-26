from pathlib import Path

import torch

from kernel_contract.model import GuardedModel, Installed, compare_model, load_kernel, trace

K = Path(__file__).parent / "kernels"


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 32)

    def forward(self, x):
        h = self.lin(x)
        a = h.softmax(-1)
        b = h.transpose(0, 1).softmax(-1)
        return a.sum() + b.sum()


def test_trace_and_guard():
    torch.manual_seed(0)
    m = Tiny().eval()
    x = torch.randn(8, 16)
    sites = trace(m, (x,), device="cpu")
    soft = [s for s in sites if s.spec.op == "torch.Tensor.softmax"]
    assert len(soft) == 2
    dense_site = next(s for s in soft if s.spec.args[0]["tensor"]["contiguous"])
    refusing = Installed(dense_site.spec, load_kernel(K / "refusing_softmax.py"), strided_ok=False)
    r = compare_model(m, (x,), [refusing], reps=2)
    assert r["routed"] == {"torch.Tensor.softmax": 1}
    assert r["fallback"] == {"torch.Tensor.softmax": 1}
    assert r["max_abs_diff"] < 1e-5
