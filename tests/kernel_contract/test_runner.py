from pathlib import Path

import torch

from kernel_contract.runner import run_contract
from kernel_contract.spec import capture

K = Path(__file__).parent / "kernels"


def softmax_spec():
    return capture("torch.softmax", (torch.randn(6, 32, dtype=torch.bfloat16), 1))


def test_correct_kernel_accepted():
    assert run_contract(softmax_spec(), K / "good_softmax.py", device="cpu").outcome == "accept"


def test_dense_pointer_is_wrong_only_on_layouts():
    v = run_contract(softmax_spec(), K / "dense_pointer_softmax.py", device="cpu")
    assert v.outcome == "wrong"
    assert {r["axis"] for r in v.failures} == {"layout"}
    assert run_contract(softmax_spec(), K / "dense_pointer_softmax.py", contract="dense", device="cpu").outcome == "accept"


def test_refusal_is_not_wrong():
    v = run_contract(softmax_spec(), K / "refusing_softmax.py", device="cpu")
    assert v.outcome == "refuse"


def test_crash_is_survived():
    v = run_contract(softmax_spec(), K / "crashing_softmax.py", device="cpu")
    assert v.outcome == "crash"
    names = [r["name"] for r in v.results]
    assert "observed" in names and "strided2" in names


def test_observed_strides_are_replayed():
    x = torch.randn(4, 64, 6).transpose(1, 2)
    spec = capture("torch.nn.functional.layer_norm", (x, (64,)))
    assert spec.args[0]["tensor"]["contiguous"] is False
    assert run_contract(spec, K / "good_softmax.py", device="cpu", cases=[{"name": "observed", "axis": "seed", "seed": 0}]).outcome in ("wrong", "refuse")


def test_gradients_checked():
    spec = capture("torch.softmax", (torch.randn(6, 32), 1))
    spec.flags["check_grads"] = True
    assert run_contract(spec, K / "good_softmax.py", device="cpu").outcome == "accept"
    bad = run_contract(spec, K / "detached_softmax.py", device="cpu")
    assert bad.outcome == "wrong" and "gradient" in bad.failures[0]["reason"]
