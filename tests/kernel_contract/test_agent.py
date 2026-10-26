from pathlib import Path

import torch

import kernel_contract.agent as agent
from kernel_contract.spec import capture
from utils.providers.base import LLMResponse

K = Path(__file__).parent / "kernels"


class ScriptedProvider:
    def __init__(self, files):
        self.files = list(files)
        self.prompts = []

    def get_response(self, model, messages, **kw):
        self.prompts.append(messages[0]["content"])
        src = (K / self.files.pop(0)).read_text()
        return LLMResponse(content=f"Here it is.\n```python\n{src}```", model=model, provider="scripted")


def test_repair_from_counterexample(monkeypatch, tmp_path):
    prov = ScriptedProvider(["dense_pointer_softmax.py", "good_softmax.py"])
    monkeypatch.setattr(agent, "get_model_provider", lambda m: prov)
    spec = capture("torch.softmax", (torch.randn(6, 32, dtype=torch.bfloat16), 1))
    events = []
    res = agent.solve(spec, "scripted", tmp_path, device="cpu", emit=events.append)
    assert res.accepted and len(res.attempts) == 2
    assert "strided2" in prov.prompts[1] and "stride tripled" in prov.prompts[1]
    assert [e["outcome"] for e in events if e["event"] == "verdict"] == ["wrong", "accept"]


def test_describe_call_reports_observed_layout():
    x = torch.randn(4, 64, 6).transpose(1, 2)
    text = agent.describe_call(capture("torch.nn.functional.layer_norm", (x, (64,))))
    assert "non-contiguous" in text and "(64,)" in text
