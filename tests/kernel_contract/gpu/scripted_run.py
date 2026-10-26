"""A full run with a scripted writer, for checking the pipeline and the dashboard without an API key."""
import sys
from pathlib import Path

import kernel_contract.agent as agent
from kernel_contract import run
from utils.providers.base import LLMResponse

HERE = Path(__file__).resolve().parent
SCRIPT = {"apply_rotary": ["rotary_flat.py", "rotary_strided.py"]}


class Scripted:
    def __init__(self):
        self.queue = {k: list(v) for k, v in SCRIPT.items()}

    def get_response(self, model, messages, **kw):
        text = messages[0]["content"]
        key = next(k for k in self.queue if k in text)
        q = self.queue[key]
        src = (HERE / (q.pop(0) if len(q) > 1 else q[0])).read_text()
        return LLMResponse(content=f"```python\n{src}```", model=model, provider="scripted",
                           usage={"completion_tokens": len(src) // 4})


prov = Scripted()
agent.get_model_provider = lambda m: prov
sys.argv = ["run", "--model", "kernel_contract.demo_models:llama_block", "--model-id", "scripted",
            "--only", "apply_rotary", "--top", "1", "--repairs", "1", "--out", sys.argv[1]]
run.main()
