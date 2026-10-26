"""GPU smoke test: trace the real Llama block, fuzz two hand-written rotary kernels."""
import json
from pathlib import Path

from kernel_contract.demo_models import llama_block
from kernel_contract.model import Installed, compare_model, load_kernel, trace
from kernel_contract.runner import run_contract

HERE = Path(__file__).parent
m, x = llama_block(seq=2048)
sites = trace(m, x)
for s in sites[:10]:
    print(f"{s.spec.name:32s} calls={s.calls} ms_total={s.ms_total:.3f}")
rot = next(s for s in sites if s.spec.name.startswith("apply_rotary"))
for k in ("rotary_strided.py", "rotary_flat.py"):
    v = run_contract(rot.spec, HERE / k)
    print(k, v.outcome, [(r["name"], r["outcome"]) for r in v.failures])
good = Installed(rot.spec, load_kernel(HERE / "rotary_strided.py"), strided_ok=True)
print(json.dumps(compare_model(m, x, [good]), indent=1))
