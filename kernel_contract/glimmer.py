"""Muse Glimmer as the target workload: load it, time greedy decoding, and trace where time goes.

  python -m kernel_contract.glimmer --probe        # load, generate, trace prefill and decode
"""

from __future__ import annotations

import argparse
import json
import time

import torch

MODEL_ID = "meta-models/Muse-Glimmer-30B"
PROMPT = "Explain, step by step, how a GPU executes a matrix multiplication, and why memory bandwidth matters."


def load(model_id: str = MODEL_ID):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tok


@torch.no_grad()
def decode_speed(model, tok, new_tokens: int = 128) -> dict:
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"prompt_tokens": ids.shape[1], "new_tokens": new_tokens, "seconds": dt,
            "tokens_per_s": new_tokens / dt, "text": tok.decode(out[0, ids.shape[1]:])[:300]}


@torch.no_grad()
def compiled_decode_speed(model, tok, new_tokens: int = 128, mode: str = "reduce-overhead") -> dict:
    """The baseline judges will ask about: static KV cache plus torch.compile with CUDA graphs."""
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    model.generation_config.cache_implementation = "static"
    model.forward = torch.compile(model.forward, mode=mode, fullgraph=False)
    t0 = time.perf_counter()
    for _ in range(3):
        model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False)
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"compile_mode": mode, "warmup_s": warm_s, "tokens_per_s": new_tokens / dt,
            "text": tok.decode(out[0, ids.shape[1]:])[:300]}


def decode_step_inputs(model, tok, context: int = 512):
    """One decode step after a prefill of `context` tokens: what generate() calls per new token."""
    ids = torch.randint(1000, 100000, (1, context), device="cuda")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)
    return (nxt,), {"past_key_values": out.past_key_values, "use_cache": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--compiled", action="store_true", help="only time torch.compile decoding")
    ap.add_argument("--model-id", default=MODEL_ID)
    a = ap.parse_args()
    t0 = time.time()
    model, tok = load(a.model_id)
    print(json.dumps({"load_s": time.time() - t0, "mem_gb": torch.cuda.memory_allocated() / 1e9}), flush=True)
    if a.compiled:
        import traceback
        for mode in ("reduce-overhead", "default"):
            try:
                print(json.dumps(compiled_decode_speed(model, tok, mode=mode)), flush=True)
            except Exception as e:
                print(json.dumps({"compile_mode": mode, "error": f"{type(e).__name__}: {e}"[:1500]}), flush=True)
                traceback.print_exc(limit=3)
            model.forward = model.__class__.forward.__get__(model)
            torch._dynamo.reset()
        return
    print(json.dumps(decode_speed(model, tok)), flush=True)

    from .model import trace

    class Step(torch.nn.Module):
        def __init__(self, m, kw):
            super().__init__()
            self.m, self.kw = m, kw

        def forward(self, x):
            import copy
            return self.m(input_ids=x, past_key_values=copy.deepcopy(self.kw["past_key_values"]), use_cache=True).logits

    args, kw = decode_step_inputs(model, tok)
    sites = trace(Step(model, kw), args)
    total = sum(s.ms_total for s in sites)
    print(json.dumps({"decode_step_op_ms": total, "n_sites": len(sites)}), flush=True)
    for s in sites[:25]:
        print(f"{s.spec.name:40s} {s.calls:4d} calls {s.ms_total:8.3f} ms {100 * s.ms_total / total:5.1f}%  {s.spec.op}", flush=True)


if __name__ == "__main__":
    main()
