"""Train a small byte-level language model whose sequence mixer is the edited layer (EDA), once with the
paper's reference loop and once with the agent's kernel, from the same init and the same batches.

  python -m kernel_contract.train_demo --layer wek --mixer reference --steps 60 --out ref.jsonl
  python -m kernel_contract.train_demo --layer wek --mixer kernel_contract/recur_cands/wek_wonder_fwdbwd.py --out k.jsonl
  python -m kernel_contract.train_demo --layer wek --mixer reduction --out red.jsonl
  python -m kernel_contract.train_demo --layer wek --grad-check kernel_contract/recur_cands/wek_wonder_fwdbwd.py
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import recur_eda, recur_vse, recur_wek
from .model import load_kernel

LAYERS = {"eda": recur_eda, "wek": recur_wek, "vse": recur_vse}

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class Mixer(nn.Module):
    def __init__(self, d, heads, head_dim, fn, channel_decay):
        super().__init__()
        self.h, self.dh, self.fn, self.cd = heads, head_dim, fn, channel_decay
        self.ng = heads * head_dim if channel_decay else heads
        self.proj = nn.Linear(d, 4 * heads * head_dim + self.ng + 2 * heads, bias=False)
        self.out = nn.Linear(heads * head_dim, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        h, dh = self.h, self.dh
        q, k, v, e, a, gates = self.proj(x).split([h * dh] * 4 + [self.ng, 2 * h], dim=-1)
        q, k, v, e = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v, e))
        k, e = F.normalize(k.float(), dim=-1), F.normalize(e.float(), dim=-1)
        q, k, v, e = (t.to(torch.bfloat16) for t in (q, k, v, e))
        b, c = gates.float().view(B, T, 2, h).permute(2, 0, 3, 1)
        a = a.float().view(B, T, h, dh).transpose(1, 2) if self.cd else a.float().transpose(1, 2)
        # bias toward slow decay so the state carries information across many steps
        g, beta, gamma = F.logsigmoid(a + 3), b.sigmoid(), c.sigmoid()
        o = self.fn(q, k, v, e, g, beta, gamma, dh ** -0.5)
        return self.out(o.float().transpose(1, 2).reshape(B, T, h * dh))


class Block(nn.Module):
    def __init__(self, d, heads, head_dim, fn, channel_decay):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.mix = Mixer(d, heads, head_dim, fn, channel_decay)
        self.up = nn.Linear(d, 8 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        a, b = self.up(self.n2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(a) * b)


class LM(nn.Module):
    def __init__(self, fn, d=512, layers=4, heads=4, head_dim=128, channel_decay=True):
        super().__init__()
        self.emb = nn.Embedding(256, d)
        self.blocks = nn.ModuleList(Block(d, heads, head_dim, fn, channel_decay) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)
        for b in self.blocks:
            for lin in (b.mix.proj, b.mix.out, b.up, b.down):
                nn.init.normal_(lin.weight, std=0.02)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return F.linear(self.norm(x), self.emb.weight)


def data(device):
    path = Path("/tmp/tinyshakespeare.txt")
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    return torch.tensor(list(path.read_bytes()), dtype=torch.long, device=device)


def layer_module(layer):
    import importlib
    return LAYERS[layer] if layer in LAYERS else importlib.import_module(layer)


def mixer_fn(layer, mixer):
    mod = layer_module(layer)
    if mixer == "reference":
        return mod.reference
    if mixer == "reduction":
        return mod.fla_reduction
    if mixer == "compiled":
        return torch.compile(mod.reference)
    return load_kernel(Path(mixer))


def batch(ids, n, seq, gen):
    starts = torch.randint(0, len(ids) - seq - 1, (n,), device=ids.device, generator=gen)
    return torch.stack([ids[s:s + seq + 1] for s in starts.tolist()])


def loss_of(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x[:, :-1])
    return F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten())


def grad_check(layer, path, batch_size, seq):
    """One step from the same init and batch: every parameter gradient, kernel vs reference."""
    ids = data("cuda")
    x = batch(ids, batch_size, seq, torch.Generator(device="cuda").manual_seed(1))
    grads = {}
    for name, fn in (("reference", mixer_fn(layer, "reference")), ("kernel", mixer_fn(layer, path))):
        torch.manual_seed(0)
        model = LM(fn, channel_decay=getattr(layer_module(layer), "CHANNEL_DECAY", True)).cuda()
        loss = loss_of(model, x)
        loss.backward()
        grads[name] = (loss.item(), {n: p.grad.float() for n, p in model.named_parameters()})
    (lr, gr), (lk, gk) = grads["reference"], grads["kernel"]
    rel = {n: float((gk[n] - gr[n]).norm() / gr[n].norm().clamp_min(1e-30)) for n in gr}
    worst = sorted(rel.items(), key=lambda t: -t[1])[:5]
    print(json.dumps({"loss_reference": lr, "loss_kernel": lk, "max_param_grad_rel_err": worst[0][1],
                      "worst": worst, "median_rel_err": sorted(rel.values())[len(rel) // 2]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="wek", help="eda, wek, vse, or a layer module path")
    ap.add_argument("--mixer", default="reference")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--grad-check")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.grad_check:
        return grad_check(a.layer, a.grad_check, a.batch, a.seq)
    torch.manual_seed(0)
    model = LM(mixer_fn(a.layer, a.mixer), channel_decay=getattr(layer_module(a.layer), "CHANNEL_DECAY", True)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    ids = data("cuda")
    split = int(len(ids) * 0.9)
    train, val = ids[:split], ids[split:]
    gen = torch.Generator(device="cuda").manual_seed(1)
    vx = batch(val, a.batch, a.seq, torch.Generator(device="cuda").manual_seed(2))
    with open(a.out, "w") as f:
        for step in range(a.steps):
            x = batch(train, a.batch, a.seq, gen)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = loss_of(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            rec = {"step": step, "loss": loss.item(), "grad_norm": gnorm.item(), "step_s": time.perf_counter() - t0,
                   "tokens_per_s": a.batch * a.seq / (time.perf_counter() - t0)}
            if step % a.eval_every == 0 or step == a.steps - 1:
                with torch.no_grad():
                    rec["val_loss"] = loss_of(model, vx).item()
                rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
