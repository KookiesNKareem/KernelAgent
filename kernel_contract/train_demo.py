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
import hashlib
import statistics
from contextlib import nullcontext
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


class TriadicMixer(nn.Module):
    """Triadic Gated DeltaNet's inputs: a second unit-norm key and query of width E and one forget gate per slice."""

    def __init__(self, d, heads, head_dim, fn, n_e):
        super().__init__()
        self.h, self.dh, self.ne, self.fn = heads, head_dim, n_e, fn
        self.proj = nn.Linear(d, 3 * heads * head_dim + 3 * heads * n_e + heads, bias=False)
        self.out = nn.Linear(heads * head_dim, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        h, dh, ne = self.h, self.dh, self.ne
        q, k, v, q2, k2, a, b = self.proj(x).split([h * dh] * 3 + [h * ne] * 3 + [h], dim=-1)
        q, k, v = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v))
        q2, k2, a = (t.view(B, T, h, ne).transpose(1, 2) for t in (q2, k2, a))
        q, k, q2, k2 = (F.normalize(t.float(), dim=-1).to(torch.bfloat16) for t in (q, k, q2, k2))
        g, beta = F.logsigmoid(a.float() + 3), b.float().transpose(1, 2).sigmoid()
        o = self.fn(q, k, v.to(torch.bfloat16), q2, k2, g, beta, dh ** -0.5)
        return self.out(o.float().transpose(1, 2).reshape(B, T, h * dh))


class Block(nn.Module):
    def __init__(self, d, heads, head_dim, fn, channel_decay, n_e=None):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.mix = TriadicMixer(d, heads, head_dim, fn, n_e) if n_e else Mixer(d, heads, head_dim, fn, channel_decay)
        self.up = nn.Linear(d, 8 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        a, b = self.up(self.n2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(a) * b)


class LM(nn.Module):
    def __init__(self, fn, d=512, layers=4, heads=4, head_dim=128, channel_decay=True, n_e=None):
        super().__init__()
        self.emb = nn.Embedding(256, d)
        self.blocks = nn.ModuleList(Block(d, heads, head_dim, fn, channel_decay, n_e) for _ in range(layers))
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


def data(device, data_path=None):
    path = Path(data_path) if data_path else Path("/tmp/tinyshakespeare.txt")
    if not path.exists():
        if data_path:
            raise FileNotFoundError(path)
        urllib.request.urlretrieve(DATA_URL, path)
    return torch.tensor(list(path.read_bytes()), dtype=torch.long, device=device)


def layer_module(layer):
    import importlib
    return LAYERS[layer] if layer in LAYERS else importlib.import_module(layer)


def make_lm(layer, fn, device="cuda", **kwargs):
    mod = layer_module(layer)
    return LM(fn, channel_decay=getattr(mod, "CHANNEL_DECAY", True), n_e=getattr(mod, "E", None), **kwargs).to(device)


def mixer_fn(layer, mixer):
    mod = layer_module(layer)
    if mixer == "reference":
        return mod.reference
    if mixer == "reduction":
        return mod.fla_reduction
    if mixer == "compiled":
        return torch.compile(mod.reference)
    if mixer == "checkpointed":  # the standard memory fix: keep only the inputs, recompute the loop in backward
        from torch.utils.checkpoint import checkpoint
        return lambda *a: checkpoint(mod.reference, *a, use_reentrant=False)
    return load_kernel(Path(mixer))


def batch(ids, n, seq, gen):
    starts = torch.randint(0, len(ids) - seq - 1, (n,), device=ids.device, generator=gen)
    return torch.stack([ids[s:s + seq + 1] for s in starts.tolist()])


def loss_of(model, x):
    with (torch.autocast("cuda", dtype=torch.bfloat16) if x.device.type == "cuda" else nullcontext()):
        logits = model(x[:, :-1])
    return F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten())


def grad_check(layer, path, batch_size, seq):
    """One step from the same init and batch: every parameter gradient, kernel vs reference."""
    ids = data("cuda")
    x = batch(ids, batch_size, seq, torch.Generator(device="cuda").manual_seed(1))
    grads = {}
    for name, fn in (("reference", mixer_fn(layer, "reference")), ("kernel", mixer_fn(layer, path))):
        torch.manual_seed(0)
        model = make_lm(layer, fn)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--data", help="Optional local dataset; its bytes are hashed in the receipt")
    ap.add_argument("--expected-data-sha256", help="Refuse a dataset that differs from the recorded bytes")
    a = ap.parse_args()
    if min(a.steps, a.batch, a.seq, a.val_batches, a.eval_every, a.width, a.layers, a.heads, a.head_dim) < 1:
        ap.error("training dimensions, steps and validation counts must be positive")
    if a.grad_check:
        return grad_check(a.layer, a.grad_check, a.batch, a.seq)
    device = a.device
    sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    torch.manual_seed(a.seed)
    model = make_lm(a.layer, mixer_fn(a.layer, a.mixer), device=device, d=a.width, layers=a.layers,
                    heads=a.heads, head_dim=a.head_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    ids = data(device, a.data)
    data_path = Path(a.data) if a.data else Path("/tmp/tinyshakespeare.txt")
    if a.expected_data_sha256 and hashlib.sha256(data_path.read_bytes()).hexdigest() != a.expected_data_sha256:
        raise ValueError("Dataset hash differs from the recorded experiment")
    split = int(len(ids) * 0.9)
    train, val = ids[:split], ids[split:]
    gen = torch.Generator(device=device).manual_seed(a.seed + 1)
    val_gen = torch.Generator(device=device).manual_seed(a.seed + 2)
    vx = [batch(val, a.batch, a.seq, val_gen) for _ in range(a.val_batches)]
    from muse_kernel.identity import environment
    data_path = Path(a.data) if a.data else Path("/tmp/tinyshakespeare.txt")
    meta = {"config": vars(a), "environment": environment(), "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "dataset_url": DATA_URL if not a.data else None, "validation": "mean over fixed disjoint-split batches",
            "started": time.time()}
    if Path(a.mixer).is_file():
        meta["kernel_sha256"] = hashlib.sha256(Path(a.mixer).read_bytes()).hexdigest()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".meta.json").write_text(json.dumps(meta, indent=2))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    with open(a.out, "w") as f:
        for step in range(a.steps):
            x = batch(train, a.batch, a.seq, gen)
            sync()
            t0 = time.perf_counter()
            try:
                loss = loss_of(model, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                sync()
            except torch.OutOfMemoryError:
                f.write(json.dumps({"step": step, "out_of_memory": True}) + "\n")
                raise
            dt = time.perf_counter() - t0
            rec = {"step": step, "seed": a.seed, "loss": loss.item(), "grad_norm": gnorm.item(),
                   "step_s": dt, "tokens_per_s": a.batch * a.seq / dt}
            if step % a.eval_every == 0 or step == a.steps - 1:
                model.eval()
                with torch.no_grad():
                    vals = [loss_of(model, v).item() for v in vx]
                model.train()
                rec.update(val_loss=statistics.mean(vals), val_loss_std=statistics.pstdev(vals), val_batches=len(vals),
                           peak_mem_gb=torch.cuda.max_memory_allocated()/1e9 if device == "cuda" else None)
            rec["elapsed_s"] = time.perf_counter() - training_start
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
