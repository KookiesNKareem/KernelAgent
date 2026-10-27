"""Train a small byte-level language model whose sequence mixer is the edited layer (EDA), once with the
paper's reference loop and once with the agent's kernel, from the same init and the same batches.

  python -m kernel_contract.train_demo --mixer reference --steps 60 --out ref.jsonl
  python -m kernel_contract.train_demo --mixer kernel_contract/recur_cands/eda_wonder_fwdbwd.py --steps 60 --out k.jsonl
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

from .model import load_kernel
from .recur_eda import reference

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class Mixer(nn.Module):
    def __init__(self, d, heads, head_dim, fn):
        super().__init__()
        self.h, self.dh, self.fn = heads, head_dim, fn
        self.proj = nn.Linear(d, 4 * heads * head_dim + 3 * heads, bias=False)
        self.out = nn.Linear(heads * head_dim, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        h, dh = self.h, self.dh
        q, k, v, e, gates = self.proj(x).split([h * dh] * 4 + [3 * h], dim=-1)
        q, k, v, e = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v, e))
        k, e = F.normalize(k.float(), dim=-1).to(x.dtype), F.normalize(e.float(), dim=-1).to(x.dtype)
        a, b, c = gates.float().view(B, T, 3, h).permute(2, 0, 3, 1)
        g, beta, gamma = F.logsigmoid(a), b.sigmoid(), c.sigmoid()
        o = self.fn(q, k, v, e, g, beta, gamma, dh ** -0.5)
        return self.out(o.to(x.dtype).transpose(1, 2).reshape(B, T, h * dh))


class Block(nn.Module):
    def __init__(self, d, heads, head_dim, fn):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.mix = Mixer(d, heads, head_dim, fn)
        self.up = nn.Linear(d, 8 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        a, b = self.up(self.n2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(a) * b)


class LM(nn.Module):
    def __init__(self, fn, d=512, layers=4, heads=4, head_dim=128):
        super().__init__()
        self.emb = nn.Embedding(256, d)
        self.blocks = nn.ModuleList(Block(d, heads, head_dim, fn) for _ in range(layers))
        self.norm = nn.RMSNorm(d)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", default="reference")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fn = reference if a.mixer == "reference" else load_kernel(Path(a.mixer))
    torch.manual_seed(0)
    model = LM(fn).cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    ids = data("cuda")
    gen = torch.Generator(device="cuda").manual_seed(1)
    with open(a.out, "w") as f:
        for step in range(a.steps):
            starts = torch.randint(0, len(ids) - a.seq - 1, (a.batch,), device="cuda", generator=gen)
            x = torch.stack([ids[s:s + a.seq + 1] for s in starts.tolist()])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(x[:, :-1]).float().flatten(0, 1), x[:, 1:].flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            rec = {"step": step, "loss": loss.item(), "step_s": time.perf_counter() - t0}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
