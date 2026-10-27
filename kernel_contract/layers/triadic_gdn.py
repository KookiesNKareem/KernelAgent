"""Triadic Gated DeltaNet (Sieberling et al., arXiv 2609.36529, Eq. 5): a gated delta rule over the joint key
kappa_t = k_t (x) k'_t, so the state is a d x E x d tensor flattened to [d*E, d].

    S_t = (I - beta_t kappa_t kappa_t^T) D_t S_{t-1} + beta_t kappa_t v_t^T,    o_t = scale * S_t^T (q_t (x) q'_t)

with one forget gate per slice e along the E axis: D_t = diag(exp(g_t)) repeated over d inside each slice.
E = 1 is exactly the gated delta rule. Here d = 128 and E = 4.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

E = 4


def reference(q, k, v, q2, k2, g, beta, scale):
    # q, k: [B, H, T, d] unit-norm; v: [B, H, T, d]; q2, k2: [B, H, T, E] unit-norm;
    # g (log decay per slice, <= 0): [B, H, T, E]; beta in (0, 1): [B, H, T]
    B, H, T, d = q.shape
    V = v.shape[-1]
    nE = k2.shape[-1]
    q, k, v, q2, k2, g, beta = (x.to(torch.promote_types(x.dtype, torch.float32))
                                for x in (q, k, v, q2, k2, g, beta))
    S = q.new_zeros(B, H, d * nE, V)
    outs = []
    for t in range(T):
        kap = (k[:, :, t, :, None] * k2[:, :, t, None, :]).flatten(-2)       # [B, H, d*E], index i*E + e
        qq = (q[:, :, t, :, None] * q2[:, :, t, None, :]).flatten(-2)
        decay = g[:, :, t, None, :].exp().expand(B, H, d, nE).flatten(-2)    # one gate per slice e
        S = S * decay[..., None]
        pred = (S * kap[..., None]).sum(-2)
        S = S + (beta[:, :, t][..., None] * (v[:, :, t] - pred))[..., None, :] * kap[..., None]
        outs.append((S * (qq * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def inputs(B=4, H=8, T=1024, D=128, device="cuda", dtype=torch.bfloat16, grad=False, Dv=None, nE=E):
    torch.manual_seed(0)
    q = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    v = torch.randn(B, H, T, Dv or D, device=device, dtype=dtype)
    q2 = F.normalize(torch.randn(B, H, T, nE, device=device), dim=-1).to(dtype)
    k2 = F.normalize(torch.randn(B, H, T, nE, device=device), dim=-1).to(dtype)
    g = F.logsigmoid(torch.randn(B, H, T, nE, device=device) + 3)
    beta = torch.rand(B, H, T, device=device).sigmoid()
    xs = [q, k, v, q2, k2, g, beta]
    if grad:
        for x in xs:
            x.requires_grad_(True)
    return xs


def spec(B=2, H=4, T=512, D=128, Dv=None):
    from ..spec import capture, with_domains
    x = inputs(B=B, H=H, T=T, D=D, Dv=Dv)
    s = capture(f"{__name__}.reference", (*x, D ** -0.5), name="triadic_gdn")
    s.flags["any_float_output"] = True
    s.flags["normwise_rtol"] = 2e-2
    return with_domains(s, {0: "l2row", 1: "l2row", 3: "l2row", 4: "l2row", 5: "logdecay", 6: "unit"})


def fla_reduction(q, k, v, q2, k2, g, beta, scale):
    """Materialize the joint keys and run FLA's per-channel gated delta rule at key dim d*E."""
    from fla.ops.kda import chunk_kda
    B, H, T, d = q.shape
    nE = k2.shape[-1]
    kap = (k[..., :, None] * k2[..., None, :]).flatten(-2)
    qq = (q[..., :, None] * q2[..., None, :]).flatten(-2)
    gg = g[..., None, :].expand(B, H, T, d, nE).flatten(-2).contiguous()
    o, _ = chunk_kda(*(x.transpose(1, 2) for x in (qq, kap, v, gg, beta)), scale=scale)
    return o.transpose(1, 2)
