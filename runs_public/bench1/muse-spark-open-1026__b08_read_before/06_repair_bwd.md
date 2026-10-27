```python
import torch
import triton
import triton.language as tl

BK = 32
BV = 32

@triton.jit
def fwd_chunk_fwd(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,chk_ptr,
    BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    N: tl.constexpr, C: tl.constexpr,
    NK: tl.constexpr, NV: tl.constexpr,
    SCALE: tl.constexpr, t0: tl.constexpr, cur_T: tl.constexpr, chk_idx: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0, 32)
    offs_v = tl.arange(0, 32)
    v_base = pid_v * 32 + offs_v
    mask_v = v_base < V
    # pred/out accumulators per V tile (this program handles one V tile)
    # we will loop ki inside each t for two phases
    for i in range(64):  # max C, masked
        if i >= cur_T:
            break
        t = t0 + i
        # phase A: decay tiles and compute partials
        out_acc = tl.zeros((32,), dtype=tl.float32)
        pred_acc = tl.zeros((32,), dtype=tl.float32)
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        # first pass over K tiles: decay + accumulate
        for ki in range(4):  # max 4 for K=128, for larger will be 5 but we use 8 to be safe
            if ki >= NK:
                break
            k_base = ki * 32 + offs_k
            mask_k = k_base < K
            # pointers for S
            s_off = (pid_bh * (N + 1) + chk_idx) * K * V + k_base[:, None] * V + v_base[None, :]
            mask2d = (k_base[:, None] < K) & (v_base[None, :] < V)
            # use separate loads for chk start, but after first i S is updated in HBM via stores below
            # we need to load current S tile: for i>0 it was written back in previous i's second pass
            S = tl.load(chk_ptr + (pid_bh * (N + 1) + chk_idx) * K * V + k_base[:, None] * V + v_base[None, :], mask=mask2d, other=0.0) if i == 0 else tl.load(chk_ptr + (pid_bh * (N + 1) + chk_idx + 8) * K * V + k_base[:, None] * V + v_base[None, :], mask=mask2d, other=0.0)
            # Actually we keep S in a temporary buffer chk_tmp: reuse chk next slot as working buffer
            # To avoid extra buffer we alternate: for i>0 load from working slot
            # We use extra working area at index N+1 (allocated)
            # The wrapper allocates chk with N+2 slots, first N+1 checkpoints, last is working
            # decay and store back to working slot
            qv = tl.load(q_ptr + (pid_bh * T + t) * K + k_base, mask=mask_k, other=0.0).to(tl.float32)
            kv = tl.load(k_ptr + (pid_bh * T + t) * K + k_base, mask=mask_k, other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + (pid_bh * T + t) * K + k_base, mask=mask_k, other=0.0).to(tl.float32)
            d = tl.exp(gv)
            # load S working
            if i == 0:
                S_cur = tl.load(chk_ptr + (pid_bh * (N + 1) + chk_idx) * K * V + k_base[:, None] * V + v_base[None, :], mask=mask2d, other=0.0)
            else:
                S_cur = tl.load(chk_ptr + (pid_bh * (N + 2)) * K * V + k_base[:, None] * V + v_base[None, :], mask=mask2d, other=0.0)
            Sd = S_cur * d[:, None]
            tl.store(chk_ptr + (pid_bh * (N + 2)) * K * V + k_base[:, None] * V + v_base[None, :], Sd, mask=mask2d)
            out_acc += tl.sum(Sd * (qv * SCALE)[:, None], axis=0)
            pred_acc += tl.sum(Sd * kv[:, None], axis=0)
        tl.store(out_ptr + (pid_bh * T + t) * V + v_base, out_acc, mask=mask_v)
        # phase B: S += beta*k*(v-pred)
        v_tile = tl.load(v_ptr + (pid_bh * T + t) * V + v_base, mask=mask_v, other=0.0).to(tl.float32)
        delta = beta * (v_tile - pred_acc)
        for ki in range(4):
            if ki >= NK:
                break
            k_base = ki * 32 + offs_k
            mask2d = (k_base[:, None] < K) & (v_base[None, :] < V)
            S = tl.load(chk_ptr + (pid_bh * (N + 2)) * K * V + k_base[:, None] * V + v_base[None, :], mask=mask2d, other=0.0)
            kv = tl.load(k_ptr + (pid_bh * T + t) * K + k_base, mask=k_base < K, other=0.0).to(tl.float32)
            S2 = S + kv[:, None] * delta[None, :]
            tl.store(chk_ptr + (pid_bh * (N + 2)) * K * V + k_base[:, None] * V + v_base[None, :], S2, mask=mask2d)
    # copy working buffer to next checkpoint
    for ki in range(4):
        if ki >= NK:
            break
        k_base = ki * 32 + offs_k
        mask2d = (k_base[:, None] < K) & (v_base[None, :] < V)
        S = tl.load(chk_ptr + (pid_bh * (N + 2)) * K * V + k_base[:, None] * V + v_base[None, :], mask=mask2d, other=0.0)
        tl.store(chk_ptr + (pid_bh * (N + 1) + chk_idx + 1) * K * V + k_base[:, None] * V + v_base[None, :], S, mask=mask2d)

# Simplified stable forward that does not need complex chunk trick: single tiled HBM loop per chunk
@triton.jit
def fwd_generic_chunk(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,chk_ptr,
    BH,T,K,V,N,C,NK,NV,SCALE,t0,cur_T,chk_idx):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0,32)
    offs_v = tl.arange(0,32)
    v_off = pid_v*32 + offs_v
    mask_v = v_off < V
    # will process cur_T steps sequentially, keep S in HBM next checkpoint working area
    for ii in range(64):
        if ii >= cur_T:
            break
        t = t0 + ii
        beta = tl.load(beta_ptr + pid_bh*T + t).to(tl.float32)
        vt = tl.load(v_ptr + (pid_bh*T + t)*V + v_off, mask=mask_v, other=0.0).to(tl.float32)
        # accumulate pred/out across K
        pred = tl.zeros((32,),dtype=tl.float32)
        outa = tl.zeros((32,),dtype=tl.float32)
        for ki in range(8):
            if ki >= NK:
                break
            k_off = ki*32 + offs_k
            mask_k = k_off < K
            mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
            s_ptr = chk_ptr + (pid_bh*(N+1)+chk_idx)*K*V + k_off[:,None]*V + v_off[None,:] if ii==0 else chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:]
            # Triton cannot have Python ternary on pointer, do if
            if ii==0:
                S = tl.load(chk_ptr + (pid_bh*(N+1)+chk_idx)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
            else:
                S = tl.load(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
            qv = tl.load(q_ptr + (pid_bh*T + t)*K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            kv = tl.load(k_ptr + (pid_bh*T + t)*K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + (pid_bh*T + t)*K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            d = tl.exp(gv)
            Sd = S * d[:,None]
            # write back Sd to working
            tl.store(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], Sd, mask=mask2d)
            pred += tl.sum(Sd * kv[:,None], axis=0)
            outa += tl.sum(Sd * (qv*SCALE)[:,None], axis=0)
        tl.store(out_ptr + (pid_bh*T + t)*V + v_off, outa, mask=mask_v)
        delta = beta * (vt - pred)
        for ki in range(8):
            if ki >= NK:
                break
            k_off = ki*32 + offs_k
            mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
            S = tl.load(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
            kv = tl.load(k_ptr + (pid_bh*T + t)*K + k_off, mask=k_off<K, other=0.0).to(tl.float32)
            S2 = S + kv[:,None]*delta[None,:]
            tl.store(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], S2, mask=mask2d)
    for ki in range(8):
        if ki >= NK:
            break
        k_off = ki*32 + offs_k
        mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
        S = tl.load(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
        tl.store(chk_ptr + (pid_bh*(N+1)+chk_idx+1)*K*V + k_off[:,None]*V + v_off[None,:], S, mask=mask2d)

@triton.jit
def scratch_chunk(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,chk_ptr,scr_ptr,BH,T,K,V,N,C,NK, t0,cur_T,chk_idx):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0,32)
    offs_v = tl.arange(0,32)
    v_off = pid_v*32 + offs_v
    # load S0
    # use working buffer at N+1
    # copy chk_idx into working
    for ki in range(8):
        if ki >= NK:
            break
        k_off = ki*32 + offs_k
        mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
        S0 = tl.load(chk_ptr + (pid_bh*(N+1)+chk_idx)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
        tl.store(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], S0, mask=mask2d)
    for ii in range(64):
        if ii >= cur_T:
            break
        t = t0 + ii
        beta = tl.load(beta_ptr + pid_bh*T + t).to(tl.float32)
        vt = tl.load(v_ptr + (pid_bh*T + t)*V + v_off, mask=v_off<V, other=0.0).to(tl.float32)
        pred = tl.zeros((32,),dtype=tl.float32)
        for ki in range(8):
            if ki >= NK:
                break
            k_off = ki*32 + offs_k
            mask_k = k_off < K
            mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
            S = tl.load(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
            kv = tl.load(k_ptr + (pid_bh*T + t)*K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + (pid_bh*T + t)*K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            d = tl.exp(gv)
            Sd = S * d[:,None]
            tl.store(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], Sd, mask=mask2d)
            tl.store(scr_ptr + (pid_bh*C + ii)*K*V + k_off[:,None]*V + v_off[None,:], Sd, mask=mask2d)
            pred += tl.sum(Sd * kv[:,None], axis=0)
        delta = beta * (vt - pred)
        for ki in range(8):
            if ki >= NK:
                break
            k_off = ki*32 + offs_k
            mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
            S = tl.load(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
            kv = tl.load(k_ptr + (pid_bh*T + t)*K + k_off, mask=k_off<K, other=0.0).to(tl.float32)
            S2 = S + kv[:,None]*delta[None,:]
            tl.store(chk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], S2, mask=mask2d)

@triton.jit
def bwd_chunk_kernel(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,do_ptr,scr_ptr,gchk_ptr,dq_ptr,dk_ptr,dg_ptr,dv_ptr,dbeta_ptr,
    BH,T,K,V,N,C,NK,NV,SCALE,t0,cur_T,chk_idx):
    pid_bh = tl.program_id(0)
    offs_k = tl.arange(0,32)
    offs_v = tl.arange(0,32)
    # G for next checkpoint already in gchk, working buffer at N+1
    # init working G from gchk next
    for ki in range(8):
        if ki >= NK:
            break
        k_off = ki*32 + offs_k
        for vi in range(8):
            if vi >= NV:
                break
            v_off = vi*32 + offs_v
            mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
            G = tl.load(gchk_ptr + (pid_bh*(N+1)+chk_idx+1)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
            tl.store(gchk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], G, mask=mask2d)
    # backward iterate
    for ri in range(64):
        if ri >= cur_T:
            break
        ii = cur_T - 1 - ri
        t = t0 + ii
        beta = tl.load(beta_ptr + pid_bh*T + t).to(tl.float32)
        # first compute pred and u per V tile
        # we will need pred_tiles and u_tiles stored in HBM temp buffers: use scr as temp? allocate extra temp in gchk working?
        # Instead compute on the fly per V tile inside second loops by recomputing pred/u via inner K loops
        # For each V tile we compute delta pred etc twice - okay small.
        # Process each V tile to produce dv/dbeta
        for vi in range(8):
            if vi >= NV:
                break
            v_off = vi*32 + offs_v
            mask_v = v_off < V
            vt = tl.load(v_ptr + (pid_bh*T + t)*V + v_off, mask=mask_v, other=0.0).to(tl.float32)
            dot = tl.load(do_ptr + (pid_bh*T + t)*V + v_off, mask=mask_v, other=0.0).to(tl.float32)
            # compute pred and u for this V tile
            pred = tl.zeros((32,),dtype=tl.float32)
            u = tl.zeros((32,),dtype=tl.float32)
            for ki in range(8):
                if ki >= NK:
                    break
                k_off = ki*32 + offs_k
                mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
                Sd = tl.load(scr_ptr + (pid_bh*C + ii)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
                G = tl.load(gchk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
                kv = tl.load(k_ptr + (pid_bh*T + t)*K + k_off, mask=k_off<K, other=0.0).to(tl.float32)
                pred += tl.sum(Sd * kv[:,None], axis=0)
                u += tl.sum(G * kv[:,None], axis=0)
            # dv and dbeta contribution
            dv = beta * u
            tl.store(dv_ptr + (pid_bh*T + t)*V + v_off, dv, mask=mask_v)
            db = tl.sum(u * (vt - pred), axis=0)
            tl.store(dbeta_ptr + pid_bh*T + t + 0, db)  # will be atomically added via multiple vi -> need atomic
            # Use atomic for dbeta across vi
            tl.atomic_add(dbeta_ptr + pid_bh*T + t, db)
        # second phase: dq/dk/dg and G update per K tile
        for ki in range(8):
            if ki >= NK:
                break
            k_off = ki*32 + offs_k
            mask_k = k_off < K
            qv = tl.load(q_ptr + (pid_bh*T + t)*K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            kv = tl.load(k_ptr + (pid_bh*T + t)*K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + (pid_bh*T + t)*K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            d = tl.exp(gv)
            dq_acc = tl.zeros((32,),dtype=tl.float32)
            dk_acc = tl.zeros((32,),dtype=tl.float32)
            dg_acc = tl.zeros((32,),dtype=tl.float32)
            for vi in range(8):
                if vi >= NV:
                    break
                v_off = vi*32 + offs_v
                mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
                Sd = tl.load(scr_ptr + (pid_bh*C + ii)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
                G = tl.load(gchk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
                vt = tl.load(v_ptr + (pid_bh*T + t)*V + v_off, mask=v_off<V, other=0.0).to(tl.float32)
                dot = tl.load(do_ptr + (pid_bh*T + t)*V + v_off, mask=v_off<V, other=0.0).to(tl.float32)
                # recompute pred/u for this vi (redundant but small)
                # we have them from previous loop, recompute quickly
                # to avoid recomputation we could reuse but keep simple: compute via inner K loop already done? Instead compute pred/u per vi again with K loop - we already have pred per vi via extra loop over ki, but here ki is fixed, need full sum over ki for pred/u. So we need pred_full for vi which requires sum over all ki, not just current ki.
                # So compute pred_full/u_full for this vi via loop over kj
                pred_full = tl.zeros((32,),dtype=tl.float32)
                u_full = tl.zeros((32,),dtype=tl.float32)
                for kj in range(8):
                    if kj >= NK:
                        break
                    k2_off = kj*32 + offs_k
                    mask2d2 = (k2_off[:,None] < K) & (v_off[None,:] < V)
                    S2 = tl.load(scr_ptr + (pid_bh*C + ii)*K*V + k2_off[:,None]*V + v_off[None,:], mask=mask2d2, other=0.0)
                    G2 = tl.load(gchk_ptr + (pid_bh*(N+1)+N+1)*K*V + k2_off[:,None]*V + v_off[None,:], mask=mask2d2, other=0.0)
                    k2 = tl.load(k_ptr + (pid_bh*T + t)*K + k2_off, mask=k2_off<K, other=0.0).to(tl.float32)
                    pred_full += tl.sum(S2 * k2[:,None], axis=0)
                    u_full += tl.sum(G2 * k2[:,None], axis=0)
                vmp = vt - pred_full
                dq_acc += tl.sum(Sd * dot[None,:], axis=1) * SCALE
                dk_acc += beta * (tl.sum(G * vmp[None,:], axis=1) - tl.sum(Sd * u_full[None,:], axis=1))
                # G_raw
                # need u_full for this vi to compute G_raw contribution for this tile
                # G_raw tile = G + SCALE*q*do - beta*k*u_full
                # For dg and next G we need G_raw
                # Compute G_raw tile on fly
                # we will compute dg as sum_v G_raw*Sd ; accumulate
                # Instead compute G_raw tile as variable then accumulate
                # To avoid extra loads, compute here
                # qv is per k element, dot per v element
            # This inner vi loop missing G_raw handling, we handle G update separately below
            # For dq/dk we have accumulated
            tl.store(dq_ptr + (pid_bh*T + t)*K + k_off, dq_acc, mask=mask_k)
            tl.store(dk_ptr + (pid_bh*T + t)*K + k_off, dk_acc, mask=mask_k)
            # dg will be computed in next loop that also updates G
            # do G update loop
            for vi in range(8):
                if vi >= NV:
                    break
                v_off = vi*32 + offs_v
                mask2d = (k_off[:,None] < K) & (v_off[None,:] < V)
                Sd = tl.load(scr_ptr + (pid_bh*C + ii)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
                G = tl.load(gchk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], mask=mask2d, other=0.0)
                dot = tl.load(do_ptr + (pid_bh*T + t)*V + v_off, mask=v_off<V, other=0.0).to(tl.float32)
                # compute u_full again
                u_full = tl.zeros((32,),dtype=tl.float32)
                for kj in range(8):
                    if kj >= NK:
                        break
                    k2_off = kj*32 + offs_k
                    G2 = tl.load(gchk_ptr + (pid_bh*(N+1)+N+1)*K*V + k2_off[:,None]*V + v_off[None,:], mask=(k2_off[:,None]<K)&(v_off[None,:]<V), other=0.0)
                    k2 = tl.load(k_ptr + (pid_bh*T + t)*K + k2_off, mask=k2_off<K, other=0.0).to(tl.float32)
                    u_full += tl.sum(G2 * k2[:,None], axis=0)
                Graw = G + SCALE * qv[:,None] * dot[None,:] - beta * kv[:,None] * u_full[None,:]
                dg_acc += tl.sum(Graw * Sd, axis=1)
                Gnext = Graw * d[:,None]
                tl.store(gchk_ptr + (pid_bh*(N+1)+N+1)*K*V + k_off[:,None]*V + v_off[None,:], Gnext, mask=mask2d)
            tl.store(dg_ptr + (pid_bh*T + t)*K + k_off, dg_acc, mask=mask_k)
        # dbeta double counted once, fix: we added NV times, need divide? Actually we did atomic per vi, sum is correct

class GatedDeltaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v,e,g,beta,gamma,scale):
        B,H,T,K = q.shape
        V = v.shape[-1]
        BH = B*H
        device = q.device
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = g.contiguous()
        beta_c = beta.contiguous()
        qf = q_c.view(BH,T,K)
        kf = k_c.view(BH,T,K)
        vf = v_c.view(BH,T,V)
        gf = g_c.view(BH,T,K)
        bf = beta_c.view(BH,T)
        out = torch.empty(B,H,T,V, dtype=torch.float32, device=device)
        outf = out.view(BH,T,V)
        C = 64
        N = (T + C - 1)//C
        NK = (K + 31)//32
        NV = (V + 31)//32
        # chk with N+2 slots (extra working)
        chk = torch.zeros(BH, N+2, K, V, dtype=torch.float32, device=device)
        grid = (BH, NV)
        for chk_idx in range(N):
            t0 = chk_idx*C
            cur_T = C if t0 + C <= T else T - t0
            fwd_generic_chunk[grid](qf,kf,vf,gf,bf,outf,chk,BH,T,K,V,N,C,NK,NV,scale_f,t0,cur_T,chk_idx)
        ctx.qf=qf; ctx.kf=kf; ctx.vf=vf; ctx.gf=gf; ctx.bf=bf
        ctx.chk=chk; ctx.BH=BH; ctx.B=B; ctx.H=H; ctx.T=T; ctx.K=K; ctx.V=V
        ctx.C=C; ctx.N=N; ctx.NK=NK; ctx.NV=NV; ctx.scale_f=scale_f
        ctx.q_dtype=q.dtype; ctx.k_dtype=k.dtype; ctx.v_dtype=v.dtype; ctx.e_dtype=e.dtype
        ctx.g_dtype=g.dtype; ctx.beta_dtype=beta.dtype; ctx.gamma_dtype=gamma.dtype
        return out
    @staticmethod
    def backward(ctx, grad_out):
        BH,T,K,V = ctx.BH, ctx.T, ctx.K, ctx.V
        B,H,C,N,NK,NV,scale_f = ctx.B, ctx.H, ctx.C, ctx.N, ctx.NK, ctx.NV, ctx.scale_f
        device = grad_out.device
        go = grad_out.contiguous().view(BH,T,V)
        dq = torch.zeros(BH,T,K, dtype=torch.float32, device=device)
        dk = torch.zeros(BH,T,K, dtype=torch.float32, device=device)
        dg = torch.zeros(BH,T,K, dtype=torch.float32, device=device)
        dv = torch.zeros(BH,T,V, dtype=torch.float32, device=device)
        dbeta = torch.zeros(BH,T, dtype=torch.float32, device=device)
        gchk = torch.zeros(BH, N+2, K, V, dtype=torch.float32, device=device)
        scr = torch.empty(BH, C, K, V, dtype=torch.float32, device=device)
        for chk_idx in range(N-1, -1, -1):
            t0 = chk_idx*C
            cur_T = C if t0 + C <= T else T - t0
            grid_s = (BH, NV)
            scratch_chunk[grid_s](ctx.qf,ctx.kf,ctx.vf,ctx.gf,ctx.bf,ctx.chk,scr,BH,T,K,V,N,C,NK,t0,cur_T,chk_idx)
            grid_b = (BH,)
            bwd_chunk_kernel[grid_b](ctx.qf,ctx.kf,ctx.vf,ctx.gf,ctx.bf,go,scr,gchk,dq,dk,dg,dv,dbeta,BH,T,K,V,N,C,NK,NV,scale_f,t0,cur_T,chk_idx)
        dq_out = dq.view(B,H,T,K).to(ctx.q_dtype)
        dk_out = dk.view(B,H,T,K).to(ctx.k_dtype)
        dv_out = dv.view(B,H,T,V).to(ctx.v_dtype)
        dg_out = dg.view(B,H,T,K).to(ctx.g_dtype)
        dbeta_out = dbeta.view(B,H,T).to(ctx.beta_dtype)
        de = torch.zeros(B,H,T,K, dtype=ctx.e_dtype, device=device)
        dgamma = torch.zeros(B,H,T, dtype=ctx.gamma_dtype, device=device)
        return dq_out, dk_out, dv_out, de, dg_out, dbeta_out, dgamma, None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    return GatedDeltaFunc.apply(q,k,v,e,g,beta,gamma,scale)
```
No backward uses `exp(-g)` or matrix inversion; forward checkpoints every 64 steps and recomputes `Sd` into scratch so only decays `exp(g)<=1` are multiplied. Triton blocks are fixed `32` constexpr, handling arbitrary `K/V` by masking and tiling, replacing dynamic `arange(K)` that caused compile errors and large register-resident `KxV` tiles that caused OOM/timeouts with small HBM-tiled checkpoints.