"""Edit the recurrence, then run this file on CUDA. See docs/EDITOR_DEMO.md."""
import json
import time

import torch
import muse_kernel
from kernel_contract.train_demo import make_lm, data, batch, loss_of

CHANNEL_DECAY = True


@muse_kernel.fast
def reference(q, k, v, e, g, beta, gamma, scale):
    """Write the prediction error, then erase memory."""
    B, H, T, K = q.shape
    tensors = (q, k, v, e, g, beta, gamma)
    q, k, v, e, g, beta, gamma = (x.to(torch.promote_types(x.dtype, torch.float32)) for x in tensors)
    state = q.new_zeros(B, H, K, v.shape[-1])
    outputs = []
    for t in range(T):
        key, value, erase = k[:, :, t], v[:, :, t], e[:, :, t]
        state = state * g[:, :, t].exp()[..., None]

        # Write: add the value prediction error at the current key.
        prediction = (state * key[..., None]).sum(-2)
        state = state + key[..., None] * (beta[:, :, t, None] * (value - prediction))[..., None, :]

        # Erase: remove memory along the erase direction.
        state = state - gamma[:, :, t, None, None] * erase[..., None] * (
            state * erase[..., None]
        ).sum(-2)[..., None, :]

        outputs.append((state * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outputs, dim=2)


def main():
    torch.manual_seed(0)
    model = make_lm(__name__, reference, device='cuda', d=64, layers=1, heads=2, head_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = data('cuda')
    train = tokens[:int(len(tokens)*.9)]
    rng = torch.Generator(device='cuda').manual_seed(1)

    def step():
        inputs = batch(train, 2, 64, rng)
        loss, recovered = muse_kernel.train_step(
            model, optimizer, lambda: loss_of(model, inputs), kernels=[reference]
        )
        torch.cuda.synchronize()
        return float(loss), recovered

    # Observe two real training calls. Pause the demo during GPU checking so its timings stay isolated.
    for _ in range(2):
        step()
    print(json.dumps({'phase':'captured', 'hash':reference.hash, 'cache':str(reference.job)}), flush=True)
    deadline = time.monotonic() + 25*60
    previous = None
    while reference.status()['state'] not in ('ready','failed'):
        state = reference.status()['state']
        if state != previous:
            print(json.dumps({'phase':state}), flush=True)
            previous = state
        if time.monotonic() > deadline:
            raise TimeoutError('Kernel generation did not finish')
        time.sleep(2)
    if reference.status()['state'] != 'ready':
        raise RuntimeError(reference.status())

    for i in range(30):
        started = time.perf_counter()
        loss, recovered = step()
        print(json.dumps({'step':i, 'loss':loss, 'step_ms':1000*(time.perf_counter()-started),
                          'recovered':recovered, 'calls':reference.status()['calls']}), flush=True)
    if not reference.status()['calls'].get('kernel',0):
        raise RuntimeError('The kernel never took over')
    print(json.dumps({'phase':'complete','status':reference.status()}), flush=True)


if __name__ == '__main__':
    main()
