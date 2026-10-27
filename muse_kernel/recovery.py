"""Retry a failed forward/backward before the optimizer mutates parameters."""
import torch


def train_step(model, optimizer, loss_closure, kernels=(), max_grad_norm=None):
    """loss_closure computes a scalar loss only (no optimizer step or parameter mutation).

    Restores RNG and model buffers, clears partial gradients, disables the specified Fast wrappers,
    and retries the entire computation once on their references. Fatal CUDA errors are not recoverable.
    """
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    optimizer.zero_grad(set_to_none=True)
    recovered = False
    try:
        loss = loss_closure()
        loss.backward()
    except Exception as error:
        if not kernels or any(s in str(error).lower() for s in ('illegal memory access', 'device-side assert')):
            raise
        for kernel in kernels:
            kernel.disable(f'training step: {type(error).__name__}')
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        with torch.no_grad():
            for n, b in model.named_buffers():
                b.copy_(buffers[n])
        optimizer.zero_grad(set_to_none=True)
        loss = loss_closure()
        loss.backward()
        recovered = True
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss; optimizer was not stepped')
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    else:
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError('Nonfinite gradient; optimizer was not stepped')
    optimizer.step()
    return loss.detach(), recovered
