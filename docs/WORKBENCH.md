# Recurrence workbench

Edit the computation, not merely its coefficients. The starting templates change erase direction, the
order of memory updates, or whether the current output can see the current write. They are existing
research/benchmark examples, not architecture discoveries. The source editor accepts a researcher-edited
PyTorch reference; the exact source is hashed before generation.

## Run locally with a CUDA worker

Install the same checkout on both machines (`pip install -e '.[research]'` on CUDA Linux). Put
`MODEL_API_KEY` or `BASETEN_API_KEY` only on the machine calling models. Then:

```bash
recurrence-workbench --remote 'ssh gpu-host' --remote-dir /workspace/recurrence
```

Open the printed localhost address. Choose or edit one to three references, set a shared model budget,
training seeds and steps, and run. No model call is made by opening the UI or inspecting recorded runs.
The model budget is enforced using atomic reservations across writers and variants. A separate wall-clock
limit bounds worker commands; GPU rental billing is not controlled by the dashboard. Stop/delete rented
GPUs yourself after retrieving outputs. GPU cost estimates exclude setup, idle time and storage.

The interface binds to loopback and requires a session token for mutations. References and generated
kernels are executable code: use an isolated worker without credentials or unrelated data. Process
isolation and reference-interface validation are not a security sandbox.

## Supported research interface

The current LM adapter uses `reference(q, k, v, e, g, beta, gamma, scale)`, returning `[B,H,T,V]`.
Inputs q/k/v/e are bf16, decay/gates fp32; k/e are normalized, g is nonpositive, beta/gamma are in [0,1].
Channel decay is required. The editor supports structural changes within this adapter, not arbitrary
model graphs or automatic adaptation of new state/input interfaces. Additional state dimensions such as
Triadic GDN require their own explicit input/training adapter; the historical Triadic result remains
available separately.

Research hypotheses belong in the reference and equation description. Neither the agent nor the UI
claims that a new update rule is scientifically better merely because its implementation is faster.

## Evidence flow

1. Freeze source, configuration, shape/dtype/layout contract and training budget.
2. Generate a forward kernel, evaluate it, return counterexamples for repair, then generate backward.
3. Select an accepted candidate by its exact hash. Keep every rejected attempt.
4. Draw fresh post-selection holdout seeds and combined domain/layout cases. No holdout feedback goes
   back to the writer. A holdout failure prevents training and remains in the comparison.
5. Train reference and selected kernel with matched initialization/batches for each seed, alternating
   execution order. Evaluate multiple batches from a disjoint validation split.
6. Report loss, memory, training step time, cold/cached time to target, and amortization; export receipts.

Generation is not guaranteed to fail before it succeeds. The UI shows actual failures if they occur and
never inserts a staged rejection to make a demo look agentic.

## Reproducible bundle

Every finished experiment, including failures, produces `reproducible-bundle.zip`. It includes source,
configuration, selected kernels, generation/check transcripts, environment and dataset hashes, training
logs, a content manifest, and `reproduce.py`. Unzip and run:

```bash
python reproduce.py                    # integrity verification, no model calls
pip install -e './source[research]'     # CUDA Linux
python reproduce.py --train            # repeat training from frozen kernels, no model calls
```

Integrity is not correctness. The training replay repeats the frozen numerical checks on the current
GPU before training and refuses a dataset whose hash differs from the original receipt. The manifest
records whether support code was snapshotted at experiment start or at export. Historical
records identify their earlier protocol and are never presented as new workbench runs.

## Recovery and caching

`muse_kernel.fast` only routes exact captured device/shape/dtype/stride/offset/grad signatures and checked
value domains. Caches bind reference dependencies, software/hardware identity and kernel bytes. Old or
changed-environment caches are not silently reused. Recreate the decorator after changing code/globals. To recheck a current-format committed cache on a new
CUDA environment without model calls, run `python -m muse_kernel.revalidate /path/to/cache`.

For backward recovery, use `muse_kernel.train_step(model, optimizer, loss_closure, kernels=[decorated_fn])`.
It restores RNG/buffers and discards partial gradients before retrying with references. The closure computes
loss only; it must not mutate parameters or call the optimizer. Fatal CUDA errors require worker restart.
