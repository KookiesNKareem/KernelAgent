---
name: fast-recurrence
description: Make a slow PyTorch function (typically a recurrent layer written as a Python loop over time) fast and trainable with a verified Triton kernel, using the muse_kernel decorator in this repository. Use when the user asks to speed up, write a kernel for, or make trainable a slow layer, recurrence, scan, or custom op in their PyTorch code.
---

# Fast recurrence

The user keeps their reference implementation. You add one decorator; a background build writes a forward and
backward Triton kernel, checks it against the reference in float64 (explicitly counted empirical cases and gradient checks), and swaps
it in only for calls it verified. Never hand-write the kernel yourself and never weaken the checker.

## Steps

1. Find the slow function. It must take tensors (and plain scalars) positionally and return a tensor.
   If the slow part is inside a module's `forward`, move it into a top-level function first and call that.
2. Add the decorator, and nothing else:

   ```python
   import muse_kernel

   @muse_kernel.fast
   def my_layer(q, k, v, g, beta, scale):
       ...  # unchanged
   ```

3. Run the user's training (or a few steps of it) on a CUDA machine. After two calls the decorator infers the
   contract from the real tensors (shapes, dtypes, unit-norm rows, ranges such as g <= 0 or beta in [0, 1])
   and starts a background build. Training keeps running on the reference meanwhile.
4. Watch `.muse_kernels/<function>_<hash>/status.json` next to the source file:
   `building` -> `ready` (kernel eligible for guarded use) or `failed` (reason given).
   `run/events.jsonl` has every attempt and the counterexamples the model was sent.
5. When it is `ready`, report accepted/unsupported/skipped/failed cases, forward+backward time versus a matched reference, and model cost separately from GPU cost.
   Call `my_layer.status()` for per-call routing counts (kernel vs reference and why).
6. A committed cache is reused only when source/dependencies, environment and kernel hashes match. A new
   GPU/software environment requires revalidation. Recreate the decorator after source edits.
7. For backward recovery, use `muse_kernel.train_step` with the decorated functions and a loss-only closure.
   Fatal CUDA context failures require restarting the worker; ordinary forward fallback cannot catch a later backward error.

## Settings (environment)

- `MUSE_KERNEL_MODEL` (default `muse-spark-1.3-contributor,muse-wonder-1.0`)
- `MUSE_KERNEL_MAX_USD` per model attempt (default 4). The workbench additionally enforces one shared experiment cap. Spend is tallied on the machine that runs the build.
- `MODEL_API_KEY` must be set (Meta Model API).
