# The researcher interface: code first

Open `examples/muse_kernel/research.py` in a code editor. The researcher's interface is the PyTorch
function with `@muse_kernel.fast`; the workbench is an evidence viewer and experiment comparison surface.
The example trains a real byte-level language model and uses `train_step` so an ordinary backward failure
can discard partial gradients and retry on the reference before the optimizer changes parameters.

Run directly on a CUDA machine with model access:

```bash
MUSE_KERNEL_MODEL=muse-spark-1.3-contributor MUSE_KERNEL_MAX_USD=1 \
  python examples/muse_kernel/research.py
```

Or keep model credentials on your laptop and execute the edited file on an installed CUDA worker:

```bash
python -m muse_kernel.remote_run examples/muse_kernel/research.py \
  --remote 'ssh gpu-host' --remote-dir /workspace/recurrence \
  --out runs/editor-demo --max-usd 1 --minutes 30
```

The controller uploads the script, reads contracts captured from actual training calls, builds with Meta
models locally, checks on the worker, and publishes a passing kernel back into the decorator's cache.
It also draws fresh post-selection checks before publishing. Unchanged source and matching hardware/software
can reuse the guarded cache on a later invocation. Model API keys are never uploaded to the worker.
The demo pauses after two training calls while checks run, avoiding overlap with timing measurements;
ordinary decorator use can continue training on its reference while generation runs.

Use an actual structural edit: move the erase block below the write block, or change which state the
readout observes. Re-run the file. The source hash and contract change; the previous implementation cannot
silently stand in for the changed equation. The current example keeps the same tensor interface. New state
or input interfaces require an appropriate model adapter.

## Three-minute walkthrough

- **0:00–0:25:** Show the recurrence in the editor. Explain the model change by moving the actual update,
  not by changing a coefficient. Keep the decorator visible.
- **0:25–0:50:** Run the file. Show the captured contract and source hash from real training calls.
- **0:50–1:35:** Show a recorded generation trace, explicitly identifying the elapsed generation time.
  If the run has a real rejected candidate, inspect that failure and subsequent repair. If it does not,
  do not fabricate one; use a separately identified historical rejection.
- **1:35–2:05:** Show the decorator's routing counts changing from reference to kernel and training
  continuing. Show fresh forward/gradient checks, not just a green status.
- **2:05–2:40:** Open the three-variant comparison. Compare validation loss, time and memory under the
  same training settings. Include failed variants and generation costs.
- **2:40–3:00:** Export the bundle and run its integrity check. Close with the concrete contribution:
  reducing the implementation work between an equation and an experiment.

Generation takes minutes, not the length of the video. Label recorded evidence and time cuts honestly.
A small experiment does not establish architecture superiority or a general end-to-end speedup.

The remote cache lives under the worker's `.muse_kernels/remote/`. The controller retains a copy in the
output directory. To reuse that copy on a different environment, copy it into your chosen cache directory,
run `python -m muse_kernel.revalidate /path/to/reference_HASH` there, and set `MUSE_KERNEL_CACHE` to its parent.
Revalidation requires CUDA, checks the candidate bytes and both development and required holdout cases,
and does not call a model.
