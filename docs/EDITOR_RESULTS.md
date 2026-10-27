# Editor and decorator validation — October 27, 2026

The runnable `examples/muse_kernel/research.py` trains a byte-level LM on an NVIDIA L4.
The reference carries `@muse_kernel.fast`; the training loop uses `muse_kernel.train_step`.
Model credentials remained on the local controller. The worker captured the actual model inputs,
including non-contiguous projection strides and storage offsets, rather than a synthetic contiguous call.

The cold run captured two reference training calls, generated and checked forward/backward kernels,
then routed all 30 subsequent training steps through the selected kernel. Its initial forward candidate
failed to import and was repaired by the model. The selected implementation passed 27 development cases
and 12 post-selection holdout cases. No kernel was manually edited.

A second invocation reused the same source/environment-bound cache: all 32 calls used the kernel,
with **zero model calls and $0 additional model-token cost**. The first run's model-token estimate was
$0.012247. Neither run required backward recovery; that path is covered by the injected-failure regression
check, which verifies that retrying before the optimizer update matches the reference optimizer result.

Raw scripts, captured contracts, generation transcripts, selected code, checks, routing counts and logs
are under `runs_public/editor/cold/` and `runs_public/editor/cached/`. The kernel SHA-256 is
`e8a3459f7c61cfe9d0ef2121f10385b28241e0f48542ca626222f978413781ae`.

This is an integration test of the researcher workflow. The stronger paired training comparison is the
[three-variant experiment](WORKBENCH_RESULTS.md). No architecture-discovery claim is made.

A separate source-edit check reversed erase/write order. It changed the source hash from
`80d4dce92143eb856dc192a9` to `d144ac93a1af80b517e5a0a1`; both calls stayed on the reference and queued
a new build. The existing kernel was not reused. This capture-only check made no model call.

The exported smoke bundle also passed integrity and numerical rechecks and replayed both training runs
on CUDA using its frozen dataset. Final validation losses matched the original receipts within 1e-6.
See `runs_public/editor/source_edit/`, `revalidation.json` and `bundle_replay.json`.
