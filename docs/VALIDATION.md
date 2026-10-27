# Validation protocol

Historical benchmark coverage is not a proof of general correctness. Report accepted, refused, skipped,
wrong and crashed cases separately. Size refusals may leave a usable kernel for the observed workload;
they must never be described as every test passing. An accepted observed case is mandatory.

For new workbench experiments, freeze each reference and settings before generation. Holdout seeds are
sampled only after a candidate is selected, then tested on that exact hash. All fresh holdout cases must
pass before training. These cases are evaluation-only and are not supplied as repair feedback. Do not
repeatedly tune against a failed holdout and continue calling it held out.

Use at least three training seeds and eight validation batches per evaluation for submission evidence.
Pair reference/kernel initialization, data sampling, optimizer and validation batches. Preserve every
failure and all resource costs. A small smoke run verifies plumbing; it does not establish training quality
or a general performance claim. Report the shape, GPU, software versions, dataset hash, number of steps,
seed count, and validation batch count with every result.

Distinguish generation, compilation/checking and cached training time. Report time to a prespecified
validation target only when reached; never extrapolate a target from a short curve. Step speedup and
amortization estimates are descriptive and must not be labeled measured end-to-end turnaround gains.

Useful controls: the unchanged reference, checkpointed reference, warm torch.compile including its cold
compile cost, and the best mathematically applicable expert implementation. A direct-generation control
must use the same writer and budget and must receive no repair feedback. Compare success and total cost,
not just the fastest successful candidate. Structural templates already studied in this repository are
not fresh equations; independent researcher edits must be identified separately.

The contribution is reducing the implementation bottleneck between a model equation and an experiment.
Architecture superiority needs adequate held-out task evaluation beyond these correctness checks.
