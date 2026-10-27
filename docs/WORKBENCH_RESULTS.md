# Structural workbench validation — October 27, 2026

All three variants completed generation, post-selection checking, paired training and export on one RunPod NVIDIA L4.
These are existing structural templates, not newly discovered architectures.

Shared cap: $3 model tokens and 65 minutes. Actual recorded model-token estimate: $0.0267.
Experiment elapsed: 25.4 minutes. Estimated GPU experiment cost: $0.250
at $0.59/hour, excluding rental setup/idle time. Billing may differ from token-price estimates.

Training: 100 steps per run; seeds 0, 1, 2; eight validation batches; batch 2 × 64 tokens; one layer,
width 64, two heads of width 16. Reference and kernel use matched initialization, batches and optimizer.
The reference uses checkpointing. Each variant received the same step/token budget and a share of the common generation cap.

| Variant | Reference validation loss, mean ± SD | Kernel validation loss, mean ± SD | Median paired step speedup | Peak GB, ref → kernel | Generation |
|---|---|---|---|---|---|
| Erase then write | 2.8674 ± 0.0353 | 2.8676 ± 0.0358 | 16.0× | 0.0286 → 0.0286 | 5.5 min |
| Write then erase | 2.8668 ± 0.0330 | 2.8669 ± 0.0333 | 15.4× | 0.0286 → 0.0286 | 8.8 min |
| Value-side erase | 2.8636 ± 0.0371 | 2.8638 ± 0.0374 | 15.8× | 0.0286 → 0.0286 | 6.0 min |

The largest paired final-validation difference was 0.000685. These short runs verify the implementation
path; their losses do not establish architectural superiority. The small workload does not show a meaningful
memory advantage. Generation cost is not amortized within these 100-step runs; per-seed break-even estimates
are recorded in `summary.json`. The prespecified 2.5 validation target was not reached.

## An actual rejected candidate and repair

`variant_2/run/02_bwd.py` passed forward generation but failed all 27 forward-plus-gradient development cases.
The observed case reported query-gradient error 3.07 for gradients up to 1.02. The model received these failures
and produced `03_repair_bwd.py`, which passed all 27 development cases. All selected variants then passed
12 fresh post-selection cases each, sampled after selection and never returned as repair feedback.
No kernel was hand-edited and no failure was inserted for the demo.

The raw source, candidates, prompts/responses, check results, hashes, environment and all 18 training logs
are in `runs_public/workbench/structural_matrix_01/`. `config.json` fixes the settings. Generate a portable bundle from this directory using
`kernel_contract.workbench.bundle.export_bundle`; the live workbench run also has an exported ZIP.
The support code in this run's bundle was collected at export; references and candidate bytes were frozen
before their checks and training. Future runs snapshot support code at experiment start.
