# Event contribution manifest

Upstream base: `e0647170da36ef9b059ac0bd3d60103aa4ed378b` from Meta's KernelAgent. The upstream agent,
Fuser, optimization pipeline, prior UI, oink, and their original tests remain credited to their authors.
They are not new hackathon features.

New event work is in:

- `kernel_contract/`: reference contracts, numerical/gradient checking, recurrence generation/repair,
  benchmark/training evaluation, event dashboard, and the research workbench.
- `muse_kernel/`: reference observation, guarded dispatch, environment-bound caches and training recovery.
- `utils/providers/meta_provider.py` and `budget.py`: Meta model adapters and shared spend accounting.
- `examples/muse_kernel/`, `.agents/skills/fast-recurrence/`, and corresponding new tests.
- `runs_public/`: recorded model calls, candidates, checks and training evidence. Historical evidence is
  explicitly distinguished from newly executed workbench validation.
- New README material, workbench documentation and visualization assets. Upstream material remains labeled.

The official event page begins October 26, 2026; the participant guide separately lists the building period
as October 27–November 4. Commit history preserves the actual October 26 start of local event work. Do not
change timestamps or describe preexisting upstream features as event contributions. Submission materials
should identify the exact demonstrated commits and clarify event timing with organizers if required.

Triadic GDN is prior research with an official H100/sm90 implementation. This project evaluates generated
Triton implementations for its own reference on an L4. It does not claim invention of Triadic GDN, absence
of prior kernels, or equivalence to every feature of the authors' implementation.
