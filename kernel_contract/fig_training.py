"""README figure: training loss of the kernel and the reference loop, by step and by wall-clock time.

  python -m kernel_contract.fig_training runs_public/train_triadic/fig_kernel.jsonl \\
      runs_public/train_triadic/fig_ckpt.jsonl assets/triadic_training.png
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KERNEL, REF = "#2a78d6", "#eb6834"


def load(path):
    rows = [json.loads(x) for x in open(path) if '"loss"' in x]
    ema, smooth, t, clock = None, [], 0.0, []
    for i, r in enumerate(rows):
        ema = r["loss"] if ema is None else 0.9 * ema + 0.1 * r["loss"]
        smooth.append(ema)
        t += r["step_s"] if i else 0  # the first step includes one-time compilation
        clock.append(t / 60)
    return [r["step"] for r in rows], smooth, clock


def main(kernel_path, ref_path, out):
    ks, kl, kt = load(kernel_path)
    rs, rl, rt = load(ref_path)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    labels = ("Muse Wonder kernel", "Reference loop (gradient checkpointing)")
    a.plot(ks, kl, color=KERNEL, linewidth=2, label=labels[0], zorder=3)
    a.plot(rs, rl, color=REF, linewidth=5, label=labels[1], zorder=2)
    a.set_xlabel("training step")
    a.set_ylabel("training loss (smoothed)")
    a.set_title("Same learning, step for step", loc="left", color=INK, fontsize=11)
    a.legend(frameon=False, loc="upper right", labelcolor=INK)

    b.plot(rt, rl, color=REF, linewidth=2)
    b.plot(kt, kl, color=KERNEL, linewidth=2)
    b.set_xlabel("wall-clock minutes")
    b.set_title(f"{rt[-1] / kt[-1]:.0f}x less time to train", loc="left", color=INK, fontsize=11)
    b.annotate(f"Kernel: {len(ks)} steps in {kt[-1] * 60:.0f} s", (kt[-1], kl[-1]), xytext=(10, 0),
               textcoords="offset points", va="center", color=INK, fontsize=9.5)
    b.annotate(f"Reference: {len(rs)} steps in {rt[-1]:.1f} min", (rt[-1], rl[-1]), xytext=(0, 14),
               textcoords="offset points", ha="right", color=INK, fontsize=9.5)
    for x, y, c in ((kt[-1], kl[-1], KERNEL), (rt[-1], rl[-1], REF)):
        b.plot(x, y, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    fig.suptitle("Training a 4-layer LM with Triadic Gated DeltaNet (arXiv 2609.36529) on one NVIDIA L4",
                 x=0.06, ha="left", color=INK, fontsize=12)
    fig.text(0.06, 0.01, "Batch 8 x 512 bytes, same init and batches; wall clock excludes the first step's one-time "
             "compile.\nThe plain PyTorch loop runs out of memory at this size; FLA does not support the layer.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    main(*sys.argv[1:4])
