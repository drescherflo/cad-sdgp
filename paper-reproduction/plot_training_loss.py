import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.ndimage import uniform_filter1d

base = Path("/home/flo/custom_dataset/ROCA_Outputs")
out  = Path(__file__).parent / "figures" / "training_loss.pdf"

models = [
    ("6-DOF_Augmentation",          "Freespace",     0, False),
    ("Conveyor_Augmentation",       "Conveyor",      1, False),
    ("6-DOF_Conveyor_Augmentation", "Freespace_Conveyor", 2, False),
]

palette = ["#1f77b4", "#d62728", "#2ca02c"]

pat = re.compile(r'iter: (\d+).*?total_loss: ([0-9.]+)')

def parse(path):
    iters, losses = [], []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(iters), np.array(losses)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

fig, ax = plt.subplots(figsize=(5.5, 1.6))

SMOOTH = 50  # rolling window in log-steps (each step = 20 iterations)

for dirname, label, cidx, is_roca in models:
    iters, losses = parse(base / dirname / "terminal_log.txt")
    smoothed = uniform_filter1d(losses.astype(float), size=SMOOTH, mode='nearest')
    ls = "--" if is_roca else "-"
    lw = 1.0 if is_roca else 1.4
    ax.plot(iters, smoothed, color=palette[cidx], linestyle=ls, linewidth=lw,
            label=label, alpha=0.9)

ax.axvline(60000, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
ylim = ax.get_ylim()
ax.text(60500, ylim[1] * 0.97, r"LR$\times$0.1", fontsize=6.5,
        color="gray", va="top", ha="left")

ax.set_xlim(0, 80000)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
ax.set_xlabel("Iteration")
ax.set_ylabel("Total Loss")
ax.set_title("Training Loss Curves", pad=6)
ax.grid(True, alpha=0.25, linewidth=0.5)

ax.legend(ncol=3, framealpha=0.9, edgecolor="0.8",
          columnspacing=1.0, handlelength=2.0)

plt.tight_layout(pad=0.4)
plt.savefig(out, format="pdf", bbox_inches="tight")
print(f"Saved: {out}")
