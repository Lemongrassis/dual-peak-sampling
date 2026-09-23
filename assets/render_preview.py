"""Render the README figure using the actual sampler's analytic density.

Run after installing dual-peak-sampling and matplotlib.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dual_peak_sampling import DualPeakConfig, density


def main():
    config = DualPeakConfig()
    grid = torch.linspace(0.0001, 0.9999, 20000, dtype=torch.float64)
    curve = density(grid, config)
    candidates = torch.where((curve[1:-1] > curve[:-2]) & (curve[1:-1] > curve[2:]))[0] + 1
    peaks = [(float(grid[i]) * 1000, float(curve[i])) for i in candidates]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=180)
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.72, bottom=0.20)
    fig.text(.08, .90, "DUAL PEAK", fontsize=25, weight="bold", color="#202936")
    fig.text(.08, .835, "Two adjustable peaks. A broad background. Room between them.",
             fontsize=13, color="#586473")
    ax.plot(grid.numpy() * 1000, curve.numpy(), color="#2e61a8", lw=2.7)
    ax.fill_between(grid.numpy() * 1000, curve.numpy(), color="#2e61a8", alpha=.09)
    for (x, y), color, offset in zip(peaks, ["#2e61a8", "#b56b35"], [(-38, 25), (-58, 24)]):
        ax.scatter([x], [y], color=color, s=50, edgecolor="#fafaf8", linewidth=1.5, zorder=4)
        ax.annotate(f"~{round(x)}", (x, y), xytext=offset, textcoords="offset points",
                    fontsize=12, color=color, weight="bold",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": .9})
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 3.65)
    ax.set_xticks(range(0, 1001, 100))
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Noise level · timestep on a 0–1000 display", labelpad=10, color="#586473")
    ax.set_ylabel("Probability density in t", labelpad=10, color="#586473")
    ax.tick_params(colors="#6b7480", labelsize=10, length=0, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d7dce0")
    ax.grid(axis="y", color="#e3e6e8", lw=.8)
    ax.set_axisbelow(True)
    fig.text(.08, .045, "DEFAULT MIX   15% middle peak  /  35% high-noise peak  /  45% broad background  /  5% uniform",
             fontsize=10, color="#586473")
    for ext in ("png", "svg"):
        target = Path(__file__).with_name(f"dual-peak.{ext}")
        fig.savefig(target, facecolor=fig.get_facecolor())
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
