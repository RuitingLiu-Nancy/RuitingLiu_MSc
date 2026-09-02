# -*- coding: utf-8 -*-
"""Graph-route variant candidate-access curves, drawn to match the Stage-1 figure."""
import csv, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main() -> None:
    D = collections.defaultdict(dict)
    with open("data.csv") as fh:
        for r in csv.DictReader(fh):
            D[r["method"]][int(r["depth"])] = float(r["oracle"])

    DEPTHS = sorted(D["dense_only"])
    MARKED = [8, 12, 20, 30, 40, 50]

    SERIES = [
        ("dense_only", "Dense only", "#1f5f9e", "o"),
        ("maintained_graph4", "Dense + primary Graph", "#159169", "s"),
        ("no_recognition_graph4", "Dense + no-recognition route", "#c26a12", "^"),
        ("fact_only_no_recognition_graph4", "Dense + fact-only route", "#8054c8", "D"),
    ]
    INK, MUTED, GRID = "#222222", "#555555", "#D9D9D9"

    fig, (ax, dx) = plt.subplots(
        2, 1, figsize=(13.2, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.10})
    fig.patch.set_facecolor("white")

    for key, label, colour, marker in SERIES:
        y = [D[key][m] for m in DEPTHS]
        ax.plot(DEPTHS, y, color=colour, linewidth=1.9, label=label, zorder=3)
        ax.plot(MARKED, [D[key][m] for m in MARKED], linestyle="none", marker=marker,
                markersize=6.2, color=colour, markeredgecolor="white",
                markeredgewidth=0.9, zorder=4)

    # lower panel: marginal gain of four graph candidates, against the same four
    # slots spent on dense instead
    for key, label, colour, marker in SERIES[1:]:
        y = [D[key][m] - D["dense_only"][m] for m in DEPTHS]
        dx.plot(DEPTHS, y, color=colour, linewidth=1.9, zorder=3)
        dx.plot(MARKED, [D[key][m] - D["dense_only"][m] for m in MARKED], linestyle="none",
                marker=marker, markersize=6.2, color=colour, markeredgecolor="white",
                markeredgewidth=0.9, zorder=4)

    matched = [m for m in DEPTHS if m + 4 in D["dense_only"]]
    dx.plot(matched, [D["dense_only"][m + 4] - D["dense_only"][m] for m in matched],
            color=MUTED, linewidth=1.7, linestyle=(0, (5, 2.5)), zorder=5,
            label="the same four slots spent on Dense instead")

    ax.set_ylabel("Oracle Utility@8", fontsize=12.5, labelpad=9)
    dx.set_ylabel("$\\Delta$ vs Dense\nat the same $M$", fontsize=10.6, labelpad=9)
    dx.set_xlabel("Candidate-pool depth, $M$", fontsize=12.5, labelpad=9)
    dx.axhline(0.0, color=MUTED, linewidth=0.9, zorder=1)
    dx.set_yscale("log")
    dx.set_ylim(0.006, 0.95)
    dx.set_yticks([0.01, 0.03, 0.1, 0.3])
    dx.set_yticklabels(["0.01", "0.03", "0.10", "0.30"])

    for a in (ax, dx):
        a.set_xlim(7.2, 51.2)
        a.set_xticks(MARKED)
        a.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85, zorder=0)
        a.grid(axis="x", visible=False)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_color(MUTED)
        a.spines["bottom"].set_color(MUTED)
        a.tick_params(colors="#333333", labelsize=10.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylim(4.49, 5.94)
    ax.annotate("the three variant curves are visually coincident:\nthey span .053 at $M{=}8$ and .0005 at $M{=}50$",
                xy=(20, 5.599), xytext=(26.5, 5.30), color=MUTED, fontsize=10.0, ha="left",
                arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.85})

    dx.annotate("four more Dense candidates always buy\nmore than four graph candidates",
                xy=(31, D["dense_only"][35] - D["dense_only"][31]), xytext=(33.0, 0.16),
                color=MUTED, fontsize=10.0, ha="left",
                arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.85})

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = dx.get_legend_handles_labels()
    fig.suptitle("Stage 1 — Graph route variants", x=0.068, y=0.978, ha="left",
                 fontsize=19, fontweight="bold", color=INK)
    fig.text(0.068, 0.938,
             "Development300 · E5 Dense prefix · four strict-native graph candidates added · higher is better",
             ha="left", fontsize=11.6, color=MUTED)
    fig.legend(handles=handles + h2, labels=labels + l2, loc="upper left",
               bbox_to_anchor=(0.062, 0.908), ncol=3, frameon=False,
               fontsize=11.0, handlelength=2.6, columnspacing=1.9, handletextpad=0.7)
    fig.subplots_adjust(left=0.088, right=0.985, bottom=0.095, top=0.800)

    fig.savefig("graph_variant_curves_e5.png", dpi=220, facecolor="white")
    fig.savefig("graph_variant_curves_e5.pdf", facecolor="white")
    print("saved")


if __name__ == "__main__":
    main()
