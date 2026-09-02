#!/usr/bin/env python3
"""Stage-1 candidate-pool figure, revised per project owner 2026-08-25.

Changes vs the executor's version:
  * no grey subtitle;
  * clearer legend wording (routes spelled out; RRF variants disambiguated);
  * bottom zoom panel now includes the three-route RRF and an explicit
    Dense baseline zero line, alongside two-route RRF and CC.
Data: stage1_candidate_pool_oracle_curves.csv (E5, Development300), verbatim.
"""
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Run from the repository root: python figures/make_stage1_curves_rrf3.py
SRC = "out/rq2b_rrf3_stage1_curves_v1/stage1_candidate_pool_oracle_curves.csv"

SERIES = {  # method -> (legend label, color, marker, linewidth, zorder)
    "dense_only":        ("Dense only", "#1f4e9c", "o", 1.4, 5),
    "graph_only":        ("Graph only", "#d67ba8", "v", 1.1, 2),
    "dense_70_graph_30": ("Dense:Graph 70:30 interleave", "#2e9e6b", "s", 1.1, 2),
    "dense_50_graph_50": ("Dense:Graph 50:50 interleave", "#e0a63a", "D", 1.1, 2),
    "dense_30_graph_70": ("Dense:Graph 30:70 interleave", "#d1603d", "^", 1.1, 2),
    "rrf":               ("RRF fusion (dense + graph)", "#63b3e4", "+", 1.4, 4),
    "cc":                ("CC fusion (dense + graph)", "#6a51a3", "x", 1.1, 3),
    "rrf3":              ("RRF fusion (dense + lexical + graph)", "#111111", ".", 1.8, 6),
}
ZOOM = ["rrf3", "rrf", "cc"]  # delta panel series (dense = zero line)


def main() -> None:
    data = defaultdict(dict)
    for r in csv.DictReader(open(SRC)):
        if r["backend"] != "e5":
            continue
        data[r["method"]][int(r["pool_depth"])] = float(r["oracle_u8"])
    depths = sorted(data["dense_only"])

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(9.2, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.12})

    for m, (label, color, marker, lw, z) in SERIES.items():
        xs = [d for d in depths if d in data[m]]
        ys = [data[m][d] for d in xs]
        mark_every = [i for i, d in enumerate(xs) if d in (8, 12, 20, 30, 40, 50)]
        ax.plot(xs, ys, color=color, lw=lw, marker=marker, markersize=4.4,
                markevery=mark_every, label=label, zorder=z)

    ax.set_ylabel("Oracle Utility@8")
    ax.set_title("Stage 1 — Candidate-pool access", loc="left",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.4, ncol=2,
              handlelength=2.2, columnspacing=1.2)

    # --- zoom panel: paired difference vs the Dense baseline -------------------
    axd.axhline(0, color="#9a9a9a", lw=1.0)
    axd.annotate("Dense only (baseline = 0)", xy=(20.5, 0),
                 xytext=(20.5, -0.0235), fontsize=8, color="#666666")
    for m in ZOOM:
        label, color, marker, lw, z = SERIES[m]
        xs = [d for d in depths if d in data[m]]
        ys = [data[m][d] - data["dense_only"][d] for d in xs]
        axd.plot(xs, ys, color=color, lw=lw, zorder=z)
    axd.annotate("RRF (dense+lexical+graph) ends +.004 above Dense",
                 xy=(50, data["rrf3"][50] - data["dense_only"][50]),
                 xytext=(29.5, 0.052), fontsize=8, color="#111111",
                 arrowprops=dict(arrowstyle="->", color="#111111", lw=0.8))
    axd.annotate("RRF (dense+graph) = Dense at M=50",
                 xy=(50, data["rrf"][50] - data["dense_only"][50]),
                 xytext=(33.5, -0.028), fontsize=8, color="#3f88bd",
                 arrowprops=dict(arrowstyle="->", color="#63b3e4", lw=0.8))
    axd.set_ylabel("$\\Delta$ vs Dense")
    axd.set_xlabel("Candidate-pool depth, $M$")
    axd.grid(axis="y", color="#e6e6e6", lw=0.7)
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)
    axd.set_xticks([8, 12, 20, 30, 40, 50])
    axd.set_ylim(-0.035, 0.085)

    fig.savefig("stage1_candidate_pool_curves_rrf3.png", dpi=220, bbox_inches="tight")
    fig.savefig("stage1_candidate_pool_curves_rrf3.pdf", bbox_inches="tight")
    print("saved: rrf3 M=8 delta =", round(data["rrf3"][8] - data["dense_only"][8], 4),
          "| M=50 delta =", round(data["rrf3"][50] - data["dense_only"][50], 4))


if __name__ == "__main__":
    main()
