#!/usr/bin/env python3
"""Publication figures for the two Stage-2 selector sweeps.

Reads only the descriptive curves the matrix builder emitted; computes nothing.
Both panels plot the change in realised Utility@8 against Direct selection, so
the zero line is "do nothing beyond ranking by predicted utility".

An important caveat is drawn onto both figures rather than left to the caption:
these full-cohort curves are DESCRIPTIVE. The r* and beta* the thesis reports
come from the nested per-fold procedure and are marked here only so the reader
can see where the nested choice landed relative to the descriptive optimum --
the curve was never used to pick them.

Style follows figures/make_stage1_curves_rrf3.py: same figure scale,
frameless legend, muted palette, PNG at 220 dpi plus a vector PDF.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "out" / "stage2_selection_matrix_complete_v1"
FIGURES = ROOT / "figures"

# Main-text arms, in the order the instruction lists them.
REPLACEMENT_ARMS = [
    ("huber7d", "Huber (clean 7D)", "#3f88bd", "o", "-"),
    ("best_lightweight_nested", "Nested lightweight family selection",
     "#111111", "s", "-"),
    ("cross_encoder_matched", "Matched cross-encoder", "#c0504d", "^", "-"),
]
RESIDUAL_ARMS = REPLACEMENT_ARMS + [
    ("lm7d_lin_g7", "LambdaMART, linear gain, 7 grades", "#4f9153", "D", "--"),
    ("lm7d_exp_g7", "LambdaMART, exponential gain, 7 grades", "#b07aa1", "v", "--"),
]

ALL_MODEL_ORDER = [
    "lw_huber", "lw_ridge", "lw_elasticnet", "lw_hist_gbr",
    "lw_xgb_regression", "lw_catboost_regression", "lw_small_mlp",
    "lw_ranknet", "lw_xgb_pairwise", "lw_lambdamart_aligned",
    "lw_lgbm_lambdarank", "lw_catboost_yetirank",
    "best_lightweight_nested", "cross_encoder_matched",
]
ALL_MODEL_LABELS = {
    "lw_huber": "Huber", "lw_ridge": "Ridge",
    "lw_elasticnet": "ElasticNet", "lw_hist_gbr": "HistGBR",
    "lw_xgb_regression": "XGB regression",
    "lw_catboost_regression": "CatBoost regression",
    "lw_small_mlp": "Small MLP", "lw_ranknet": "RankNet",
    "lw_xgb_pairwise": "XGB pairwise",
    "lw_lambdamart_aligned": "LambdaMART",
    "lw_lgbm_lambdarank": "LightGBM LambdaRank",
    "lw_catboost_yetirank": "CatBoost YetiRank",
    "best_lightweight_nested": "Nested family selection",
    "cross_encoder_matched": "MiniLM cross-encoder",
}


def _read(name: str) -> list[dict[str, str]]:
    with (SOURCE / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _chosen() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for row in _read("STAGE2_SELECTION_MATRIX.csv"):
        out[(row["scorer"], "r")] = float(row["r_mean"])
        out[(row["scorer"], "beta")] = float(row["beta_mean"])
    return out


def _panel(ax, rows, arms, xkey, chosen, xlabel, title, cast):
    by_scorer: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        by_scorer.setdefault(row["scorer"], []).append(
            (cast(row[xkey]), float(row["delta_vs_direct"])))
    ax.axhline(0.0, color="#999999", linewidth=0.9, zorder=1)
    for scorer, label, colour, marker, style in arms:
        points = sorted(by_scorer[scorer])
        ax.plot([p[0] for p in points], [p[1] for p in points],
                linestyle=style, color=colour, linewidth=1.6, marker=marker,
                markersize=4.2, label=label, zorder=3)
        mark = chosen[(scorer, "r" if xkey == "r" else "beta")]
        ax.axvline(mark, color=colour, linewidth=0.8, linestyle=":", alpha=0.55,
                   zorder=2)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(r"$\Delta$ Utility@8 vs Direct", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=9)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _summary_map(path: Path, method: str, value: str = "chosen_mean") -> dict[str, float]:
    frame = pd.read_csv(path / "hyperparameter_summary.csv")
    return {
        str(row.scorer): float(getattr(row, value))
        for row in frame[frame.method.eq(method)].itertuples()
    }


def _small_multiples(
    rows: pd.DataFrame, x: str, title: str, xlabel: str,
    primary: dict[str, float], endpoint: dict[str, float] | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(4, 4, figsize=(12.2, 10.6), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    for ax, scorer in zip(axes_flat, ALL_MODEL_ORDER):
        view = rows[rows.scorer.eq(scorer)].sort_values(x)
        ax.axhline(0.0, color="#999999", linewidth=0.7)
        ax.plot(view[x], view.delta_vs_direct, color="#2563A5", linewidth=1.45,
                marker="o" if x == "r" else None, markersize=3.2)
        ax.axvline(primary[scorer], color="#D55E00", linestyle=":",
                   linewidth=1.0)
        if endpoint is not None:
            ax.axvline(endpoint[scorer], color="#009E73", linestyle="--",
                       linewidth=0.9)
        # Keep the model name inside its own axes. Matplotlib can otherwise
        # paint subplot titles underneath the neighbouring axes when a dense
        # shared-axis grid is exported to PDF/PNG.
        ax.text(.5, .96, ALL_MODEL_LABELS[scorer], transform=ax.transAxes,
                ha="center", va="top", fontsize=9.1, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2,
                      "alpha": .88}, zorder=10)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.55)
        ax.tick_params(labelsize=7.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes_flat[len(ALL_MODEL_ORDER):]:
        ax.axis("off")
    if x == "r":
        for ax in axes_flat[:len(ALL_MODEL_ORDER)]:
            ax.set_xticks(range(0, 9))
    else:
        for ax in axes_flat[:len(ALL_MODEL_ORDER)]:
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .25, .5, .75, 1])
    fig.suptitle(title, fontsize=15, fontweight="bold", y=.985)
    fig.supxlabel(xlabel, fontsize=11, y=.038)
    fig.supylabel(r"$\Delta$ Utility@8 vs Direct", fontsize=11)
    note = ("orange dotted = mean nested choice for r=1--7; green dashed = mean "
            "nested choice when r=8 is included; r=8 equals Direct"
            if endpoint is not None else
            r"orange dotted = mean nested choice $\beta^*$; $\beta=0$ equals Direct")
    fig.text(.5, .010, note, ha="center", fontsize=8.5, color="#555555")
    # Leave enough separation for the first-row model names below the title.
    fig.subplots_adjust(left=.075, right=.99, bottom=.095, top=.91,
                        wspace=.22, hspace=.34)
    return fig


def render_all_models(
    selected_sets: Path, residual_sweep: Path, primary_symmetric: Path,
    r8_symmetric: Path, output_dir: Path,
) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    ladder = pd.read_parquet(selected_sets)
    residual = pd.read_parquet(residual_sweep)
    scorers = sorted(set(map(str, ladder.scorer)))
    if scorers != sorted(ALL_MODEL_ORDER):
        raise ValueError(f"Stage-2 scorer set changed: {scorers}")
    replacement_rows = (
        ladder.groupby(["scorer", "replacement_budget"], as_index=False)
        .selected_utility_at8.mean()
        .rename(columns={"replacement_budget": "r",
                         "selected_utility_at8": "utility_at8"})
    )
    direct = replacement_rows[replacement_rows.r.eq(8)].set_index("scorer").utility_at8
    replacement_rows["delta_vs_direct"] = [
        float(u - direct[s]) for s, u in
        zip(replacement_rows.scorer, replacement_rows.utility_at8)
    ]
    residual_rows = (
        residual.groupby(["scorer", "entry_weight_alpha"], as_index=False)
        .selected_utility_at8.mean()
        .rename(columns={"entry_weight_alpha": "beta",
                         "selected_utility_at8": "utility_at8"})
    )
    beta0 = residual_rows[residual_rows.beta.eq(0)].set_index("scorer").utility_at8
    residual_rows["delta_vs_direct"] = [
        float(u - beta0[s]) for s, u in
        zip(residual_rows.scorer, residual_rows.utility_at8)
    ]
    primary_r = _summary_map(primary_symmetric, "anchored_swap")
    r8_r = _summary_map(r8_symmetric, "anchored_swap")
    beta = _summary_map(primary_symmetric, "residual_prior")
    old_u = _summary_map(primary_symmetric, "anchored_swap", "held_out_utility_at8")
    r8_u = _summary_map(r8_symmetric, "anchored_swap", "held_out_utility_at8")
    sensitivity = pd.DataFrame([
        {"scorer": scorer, "label": ALL_MODEL_LABELS[scorer],
         "primary_r_max7_mean": primary_r[scorer],
         "primary_replacement_utility_at8": old_u[scorer],
         "r8_included_mean": r8_r[scorer],
         "r8_included_utility_at8": r8_u[scorer],
         "direct_utility_at8": float(direct[scorer])}
        for scorer in ALL_MODEL_ORDER
    ])
    output_dir.mkdir(parents=True)
    replacement_rows.to_csv(output_dir / "replacement_curves_all_models.csv", index=False)
    residual_rows.to_csv(output_dir / "residual_curves_all_models.csv", index=False)
    sensitivity.to_csv(output_dir / "r8_endpoint_sensitivity.csv", index=False)
    rep_fig = _small_multiples(
        replacement_rows, "r", "Replacement curves for all Stage-2 models",
        "replacement capacity r (0 = Stage-1 Top-8; 8 = Direct)",
        primary_r, r8_r)
    res_fig = _small_multiples(
        residual_rows, "beta", "Residual-prior curves for all Stage-2 models",
        r"residual-prior weight $\beta$ (0 = Direct; 1 = Stage-1 score only)",
        beta)
    for fig, stem in ((rep_fig, "replacement_curves_all_models"),
                      (res_fig, "residual_curves_all_models")):
        fig.savefig(output_dir / f"{stem}.png", dpi=260, facecolor="white")
        fig.savefig(output_dir / f"{stem}.pdf", facecolor="white",
                    metadata={"Creator": "GraphRAG ADHD project",
                              "CreationDate": None, "ModDate": None})
    with PdfPages(output_dir / "stage2_all_models_r_beta_curves.pdf") as pdf:
        pdf.savefig(rep_fig, facecolor="white")
        pdf.savefig(res_fig, facecolor="white")
    plt.close(rep_fig); plt.close(res_fig)
    manifest = {
        "status": "COMPLETE", "models": len(ALL_MODEL_ORDER),
        "replacement_grid": list(range(9)),
        "residual_grid": sorted(map(float, residual_rows.beta.unique())),
        "interpretation": {
            "r8": "Direct endpoint, included only as a sensitivity option",
            "primary_r": "nested selection among constrained values 1..7",
            "curves": "full-cohort descriptive; never used to select r or beta",
        },
        "boundaries": {"external_calls": 0, "frozen_test_read": False,
                       "model_training": False, "selected_sets_modified": False},
        "inputs": {str(p): _sha(p) for p in (
            selected_sets, residual_sweep,
            primary_symmetric / "hyperparameter_summary.csv",
            r8_symmetric / "hyperparameter_summary.csv")},
    }
    manifest["outputs"] = {
        p.name: _sha(p) for p in sorted(output_dir.iterdir()) if p.is_file()
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--selected-sets", type=Path)
    parser.add_argument("--residual-sweep", type=Path)
    parser.add_argument("--primary-symmetric", type=Path)
    parser.add_argument("--r8-symmetric", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.all_models:
        required = (args.selected_sets, args.residual_sweep,
                    args.primary_symmetric, args.r8_symmetric, args.output_dir)
        if any(v is None for v in required):
            parser.error("--all-models requires all source and output paths")
        print(json.dumps(render_all_models(
            args.selected_sets, args.residual_sweep, args.primary_symmetric,
            args.r8_symmetric, args.output_dir), indent=2, sort_keys=True))
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    chosen = _chosen()

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _panel(ax, _read("replacement_frontier.csv"), REPLACEMENT_ARMS, "r", chosen,
           "replacement capacity $r$  (candidates allowed to differ from the "
           "first-stage Top-8)",
           "Replacement-constrained selection", lambda v: int(float(v)))
    ax.set_xticks(range(0, 9))
    # Annotations sit in the empty lower-middle, clear of every curve.
    ax.set_ylim(-0.72, 0.05)   # reserve a clear band under the data for notes
    ax.annotate("$r=8$ is the unrestricted endpoint and equals Direct exactly;  "
                "dotted lines are the nested mean $r^{\\star}$, chosen on inner "
                "folds \u2014 not on this curve",
                xy=(0, 0), xytext=(-0.35, -0.685), fontsize=7.8, color="#666666")
    ax.legend(loc="center right", frameon=False, fontsize=8.4)
    # The whole selector question lives in the last three steps; inset it.
    inset = ax.inset_axes([0.13, 0.20, 0.32, 0.28], facecolor="white")
    inset.set_zorder(5); inset.patch.set_alpha(1.0)
    for scorer, _label, colour, marker, style in REPLACEMENT_ARMS:
        points = sorted((int(float(r["r"])), float(r["delta_vs_direct"]))
                        for r in _read("replacement_frontier.csv")
                        if r["scorer"] == scorer)
        points = [p for p in points if p[0] >= 5]
        inset.plot([p[0] for p in points], [p[1] for p in points],
                   linestyle=style, color=colour, linewidth=1.4, marker=marker,
                   markersize=3.6)
    inset.axhline(0.0, color="#999999", linewidth=0.8)
    inset.set_xticks([5, 6, 7, 8])
    inset.tick_params(labelsize=7.2)
    inset.set_title("zoom: $r\\geq 5$", fontsize=7.6, pad=3)
    inset.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage2_replacement_frontier.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "stage2_replacement_frontier.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _panel(ax, _read("residual_prior_sweep.csv"), RESIDUAL_ARMS, "beta", chosen,
           r"residual-prior weight $\beta$  (0 = Direct, 1 = first-stage order only)",
           "Residual-prior selection", float)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.72, 0.07)   # reserve a clear band under the data for notes
    ax.annotate(r"$\beta=0$ reproduces Direct exactly;  dotted lines are the "
                "nested mean $\\beta^{\\star}$, chosen on inner folds \u2014 not "
                "on this curve", xy=(0, 0), xytext=(-0.045, -0.685),
                fontsize=7.8, color="#666666")
    ax.legend(loc="lower left", frameon=False, fontsize=8.0,
              bbox_to_anchor=(0.0, 0.08))
    # Every selected beta lies below .16; the rest of the sweep is a monotone
    # decline that would otherwise squash the informative region flat.
    inset = ax.inset_axes([0.53, 0.58, 0.33, 0.36], facecolor="white")
    inset.set_zorder(5); inset.patch.set_alpha(1.0)
    for scorer, _label, colour, marker, style in RESIDUAL_ARMS:
        points = sorted((float(r["beta"]), float(r["delta_vs_direct"]))
                        for r in _read("residual_prior_sweep.csv")
                        if r["scorer"] == scorer)
        points = [p for p in points if p[0] <= 0.25]
        inset.plot([p[0] for p in points], [p[1] for p in points],
                   linestyle=style, color=colour, linewidth=1.4, marker=marker,
                   markersize=3.6)
        inset.axvline(chosen[(scorer, "beta")], color=colour, linewidth=0.8,
                      linestyle=":", alpha=0.55)
    inset.axhline(0.0, color="#999999", linewidth=0.8)
    inset.tick_params(labelsize=7.2)
    inset.set_title(r"zoom: $\beta\leq 0.25$", fontsize=7.6, pad=3)
    inset.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage2_residual_sweep.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "stage2_residual_sweep.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES/'stage2_replacement_frontier.pdf'}")
    print(f"wrote {FIGURES/'stage2_residual_sweep.pdf'}")


if __name__ == "__main__":
    main()
