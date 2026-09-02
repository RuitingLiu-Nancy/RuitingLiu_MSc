#!/usr/bin/env python3
"""Select the swap radius r and the residual weight beta by the same procedure.

The Stage-2 comparison asks whether reintroducing the first-stage ordering into
an otherwise unconstrained selector helps.  Two methods do that: an anchored
swap, which constrains the feasible set and whose hyperparameter is the radius
r; and a residual prior, which reweights the objective and whose hyperparameter
is the weight beta.  Until now r was reported at a fixed set of values while
beta was tuned, so a comparison between the methods confounded a tuned quantity
with a fixed one.

This runner puts both through one code path: the same query-grouped 5x5 folds,
the same grid search inside training folds, the same one-standard-error
selection rule, and the same cross-fitted application to held-out queries.

The conservative direction differs between the two parameterisations and this
is deliberate, not an inconsistency.  "Conservative" means relying less on the
first-stage prior.  For beta that is the SMALLER value; for r, whose upper
endpoint r = K is the unconstrained selector, it is the LARGER value.

Nothing is fitted here.  Every selected set is read from a frozen artefact, the
scoring models are frozen out-of-fold predictions, and the only quantities
chosen are the two scalars.  Test200 is never opened.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEY = "rq2b_symmetric_hyperparameter_selection_dev300"

try:
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file

DEFAULT_SCORERS = ["candidate_huber", "candidate_small_mlp", "candidate_lambdamart"]


def _scorers(cfg: dict) -> list[str]:
    """The arms this configuration selects hyperparameters for.

    The frozen-family requirement guards the replay of the frozen eight-feature
    contract, where the three original scorers must always be rebuilt so drift
    is caught.  A configuration built on a different scorer contract - the
    Stage-2 redesign's eleven-feature arms, which do not include Small MLP -
    can waive it by setting ``require_frozen_scorer_families: false``; the
    default keeps every existing configuration's behaviour unchanged.
    """
    names = [str(name) for name in cfg.get("scorers", DEFAULT_SCORERS)]
    if not names or len(set(names)) != len(names):
        raise ValueError("the scorer list must be non-empty and unique")
    if bool(cfg.get("require_frozen_scorer_families", True)) and not set(
        DEFAULT_SCORERS
    ).issubset(names):
        raise ValueError("the scorer list must retain the three frozen families")
    return names
SWAP, RESIDUAL = "anchored_swap", "residual_prior"


# --------------------------------------------------------------------------- config

def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _load_config(config_key: str) -> dict[str, Any]:
    raw = dict(project_config.load()[config_key])
    if raw.get("allow_external_calls") or raw.get("allow_frozen_test"):
        raise ValueError("hyperparameter selection must stay local and development-only")
    for key in ("stage2_selected_sets", "residual_beta_sweep", "split_manifest",
                "output_dir"):
        raw[key] = _resolve(raw[key])
        if key != "output_dir" and "test" in str(raw[key]).lower():
            raise ValueError(f"{key} resolves to a path that resembles frozen test scope")
    if int(raw["final_k"]) != 8:
        raise ValueError("the evidence budget must remain K=8")
    swap_grid = list(map(int, raw["swap_grid"]))
    include_direct_endpoint = bool(
        raw.get("include_direct_endpoint_in_swap_grid", False))
    if swap_grid != sorted(set(swap_grid)) or not all(
        0 < r <= int(raw["final_k"]) if include_direct_endpoint
        else 0 < r < int(raw["final_k"])
        for r in swap_grid
    ):
        raise ValueError(
            "the swap grid must be strictly increasing, exclude 0, and "
            + ("may include the Direct endpoint " if include_direct_endpoint
               else "exclude the Direct endpoint ")
            + f"{int(raw['final_k'])}; got {swap_grid}"
        )
    beta_grid = [round(float(b), 6) for b in raw["beta_grid"]]
    if beta_grid != sorted(set(beta_grid)) or beta_grid[0] < 0 or beta_grid[-1] > 1:
        raise ValueError("the beta grid must be strictly increasing within [0, 1]")
    if str(raw["selection_rule"]) != "one_standard_error":
        raise ValueError("only the one-standard-error rule is registered")
    if int(raw["inner_folds"]) < 2:
        raise ValueError("nested selection needs at least two inner folds")
    raw["swap_grid"], raw["beta_grid"] = swap_grid, beta_grid
    return raw


# --------------------------------------------------------------------------- inputs

def _load_folds(path: Path, qids: set[str]) -> dict[tuple[int, int], tuple[list[str], list[str]]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    audit = manifest["audit"]
    if (
        int(audit["outer_repeats"]) != 5
        or int(audit["outer_folds"]) != 5
        or int(audit["query_overlap_per_fold"]) != 0
    ):
        raise ValueError("the frozen 5x5 query-grouped split contract changed")
    folds: dict[tuple[int, int], tuple[list[str], list[str]]] = {}
    for row in manifest["rows"]:
        train = [str(q) for q in row["train_query_ids"]]
        validation = [str(q) for q in row["validation_query_ids"]]
        if set(train) & set(validation) or set(train) | set(validation) != qids:
            raise ValueError("split leakage or coverage failure")
        folds[(int(row["repeat"]), int(row["fold"]))] = (train, validation)
    if set(folds) != {(r, f) for r in range(5) for f in range(5)}:
        raise ValueError("the split is not 5x5")
    return folds


def _swap_utilities(cfg: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    """scorer -> r -> query -> realised utility, at the registered depth and entry."""
    depth, entry = int(cfg["pool_depth"]), str(cfg["entry_ranking"])
    out: dict[str, dict[int, dict[str, float]]] = {s: {} for s in _scorers(cfg)}
    for row in pq.read_table(cfg["stage2_selected_sets"]).to_pylist():
        if int(row["pool_depth"]) != depth or str(row["entry_ranking"]) != entry:
            continue
        scorer = str(row["scorer"])
        if scorer not in out:
            continue
        out[scorer].setdefault(int(row["replacement_budget"]), {})[
            str(row["query_id"])] = float(row["selected_utility_at8"])
    needed = set(cfg["swap_grid"]) | {0, int(cfg["final_k"])}
    for scorer, byr in out.items():
        missing = needed - set(byr)
        if missing:
            raise ValueError(
                f"{scorer}: the selected-set artefact does not cover r in {sorted(missing)}; "
                "run the full replacement ladder first"
            )
    return out


def _residual_utilities(cfg: dict[str, Any]) -> dict[str, dict[float, dict[str, float]]]:
    """scorer -> beta -> query -> realised utility."""
    out: dict[str, dict[float, dict[str, float]]] = {s: {} for s in _scorers(cfg)}
    for row in pq.read_table(cfg["residual_beta_sweep"]).to_pylist():
        scorer = str(row["scorer"])
        if scorer not in out:
            continue
        beta = round(float(row["entry_weight_alpha"]), 6)
        out[scorer].setdefault(beta, {})[str(row["query_id"])] = float(
            row["selected_utility_at8"])
    for scorer, byb in out.items():
        missing = set(cfg["beta_grid"]) - set(byb)
        if missing:
            raise ValueError(
                f"{scorer}: the residual artefact does not cover beta in "
                f"{sorted(missing)[:5]}...; regenerate it over the registered grid"
            )
    return out


# --------------------------------------------------------------------- selection rule

def _inner_folds(train: list[str], count: int, seed: int) -> list[list[str]]:
    """Deterministic query-grouped inner split of one outer training fold."""
    order = sorted(train)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(order))
    folds: list[list[str]] = [[] for _ in range(count)]
    for position, index in enumerate(permutation):
        folds[position % count].append(order[int(index)])
    if sorted(q for fold in folds for q in fold) != order:
        raise ValueError("inner split lost or duplicated queries")
    if any(not fold for fold in folds):
        raise ValueError("an inner fold is empty")
    return folds


def _nested_choice(
    grid: list[Any],
    utilities: dict[Any, dict[str, float]],
    train: list[str],
    inner_count: int,
    seed: int,
    conservative: str,
) -> tuple[Any, dict[str, float], dict[Any, tuple[float, float]]]:
    """Choose a hyperparameter by nested cross-validation inside one outer fold.

    The outer training queries are split again, every grid value is scored on
    each inner validation fold, and the selection statistic is the mean of
    those inner estimates.  The one-standard-error tolerance is the standard
    error ACROSS INNER FOLDS of the paired difference against the inner-CV
    best, which is the rule in its textbook form: the spread being measured is
    the uncertainty of the cross-validated estimate, not the spread of utility
    over queries.  It is paired because every grid value is scored on the same
    inner folds.

    `conservative` is "low" when relying less on the first-stage prior means a
    smaller hyperparameter (beta) and "high" when it means a larger one (the
    swap radius, whose upper endpoint is the unconstrained selector).
    """
    folds = _inner_folds(train, inner_count, seed)
    per_fold = {
        g: [statistics.fmean([utilities[g][q] for q in fold]) for fold in folds]
        for g in grid
    }
    cv_mean = {g: statistics.fmean(per_fold[g]) for g in grid}
    best = max(grid, key=lambda g: cv_mean[g])
    admissible: list[Any] = []
    errors: dict[Any, float] = {}
    for g in grid:
        paired = [a - b for a, b in zip(per_fold[g], per_fold[best])]
        errors[g] = (
            statistics.stdev(paired) / len(paired) ** 0.5 if len(paired) > 1 else 0.0
        )
        if cv_mean[best] - cv_mean[g] <= errors[g]:
            admissible.append(g)
    if not admissible:
        admissible = [best]
    chosen = min(admissible) if conservative == "low" else max(admissible)
    curve = {g: (cv_mean[g], errors[g]) for g in grid}
    return chosen, {
        "inner_cv_best": float(cv_mean[best]),
        "inner_cv_best_grid_value": float(best),
        "inner_cv_at_chosen": float(cv_mean[chosen]),
        "paired_standard_error_at_chosen": float(errors[chosen]),
        "deficit_at_chosen": float(cv_mean[best] - cv_mean[chosen]),
        "admissible_count": len(admissible),
    }, curve


def _cross_fit(
    grid: list[Any],
    utilities: dict[Any, dict[str, float]],
    folds: dict[tuple[int, int], tuple[list[str], list[str]]],
    conservative: str,
    inner_count: int,
    inner_seed: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], list[dict[str, Any]]]:
    held: dict[str, list[float]] = {}
    picks: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for (repeat, fold), (train, validation) in sorted(folds.items()):
        seed = inner_seed + 100 * repeat + fold
        chosen, audit, curve = _nested_choice(
            grid, utilities, train, inner_count, seed, conservative,
        )
        picks.append({"repeat": repeat, "fold": fold, "chosen": float(chosen),
                      "inner_seed": seed, **audit})
        for g in grid:
            mean, error = curve[g]
            curve_rows.append({
                "repeat": repeat, "fold": fold, "grid_value": float(g),
                "inner_cv_mean": mean, "paired_standard_error": error,
            })
        for q in validation:
            held.setdefault(q, []).append(utilities[chosen][q])
    return held, picks, curve_rows


def _paired(left: dict[str, float], right: dict[str, float], draws: int, seed: int) -> dict[str, float]:
    qids = sorted(set(left) & set(right))
    diff = np.array([left[q] - right[q] for q in qids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(draws, diff.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_delta": float(diff.mean()), "ci_low": float(low), "ci_high": float(high),
        "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()), "queries": len(qids),
    }


# --------------------------------------------------------------------------- driver

def run(config_key: str = CONFIG_KEY, output_dir: Path | None = None) -> dict[str, Any]:
    cfg = _load_config(config_key)
    destination = Path(output_dir or cfg["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    swap = _swap_utilities(cfg)
    residual = _residual_utilities(cfg)
    qids = set(swap[_scorers(cfg)[0]][int(cfg["final_k"])])
    if len(qids) != int(cfg["expected_queries"]):
        raise ValueError(f"expected {cfg['expected_queries']} queries, found {len(qids)}")
    folds = _load_folds(cfg["split_manifest"], qids)
    draws, seed = int(cfg["bootstrap_draws"]), int(cfg["bootstrap_seed"])
    K = int(cfg["final_k"])

    summaries, contrasts, picks_rows, curve_rows = [], [], [], []
    for scorer in _scorers(cfg):
        direct = swap[scorer][K]
        summaries.append({
            "scorer": scorer, "method": "direct", "hyperparameter": "", "chosen_mean": "",
            "chosen_median": "", "queries": len(direct),
            "held_out_utility_at8": statistics.fmean(direct.values()),
        })
        for method, grid, utilities, conservative, name in (
            (SWAP, cfg["swap_grid"], swap[scorer], "high", "r"),
            (RESIDUAL, cfg["beta_grid"], residual[scorer], "low", "beta"),
        ):
            held, picks, curve = _cross_fit(
                grid, utilities, folds, conservative,
                int(cfg["inner_folds"]), int(cfg["inner_split_seed"]),
            )
            per_query = {q: statistics.fmean(v) for q, v in held.items()}
            chosen = [p["chosen"] for p in picks]
            summaries.append({
                "scorer": scorer, "method": method, "hyperparameter": name,
                "chosen_mean": statistics.fmean(chosen),
                "chosen_median": statistics.median(chosen),
                "queries": len(per_query),
                "held_out_utility_at8": statistics.fmean(per_query.values()),
            })
            contrasts.append({
                "scorer": scorer, "method": method, "comparator": "direct",
                **_paired(per_query, direct, draws, seed),
            })
            picks_rows += [{"scorer": scorer, "method": method, **p} for p in picks]
            curve_rows += [{"scorer": scorer, "method": method, **c} for c in curve]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as tmp:
        out = Path(tmp)
        names = ("hyperparameter_summary.csv", "hyperparameter_contrasts.csv",
                 "hyperparameter_fold_choices.csv", "hyperparameter_tuning_curves.csv")
        for name, rows in zip(names, (summaries, contrasts, picks_rows, curve_rows)):
            header = list(rows[0])
            body = [",".join(header)] + [
                ",".join(str(row[key]) for key in header) for row in rows]
            (out / name).write_text("\n".join(body) + "\n", encoding="utf-8")
        pq.write_table(pa.Table.from_pylist(curve_rows),
                       out / "hyperparameter_tuning_curves.parquet", compression="zstd")
        manifest = {
            "schema": "rq2b-symmetric-hyperparameter-selection-dev300-v1",
            "version": str(cfg["version"]), "status": "COMPLETE", "created_utc": now(),
            "estimand": {
                "question": "does either method of reintroducing the first-stage "
                            "ordering beat the unconstrained selector, when both "
                            "hyperparameters are chosen by the same procedure",
                "reference": "direct selection (r = K, equivalently beta = 0)",
                "cohort": "Development300", "entry_ranking": str(cfg["entry_ranking"]),
                "pool_depth": int(cfg["pool_depth"]), "final_k": K,
            },
            "procedure": {
                "outer_folds": "frozen 5x5 query-grouped split",
                "inner_folds": int(cfg["inner_folds"]),
                "inner_split": "deterministic query-grouped partition of each outer "
                               "training fold; seed = inner_split_seed + 100*repeat + fold",
                "search": "exhaustive grid, scored on inner validation folds",
                "rule": "nested cross-validation; the selection statistic is the mean "
                        "over inner folds, and the tolerance is one standard error "
                        "ACROSS INNER FOLDS of the paired difference against the "
                        "inner-CV best, then the most conservative admissible value",
                "conservative_direction": {
                    "beta": "smaller (less weight on the first-stage prior)",
                    "r": "larger (r = K is the unconstrained selector)",
                },
                "swap_grid": cfg["swap_grid"], "beta_grid": cfg["beta_grid"],
            },
            "counts": {"scorers": len(_scorers(cfg)), "queries": len(qids),
                       "summary_rows": len(summaries), "contrast_rows": len(contrasts),
                       "fold_choice_rows": len(picks_rows), "curve_rows": len(curve_rows)},
            "boundaries": {"external_requests_made": 0, "frozen_test_read": False,
                           "model_training_performed": False,
                           "evaluated_query_contributes_to_its_own_hyperparameter": False,
                           "community_response_used": False},
            "inputs": {k: {"path": str(cfg[k]), "sha256": sha256_file(cfg[k])}
                       for k in ("stage2_selected_sets", "residual_beta_sweep",
                                 "split_manifest")},
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                         "pyarrow": pa.__version__},
            "outputs": {n: sha256_file(out / n) for n in
                        names + ("hyperparameter_tuning_curves.parquet",)},
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        Path(out).rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-key", default=CONFIG_KEY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = run(args.config_key, args.output_dir)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"],
                      "procedure": manifest["procedure"]},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
