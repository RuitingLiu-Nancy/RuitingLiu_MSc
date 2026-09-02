#!/usr/bin/env python3
"""Community-response correspondence for every CURRENT Stage-2 system/selector.

Post-hoc evaluation only.  No model is fitted, no hyperparameter is chosen, no
candidate set is selected here: the selected comment identities are read from
the frozen Stage-2 selection artefacts and joined against the withheld
community replies.  Utility@8 is likewise read, never recomputed.

Two design points that matter for correctness:

* **Cross-fitting is mirrored exactly.**  Under the frozen 5x5 design a query
  is a validation query in five outer folds, and the Replacement / Residual
  selectors may choose a different r or beta in each.  The reported Utility@8
  is the mean over those five fold-specific values, so the community metrics
  are computed the same way: per (query, fold) on that fold's selected set,
  then averaged over the five folds.  Direct is invariant across folds, so its
  five sets coincide.

* **Community signal is never an input.**  The replies enter after every
  selected set is already frozen.  They were excluded from retrieval, pool
  construction, scorer features, fitting, family selection and both selector
  tunings; the audit re-asserts this from the upstream manifests rather than
  assuming it.

Reuse: `alignment` and `bidirectional_f` come from
``evaluation/community_reply_auxiliary`` unchanged - the CRA / RCC / BestAlign /
ReplyCoverage definitions are not reimplemented here - as does ``encode_texts``
with the project's frozen BGE-M3 configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEY = "stage2_community_dev300_complete"
DIRECT_BUDGET = 8
SWAP, RESIDUAL, DIRECT = "anchored_swap", "residual_prior", "direct"

sys.path.insert(0, str(ROOT))
from evidence_selection import run_rq2b_symmetric_hyperparameter_selection as sym  # noqa: E402
from evaluation.run_m50_community_frontier_analysis import (  # noqa: E402
    encode_with_verified_prior_cache,
)

try:
    import configuration as project_config
    from evaluation import community_reply_auxiliary as caux
    from evaluation.community_reply_auxiliary import now, sha256_file
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    import configuration as project_config
    from evaluation import community_reply_auxiliary as caux
    from evaluation.community_reply_auxiliary import now, sha256_file

# scorer -> (learning formulation, human name)
FORMULATION: dict[str, tuple[str, str]] = {
    "huber7d": ("Pointwise utility prediction", "Huber (frozen small grid)"),
    "ranknet7d": ("Pairwise utility-aware ranking", "RankNet (frozen small grid)"),
    "lw_huber": ("Pointwise utility prediction", "Huber"),
    "lw_ridge": ("Pointwise utility prediction", "Ridge"),
    "lw_elasticnet": ("Pointwise utility prediction", "ElasticNet"),
    "lw_hist_gbr": ("Pointwise utility prediction", "HistGradientBoosting"),
    "lw_xgb_regression": ("Pointwise utility prediction", "XGBoost regression"),
    "lw_catboost_regression": ("Pointwise utility prediction", "CatBoost regression"),
    "lw_small_mlp": ("Pointwise utility prediction", "Small MLP"),
    "lw_ranknet": ("Pairwise utility-aware ranking", "RankNet"),
    "lw_xgb_pairwise": ("Pairwise utility-aware ranking", "XGBoost pairwise"),
    "lw_lambdamart_aligned": ("Query-grouped utility-aware ranking",
                              "LambdaMART (aligned, linear gain)"),
    "lw_lambdamart_exp_gain": ("Query-grouped utility-aware ranking",
                               "LambdaMART (matched exponential gain)"),
    "lw_lgbm_lambdarank": ("Query-grouped utility-aware ranking",
                           "LightGBM LambdaRank"),
    "lw_catboost_yetirank": ("Query-grouped utility-aware ranking",
                             "CatBoost YetiRank"),
    "lm7d_lin_g7": ("Query-grouped utility-aware ranking",
                    "LambdaMART, linear gain, 7 grades"),
    "lm7d_lin_g25": ("Query-grouped utility-aware ranking",
                     "LambdaMART, linear gain, 25 grades"),
    "lm7d_exp_g7": ("Query-grouped utility-aware ranking",
                    "LambdaMART, exponential gain, 7 grades"),
    "lm7d_exp_g25": ("Query-grouped utility-aware ranking",
                     "LambdaMART, exponential gain, 25 grades"),
    "lambdamart7d": ("Query-grouped utility-aware ranking",
                     "LambdaMART, exponential gain, 7 grades (alias)"),
    "best_lightweight_nested": ("Nested model-family SELECTION PROCEDURE",
                                "Nested lightweight family selection"),
    "cross_encoder_matched": ("Neural text interaction",
                              "MiniLM cross-encoder (matched protocol)"),
    "cross_encoder": ("Neural text interaction",
                      "MiniLM cross-encoder (superseded protocol)"),
}
ALIAS_OF = {"lambdamart7d": "lm7d_exp_g7"}
SUPERSEDED = {"cross_encoder"}
DENSE_BASELINE = "raw_e5_dense_top8"
METRICS = ("u_at_8", "cra", "rcc", "bialign_f1", "best_align")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _ids_hash(ids) -> str:
    return hashlib.sha256("\0".join(map(str, ids)).encode("utf-8")).hexdigest()[:16]


def _paired(left: dict[str, float], right: dict[str, float], draws: int,
            seed: int) -> dict[str, Any]:
    """The project's whole-query paired bootstrap, delegated unchanged."""
    return sym._paired(left, right, draws, seed)


def _sets_view(directory: Path, cfg_key: str) -> dict[str, Any]:
    cfg = dict(project_config.load()[cfg_key])
    return {"stage2_selected_sets": directory / "stage2_selected_sets.parquet",
            "residual_beta_sweep": directory / "residual_beta_sweep.parquet",
            "backend": cfg["backend"], "pool_depth": cfg["pool_depth"],
            "entry_ranking": cfg["entry_ranking"], "final_k": cfg["final_k"],
            "swap_grid": list(range(1, int(cfg["final_k"]))),
            "beta_grid": [round(float(b), 6) for b in cfg["beta_grid"]],
            "scorers": sorted({str(s) for s in cfg["expected_scorers"]}),
            "require_frozen_scorer_families": False}


def run(config_key: str = CONFIG_KEY, output_dir: Path | None = None,
        limit_systems: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = dict(project_config.load()[config_key])
    if cfg.get("allow_external_calls") or cfg.get("allow_frozen_test"):
        raise ValueError("the community join is post-hoc and offline")
    for key in ("output_dir", "corpus", "reference_texts", "triangulation_manifest",
                "utility_registry", "dense_memberships", "split_manifest"):
        cfg[key] = _resolve(cfg[key])
        if key != "output_dir" and "test" in str(cfg[key]).lower():
            raise ValueError(f"{key} resolves to a path resembling test scope")
    for key in ("encoder_params_file", "embedding_prior_registry",
                "embedding_prior_output"):
        if cfg.get(key):
            cfg[key] = _resolve(cfg[key])
    destination = Path(output_dir or cfg["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    audit: dict[str, Any] = {"checks": {}, "failures": []}

    # ---- gate 0: the current Development300 / 4,143-reply universe --------
    triangulation = json.loads(Path(cfg["triangulation_manifest"]).read_text())
    reference = _read_jsonl(cfg["reference_texts"])
    replies_by_query = {str(r["query_id"]): list(r["replies"]) for r in reference}
    reply_total = sum(len(v) for v in replies_by_query.values())
    audit["checks"]["reference_query_count"] = len(replies_by_query)
    audit["checks"]["valid_hidden_reply_count"] = reply_total
    audit["checks"]["manifest_valid_hidden_replies"] = int(
        triangulation["counts"]["valid_hidden_replies"])
    audit["checks"]["queries_with_at_least_one_reply"] = sum(
        1 for v in replies_by_query.values() if v)
    audit["checks"]["min_replies_per_query"] = min(
        len(v) for v in replies_by_query.values())
    audit["checks"]["upstream_leakage_query_count"] = int(
        triangulation["reference_audit"]["leakage_query_count"])
    audit["checks"]["upstream_community_used_for_selection"] = bool(
        triangulation["invariants"]["community_used_for_retrieval_or_selection"])
    for name, ok in (
        ("query_count_is_300", len(replies_by_query) == int(cfg["expected_queries"])),
        ("reply_count_is_expected", reply_total == int(cfg["expected_replies"])),
        ("manifest_agrees_on_reply_count",
         int(triangulation["counts"]["valid_hidden_replies"])
         == int(cfg["expected_replies"])),
        ("every_query_has_a_reply",
         all(len(v) >= 1 for v in replies_by_query.values())),
        ("no_upstream_leakage",
         int(triangulation["reference_audit"]["leakage_query_count"]) == 0),
        ("community_not_used_for_selection",
         not triangulation["invariants"]["community_used_for_retrieval_or_selection"]),
    ):
        if not ok:
            audit["failures"].append(name)
    if audit["failures"]:
        raise ValueError(f"cohort identity gate failed: {audit['failures']}. "
                         "Refusing to switch cohorts silently.")

    corpus = {str(r["title"]): str(r["text"])
              for r in json.loads(Path(cfg["corpus"]).read_text(encoding="utf-8"))}
    registry = {}
    for row in _read_jsonl(cfg["utility_registry"]):
        registry[(str(row["query_id"]), str(row["comment_id"]))] = row
    dense_rank: dict[tuple[str, str], int] = {}
    for row in _read_jsonl(cfg["dense_memberships"]):
        if str(row["backend"]) == str(cfg["backend"]):
            dense_rank[(str(row["query_id"]), str(row["comment_id"]))] = int(row["rank"])

    # ---- assemble every current system's per-(query, fold) selected set ---
    systems: dict[str, dict[str, Any]] = {}
    per_fold: dict[str, dict[str, list[tuple[str, tuple[str, ...], float]]]] = {}
    utilities: dict[str, dict[str, float]] = {}
    provenance: dict[str, dict[str, str]] = {}

    for source in cfg["sources"]:
        sets_dir = _resolve(source["sets"])
        view = _sets_view(sets_dir, str(source["sets_key"]))
        ladder_rows = pq.read_table(view["stage2_selected_sets"]).to_pylist()
        sweep_rows = pq.read_table(view["residual_beta_sweep"]).to_pylist()
        ladder: dict[str, dict[int, dict[str, dict]]] = defaultdict(
            lambda: defaultdict(dict))
        for row in ladder_rows:
            ladder[str(row["scorer"])][int(row["replacement_budget"])][
                str(row["query_id"])] = row
        sweep: dict[str, dict[float, dict[str, dict]]] = defaultdict(
            lambda: defaultdict(dict))
        for row in sweep_rows:
            sweep[str(row["scorer"])][round(float(row["entry_weight_alpha"]), 6)][
                str(row["query_id"])] = row
        choices = defaultdict(list)
        with (_resolve(source["symmetric"]) /
              "hyperparameter_fold_choices.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                choices[(row["scorer"], row["method"])].append(row)

        qids = sorted(ladder[next(iter(ladder))][DIRECT_BUDGET])
        folds = sym._load_folds(cfg["split_manifest"], set(qids))

        # dense baseline: identical in every ladder row, so read it once
        if DENSE_BASELINE not in systems:
            any_scorer = next(iter(ladder))
            base = ladder[any_scorer][DIRECT_BUDGET]
            per_fold[DENSE_BASELINE] = {
                q: [("raw", tuple(map(str, base[q]["dense8_baseline_ids"])),
                     float(base[q]["dense8_baseline_utility_at8"]))] for q in qids}
            systems[DENSE_BASELINE] = {
                "scorer": DENSE_BASELINE, "strategy": "raw_dense",
                "formulation": "First-stage retrieval baseline",
                "name": "Raw E5 Dense Top-8", "r_mean": "", "r_median": "",
                "beta_mean": "", "beta_median": ""}
            provenance[DENSE_BASELINE] = {
                "selected_set_source": str(sets_dir.relative_to(ROOT))
                + "/stage2_selected_sets.parquet (dense8_baseline_ids)",
                "status": "reused"}

        for scorer in sorted(ladder):
            base_prov = {"selected_set_source": str(sets_dir.relative_to(ROOT)),
                         "symmetric_source": str(source["symmetric"]),
                         "status": "reused"}
            # Direct
            sid = f"{scorer}::direct"
            per_fold[sid] = {q: [("all", tuple(map(str, ladder[scorer][DIRECT_BUDGET][q]
                                                   ["selected_comment_ids"])),
                                 float(ladder[scorer][DIRECT_BUDGET][q]
                                       ["selected_utility_at8"]))] for q in qids}
            systems[sid] = {"scorer": scorer, "strategy": "direct",
                            "formulation": FORMULATION[scorer][0],
                            "name": FORMULATION[scorer][1],
                            "r_mean": "", "r_median": "",
                            "beta_mean": "", "beta_median": ""}
            provenance[sid] = dict(base_prov)

            # Replacement and Residual, cross-fitted exactly as U@8 is
            for method, table, key in ((SWAP, ladder[scorer], "r"),
                                       (RESIDUAL, sweep[scorer], "beta")):
                rows = choices[(scorer, method)]
                if not rows:
                    continue
                picked = {(int(r["repeat"]), int(r["fold"])): float(r["chosen"])
                          for r in rows}
                sid = f"{scorer}::{'replacement' if method == SWAP else 'residual'}"
                collected: dict[str, list] = defaultdict(list)
                for (repeat, fold), (_train, validation) in sorted(folds.items()):
                    chosen = picked[(repeat, fold)]
                    lookup = table[int(chosen)] if key == "r" \
                        else table[round(chosen, 6)]
                    for q in validation:
                        row = lookup[q]
                        collected[q].append((f"r{repeat}f{fold}",
                                             tuple(map(str, row["selected_comment_ids"])),
                                             float(row["selected_utility_at8"])))
                per_fold[sid] = dict(collected)
                values = [float(r["chosen"]) for r in rows]
                systems[sid] = {
                    "scorer": scorer,
                    "strategy": "replacement" if method == SWAP else "residual",
                    "formulation": FORMULATION[scorer][0],
                    "name": FORMULATION[scorer][1],
                    "r_mean": f"{statistics.fmean(values):.3f}" if key == "r" else "",
                    "r_median": f"{statistics.median(values):.3f}" if key == "r" else "",
                    "beta_mean": "" if key == "r" else f"{statistics.fmean(values):.4f}",
                    "beta_median": "" if key == "r"
                                   else f"{statistics.median(values):.4f}"}
                provenance[sid] = dict(base_prov)

    # Optional direct controls whose Top-8 sets were frozen by another runner.
    # This keeps community replies downstream of selection while allowing a
    # supervision control (for example, the unfitted MS MARCO CE) to reuse the
    # canonical community evaluation without fabricating a Stage-2 selector.
    for source in cfg.get("supplemental_direct_systems", []):
        sid = str(source["system_id"])
        if sid in systems:
            raise ValueError(f"duplicate supplemental system id: {sid}")
        selected_path = _resolve(source["selected_sets_csv"])
        if "test" in str(selected_path).lower():
            raise ValueError("supplemental direct system points at test scope")
        with selected_path.open(encoding="utf-8") as handle:
            selected_rows = list(csv.DictReader(handle))
        if len(selected_rows) != int(cfg["expected_queries"]):
            raise ValueError(
                f"supplemental system {sid} has {len(selected_rows)} queries")
        direct: dict[str, list[tuple[str, tuple[str, ...], float]]] = {}
        for row in selected_rows:
            qid = str(row["query_id"])
            ids = tuple(filter(None, str(row["selected_comment_ids"]).split(";")))
            if len(ids) != DIRECT_BUDGET or len(set(ids)) != DIRECT_BUDGET:
                raise ValueError(f"supplemental system {sid}/{qid} is not Top-8 unique")
            joined_utility = statistics.fmean(
                float(registry[(qid, cid)]["utility"]) for cid in ids)
            recorded_utility = float(row["utility_at8"])
            if abs(joined_utility - recorded_utility) > 1e-10:
                raise ValueError(
                    f"supplemental utility mismatch for {sid}/{qid}: "
                    f"{recorded_utility} vs {joined_utility}")
            direct[qid] = [("all", ids, joined_utility)]
        if set(direct) != set(per_fold[DENSE_BASELINE]):
            raise ValueError(f"supplemental system {sid} query identity mismatch")
        per_fold[sid] = direct
        systems[sid] = {
            "scorer": str(source["scorer"]), "strategy": "direct",
            "formulation": str(source["formulation"]),
            "name": str(source["name"]),
            "r_mean": "", "r_median": "", "beta_mean": "", "beta_median": "",
        }
        provenance[sid] = {
            "selected_set_source": str(selected_path.relative_to(ROOT)),
            "status": "supplemental frozen direct control",
        }

    if limit_systems is not None:
        keep = sorted(systems)[:limit_systems]
        systems = {k: systems[k] for k in keep}
        per_fold = {k: per_fold[k] for k in keep}

    # ---- texts needed, then the frozen encoder --------------------------
    needed_comments = {cid for sets in per_fold.values() for rows in sets.values()
                       for _tag, ids, _u in rows for cid in ids}
    missing = sorted(c for c in needed_comments if c not in corpus)
    audit["checks"]["selected_comments_distinct"] = len(needed_comments)
    audit["checks"]["selected_comments_missing_from_corpus"] = len(missing)
    if missing:
        raise ValueError(f"{len(missing)} selected comments are absent from the corpus")

    reply_texts = {q: [str(rp["text"]) for rp in rows]
                   for q, rows in replies_by_query.items()}
    texts = sorted({corpus[c] for c in needed_comments}
                   | {t for v in reply_texts.values() for t in v})
    encoder_source = (
        project_config.load(str(cfg["encoder_params_file"]), force=True)
        if cfg.get("encoder_params_file") else project_config.load()
    )
    encoder_cfg = {"semantic_encoder": dict(
        encoder_source[str(cfg["encoder_config_key"])]["semantic_encoder"])}
    cache_dir = destination.parent / f".{destination.name}.embcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("embedding_prior_registry") and cfg.get("embedding_prior_output"):
        prior_ids = {
            str(row["comment_id"])
            for row in _read_jsonl(cfg["embedding_prior_registry"])
        }
        missing_prior_ids = sorted(prior_ids - set(corpus))
        if missing_prior_ids:
            raise ValueError(
                f"{len(missing_prior_ids)} prior-cache comments are absent from corpus")
        prior_texts = [t for values in reply_texts.values() for t in values]
        prior_texts.extend(corpus[cid] for cid in sorted(prior_ids))
        embeddings, embed_audit = encode_with_verified_prior_cache(
            config=encoder_cfg, texts=texts, prior_texts=prior_texts,
            prior_output=cfg["embedding_prior_output"], output=cache_dir)
    else:
        embeddings, embed_audit = caux.encode_texts(encoder_cfg, texts, cache_dir)
    audit["embedding"] = embed_audit

    # ---- metrics, mirroring the cross-fitted estimand --------------------
    threshold = float(encoder_cfg["semantic_encoder"]["threshold"])
    beta_f = float(cfg["bidirectional_beta"])
    per_query_rows: list[dict] = []
    items_rows: list[dict] = []
    metric_by_system: dict[str, dict[str, dict[str, float]]] = {}

    for sid in sorted(per_fold):
        spec = systems[sid]
        acc: dict[str, dict[str, float]] = {m: {} for m in METRICS}
        for qid in sorted(per_fold[sid]):
            entries = per_fold[sid][qid]
            replies = reply_texts[qid]
            folded = []
            for tag, ids, utility in entries:
                stats = caux.alignment([corpus[c] for c in ids], replies,
                                       embeddings, threshold)
                folded.append({
                    "tag": tag, "ids": ids, "u_at_8": utility,
                    "cra": stats["cra_at8"], "rcc": stats["rcc_at8"],
                    "best_align": stats["best_align_at8"],
                    "reply_coverage": stats["reply_coverage_at8"],
                    "bialign_f1": caux.bidirectional_f(
                        stats["cra_at8"], stats["rcc_at8"], beta_f)})
            mean = {k: statistics.fmean(f[k] for f in folded)
                    for k in ("u_at_8", "cra", "rcc", "best_align",
                              "bialign_f1", "reply_coverage")}
            for m in METRICS:
                acc[m][qid] = mean[m]
            per_query_rows.append({
                "query_id": qid, "system_id": sid, "scorer": spec["scorer"],
                "learning_formulation": spec["formulation"],
                "selection_strategy": spec["strategy"],
                "selected_comment_ids": ";".join(folded[0]["ids"]),
                "selected_comment_ids_hash": _ids_hash(folded[0]["ids"]),
                "distinct_selected_sets_across_folds":
                    len({f["ids"] for f in folded}),
                "selected_count": len(folded[0]["ids"]),
                "u_at_8": f"{mean['u_at_8']:.6f}", "cra": f"{mean['cra']:.6f}",
                "rcc": f"{mean['rcc']:.6f}",
                "bialign_f1": f"{mean['bialign_f1']:.6f}",
                "best_align": f"{mean['best_align']:.6f}",
                "reply_coverage": f"{mean['reply_coverage']:.6f}",
                "reply_count": len(replies),
                "community_reply_ids_hash": _ids_hash(
                    [rp["reply_id"] for rp in replies_by_query[qid]]),
                "folds_contributing": len(folded),
                "r_selected_for_folds": ";".join(
                    f"{f['tag']}" for f in folded) if spec["strategy"] != "direct"
                    else "",
            })
            # per-item long form, first fold's set (identical for Direct)
            reference_ids = folded[0]["ids"]
            reply_vectors = np.stack([embeddings[caux.text_sha(t)] for t in replies])
            for position, cid in enumerate(reference_ids, start=1):
                vector = embeddings[caux.text_sha(corpus[cid])]
                similarity = reply_vectors @ vector
                best = int(np.argmax(similarity))
                entry = registry.get((qid, cid))
                items_rows.append({
                    "query_id": qid, "system_id": sid,
                    "selection_strategy": spec["strategy"],
                    "selected_position": position, "comment_id": cid,
                    "source_post_id": (str(entry.get("card_id", "")) if entry else ""),
                    "true_utility": (f"{float(entry['utility']):.4f}"
                                     if entry else ""),
                    "dense_rank": dense_rank.get((qid, cid), ""),
                    "candidate_cra_max": f"{float(similarity.max()):.6f}",
                    "best_reply_id": str(replies_by_query[qid][best]["reply_id"]),
                })
        metric_by_system[sid] = acc

    return _emit(destination, started, cfg, systems, provenance, metric_by_system,
                 per_query_rows, items_rows, audit, replies_by_query, reply_total)


def _emit(destination, started, cfg, systems, provenance, metric_by_system,
          per_query_rows, items_rows, audit, replies_by_query, reply_total):
    draws, seed = int(cfg["bootstrap_draws"]), int(cfg["bootstrap_seed"])
    matrix, contrasts, quadrants = [], [], []

    for sid in sorted(systems):
        spec, acc = systems[sid], metric_by_system[sid]
        canonical = ALIAS_OF.get(spec["scorer"])
        status = ("alias of " + canonical + "::" + spec["strategy"]) if canonical \
            else ("superseded protocol, provenance only"
                  if spec["scorer"] in SUPERSEDED
                  else "current clean-contract protocol")
        coverage = [float(r["reply_coverage"]) for r in per_query_rows
                    if r["system_id"] == sid]
        matrix.append({
            "learning_formulation": spec["formulation"], "scorer": spec["scorer"],
            "canonical_system_id": sid, "selection_strategy": spec["strategy"],
            "r_selected_mean": spec["r_mean"] or "N/A",
            "r_selected_median": spec["r_median"] or "N/A",
            "beta_selected_mean": spec["beta_mean"] or "N/A",
            "beta_selected_median": spec["beta_median"] or "N/A",
            "n_queries": len(acc["u_at_8"]),
            **{m: f"{statistics.fmean(acc[m].values()):.6f}" for m in METRICS},
            "reply_coverage_if_available": f"{statistics.fmean(coverage):.6f}",
            "selected_set_source": provenance[sid]["selected_set_source"],
            "utility_source": "frozen utility-v2 joined in the selection artefact",
            "community_embedding_source": audit["embedding"]["cache_path"],
            "protocol_status": status,
        })

    def contrast(left_id, right_id, tag):
        for metric in METRICS:
            row = sym._paired(metric_by_system[left_id][metric],
                              metric_by_system[right_id][metric], draws, seed)
            contrasts.append({
                "contrast": tag, "arm": left_id, "comparator": right_id,
                "metric": metric, "mean_delta": f"{row['mean_delta']:+.6f}",
                "ci_low": f"{row['ci_low']:+.6f}", "ci_high": f"{row['ci_high']:+.6f}",
                "wins": row["wins"], "ties": row["ties"], "losses": row["losses"],
                "n_queries": row["queries"],
                "interval_excludes_zero":
                    "yes" if row["ci_low"] * row["ci_high"] > 0 else "no"})

    # within-scorer selector contrasts
    for sid in sorted(systems):
        spec = systems[sid]
        if spec["strategy"] in ("replacement", "residual"):
            base = f"{spec['scorer']}::direct"
            if base in metric_by_system:
                contrast(sid, base, f"{spec['strategy']}_minus_direct")

    # principal cross-system contrasts (instruction section 10)
    cross = [
        ("best_lightweight_nested::direct", DENSE_BASELINE, "A_nested_direct_vs_dense"),
        ("cross_encoder_matched::direct", DENSE_BASELINE, "B_ce_direct_vs_dense"),
        ("cross_encoder_matched::direct", "best_lightweight_nested::direct",
         "C_ce_direct_vs_nested_direct"),
        ("cross_encoder_matched::residual", "cross_encoder_matched::direct",
         "D_ce_residual_vs_ce_direct"),
        ("best_lightweight_nested::residual", "best_lightweight_nested::direct",
         "E_nested_residual_vs_nested_direct"),
        ("lm7d_exp_g7::residual", "lm7d_exp_g7::direct",
         "F_expLambda_residual_vs_direct"),
        ("lm7d_lin_g7::direct", "lm7d_exp_g7::direct",
         "G_linLambda_direct_vs_expLambda_direct"),
    ]
    cross.extend(
        (str(row["left"]), str(row["right"]), str(row["tag"]))
        for row in cfg.get("supplemental_contrasts", [])
    )
    for left, right, tag in cross:
        if left in metric_by_system and right in metric_by_system:
            contrast(left, right, tag)
            for metric in ("cra", "rcc", "bialign_f1"):
                counts = {"pp": 0, "pn": 0, "np": 0, "nn": 0}
                for qid in metric_by_system[left]["u_at_8"]:
                    du = (metric_by_system[left]["u_at_8"][qid]
                          - metric_by_system[right]["u_at_8"][qid])
                    dc = (metric_by_system[left][metric][qid]
                          - metric_by_system[right][metric][qid])
                    counts["pp" if du > 0 and dc > 0 else
                           "pn" if du > 0 else
                           "np" if dc > 0 else "nn"] += 1
                total = sum(counts.values())
                quadrants.append({
                    "contrast": tag, "arm": left, "comparator": right,
                    "community_metric": metric,
                    "dU_pos_dC_pos": counts["pp"], "dU_pos_dC_nonpos": counts["pn"],
                    "dU_nonpos_dC_pos": counts["np"],
                    "dU_nonpos_dC_nonpos": counts["nn"], "n_queries": total,
                    "share_dU_pos_dC_pos": f"{counts['pp']/total:.4f}",
                    "share_dU_pos_dC_nonpos": f"{counts['pn']/total:.4f}"})

    # system-level descriptive relationship
    current = [r for r in matrix if r["protocol_status"].startswith("current")]
    relationship = {}
    for metric in ("cra", "rcc", "bialign_f1"):
        relationship[f"u_at_8_vs_{metric}"] = {
            "pearson": caux.finite_spearman  # placeholder replaced below
        }
    xs = [float(r["u_at_8"]) for r in current]
    for metric in ("cra", "rcc", "bialign_f1"):
        ys = [float(r[metric]) for r in current]
        relationship[f"u_at_8_vs_{metric}"] = {
            "spearman": caux.finite_spearman(xs, ys),
            "pearson": (float(np.corrcoef(xs, ys)[0, 1])
                        if len(set(xs)) > 1 and len(set(ys)) > 1 else None),
            "n_system_conditions": len(xs),
            "caveat": "small, dependent set of conditions; descriptive only"}

    audit["checks"]["systems_evaluated"] = len(systems)
    audit["checks"]["per_query_rows"] = len(per_query_rows)
    audit["checks"]["every_set_has_eight_unique"] = all(
        r["selected_count"] == 8 and len(set(r["selected_comment_ids"].split(";"))) == 8
        for r in per_query_rows)
    audit["checks"]["aliases_detected"] = ALIAS_OF
    audit["checks"]["superseded_flagged"] = sorted(SUPERSEDED)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        out = Path(tmp)

        def write(name, rows):
            header = list(rows[0]) if rows else []
            with (out / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(header)
                writer.writerows([str(r.get(k, "")) for k in header] for r in rows)

        write("CURRENT_STAGE2_UTILITY_COMMUNITY_MATRIX.csv", matrix)
        write("CURRENT_STAGE2_COMMUNITY_CONTRASTS.csv", contrasts)
        write("CURRENT_STAGE2_COMMUNITY_PER_QUERY.csv", per_query_rows)
        write("CURRENT_STAGE2_SELECTED_ITEMS_LONG.csv", items_rows)
        write("CURRENT_STAGE2_UC_QUADRANTS.csv", quadrants)
        principal_ids = {
            DENSE_BASELINE,
            "best_lightweight_nested::direct", "best_lightweight_nested::replacement",
            "best_lightweight_nested::residual",
            "cross_encoder_matched::direct", "cross_encoder_matched::replacement",
            "cross_encoder_matched::residual",
            "lm7d_exp_g7::direct", "lm7d_exp_g7::residual",
            "lm7d_lin_g7::direct", "lm7d_lin_g7::residual"}
        principal_ids.update(map(str, cfg.get("supplemental_principal_system_ids", [])))
        principal = [r for r in matrix if r["canonical_system_id"] in principal_ids]
        write("THESIS_COMMUNITY_PRINCIPAL_SYSTEMS.csv", [{
            "System": r["scorer"], "Selection": r["selection_strategy"],
            "U@8": r["u_at_8"], "CRA": r["cra"], "RCC": r["rcc"],
            "BiAlignF1": r["bialign_f1"], "BestAlign": r["best_align"]}
            for r in principal])

        manifest = {
            "schema": "stage2-community-dev300-complete-v1",
            "version": str(cfg["version"]), "status": "COMPLETE",
            "created_utc": now(),
            "task_type": "post-hoc evaluation; zero model fitting, zero selector "
                         "tuning, zero LLM judging",
            "cohort": {"queries": len(replies_by_query),
                       "valid_hidden_replies": reply_total,
                       "candidate_pool": "RRF2 / E5 / M=50", "final_k": 8},
            "estimand": "community metrics are cross-fitted exactly as Utility@8 "
                        "is: per (query, outer fold) on that fold's selected set, "
                        "then averaged over the five folds in which the query is "
                        "held out",
            "audit": audit,
            "system_level_relationship": relationship,
            "counts": {"systems": len(systems), "matrix_rows": len(matrix),
                       "contrast_rows": len(contrasts),
                       "per_query_rows": len(per_query_rows),
                       "item_rows": len(items_rows),
                       "quadrant_rows": len(quadrants)},
            "bootstrap": {"draws": draws, "seed": seed, "unit": "whole query"},
            "boundaries": {"external_requests_made": 0, "frozen_test_read": False,
                           "model_training_performed": False,
                           "selection_performed": False,
                           "selector_retuned": False,
                           "utility_recomputed": False,
                           "hidden_replies_used_as_features": False,
                           "community_joined_after_selection_frozen": True},
            "definition_provenance": {
                "module": "evaluation/community_reply_auxiliary.py",
                "functions": ["alignment", "bidirectional_f", "encode_texts",
                              "text_sha"],
                "note": "CRA/RCC/BestAlign/ReplyCoverage and the encoder are "
                        "imported unchanged, not reimplemented"},
            "outputs": {name: sha256_file(out / name)
                        for name in sorted(p.name for p in out.iterdir())},
            "software": {"python": platform.python_version(),
                         "numpy": np.__version__},
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (out / "COMMUNITY_EVALUATION_AUDIT.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        manifest["outputs"] = {name: sha256_file(out / name)
                               for name in sorted(p.name for p in out.iterdir())}
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        Path(out).rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-key", default=CONFIG_KEY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-systems", type=int)
    args = parser.parse_args()
    manifest = run(args.config_key, args.output_dir, args.limit_systems)
    print(json.dumps({"status": manifest["status"], "cohort": manifest["cohort"],
                      "counts": manifest["counts"],
                      "boundaries": manifest["boundaries"]},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
