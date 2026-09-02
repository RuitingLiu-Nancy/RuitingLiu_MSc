#!/usr/bin/env python3
"""Expanded, fairly tuned lightweight scorer search on the frozen clean 7D contract.

Answers one question: given the frozen compact representation, how much extra
realised Utility@8 can a broader and better-tuned lightweight learner recover,
and does the matched cross-encoder advantage survive that stronger search?

Nothing about the representation, the candidate pool, the labels, the outer
folds or the cross-encoder is touched here.  Only the learner family and its
hyperparameters vary.

Reuse: every model is fitted through ``run_rq2b_scorer_family_oof_dev300``'s
``FITTERS`` registry, every design matrix through
``run_selection_action_space_repair._candidate_arrays``, every inner split
through the canonical ``inner_folds``, and both inner criteria through the
canonical ``_query_mean_mae`` / ``_query_mean_ndcg_at8``.  No loss, optimiser,
split or metric is reimplemented.

What IS new here is the nested model-family selection layer.  The existing
runners tune hyperparameters inside each outer training fold but always report
one arm per run; choosing the best FAMILY by looking at the finished
Development300 table would be selection-on-the-test-of-record.  So for every
outer fold this runner:

  1. tunes each family on inner folds of the outer-training queries only,
     using that family's own role-appropriate criterion;
  2. re-reads the selected configuration's inner out-of-fold predictions -
     which cover exactly the outer-training queries - and scores every family
     on a COMMON criterion, inner realised Utility@8;
  3. picks one family for this fold on that common criterion;
  4. reports that family's outer-validation predictions.

The outer-validation queries are never consulted at any step, so the resulting
BEST_LIGHTWEIGHT_NESTED arm is fully out-of-fold, family choice included.
"""
from __future__ import annotations

import argparse
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
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEY = "lightweight_scorer_search_dev300_rawtext"
BEST = "best_lightweight_nested"

sys.path.insert(0, str(ROOT))
from utility_scoring import run_rq2b_scorer_family_oof_dev300 as family  # noqa: E402
from evidence_selection import run_selection_action_space_repair as repair  # noqa: E402
from utility_scoring.stage2_training_contract import (  # noqa: E402
    load_direct_training_contract,
)

try:
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from utility_scoring.learned_diffusion import reranker_validation as canonical
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from utility_scoring.learned_diffusion import reranker_validation as canonical


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _spearman(predicted, actual) -> float | None:
    from scipy.stats import spearmanr
    if len(set(actual)) < 2 or len(set(predicted)) < 2:
        return None
    value = float(spearmanr(predicted, actual).statistic)
    return None if np.isnan(value) else value


def _mean_within_query_spearman(predictions, qids, candidate_ids, registry) -> float:
    values = []
    for qid in qids:
        ids = list(map(str, candidate_ids[qid]))
        rho = _spearman([predictions[(qid, cid)] for cid in ids],
                        [float(registry[(qid, cid)]["utility"]) for cid in ids])
        if rho is not None:
            values.append(rho)
    return statistics.fmean(values) if values else 0.0


def _mean_utility_at8(predictions, qids, candidate_ids, registry, k: int = 8) -> float:
    """The realised set-level metric, on whatever query subset is passed.

    Top-k by predicted score with a deterministic id tie-break, then the
    canonical ``utility_at8``.  Used here only on INNER validation queries, to
    put families with incomparable native criteria on one common footing.
    """
    values = []
    for qid in qids:
        ids = list(map(str, candidate_ids[qid]))
        ranked = sorted(ids, key=lambda cid: (-float(predictions[(qid, cid)]), cid))
        values.append(float(repair.utility_at8(ranked[:k], qid, registry)))
    return statistics.fmean(values)


def _load_universe(cfg: dict[str, Any]) -> dict[str, Any]:
    names = list(map(str, cfg["feature_names"]))
    rows = pq.read_table(cfg["features_parquet"]).to_pylist()
    static: dict[tuple[str, str], dict[str, float]] = {}
    pool: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        qid, cid = str(row["query_id"]), str(row["candidate_id"])
        static[(qid, cid)] = {name: float(row[name]) for name in names}
        pool[qid].append(cid)
    for qid in pool:
        pool[qid] = sorted(pool[qid])
    source = project_config.load()[str(cfg["source_config_key"])]
    contract = (load_direct_training_contract(ROOT, source, set(pool))
                if cfg.get("direct_split_contract")
                else repair._load_contract(source))
    if set(contract["qids"]) != set(pool):
        raise ValueError("feature matrix cohort differs from the frozen contract")
    return {"static": static, "pool": dict(pool), "contract": contract,
            "registry": contract["registry"], "splits": contract["splits"],
            "feature_names": names, "rows": len(rows)}


def _audit(cfg: dict[str, Any], universe: dict[str, Any]) -> dict[str, Any]:
    pool, static = universe["pool"], universe["static"]
    keys = set(static)
    reference = {(str(r["query_id"]), str(r["candidate_id"]))
                 for r in pq.read_table(
                     _resolve(cfg["reference_pool_parquet"])).to_pylist()}
    per_query = sorted({len(v) for v in pool.values()})
    labelled = sum(1 for key in keys if key in universe["registry"])
    folds = []
    for index, split in enumerate(universe["splits"]):
        train = list(map(str, split["train_query_ids"]))
        valid = list(map(str, split["validation_query_ids"]))
        folds.append({
            "outer_fold_index": index, "repeat": int(split["repeat"]),
            "fold": int(split["fold"]),
            "train_queries": len(train), "validation_queries": len(valid),
            "train_pairs": sum(len(pool[q]) for q in train),
            "validation_pairs": sum(len(pool[q]) for q in valid),
            "train_validation_query_overlap": len(set(train) & set(valid)),
        })
    gates = {
        "queries": len(pool), "rows": universe["rows"], "unique_pairs": len(keys),
        "duplicate_pairs": universe["rows"] - len(keys),
        "candidates_per_query_distinct": per_query,
        "candidates_per_query_is_exactly_50": per_query == [50],
        "label_coverage": f"{labelled}/{len(keys)}",
        "label_coverage_complete": labelled == len(keys),
        "key_identity_with_clean7d_universe": keys == reference,
        "reference_pool_rows": len(reference),
        "outer_folds": len(folds),
        "any_train_validation_overlap": any(
            f["train_validation_query_overlap"] for f in folds),
        "every_query_validated_five_times": sorted({
            sum(1 for s in universe["splits"]
                if qid in set(map(str, s["validation_query_ids"])))
            for qid in pool}) == [5],
        "feature_columns": universe["feature_names"],
        "constant_feature_columns": sorted(
            name for name in universe["feature_names"]
            if len({static[k][name] for k in keys}) == 1),
    }
    failed = [name for name, ok in (
        ("expected_queries", gates["queries"] == int(cfg["expected_queries"])),
        ("expected_rows", gates["rows"] == int(cfg["expected_pairs"])),
        ("no_duplicate_pairs", gates["duplicate_pairs"] == 0),
        ("candidates_per_query_is_exactly_50",
         gates["candidates_per_query_is_exactly_50"]),
        ("label_coverage_complete", gates["label_coverage_complete"]),
        ("key_identity_with_clean7d_universe",
         gates["key_identity_with_clean7d_universe"]),
        ("no_train_validation_overlap", not gates["any_train_validation_overlap"]),
        ("every_query_validated_five_times",
         gates["every_query_validated_five_times"]),
    ) if not ok]
    return {"gates": gates, "folds": folds, "failed_gates": failed}


def _tune_family(spec, universe, train_qids, inner_splits, seed):
    """Inner-fold search for one family, returning its selected configuration
    together with the inner out-of-fold predictions that configuration makes.

    The per-configuration selection criterion is the family's own, exactly as
    the canonical tuner defines it; the returned predictions are what the
    common cross-family criterion is later computed from.
    """
    pool, static, registry = universe["pool"], universe["static"], universe["registry"]
    names = universe["feature_names"]
    fitter = family.FITTERS[str(spec["fitter"])]
    kind = str(spec["family"])
    traces = []
    for config_index, setting in enumerate(spec["grid"]):
        predictions: dict[tuple[str, str], float] = {}
        scores = []
        for inner_index, (inner_train, inner_valid) in enumerate(inner_splits):
            model = fitter(inner_train, pool, static, registry, dict(setting),
                           seed + 10_000 * config_index + inner_index, names)
            predicted = family._predict_pairs(model, inner_valid, pool, static, names)
            predictions.update(predicted)
            scores.append(family._inner_score(kind, predicted, inner_valid,
                                              pool, registry))
        traces.append({"config_index": config_index, "setting": dict(setting),
                       "inner_mean": statistics.fmean(scores),
                       "inner_folds": scores, "predictions": predictions})
    best = (max(traces, key=lambda r: (r["inner_mean"], -r["config_index"]))
            if kind == family.RANKING
            else min(traces, key=lambda r: (r["inner_mean"], r["config_index"])))
    covered = sorted({qid for qid, _ in best["predictions"]})
    if covered != sorted(train_qids):
        raise AssertionError("inner predictions do not cover the outer-training queries")
    return {
        "selected_config_index": best["config_index"],
        "selected_setting": best["setting"],
        "inner_criterion": ("inner_query_mean_ndcg_at8" if kind == family.RANKING
                            else "inner_query_mean_mae"),
        "inner_criterion_value": best["inner_mean"],
        "inner_utility_at8": _mean_utility_at8(best["predictions"], train_qids,
                                               pool, registry),
        "inner_spearman": _mean_within_query_spearman(best["predictions"],
                                                      train_qids, pool, registry),
        "grid_trace": [{k: v for k, v in t.items() if k != "predictions"}
                       for t in traces],
    }


def run(config_key: str = CONFIG_KEY, output_dir: Path | None = None,
        limit_folds: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = dict(project_config.load()[config_key])
    if cfg.get("allow_external_calls") or cfg.get("allow_frozen_test"):
        raise ValueError("the lightweight search is development-only and offline")
    for key in ("output_dir", "journal_dir", "features_parquet",
                "reference_pool_parquet"):
        cfg[key] = _resolve(cfg[key])
        if key not in ("output_dir", "journal_dir") and "test" in str(cfg[key]).lower():
            raise ValueError(f"{key} resolves to a path resembling test scope")
    destination = Path(output_dir or cfg["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    universe = _load_universe(cfg)
    audit = _audit(cfg, universe)
    if audit["failed_gates"]:
        raise ValueError(f"pool audit failed: {audit['failed_gates']}")

    models = list(cfg["models"])
    family._preflight_dependencies(
        [{"scorer": str(m["name"]), "fitter": str(m["fitter"])} for m in models])
    journal = Path(cfg["journal_dir"])
    journal.mkdir(parents=True, exist_ok=True)

    pool, registry = universe["pool"], universe["registry"]
    splits = universe["splits"] if limit_folds is None \
        else universe["splits"][:limit_folds]
    raw: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list))
    inner_rows, fold_choice_rows, timing = [], [], defaultdict(float)

    for index, split in enumerate(splits):
        repeat, fold = int(split["repeat"]), int(split["fold"])
        tag = f"r{repeat}f{fold}"
        cached = journal / f"{tag}.json"
        if cached.exists():
            payload = json.loads(cached.read_text(encoding="utf-8"))
            print(json.dumps({"fold": tag, "journal": "reused"}), flush=True)
        else:
            fold_seed = int(split["seed"])
            train = list(map(str, split["train_query_ids"]))
            valid = list(map(str, split["validation_query_ids"]))
            if set(train) & set(valid):
                raise AssertionError("a held-out query entered training")
            inner = canonical.inner_folds(
                train, int(cfg["inner_folds"]),
                fold_seed + int(cfg["inner_split_seed_offset"]))
            seed = fold_seed + 700_000 + 1_000 * index
            payload = {"fold": tag, "repeat": repeat, "fold_index": fold,
                       "train_queries": len(train), "validation_queries": len(valid),
                       "models": {}}
            for spec in models:
                name = str(spec["name"])
                model_started = time.perf_counter()
                tuned = _tune_family(spec, universe, train, inner, seed)
                fitter = family.FITTERS[str(spec["fitter"])]
                model = fitter(train, pool, universe["static"], registry,
                               dict(tuned["selected_setting"]), seed + 500_000,
                               universe["feature_names"])
                predicted = family._predict_pairs(model, valid, pool,
                                                  universe["static"],
                                                  universe["feature_names"])
                expected = {(q, c) for q in valid for c in pool[q]}
                if set(predicted) != expected:
                    raise ValueError(f"{name}: held-out coverage changed")
                payload["models"][name] = {
                    **{k: v for k, v in tuned.items() if k != "grid_trace"},
                    "grid_trace": tuned["grid_trace"],
                    "validation_predictions": {f"{q}\t{c}": float(v)
                                               for (q, c), v in predicted.items()},
                    "elapsed_seconds": round(time.perf_counter() - model_started, 2),
                }
            cached.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            print(json.dumps({"fold": tag, "elapsed_seconds": round(
                sum(m["elapsed_seconds"] for m in payload["models"].values()), 1)}),
                flush=True)

        # ---- common-criterion family selection, inner data only ----------
        ranked = sorted(
            payload["models"].items(),
            key=lambda kv: (-kv[1]["inner_utility_at8"], -kv[1]["inner_spearman"],
                            kv[0]))
        chosen = ranked[0][0]
        for name, record in payload["models"].items():
            inner_rows.append({
                "outer_fold": payload["fold"], "model": name,
                "selected_config_index": record["selected_config_index"],
                "selected_hyperparameters": json.dumps(record["selected_setting"],
                                                       sort_keys=True),
                "inner_criterion": record["inner_criterion"],
                "inner_criterion_value": record["inner_criterion_value"],
                "inner_utility_at8": record["inner_utility_at8"],
                "inner_spearman": record["inner_spearman"],
                "selected_family_this_fold": name == chosen,
                "elapsed_seconds": record["elapsed_seconds"],
            })
            timing[name] += float(record["elapsed_seconds"])
            for key, value in record["validation_predictions"].items():
                qid, cid = key.split("\t")
                raw[name][(qid, cid)].append(float(value))
        fold_choice_rows.append({
            "outer_fold": payload["fold"], "selected_family": chosen,
            "inner_utility_at8": payload["models"][chosen]["inner_utility_at8"],
            "inner_spearman": payload["models"][chosen]["inner_spearman"],
            "runner_up": ranked[1][0] if len(ranked) > 1 else "",
            "runner_up_inner_utility_at8": (ranked[1][1]["inner_utility_at8"]
                                            if len(ranked) > 1 else ""),
        })
        for key, value in payload["models"][chosen]["validation_predictions"].items():
            qid, cid = key.split("\t")
            raw[BEST][(qid, cid)].append(float(value))

    if limit_folds is not None:
        return {"status": "PROBE", "folds": fold_choice_rows,
                "elapsed_seconds": round(time.perf_counter() - started, 2)}

    expected_pairs = sum(len(v) for v in pool.values())
    prediction_rows = []
    for name in sorted(raw):
        values = raw[name]
        if len(values) != expected_pairs:
            raise ValueError(f"{name}: {len(values)} pairs, expected {expected_pairs}")
        if any(len(v) != 5 for v in values.values()):
            raise ValueError(f"{name}: a pair lacks five out-of-fold repeats")
        for (qid, cid), repeats in sorted(values.items()):
            prediction_rows.append({
                "backend": str(cfg["backend"]), "scorer": name, "run": name,
                "feature_set": "clean7d" if name != BEST else "clean7d_nested",
                "query_id": qid, "candidate_id": cid,
                "oof_prediction_mean": statistics.fmean(repeats),
                "oof_repeats": len(repeats),
            })

    grid_rows = [{
        "model_family": str(spec["name"]), "fitter": str(spec["fitter"]),
        "configuration_id": i, "hyperparameters": json.dumps(dict(setting),
                                                             sort_keys=True),
        "intended_objective": str(spec["objective"]),
        "calibrated_output": "yes" if str(spec["family"]) == family.CALIBRATED
                             else "no",
        "inner_criterion": ("inner_query_mean_ndcg_at8"
                            if str(spec["family"]) == family.RANKING
                            else "inner_query_mean_mae"),
    } for spec in models for i, setting in enumerate(spec["grid"])]

    counts = defaultdict(int)
    for row in fold_choice_rows:
        counts[row["selected_family"]] += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        out = Path(tmp)
        pq.write_table(pa.Table.from_pylist(prediction_rows),
                       out / "oof_predictions.parquet", compression="zstd")
        import csv
        for name, rows_ in (("LIGHTWEIGHT_MODEL_GRID.csv", grid_rows),
                            ("LIGHTWEIGHT_INNER_SELECTION.csv", inner_rows),
                            ("BEST_LIGHTWEIGHT_FOLD_CHOICES.csv", fold_choice_rows),
                            ("LIGHTWEIGHT_POOL_AUDIT_FOLDS.csv", audit["folds"])):
            header = sorted({k for r in rows_ for k in r})
            with (out / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(header)
                writer.writerows([str(r.get(k, "")) for k in header] for r in rows_)
        manifest = {
            "schema": "lightweight-scorer-search-dev300-v1",
            "version": str(cfg["version"]), "status": "COMPLETE",
            "created_utc": now(),
            "research_question": str(cfg["research_question"]),
            "feature_contract": universe["feature_names"],
            "pool_audit": audit["gates"],
            "nested_family_selection": {
                "criterion": "inner realised Utility@8 on inner-validation "
                             "queries only; tie-break mean within-query Spearman, "
                             "then model name",
                "selection_counts": dict(sorted(counts.items())),
                "outer_validation_consulted": False,
            },
            "models": [{"name": str(s["name"]), "fitter": str(s["fitter"]),
                        "family": str(s["family"]),
                        "objective": str(s["objective"]),
                        "grid_size": len(s["grid"]),
                        "total_fit_seconds": round(timing[str(s["name"])], 1)}
                       for s in models],
            "counts": {"outer_folds": len(splits), "queries": len(pool),
                       "pairs": expected_pairs,
                       "arms": len(raw),
                       "configurations": len(grid_rows)},
            "inner_folds": int(cfg["inner_folds"]),
            "boundaries": {"external_requests_made": 0, "frozen_test_read": False,
                           "representation_modified": False,
                           "cross_encoder_touched": False,
                           "selection_hyperparameter_touched": False},
            "inputs": {str(cfg["features_parquet"]):
                       sha256_file(cfg["features_parquet"])},
            "outputs": {name: sha256_file(out / name)
                        for name in sorted(p.name for p in out.iterdir())},
            "software": {"python": platform.python_version(),
                         "numpy": np.__version__},
            "elapsed_seconds": round(time.perf_counter() - started, 2),
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
    parser.add_argument("--limit-folds", type=int)
    args = parser.parse_args()
    manifest = run(args.config_key, args.output_dir, args.limit_folds)
    print(json.dumps({k: manifest[k] for k in
                      ("status", "counts", "nested_family_selection", "folds",
                       "elapsed_seconds") if k in manifest},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
