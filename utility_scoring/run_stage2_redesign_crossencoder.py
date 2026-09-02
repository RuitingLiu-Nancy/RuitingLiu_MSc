#!/usr/bin/env python3
"""Experiment 5: the utility-aware MiniLM cross-encoder reference.

SPEC section 16 asks for exactly one lightweight neural text-interaction model,
trained against the same composite utility u(q,c) with pointwise robust
regression, to test whether direct token-level query-candidate interaction
carries more utility-prediction capacity than the engineered retrieval features.
It is a representation-capacity baseline, not a second utility definition.

Reuse: the Sentence-Transformers ``CrossEncoder`` / ``CrossEncoderTrainer``
stack that ``evaluation/crossencoder_utility_cv.py`` established for this
project.  No training loop is written here.  The loss is the library's own
``MSELoss`` module with its regression criterion replaced by Smooth L1, which
is the robust pointwise objective SPEC section 16 prefers and which the
library's kwargs cannot otherwise express; the trainer, the forward pass and
the optimiser are the library's.

Two deliberate departures from the feasibility script, both recorded in
DECISIONS D-20260827-02:
  * that script trains with the listwise LambdaLoss; SPEC section 16 requires a
    pointwise robust regression objective, which is what runs here;
  * that script splits with GroupKFold over annotation cards; this runner uses
    the frozen 5x5 query-grouped outer splits so the cross-encoder is
    comparable, fold for fold, with the feature-based scorers of Experiments 3
    and 4 and averages five out-of-fold repeats per pair exactly as they do.

Hyperparameters are fixed in configuration rather than searched: a nested grid
over 25 folds of a 22M-parameter transformer is not affordable here.  The
asymmetry against the feature scorers, which do get an inner-fold search, is
disclosed in the manifest and must be stated wherever this arm is compared.
"""
from __future__ import annotations

import argparse
import gc
import json
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
CONFIG_KEY = "stage2_redesign_crossencoder_rrf2_matched_rawtext_len256"

sys.path.insert(0, str(ROOT))
from utility_scoring.stage2_training_contract import (  # noqa: E402
    load_direct_training_contract,
)

try:
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from evaluation.judgment_completeness import complete_utility_v2_rows
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from evaluation.judgment_completeness import complete_utility_v2_rows


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _universe(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pairs, texts, labels and the frozen split contract for one training pool.

    Extracted verbatim from ``run`` so the pool audit, the token-length audit
    and the tuning pass all read the universe through exactly the same code
    path as the fold loop; the pool a run trains on is therefore the pool the
    audits describe, by construction rather than by convention.
    """
    rows = pq.read_table(cfg["features_parquet"]).to_pylist()
    pairs = [(str(r["query_id"]), str(r["candidate_id"])) for r in rows]
    by_query: dict[str, list[str]] = defaultdict(list)
    for qid, cid in pairs:
        by_query[qid].append(cid)

    corpus_text = {
        str(r["title"]): str(r["text"])
        for r in json.loads(Path(cfg["corpus"]).read_text(encoding="utf-8"))
    }
    query_text = {
        str(r["id"]): str(r["question"])
        for r in json.loads(Path(cfg["queries"]).read_text(encoding="utf-8"))
    }
    _complete, registry = complete_utility_v2_rows(
        _read_jsonl(cfg["utility_registry"]))
    if len(registry) != int(cfg["expected_registry_rows"]):
        raise ValueError("utility registry identity changed")
    missing = [p for p in pairs if p not in registry]
    if missing:
        raise ValueError(f"{len(missing)} pairs lack a utility label")
    utility = {p: float(registry[p]["utility"]) for p in pairs}

    source_raw = project_config.load()[str(cfg["source_config_key"])]
    if cfg.get("direct_split_contract"):
        contract = load_direct_training_contract(ROOT, source_raw, set(by_query))
    else:
        import run_selection_action_space_repair as repair
        contract = repair._load_contract(source_raw)
    if set(contract["qids"]) != set(by_query):
        raise ValueError("feature matrix cohort differs from the frozen contract")
    return {"pairs": pairs, "by_query": dict(by_query), "corpus_text": corpus_text,
            "query_text": query_text, "utility": utility, "contract": contract,
            "splits": contract["splits"]}


def _texts_for(universe: dict[str, Any], qids: list[str]):
    q, c, y, keys = [], [], [], []
    for qid in qids:
        for cid in universe["by_query"][qid]:
            q.append(universe["query_text"][qid])
            c.append(universe["corpus_text"][cid])
            y.append(universe["utility"][(qid, cid)])
            keys.append((qid, cid))
    return q, c, y, keys


def _spearman(predicted, actual) -> float | None:
    """The evaluation runner's within-query criterion, same definition."""
    from scipy.stats import spearmanr
    if len(set(actual)) < 2 or len(set(predicted)) < 2:
        return None
    value = float(spearmanr(predicted, actual).statistic)
    return None if np.isnan(value) else value


def _smooth_l1_loss_class():
    """The library's pointwise regression loss with a robust criterion.

    Only ``loss_fct`` changes; ``forward``, the preprocessing and the model call
    are inherited from Sentence-Transformers unchanged.
    """
    import torch
    from sentence_transformers.cross_encoder import losses as celoss

    class SmoothL1PointwiseLoss(celoss.MSELoss):
        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            self.loss_fct = torch.nn.SmoothL1Loss()

    return SmoothL1PointwiseLoss


def _stabilize_cross_encoder(model) -> dict[str, float]:
    """Apply the repository's verified local-load fix and fail closed.

    On the pinned macOS torch/transformers stack this checkpoint can expose
    finite parameters whose backing storage produces NaN during GEMM.  The
    same checkpoint is already repaired this way in
    ``remediation/runners/S1_build_rawtext_arms.py``.  Keep the workaround in
    one helper so OOF and tuning use the identical load contract.
    """
    import torch

    inner = getattr(model, "model", model)
    with torch.no_grad():
        for parameter in inner.parameters():
            parameter.data = parameter.data.detach().clone()

    probe = np.asarray(model.predict(
        [("how do I stop procrastinating on chores",
          "Break the task into two-minute steps and start a timer."),
         ("how do I stop procrastinating on chores",
          "My cat is orange and sleeps on the windowsill.")],
        batch_size=2, show_progress_bar=False), dtype=np.float64)
    if not np.isfinite(probe).all() or probe[0] <= probe[1]:
        raise RuntimeError(
            f"cross-encoder failed its discrimination probe: {probe.tolist()}")
    return {"relevant": float(probe[0]), "irrelevant": float(probe[1])}


def run(config_key: str = CONFIG_KEY, output_dir: Path | None = None,
        limit_folds: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = dict(project_config.load()[config_key])
    if cfg.get("allow_external_calls") or cfg.get("allow_frozen_test"):
        raise ValueError("cross-encoder training is local and development-only")
    for key in ("output_dir", "journal_dir", "zero_shot_output_dir",
                "features_parquet", "corpus", "queries", "utility_registry"):
        cfg[key] = _resolve(cfg[key])
        if key not in ("output_dir", "journal_dir") and "test" in str(cfg[key]).lower():
            raise ValueError(f"{key} resolves to a path resembling test scope")
    destination = Path(output_dir or cfg["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer, CrossEncoderTrainingArguments,
    )
    from transformers import set_seed

    # ---- universe, texts and labels --------------------------------------
    universe = _universe(cfg)
    pairs = universe["pairs"]
    by_query = universe["by_query"]
    utility = universe["utility"]
    splits = universe["splits"]

    model_name = str(cfg["model"])
    max_length = int(cfg["max_length"])
    seed = int(cfg["seed"])
    journal = Path(cfg["journal_dir"])
    journal.mkdir(parents=True, exist_ok=True)

    def texts_for(qids: list[str]) -> tuple[list[str], list[str], list[float],
                                            list[tuple[str, str]]]:
        return _texts_for(universe, qids)

    loss_class = _smooth_l1_loss_class()
    raw_predictions: dict[tuple[str, str], list[float]] = defaultdict(list)
    fold_rows: list[dict] = []
    planned = splits if limit_folds is None else splits[:limit_folds]
    for index, split in enumerate(planned):
        repeat, fold = int(split["repeat"]), int(split["fold"])
        tag = f"r{repeat}f{fold}"
        fold_path = journal / f"{tag}.parquet"
        meta_path = journal / f"{tag}_meta.json"
        if fold_path.exists() and meta_path.exists():
            cached = pq.read_table(fold_path).to_pylist()
            for row in cached:
                raw_predictions[(str(row["query_id"]),
                                 str(row["candidate_id"]))].append(
                    float(row["prediction"]))
            fold_rows.append(json.loads(meta_path.read_text(encoding="utf-8")))
            print(json.dumps({"fold": tag, "journal": "reused"}), flush=True)
            continue

        fold_started = time.perf_counter()
        train_qids = list(map(str, split["train_query_ids"]))
        valid_qids = list(map(str, split["validation_query_ids"]))
        if set(train_qids) & set(valid_qids):
            raise AssertionError("a held-out query entered training")
        set_seed(seed + index)
        model = CrossEncoder(model_name, num_labels=1, local_files_only=True,
                             max_length=max_length)
        probe = _stabilize_cross_encoder(model)
        print(json.dumps({"fold": tag, "cross_encoder_probe": probe},
                         sort_keys=True), flush=True)
        tq, tc, ty, _ = texts_for(train_qids)
        dataset = Dataset.from_dict({"query": tq, "candidate": tc, "label": ty})
        with tempfile.TemporaryDirectory(prefix=f".ce_{tag}.") as trainer_dir:
            args_ = CrossEncoderTrainingArguments(
                output_dir=trainer_dir,
                num_train_epochs=float(cfg["epochs"]),
                per_device_train_batch_size=int(cfg["batch_size"]),
                learning_rate=float(cfg["learning_rate"]),
                warmup_ratio=float(cfg["warmup_ratio"]),
                optim="adamw_torch",
                save_strategy="no", eval_strategy="no", logging_strategy="no",
                report_to="none", disable_tqdm=True,
                seed=seed + index, data_seed=seed + index,
            )
            trainer = CrossEncoderTrainer(
                model=model, args=args_, train_dataset=dataset,
                loss=loss_class(model),
            )
            train_output = trainer.train()
        vq, vc, vy, vkeys = texts_for(valid_qids)
        scores = np.asarray(
            model.predict(list(zip(vq, vc)),
                          batch_size=int(cfg["predict_batch_size"]),
                          show_progress_bar=False),
            dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"non-finite cross-encoder scores in fold {tag}")
        cached = [
            {"query_id": qid, "candidate_id": cid, "prediction": float(value)}
            for (qid, cid), value in zip(vkeys, scores)
        ]
        pq.write_table(pa.Table.from_pylist(cached), fold_path,
                       compression="zstd")
        meta = {
            "fold": tag, "repeat": repeat, "fold_index": fold,
            "train_queries": len(train_qids), "validation_queries": len(valid_qids),
            "train_pairs": len(ty), "validation_pairs": len(vy),
            "training_loss": float(train_output.training_loss),
            "elapsed_seconds": round(time.perf_counter() - fold_started, 2),
        }
        meta_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
        for row in cached:
            raw_predictions[(str(row["query_id"]),
                             str(row["candidate_id"]))].append(
                float(row["prediction"]))
        fold_rows.append(meta)
        print(json.dumps(meta, sort_keys=True), flush=True)
        del trainer, model, dataset
        gc.collect()

    if limit_folds is not None:
        return {"status": "PROBE", "folds": fold_rows}

    run_name = str(cfg["run_name"])
    prediction_rows = [
        {
            "backend": str(cfg["backend"]), "scorer": run_name, "run": run_name,
            "feature_set": "cross_encoder_text",
            "query_id": qid, "candidate_id": cid,
            "oof_prediction_mean": statistics.fmean(values),
            "oof_repeats": len(values),
        }
        for (qid, cid), values in sorted(raw_predictions.items())
    ]
    if len(prediction_rows) != len(pairs):
        raise ValueError(
            f"{len(prediction_rows)} predicted pairs, expected {len(pairs)}")
    if any(row["oof_repeats"] != 5 for row in prediction_rows):
        raise ValueError("a pair lacks five out-of-fold repeats")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        out = Path(tmp)
        pq.write_table(pa.Table.from_pylist(prediction_rows),
                       out / "oof_predictions.parquet", compression="zstd")
        header = list(fold_rows[0])
        (out / "fold_training.csv").write_text(
            "\n".join([",".join(header)] + [
                ",".join(str(r[k]) for k in header) for r in fold_rows
            ]) + "\n", encoding="utf-8")
        manifest = {
            "schema": "stage2-redesign-crossencoder-v1",
            "version": str(cfg["version"]),
            "status": "COMPLETE",
            "created_utc": now(),
            "experiment": "SPEC Experiment 5 - utility-aware cross-encoder",
            "track": "A_rrf2_principal",
            "model": {
                "checkpoint": model_name,
                "local_files_only": True,
                "num_labels": 1,
                "max_length": max_length,
                "parameters_millions": 22.7,
            },
            "objective": {
                "target": "frozen composite utility u(q,c), 1-7 scale, unscaled",
                "loss": ("Sentence-Transformers CrossEncoder MSELoss module with "
                         "its criterion replaced by torch SmoothL1Loss "
                         "(SPEC section 16 pointwise robust regression)"),
                "trainer": "sentence_transformers CrossEncoderTrainer",
                "hyperparameter_search": False,
                "hyperparameter_disclosure": (
                    "fixed configuration, no inner-fold grid search, unlike the "
                    "feature-based scorers of Experiments 3 and 4; a nested "
                    "search over 25 folds of a transformer was not affordable. "
                    "State this asymmetry wherever this arm is compared"
                ),
                "epochs": float(cfg["epochs"]),
                "batch_size": int(cfg["batch_size"]),
                "learning_rate": float(cfg["learning_rate"]),
                "warmup_ratio": float(cfg["warmup_ratio"]),
                "seed": seed,
            },
            "folds": fold_rows,
            "counts": {
                "outer_folds": len(splits),
                "queries": len(by_query),
                "pairs": len(prediction_rows),
            },
            "boundaries": {
                "external_requests_made": 0,
                "frozen_test_read": False,
                "engineered_features_used": False,
                "fusion_entry_rank_used_as_feature": False,
            },
            "inputs": {
                str(cfg["features_parquet"]): sha256_file(cfg["features_parquet"]),
            },
            "outputs": {
                name: sha256_file(out / name)
                for name in ("oof_predictions.parquet", "fold_training.csv")
            },
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        Path(out).rename(destination)
    return manifest


def _prepare(config_key: str) -> dict[str, Any]:
    cfg = dict(project_config.load()[config_key])
    if cfg.get("allow_external_calls") or cfg.get("allow_frozen_test"):
        raise ValueError("cross-encoder work is local and development-only")
    for key in ("output_dir", "journal_dir", "features_parquet", "corpus",
                "queries", "utility_registry"):
        if key not in cfg:
            continue
        cfg[key] = _resolve(cfg[key])
        if key not in ("output_dir", "journal_dir") and "test" in str(cfg[key]).lower():
            raise ValueError(f"{key} resolves to a path resembling test scope")
    return cfg


def zero_shot(config_key: str, output_dir: Path | None = None) -> dict[str, Any]:
    """Score the matched Development300 pool with the unfitted base CE.

    This is the architecture/supervision control requested for the Stage-2
    analysis. It uses the exact query/candidate texts, M=50 pool and
    max-length contract of the utility-trained arm, but performs no fitting.
    The score file is deliberately named ``predictions.parquet`` rather than
    OOF predictions: every score comes from the unchanged public MS MARCO
    checkpoint and has no Development300 training exposure.
    """
    started = time.perf_counter()
    cfg = _prepare(config_key)
    if "zero_shot_output_dir" not in cfg and output_dir is None:
        raise KeyError("zero_shot_output_dir is required for zero-shot mode")
    destination = Path(output_dir or cfg["zero_shot_output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    from sentence_transformers import CrossEncoder

    universe = _universe(cfg)
    qids = sorted(universe["by_query"])
    query, candidate, actual, keys = _texts_for(universe, qids)
    model = CrossEncoder(
        str(cfg["model"]), num_labels=1, local_files_only=True,
        max_length=int(cfg["max_length"]),
    )
    probe = _stabilize_cross_encoder(model)
    scores = np.asarray(
        model.predict(
            list(zip(query, candidate)),
            batch_size=int(cfg["predict_batch_size"]),
            show_progress_bar=True,
        ),
        dtype=np.float64,
    )
    if scores.shape != (len(keys),) or not np.isfinite(scores).all():
        raise RuntimeError("zero-shot cross-encoder returned invalid scores")

    prediction_rows = [{
        "backend": str(cfg["backend"]),
        "scorer": "cross_encoder_ms_marco_zero_shot",
        "run": "cross_encoder_ms_marco_zero_shot",
        "feature_set": "cross_encoder_text_zero_shot",
        "query_id": qid,
        "candidate_id": cid,
        "prediction": float(score),
    } for (qid, cid), score in zip(keys, scores, strict=True)]

    by_query_prediction: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for (qid, cid), score, target in zip(keys, scores, actual, strict=True):
        by_query_prediction[qid].append((cid, float(score), float(target)))
    per_query_rows: list[dict[str, Any]] = []
    for qid in qids:
        values = by_query_prediction[qid]
        selected = sorted(values, key=lambda row: (-row[1], row[0]))[:8]
        rho = _spearman([row[1] for row in values], [row[2] for row in values])
        per_query_rows.append({
            "query_id": qid,
            "candidate_count": len(values),
            "selected_comment_ids": ";".join(row[0] for row in selected),
            "utility_at8": statistics.fmean(row[2] for row in selected),
            "within_query_spearman": "" if rho is None else rho,
        })
    if {row["candidate_count"] for row in per_query_rows} != {50}:
        raise ValueError("zero-shot scoring did not preserve the M=50 pool")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        import csv
        out = Path(tmp)
        pq.write_table(
            pa.Table.from_pylist(prediction_rows),
            out / "predictions.parquet", compression="zstd",
        )
        with (out / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_query_rows[0]))
            writer.writeheader()
            writer.writerows(per_query_rows)
        defined_rho = [float(row["within_query_spearman"])
                       for row in per_query_rows
                       if row["within_query_spearman"] != ""]
        manifest = {
            "schema": "stage2-crossencoder-zero-shot-control-v1",
            "version": str(cfg["version"]),
            "status": "COMPLETE",
            "created_utc": now(),
            "model": {
                "checkpoint": str(cfg["model"]),
                "local_files_only": True,
                "max_length": int(cfg["max_length"]),
                "supervision": "public MS MARCO relevance only; no Development300 utility fitting",
                "stability_probe": probe,
            },
            "counts": {"queries": len(qids), "pairs": len(keys),
                       "candidates_per_query": 50, "selected_per_query": 8},
            "summary": {
                "mean_utility_at8": statistics.fmean(
                    float(row["utility_at8"]) for row in per_query_rows),
                "mean_within_query_spearman": statistics.fmean(defined_rho),
                "queries_with_defined_spearman": len(defined_rho),
            },
            "boundaries": {"external_requests_made": 0,
                           "frozen_test_read": False,
                           "model_training_performed": False,
                           "community_used_for_selection": False},
            "inputs": {str(cfg["features_parquet"]):
                       sha256_file(cfg["features_parquet"])},
            "outputs": {
                "predictions.parquet": sha256_file(out / "predictions.parquet"),
                "per_query.csv": sha256_file(out / "per_query.csv"),
            },
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(out).rename(destination)
    return manifest


def closed600_transfer(config_key: str, output_dir: Path | None = None) -> dict[str, Any]:
    """Apply the final utility-trained CE to the repaired Closed600 E5@100.

    This is a local, supplemental cross-task diagnostic. Closed600's native
    reply-recovery qrels measure structural/semantic relevance rather than the
    Development300 utility target, so the report is explicitly split into
    Development300-overlap and non-overlap queries and is never used to select
    the Stage-2 model.
    """
    import csv
    import torch
    from sentence_transformers import CrossEncoder
    from run_closed600_e5_lopo_preflight import structural_profile
    from score_multihop_retrieval import paired_bootstrap_delta

    started = time.perf_counter()
    cfg = _prepare(config_key)
    transfer = dict(cfg["closed600_transfer"])
    for key in ("output_dir", "checkpoint", "heldout", "corpus_map",
                "e5_rankings", "zero_shot_rankings", "development_queries"):
        transfer[key] = _resolve(transfer[key])
        if "test" in str(transfer[key]).lower():
            raise ValueError(f"closed600 transfer input {key} resembles test scope")
    destination = Path(output_dir or transfer["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    with Path(transfer["heldout"]).open(encoding="utf-8") as handle:
        heldout = list(csv.DictReader(handle))
    query_text = {str(row["post_id"]): str(row["query_text"]) for row in heldout}
    gold = {str(row["post_id"]): {
        cid for cid in str(row["gold_comment_ids"]).split("|") if cid
    } for row in heldout}
    with Path(transfer["corpus_map"]).open(encoding="utf-8") as handle:
        comment_to_post = {
            str(row["comment_id"]): str(row["post_id"])
            for row in csv.DictReader(handle)
        }
    corpus = {str(row["title"]): str(row["text"])
              for row in json.loads(Path(cfg["corpus"]).read_text(encoding="utf-8"))}

    def load_rankings(path: Path) -> dict[str, list[str]]:
        result = {str(row["id"]): list(map(str, row["retrieved_titles"]))
                  for row in _read_jsonl(path)}
        if set(result) != set(query_text):
            raise ValueError(f"ranking query identity mismatch: {path}")
        return result

    e5 = load_rankings(transfer["e5_rankings"])
    zero = load_rankings(transfer["zero_shot_rankings"])
    if any(len(e5[qid]) < 100 for qid in e5):
        raise ValueError("E5 first stage is shallower than 100")

    model = CrossEncoder(
        str(transfer["checkpoint"]), num_labels=1, local_files_only=True,
        max_length=int(cfg["max_length"]),
    )
    inner = getattr(model, "model", model)
    with torch.no_grad():
        for parameter in inner.parameters():
            parameter.data = parameter.data.detach().clone()
    probe = np.asarray(model.predict(
        [("how do I stop procrastinating on chores",
          "Break the task into two-minute steps and start a timer."),
         ("how do I stop procrastinating on chores",
          "My cat is orange and sleeps on the windowsill.")],
        batch_size=2, show_progress_bar=False), dtype=np.float64)
    if not np.isfinite(probe).all() or np.ptp(probe) == 0:
        raise RuntimeError(f"utility-trained CE failed finite probe: {probe.tolist()}")

    trained: dict[str, list[str]] = {}
    ranking_rows: list[dict[str, Any]] = []
    scored = 0
    qids = sorted(query_text)
    for index, qid in enumerate(qids, start=1):
        pool = e5[qid][:100]
        scores = np.asarray(model.predict(
            [(query_text[qid], corpus[cid]) for cid in pool],
            batch_size=int(cfg["predict_batch_size"]),
            show_progress_bar=False,
        ), dtype=np.float64)
        if scores.shape != (100,) or not np.isfinite(scores).all():
            raise RuntimeError(f"invalid utility-trained CE scores for {qid}")
        order = sorted(range(100), key=lambda i: (-float(scores[i]), pool[i]))
        trained[qid] = [pool[i] for i in order]
        ranking_rows.append({"id": qid, "retrieved_titles": trained[qid]})
        scored += 100
        if index % 100 == 0 or index == len(qids):
            print(json.dumps({"mode": "closed600-transfer", "queries_done": index,
                              "pairs_scored": scored,
                              "elapsed_seconds": round(time.perf_counter() - started, 1)}),
                  flush=True)

    development_ids = {
        str(row["id"]) for row in json.loads(
            Path(transfer["development_queries"]).read_text(encoding="utf-8"))
    }
    overlap = sorted(set(qids) & development_ids)
    nonoverlap = sorted(set(qids) - development_ids)
    systems = {"e5_dense": e5, "ce_ms_marco_zero_shot": zero,
               "ce_utility_trained": trained}
    groups = {"all600": qids, "development300_overlap": overlap,
              "nonoverlap_primary": nonoverlap}
    profiles: dict[str, Any] = {}
    per_query: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    comparisons: list[dict[str, Any]] = []
    for group_name, group_qids in groups.items():
        if not group_qids:
            profiles[group_name] = {}
            per_query[group_name] = {}
            continue
        subset_gold = {qid: gold[qid] for qid in group_qids}
        profiles[group_name] = {}
        per_query[group_name] = {}
        for system_name, rankings in systems.items():
            profile, values = structural_profile(rankings, subset_gold, comment_to_post)
            profiles[group_name][system_name] = profile
            per_query[group_name][system_name] = values
        for left, right in (
            ("ce_ms_marco_zero_shot", "e5_dense"),
            ("ce_utility_trained", "e5_dense"),
            ("ce_utility_trained", "ce_ms_marco_zero_shot"),
        ):
            for metric in ("nDCG@10", "Recall@5", "MRR", "Hit@10"):
                result = paired_bootstrap_delta(
                    {qid: values[metric] for qid, values
                     in per_query[group_name][left].items()},
                    {qid: values[metric] for qid, values
                     in per_query[group_name][right].items()},
                    n_boot=int(transfer["bootstrap_draws"]),
                    seed=int(transfer["bootstrap_seed"]),
                )
                comparisons.append({"group": group_name, "left": left,
                                    "right": right, "metric": metric,
                                    **(result or {})})

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        out = Path(tmp)
        with (out / "ce_utility_trained_rerank_closed600.jsonl").open(
                "w", encoding="utf-8") as handle:
            for row in ranking_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = {"profiles": profiles, "paired_comparisons": comparisons,
                  "primary_group": "nonoverlap_primary",
                  "claim_boundary": (
                      "Supplemental cross-task diagnostic only. Closed600 qrels are native "
                      "reply-recovery relevance, not the Development300 utility target; "
                      "the non-overlap subset is primary because Development300 queries "
                      "were used to fit the utility-trained checkpoint.")}
        (out / "structural_transfer_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        manifest = {
            "schema": "closed600-utility-ce-transfer-v1",
            "version": str(cfg["version"]), "status": "COMPLETE",
            "created_utc": now(),
            "model": {"checkpoint": str(transfer["checkpoint"]),
                      "max_length": int(cfg["max_length"]),
                      "training_target": "Development300 utility-v2"},
            "counts": {"queries": len(qids), "pairs_scored": scored,
                       "development300_overlap_queries": len(overlap),
                       "nonoverlap_queries": len(nonoverlap)},
            "boundaries": {"external_requests_made": 0,
                           "frozen_test_read": False,
                           "model_fitted_in_this_run": False,
                           "used_for_model_selection": False},
            "inputs": {
                **{str(transfer[key]): sha256_file(transfer[key])
                   for key in ("heldout", "corpus_map", "e5_rankings",
                               "zero_shot_rankings", "development_queries")},
                str(transfer["checkpoint"] / "model.safetensors"):
                    sha256_file(transfer["checkpoint"] / "model.safetensors"),
            },
            "outputs": {
                "ce_utility_trained_rerank_closed600.jsonl": sha256_file(
                    out / "ce_utility_trained_rerank_closed600.jsonl"),
                "structural_transfer_report.json": sha256_file(
                    out / "structural_transfer_report.json"),
            },
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        Path(out).rename(destination)
    return manifest


def _tuning_split(universe: dict[str, Any], cfg: dict[str, Any]
                  ) -> tuple[list[str], list[str], dict[str, Any]]:
    """One deterministic query-grouped split for configuration selection.

    The universe is the TRAINING side of one frozen outer fold, so no query
    that the chosen configuration is scored on here was held out of that fold.
    That bounds, but does not eliminate, the overlap: under a 5x5 design every
    query is a validation query in five of the twenty-five folds, so the tuning
    queries do reappear as out-of-fold predictions later.  The configuration
    choice is therefore development-tuned but NOT fully nested, which is a
    smaller asymmetry than the previous fixed configuration rather than none.
    The manifest states this and the thesis must repeat it.
    """
    tune_cfg = dict(cfg["tuning"])
    index = int(tune_cfg["universe_split_index"])
    split = universe["splits"][index]
    pool = sorted(map(str, split["train_query_ids"]))
    rng = np.random.default_rng(int(tune_cfg["split_seed"]))
    order = list(rng.permutation(len(pool)))
    holdout = int(tune_cfg["holdout_queries"])
    if not 0 < holdout < len(pool):
        raise ValueError("holdout_queries must sit strictly inside the pool")
    valid = sorted(pool[i] for i in order[:holdout])
    train = sorted(pool[i] for i in order[holdout:])
    if set(train) & set(valid):
        raise AssertionError("tuning split overlaps")
    audit = {
        "universe": f"train side of frozen outer split index {index} "
                    f"(repeat {split['repeat']}, fold {split['fold']})",
        "universe_queries": len(pool),
        "tuning_train_queries": len(train),
        "tuning_validation_queries": len(valid),
        "split_seed": int(tune_cfg["split_seed"]),
        "overlap_queries": 0,
        "nesting_disclosure": (
            "one global configuration selected on this single query-grouped "
            "split; not a fully nested per-outer-fold search. Every query is a "
            "validation query in 5 of the 25 outer folds, so these tuning "
            "queries reappear in the reported out-of-fold predictions"),
    }
    return train, valid, audit


def audit_pool(config_key: str, output_dir: Path | None = None) -> dict[str, Any]:
    """Section 2 and 3 gates: pool identity, fold identity, label coverage."""
    started = time.perf_counter()
    cfg = _prepare(config_key)
    destination = Path(output_dir or cfg["audit_output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    universe = _universe(cfg)
    pairs, by_query = universe["pairs"], universe["by_query"]

    reference = pq.read_table(_resolve(cfg["reference_pool_parquet"])).to_pylist()
    reference_keys = {(str(r["query_id"]), str(r["candidate_id"]))
                      for r in reference}
    per_query = sorted({len(v) for v in by_query.values()})
    gates = {
        "unique_queries": len(by_query),
        "unique_pairs": len(set(pairs)),
        "total_rows": len(pairs),
        "duplicate_pairs": len(pairs) - len(set(pairs)),
        "candidates_per_query_distinct_values": per_query,
        "candidates_per_query_is_exactly_50": per_query == [50],
        "label_coverage": f"{len(universe['utility'])}/{len(pairs)}",
        "label_coverage_complete": len(universe["utility"]) == len(pairs),
        "key_identity_with_feature_scorer_pool":
            set(pairs) == reference_keys,
        "reference_pool_rows": len(reference_keys),
        "rrf3_only_candidates_present": len(set(pairs) - reference_keys),
    }
    folds = []
    for index, split in enumerate(universe["splits"]):
        train = list(map(str, split["train_query_ids"]))
        valid = list(map(str, split["validation_query_ids"]))
        folds.append({
            "outer_fold_index": index,
            "repeat": int(split["repeat"]), "fold": int(split["fold"]),
            "train_queries": len(train), "validation_queries": len(valid),
            "train_pairs": sum(len(by_query[q]) for q in train),
            "validation_pairs": sum(len(by_query[q]) for q in valid),
            "train_validation_query_overlap": len(set(train) & set(valid)),
        })
    gates["outer_folds"] = len(folds)
    gates["any_train_validation_overlap"] = any(
        f["train_validation_query_overlap"] for f in folds)
    gates["every_query_validated_five_times"] = sorted({
        sum(1 for s in universe["splits"]
            if qid in set(map(str, s["validation_query_ids"])))
        for qid in by_query}) == [5]

    failed = [
        name for name, ok in (
            ("candidates_per_query_is_exactly_50",
             gates["candidates_per_query_is_exactly_50"]),
            ("label_coverage_complete", gates["label_coverage_complete"]),
            ("key_identity_with_feature_scorer_pool",
             gates["key_identity_with_feature_scorer_pool"]),
            ("no_duplicate_pairs", gates["duplicate_pairs"] == 0),
            ("no_rrf3_only_candidates",
             gates["rrf3_only_candidates_present"] == 0),
            ("expected_pairs", len(pairs) == int(cfg["expected_pairs"])),
            ("expected_queries", len(by_query) == int(cfg["expected_queries"])),
            ("no_train_validation_overlap",
             not gates["any_train_validation_overlap"]),
            ("every_query_validated_five_times",
             gates["every_query_validated_five_times"]),
        ) if not ok
    ]
    return _emit(destination, started, cfg, {
        "schema": "stage2-crossencoder-matched-pool-audit-v1",
        "status": "FAILED_GATE" if failed else "COMPLETE",
        "failed_gates": failed,
        "gates": gates,
        "folds": folds,
    }, tables={"CE_MATCHED_POOL_AUDIT_FOLDS.csv": folds})


def audit_tokens(config_key: str, output_dir: Path | None = None) -> dict[str, Any]:
    """Section 6 gate: is truncation at the configured max_length material?"""
    started = time.perf_counter()
    cfg = _prepare(config_key)
    destination = Path(output_dir or cfg["token_audit_output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    from transformers import AutoTokenizer

    universe = _universe(cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(cfg["model"]),
                                              local_files_only=True)
    q, c, _y, _keys = _texts_for(universe, sorted(universe["by_query"]))
    lengths = np.asarray([
        len(ids) for ids in tokenizer(q, c, truncation=False)["input_ids"]
    ], dtype=int)
    limits = [int(v) for v in cfg["token_audit_limits"]]
    percentiles = {
        f"p{p}": float(np.percentile(lengths, p)) for p in (50, 75, 90, 95, 99)
    }
    rows = [{
        "statistic": name, "value": value
    } for name, value in [
        ("pairs", int(lengths.size)), ("min", int(lengths.min())),
        ("mean", float(lengths.mean())), ("median", float(np.median(lengths))),
        *[(k, v) for k, v in percentiles.items()],
        ("max", int(lengths.max())),
    ]]
    for limit in limits:
        rows.append({"statistic": f"fraction_truncated_at_{limit}",
                     "value": float((lengths > limit).mean())})
        rows.append({"statistic": f"pairs_truncated_at_{limit}",
                     "value": int((lengths > limit).sum())})
    configured = int(cfg["max_length"])
    fraction = float((lengths > configured).mean())
    return _emit(destination, started, cfg, {
        "schema": "stage2-crossencoder-token-audit-v1",
        "status": "COMPLETE",
        "tokenizer": str(cfg["model"]),
        "configured_max_length": configured,
        "fraction_truncated_at_configured_max_length": fraction,
        "decision_rule": ("keep max_length when under ~10% of pairs truncate; "
                          "otherwise compare {192, 256} and no wider"),
        "keep_configured_max_length": fraction < 0.10,
        "percentiles": percentiles,
    }, tables={"CE_TOKEN_LENGTH_AUDIT.csv": rows})


def tune(config_key: str, output_dir: Path | None = None) -> dict[str, Any]:
    """Sections 6-8: a small, bounded, development-only configuration search."""
    started = time.perf_counter()
    cfg = _prepare(config_key)
    tune_cfg = dict(cfg["tuning"])
    destination = Path(output_dir or tune_cfg["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer, CrossEncoderTrainingArguments,
    )
    from transformers import set_seed

    universe = _universe(cfg)
    train_qids, valid_qids, split_audit = _tuning_split(universe, cfg)
    tq, tc, ty, _ = _texts_for(universe, train_qids)
    vq, vc, vy, vkeys = _texts_for(universe, valid_qids)
    loss_class = _smooth_l1_loss_class()
    journal = Path(_resolve(tune_cfg["journal_dir"]))
    journal.mkdir(parents=True, exist_ok=True)

    grid_rows: list[dict] = []
    for index, setting in enumerate(tune_cfg["grid"]):
        learning_rate = float(setting["learning_rate"])
        epochs = float(setting["epochs"])
        max_length = int(setting.get("max_length", cfg["max_length"]))
        tag = f"cfg{index:02d}_lr{learning_rate:.0e}_ep{epochs:g}_ml{max_length}"
        cached_path = journal / f"{tag}.json"
        if cached_path.exists():
            row = json.loads(cached_path.read_text(encoding="utf-8"))
            row["journal_reused"] = True
            grid_rows.append(row)
            print(json.dumps({"config": tag, "journal": "reused"}), flush=True)
            continue

        config_started = time.perf_counter()
        seed = int(cfg["seed"]) + index
        set_seed(seed)
        model = CrossEncoder(str(cfg["model"]), num_labels=1,
                             local_files_only=True, max_length=max_length)
        probe = _stabilize_cross_encoder(model)
        print(json.dumps({"config": tag, "cross_encoder_probe": probe},
                         sort_keys=True), flush=True)
        dataset = Dataset.from_dict({"query": tq, "candidate": tc, "label": ty})
        with tempfile.TemporaryDirectory(prefix=f".ce_tune_{index}.") as trainer_dir:
            args_ = CrossEncoderTrainingArguments(
                output_dir=trainer_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=int(cfg["batch_size"]),
                learning_rate=learning_rate,
                warmup_ratio=float(cfg["warmup_ratio"]),
                optim="adamw_torch",
                save_strategy="no", eval_strategy="no", logging_strategy="no",
                report_to="none", disable_tqdm=True,
                seed=seed, data_seed=seed,
            )
            trainer = CrossEncoderTrainer(model=model, args=args_,
                                          train_dataset=dataset,
                                          loss=loss_class(model))
            train_output = trainer.train()
        scores = np.asarray(
            model.predict(list(zip(vq, vc)),
                          batch_size=int(cfg["predict_batch_size"]),
                          show_progress_bar=False), dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"non-finite scores for configuration {tag}")
        by_q: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (qid, _cid), predicted, actual in zip(vkeys, scores, vy):
            by_q[qid].append((float(predicted), float(actual)))
        rhos = [r for r in (
            _spearman([p for p, _ in v], [a for _, a in v]) for v in by_q.values()
        ) if r is not None]
        row = {
            "config_index": index, "tag": tag,
            "learning_rate": learning_rate, "epochs": epochs,
            "max_length": max_length, "seed": seed,
            "training_loss": float(train_output.training_loss),
            "validation_queries": len(by_q),
            "validation_pairs": len(vy),
            "queries_with_defined_spearman": len(rhos),
            "within_query_spearman_mean": statistics.fmean(rhos),
            "within_query_spearman_median": statistics.median(rhos),
            "mean_absolute_error": float(np.mean(np.abs(scores - np.asarray(vy)))),
            "elapsed_seconds": round(time.perf_counter() - config_started, 2),
            "journal_reused": False,
        }
        cached_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
        grid_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del trainer, model, dataset
        gc.collect()

    # primary criterion: within-query Spearman; tie-break on calibrated error
    best = max(grid_rows, key=lambda r: (round(r["within_query_spearman_mean"], 6),
                                         -r["mean_absolute_error"]))
    for row in grid_rows:
        row["selected"] = row["config_index"] == best["config_index"]
    frozen = {
        "learning_rate": best["learning_rate"], "epochs": best["epochs"],
        "max_length": best["max_length"],
        "batch_size": int(cfg["batch_size"]),
        "warmup_ratio": float(cfg["warmup_ratio"]),
        "seed": int(cfg["seed"]),
        "model": str(cfg["model"]),
        "selected_on": "within_query_spearman_mean, tie-break mean_absolute_error",
        "selected_tag": best["tag"],
    }
    return _emit(destination, started, cfg, {
        "schema": "stage2-crossencoder-tuning-v1",
        "status": "COMPLETE",
        "tuning_split": split_audit,
        "criterion": {
            "primary": "within-query Spearman between predicted score and "
                       "frozen utility, mean over tuning-validation queries",
            "tiebreaker": "mean absolute error on the continuous utility scale",
            "not_used": "Development300 Utility@8 was not consulted",
        },
        "grid_size": len(grid_rows),
        "selected": frozen,
    }, tables={"CE_TUNING_GRID.csv": grid_rows},
       extra_files={"CE_MATCHED_CONFIG.json": frozen})


def _emit(destination: Path, started: float, cfg: dict[str, Any],
          body: dict[str, Any], tables: dict[str, list[dict]] | None = None,
          extra_files: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write tables, extra json files and a manifest atomically."""
    import csv
    import platform

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        out = Path(tmp)
        for name, rows in (tables or {}).items():
            header = sorted({key for row in rows for key in row})
            with (out / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(header)
                writer.writerows([str(row.get(k, "")) for k in header]
                                 for row in rows)
        for name, payload in (extra_files or {}).items():
            (out / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2,
                           sort_keys=True) + "\n", encoding="utf-8")
        produced = sorted(p.name for p in out.iterdir())
        manifest = {
            **body,
            "version": str(cfg["version"]),
            "created_utc": now(),
            "training_pool": str(cfg["features_parquet"]),
            "boundaries": {"external_requests_made": 0, "frozen_test_read": False,
                           "engineered_features_used": False},
            "inputs": {str(cfg["features_parquet"]):
                       sha256_file(cfg["features_parquet"])},
            "outputs": {name: sha256_file(out / name) for name in produced},
            "software": {"python": platform.python_version()},
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        Path(out).rename(destination)
    return manifest


MODES = {"oof": run, "audit-pool": audit_pool, "audit-tokens": audit_tokens,
         "tune": tune, "zero-shot": zero_shot,
         "closed600-transfer": closed600_transfer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-key", default=CONFIG_KEY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-folds", type=int)
    parser.add_argument("--mode", choices=sorted(MODES), default="oof")
    args = parser.parse_args()
    if args.mode == "oof":
        manifest = run(args.config_key, args.output_dir, args.limit_folds)
    else:
        if args.limit_folds is not None:
            parser.error("--limit-folds applies only to --mode oof")
        manifest = MODES[args.mode](args.config_key, args.output_dir)
    print(json.dumps({k: manifest[k] for k in
                      ("status", "counts", "gates", "failed_gates", "selected",
                       "keep_configured_max_length",
                       "fraction_truncated_at_configured_max_length",
                       "elapsed_seconds") if k in manifest},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
