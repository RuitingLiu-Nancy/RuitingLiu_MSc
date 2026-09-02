"""Strict development-only fusion ablation over frozen dev100-v2 artifacts.

The module reuses canonical RRF/CC, utility completeness, graded nDCG,
query-level bootstrap, frozen Linear-basic OOF predictions and the independent
BGE-M3 encoder.  It never retrieves new candidates, calls an external model,
or reads a frozen-test artifact.

Provenance: adopted Cormack RRF and canonical project CC; adapted nested
query-level model selection and MMR/RichRAG-style set selection; own reporting
and audit glue for the controlled-tail estimand.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from urllib.parse import urlparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import cohen_kappa_score

import configuration as project_config
from evaluation.community_reply_auxiliary import (
    encode_texts,
    load_config as load_reply_config,
    text_sha,
)
from evaluation.ir_metrics import graded_ndcg_at
from evaluation.judgment_completeness import complete_utility_v2_rows
from evaluation.statistics import bootstrap_ci
from fusion.ranking import cc_scores, rrf_scores


BACKENDS = ("SBERT", "Cohere")
QUOTAS = (0, 1, 2, 4)
THRESHOLDS = {"acceptable": 4.0, "useful": 4.5, "high_quality": 5.0}

PRIMARY_CONTRASTS = (
    ("CTS-Graph-Linear-q2_vs_Dense-Top8", "CTS-Graph-Linear-q2", "Dense-Top8"),
    ("CTS-Graph-Linear-q2_vs_RRF-DenseGraph", "CTS-Graph-Linear-q2", "RRF-DenseGraph"),
    ("CTS-Graph-Linear-q2_vs_CC-DenseGraph", "CTS-Graph-Linear-q2", "CC-DenseGraph"),
    ("CTS-Graph-Linear-q2_vs_Global-Linear", "CTS-Graph-Linear-q2", "Global-Linear"),
    ("CTS-Graph-Linear-q2_vs_CTS-Graph-Raw-q2", "CTS-Graph-Linear-q2", "CTS-Graph-Raw-q2"),
    ("CTS-Graph-Linear-q2_vs_CTS-Deep-Linear-q2", "CTS-Graph-Linear-q2", "CTS-Deep-Linear-q2"),
    ("CTS-Graph-Linear-q1_vs_q2", "CTS-Graph-Linear-q1", "CTS-Graph-Linear-q2"),
    ("CTS-Graph-Linear-q4_vs_q2", "CTS-Graph-Linear-q4", "CTS-Graph-Linear-q2"),
)

DIRECT_IDENTIFIER_PATTERNS = (
    ("email", re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?!\w)"), "[EMAIL]"),
    ("reddit_or_social_username", re.compile(r"(?i)(?<!\w)(?:/?u/|@)[A-Za-z0-9_-]{2,30}\b"), "[USERNAME]"),
    ("ipv4_address", re.compile(r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"), "[IP_ADDRESS]"),
    ("phone_number", re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){7,15}(?!\w)"), "[PHONE]"),
    ("precise_street_address", re.compile(
        r"(?i)\b\d{1,6}\s+(?:[A-Za-z0-9.'-]+\s+){0,5}"
        r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|court|ct|way|place|pl|terrace|close)\b[.,]?"),
     "[STREET_ADDRESS]"),
    ("uk_postcode", re.compile(r"(?i)\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"), "[POSTCODE]"),
    ("us_zip_code", re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[POSTCODE]"),
    ("explicit_person_name", re.compile(
        r"\b(?i:my name is|call me|i am named)\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)?)"),
     "[PERSON_NAME]"),
    ("named_clinician", re.compile(r"\b(?i:dr\.?|doctor)\s+[A-Z][A-Za-z'-]+\b"), "[CLINICIAN_NAME]"),
)

URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>()]+")

OFFICIAL_HEALTH_DOMAINS = {
    "nhs.uk": ("NHS", "HEALTH_GUIDANCE"),
    "nice.org.uk": ("NICE", "CLINICAL_GUIDANCE"),
    "cdc.gov": ("CDC", "PUBLIC_HEALTH_GUIDANCE"),
    "nih.gov": ("NIH", "HEALTH_INFORMATION"),
    "nimh.nih.gov": ("NIMH", "MENTAL_HEALTH_INFORMATION"),
    "who.int": ("WHO", "PUBLIC_HEALTH_GUIDANCE"),
    "health.gov": ("US_HEALTH_GOVERNMENT", "PUBLIC_GUIDANCE"),
    "samhsa.gov": ("SAMHSA", "MENTAL_HEALTH_RESOURCE"),
    "fda.gov": ("FDA", "MEDICATION_INFORMATION"),
    "medlineplus.gov": ("MEDLINEPLUS", "HEALTH_INFORMATION"),
    "mayoclinic.org": ("MAYO_CLINIC", "HEALTH_INFORMATION"),
    "clevelandclinic.org": ("CLEVELAND_CLINIC", "HEALTH_INFORMATION"),
    "chadd.org": ("CHADD", "ADHD_SUPPORT_RESOURCE"),
    "adhdfoundation.org.uk": ("ADHD_FOUNDATION", "ADHD_SUPPORT_RESOURCE"),
    "mind.org.uk": ("MIND", "MENTAL_HEALTH_SUPPORT"),
    "apa.org": ("APA", "MENTAL_HEALTH_INFORMATION"),
    "psychiatry.org": ("APA", "PSYCHIATRY_INFORMATION"),
    "understood.org": ("UNDERSTOOD", "NEURODIVERSITY_SUPPORT_RESOURCE"),
}

PUBLIC_INFORMATION_DOMAINS = {
    "gov.uk": "UK_GOVERNMENT",
    "usa.gov": "US_GOVERNMENT",
    "wikipedia.org": "WIKIPEDIA",
    "citizensadvice.org.uk": "CITIZENS_ADVICE",
    "acas.org.uk": "ACAS",
}

RESEARCH_DOMAINS = (
    "doi.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "arxiv.org",
    "researchgate.net", "scholar.google.com", "sciencedirect.com", "springer.com",
    "nature.com", "wiley.com", "frontiersin.org", "tandfonline.com", "jstor.org",
)

TOOL_APP_DOMAINS = (
    "apps.apple.com", "play.google.com", "todoist.com", "notion.so", "notion.com",
    "trello.com", "focusmate.com", "goblin.tools", "ticktick.com", "finchcare.com",
)

SENSITIVITY_PATTERNS = {
    "self-harm/suicide": re.compile(r"(?i)\b(?:suicid\w*|self[- ]?harm\w*|kill myself|want to die|end my life)\b"),
    "abuse/violence": re.compile(r"(?i)\b(?:abus\w*|violen\w*|assault\w*|beaten|hit me|domestic violence|rape\w*)\b"),
    "sexual content": re.compile(r"(?i)\b(?:sex(?:ual|ually)?|porn\w*|masturbat\w*|rape\w*)\b"),
    "minors": re.compile(r"(?i)\b(?:minor|child(?:ren)?|kid(?:s)?|teen(?:ager)?s?|my son|my daughter)\b"),
    "detailed medical history": re.compile(
        r"(?i)\b(?:diagnos(?:is|ed)|medicat\w*|prescri\w*|\d+(?:\.\d+)?\s?mg|hospitali[sz]\w*|psychiatr\w*|therap\w*)\b"),
    "illegal activity": re.compile(r"(?i)\b(?:illegal|arrest\w*|crime|criminal|theft|stole|steal\w*|drug dealer|cocaine|meth(?:amphetamine)?|police)\b"),
}

URL_CATEGORIES = (
    "OFFICIAL_HEALTH_RESOURCE", "PUBLIC_INFORMATION_RESOURCE", "ADHD_COMMUNITY_RESOURCE",
    "TOOL_OR_APP", "PRODUCT_LINK", "RESEARCH_OR_ARTICLE", "PERSONAL_PROFILE",
    "SOURCE_POST", "LOCATION_OR_MAP", "OTHER_URL",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _load_corpus(path: Path) -> dict[str, str]:
    raw = _read_json(path)
    rows = raw if isinstance(raw, list) else raw.get("docs", raw.get("corpus", []))
    result = {}
    for row in rows:
        cid = str(row.get("comment_id") or row.get("title") or row.get("id"))
        text = str(row.get("text") or row.get("content") or "")
        if cid and text:
            result[cid] = text
    return result


def _load_queries(path: Path) -> dict[str, str]:
    rows = _read_json(path)
    return {str(row.get("id") or row["query_id"]): str(row.get("question") or row.get("query_text"))
            for row in rows}


def _load_query_types(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {"single_need": "single", "multi_need": "multi"}
    result = {str(row["query_id"]): mapping[str(row["llm_single_multi_label"])] for row in rows}
    if Counter(result.values()) != Counter({"single": 50, "multi": 50}):
        raise ValueError(f"expected 50/50 query types, got {Counter(result.values())}")
    return result


def _load_run(path: Path) -> dict[str, list[dict]]:
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(path):
        by_query[str(row["query_id"])].append({
            "comment_id": str(row["comment_id"]), "rank": int(row["rank"]),
            "score": float(row.get("score", row.get("raw_score", 0.0))),
        })
    for qid in by_query:
        by_query[qid].sort(key=lambda row: (row["rank"], row["comment_id"]))
    return dict(by_query)


def _load_graph(path: Path) -> dict[str, list[dict]]:
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(path):
        by_query[str(row["query_id"])].append({
            "comment_id": str(row["comment_id"]),
            "score": float(row["graph_rrf_score"]),
            "source_memberships": row.get("source_graph_memberships", []),
        })
    for qid, rows in by_query.items():
        rows.sort(key=lambda row: (-row["score"], row["comment_id"]))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
    return dict(by_query)


def _load_oof(path: Path) -> dict[tuple[str, str], dict]:
    return {(str(row["query_id"]), str(row["comment_id"])): row for row in _read_jsonl(path)}


def _cohere_path(registry_path: Path) -> Path:
    frozen = _read_json(registry_path)
    return Path(next(row["path"] for row in frozen["runs"] if row["method"] == "cohere_dense"))


def _needs(path: Path) -> dict[str, list[str]]:
    result = {}
    for row in _read_jsonl(path):
        items = []
        primary = row.get("primary_need") or {}
        if primary.get("text"):
            items.append(str(primary["text"]))
        items.extend(str(item["text"]) for item in row.get("additional_needs", []) if item.get("text"))
        result[str(row["query_id"])] = items or [""]
    return result


def _folds(path: Path) -> dict:
    split = _read_json(path)
    if split["audit"]["verdict"] != "PASS" or split["audit"]["pair_level_random_split"]:
        raise ValueError("frozen grouped-query split failed its own audit")
    return split


def _inner_validation_sets(train_qids: list[str], seed: int, folds: int = 4) -> list[list[str]]:
    ordered = list(sorted(train_qids))
    np.random.default_rng(seed).shuffle(ordered)
    return [ordered[index::folds] for index in range(folds)]


def _metric_complete(ids: list[str], qid: str, qrels: dict[tuple[str, str], dict]) -> bool:
    return len(ids) == 8 and len(set(ids)) == 8 and all((qid, cid) in qrels for cid in ids)


def _utility(ids: list[str], qid: str, qrels: dict[tuple[str, str], dict]) -> float | None:
    if not _metric_complete(ids, qid, qrels):
        return None
    return statistics.fmean(float(qrels[(qid, cid)]["utility"]) for cid in ids)


def _select_tail(dense: list[dict], extension: list[dict], quota: int,
                 score_key: str) -> list[str]:
    if quota == 0:
        return [row["comment_id"] for row in dense[:8]]
    protected = [row["comment_id"] for row in dense[:8 - quota]]
    ordered = sorted(extension, key=lambda row: (-float(row[score_key]), row["comment_id"]))
    additions = []
    for row in ordered:
        cid = row["comment_id"]
        if cid not in protected and cid not in additions:
            additions.append(cid)
        if len(additions) == quota:
            break
    return protected + additions


def _system_row(backend: str, system: str, qid: str, ids: list[str],
                qrels: dict[tuple[str, str], dict], qrels_by_query: dict[str, dict[str, float]],
                query_type: str, metadata: dict | None = None) -> dict:
    complete = _metric_complete(ids, qid, qrels)
    utilities = [float(qrels[(qid, cid)]["utility"]) for cid in ids if (qid, cid) in qrels]
    row = {
        "backend": backend, "system": system, "query_id": qid, "query_type": query_type,
        "comment_ids": ids, "output_count": len(ids), "unique_output_count": len(set(ids)),
        "unjudged_count": sum((qid, cid) not in qrels for cid in ids),
        "judgment_coverage_at8": sum((qid, cid) in qrels for cid in ids) / 8.0,
        "complete_top8": complete, "metadata": metadata or {},
    }
    if complete:
        row.update({
            "mean_utility_at8": statistics.fmean(utilities),
            "common_idcg_ndcg_at8": graded_ndcg_at(ids, qrels_by_query[qid], 8),
            "acceptable_count_at8": sum(value >= 4.0 for value in utilities),
            "useful_count_at8": sum(value >= 4.5 for value in utilities),
            "high_quality_count_at8": sum(value >= 5.0 for value in utilities),
            "success_at1": float(any(value >= 4.5 for value in utilities[:1])),
            "success_at3": float(any(value >= 4.5 for value in utilities[:3])),
            "success_at8": float(any(value >= 4.5 for value in utilities)),
            "zero_useful_query": float(not any(value >= 4.5 for value in utilities)),
        })
    else:
        for key in ("mean_utility_at8", "common_idcg_ndcg_at8", "acceptable_count_at8",
                    "useful_count_at8", "high_quality_count_at8", "success_at1", "success_at3",
                    "success_at8", "zero_useful_query"):
            row[key] = None
    return row


def load_inputs(root: Path, cfg: dict) -> dict:
    queries = _load_queries(_resolve(root, cfg["queries"]))
    qtypes = _load_query_types(_resolve(root, cfg["query_admin"]))
    corpus = _load_corpus(_resolve(root, cfg["corpus"]))
    _, qrels = complete_utility_v2_rows(_read_jsonl(_resolve(root, cfg["utility_registry"])))
    qrels_by_query: dict[str, dict[str, float]] = defaultdict(dict)
    for (qid, cid), row in qrels.items():
        qrels_by_query[qid][cid] = float(row["utility"])
    sbert = _load_run(_resolve(root, cfg["sbert_dense_top100"]))
    cohere = _load_run(_cohere_path(_resolve(root, cfg["frozen_run_registry"])))
    graph = _load_graph(_resolve(root, cfg["graph_union_pool"]))
    oof = {
        "SBERT": _load_oof(_resolve(root, cfg["sbert_linear_oof"])),
        "Cohere": _load_oof(_resolve(root, cfg["cohere_linear_oof"])),
    }
    if set(queries) != set(qtypes) or any(set(run) != set(queries) for run in (sbert, cohere, graph)):
        raise ValueError("query identity mismatch across frozen inputs")
    return {"queries": queries, "qtypes": qtypes, "corpus": corpus, "qrels": qrels,
            "qrels_by_query": dict(qrels_by_query), "dense": {"SBERT": sbert, "Cohere": cohere},
            "graph": graph, "oof": oof, "needs": _needs(_resolve(root, cfg["summaries"])),
            "split": _folds(_resolve(root, cfg["split_manifest"]))}


def build_candidate_registry(data: dict) -> list[dict]:
    rows = []
    for backend in BACKENDS:
        for qid in sorted(data["queries"]):
            dense_map = {row["comment_id"]: row for row in data["dense"][backend][qid]}
            graph_map = {row["comment_id"]: row for row in data["graph"][qid]}
            for cid in sorted(set(dense_map) | set(graph_map)):
                dense = dense_map.get(cid); graph = graph_map.get(cid)
                prediction = data["oof"][backend].get((qid, cid))
                rows.append({
                    "query_id": qid, "backend": backend, "candidate_id": cid,
                    "dense_rank": dense["rank"] if dense else None,
                    "dense_score": dense["score"] if dense else None,
                    "graph_rank": graph["rank"] if graph else None,
                    "graph_score": graph["score"] if graph else None,
                    "in_dense_top8": bool(dense and dense["rank"] <= 8),
                    "in_dense_top100": bool(dense), "in_graph_pool": bool(graph),
                    "utility_judged": (qid, cid) in data["qrels"],
                    "utility": (float(data["qrels"][(qid, cid)]["utility"])
                                if (qid, cid) in data["qrels"] else None),
                    "features_available": prediction is not None,
                    "linear_oof_score": (float(prediction["linear_basic_oof_score"])
                                         if prediction is not None else None),
                    "source_memberships": graph.get("source_memberships", []) if graph else [],
                })
    return rows


def common_idcg_rows(data: dict) -> list[dict]:
    rows = []
    for qid in sorted(data["queries"]):
        utilities = sorted(data["qrels_by_query"][qid].values(), reverse=True)
        top = utilities[:8]
        idcg = sum((2.0 ** value - 1.0) / math.log2(index + 2) for index, value in enumerate(top))
        rows.append({"query_id": qid, "judged_candidate_count": len(utilities),
                     "top8_utility_for_idcg": top, "idcg_at8": idcg,
                     "gain": "2^utility-1", "discount": "1/log2(rank+1)",
                     "missing_judgment_policy": "system metric undefined unless complete Top-8"})
    return rows


def _graph_extension(data: dict, backend: str, qid: str) -> list[dict]:
    dense8 = {row["comment_id"] for row in data["dense"][backend][qid][:8]}
    result = []
    for row in data["graph"][qid]:
        if row["comment_id"] in dense8:
            continue
        pred = data["oof"][backend].get((qid, row["comment_id"]))
        result.append({**row, "linear": float(pred["linear_basic_oof_score"]) if pred else -math.inf})
    return result


def _deep_extension(data: dict, backend: str, qid: str) -> list[dict]:
    result = []
    for row in data["dense"][backend][qid][8:100]:
        pred = data["oof"][backend].get((qid, row["comment_id"]))
        result.append({**row, "linear": float(pred["linear_basic_oof_score"]) if pred else -math.inf})
    return result


def _global_linear(data: dict, backend: str, qid: str, large: bool) -> list[str]:
    dense = data["dense"][backend][qid][:100 if large else 8]
    candidates = {row["comment_id"] for row in dense} | {row["comment_id"] for row in data["graph"][qid]}
    scored = []
    for cid in candidates:
        prediction = data["oof"][backend].get((qid, cid))
        scored.append((float(prediction["linear_basic_oof_score"]) if prediction else -math.inf, cid))
    return [cid for _, cid in sorted(scored, key=lambda item: (-item[0], item[1]))[:8]]


def _oracle(data: dict, backend: str, qid: str, tail_q: int | None) -> list[str]:
    dense = data["dense"][backend][qid]
    if tail_q is not None:
        protected = [row["comment_id"] for row in dense[:8 - tail_q]]
        pool = [row["comment_id"] for row in data["graph"][qid] if row["comment_id"] not in protected]
        judged = [(float(data["qrels"][(qid, cid)]["utility"]), cid) for cid in pool
                  if (qid, cid) in data["qrels"]]
        return protected + [cid for _, cid in sorted(judged, key=lambda item: (-item[0], item[1]))[:tail_q]]
    candidates = {row["comment_id"] for row in dense[:100]} | {row["comment_id"] for row in data["graph"][qid]}
    judged = [(float(data["qrels"][(qid, cid)]["utility"]), cid) for cid in candidates
              if (qid, cid) in data["qrels"]]
    return [cid for _, cid in sorted(judged, key=lambda item: (-item[0], item[1]))[:8]]


def _rank_fusion(data: dict, backend: str, qid: str, graph_weight: float,
                 include_deep: bool, k0: int) -> list[str]:
    dense = data["dense"][backend][qid]
    graph = data["graph"][qid]
    runs = {"dense": dense, "graph": graph}
    weights = {"dense": 1.0, "graph": graph_weight}
    if include_deep:
        # This is intentionally registered as a degenerate duplicate view of
        # the same Dense ordering, not as an independent retriever.
        runs["deep_dense"] = dense[8:100]
        weights["deep_dense"] = 1.0
    scores = rrf_scores(runs, weights=weights, k0=k0)
    return [cid for cid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:8]]


def _rank_dense_deep(data: dict, backend: str, qid: str, k0: int) -> list[str]:
    """Required diagnostic: the second route is a slice of the first route."""
    dense = data["dense"][backend][qid]
    scores = rrf_scores({"dense": dense, "deep_dense": dense[8:100]},
                        weights={"dense": 1.0, "deep_dense": 1.0}, k0=k0)
    return [cid for cid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:8]]


def _score_fusion(data: dict, backend: str, qid: str, dense_weight: float,
                  normalization: str) -> list[str]:
    scores = cc_scores({"dense": data["dense"][backend][qid], "graph": data["graph"][qid]},
                       {"dense": dense_weight, "graph": 1.0 - dense_weight}, normalization)
    return [cid for cid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:8]]


def _tune_configs(data: dict, backend: str, fold: dict, configs: list[Any],
                  selector, seed: int) -> tuple[Any, dict]:
    train_qids = list(fold["train_query_ids"])
    inner_sets = _inner_validation_sets(train_qids, seed)
    records = []
    for config in configs:
        fold_values = []
        fold_n = []
        for validation in inner_sets:
            values = [_utility(selector(qid, config), qid, data["qrels"]) for qid in validation]
            complete = [value for value in values if value is not None]
            fold_values.append(statistics.fmean(complete) if complete else -math.inf)
            fold_n.append(len(complete))
        records.append({"config": config, "inner_fold_mean_utility": fold_values,
                        "inner_complete_query_n": fold_n,
                        "selection_score": statistics.fmean(fold_values)})
    best = sorted(records, key=lambda row: (-row["selection_score"], json.dumps(row["config"], sort_keys=True)))[0]
    return best["config"], {"backend": backend, "repeat": fold["repeat"], "fold": fold["fold"],
                            "train_query_n": len(train_qids), "inner_folds": len(inner_sets),
                            "selected": best["config"], "candidates": records}


def _crossfit_hyperparameters(data: dict, cfg: dict) -> tuple[dict, list[dict]]:
    choices: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    audit = []
    rrf_configs = [{"graph_weight": value} for value in cfg["rrf_graph_weights"]]
    cc_configs = [{"dense_weight": value, "normalization": normalization}
                  for normalization in cfg["cc_normalizations"] for value in cfg["cc_lambdas"]]
    for backend in BACKENDS:
        for fold in data["split"]["rows"]:
            repeat, number = int(fold["repeat"]), int(fold["fold"])
            rrf, record = _tune_configs(
                data, backend, fold, rrf_configs,
                lambda qid, item: _rank_fusion(data, backend, qid, item["graph_weight"], False,
                                               int(cfg["rrf_k0"])),
                int(cfg["bootstrap_seed"]) + repeat * 1009 + number * 101)
            record["method"] = "RRF-DenseGraph"; audit.append(record)
            cc, record = _tune_configs(
                data, backend, fold, cc_configs,
                lambda qid, item: _score_fusion(data, backend, qid, item["dense_weight"],
                                                item["normalization"]),
                int(cfg["bootstrap_seed"]) + 5000 + repeat * 1009 + number * 101)
            record["method"] = "CC-DenseGraph"; audit.append(record)
            for qid in fold["validation_query_ids"]:
                choices[(backend, qid, "rrf")].append(rrf)
                choices[(backend, qid, "cc")].append(cc)
    selected = {}
    for key, values in choices.items():
        if len(values) != int(data["split"]["repeats"]):
            raise ValueError(f"outer prediction count mismatch: {key}={len(values)}")
        counts = Counter(json.dumps(value, sort_keys=True) for value in values)
        selected[key] = json.loads(sorted(counts, key=lambda value: (-counts[value], value))[0])
    return selected, audit


def fixed_systems(data: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    tuned, tuning_audit = _crossfit_hyperparameters(data, cfg)
    rows = []
    for backend in BACKENDS:
        for qid in sorted(data["queries"]):
            dense = data["dense"][backend][qid]
            graph = _graph_extension(data, backend, qid)
            deep = _deep_extension(data, backend, qid)
            systems: dict[str, tuple[list[str], dict]] = {
                "Dense-Top8": ([row["comment_id"] for row in dense[:8]], {}),
                "RRF-DenseGraph-Equal": (_rank_fusion(data, backend, qid, 1.0, False, int(cfg["rrf_k0"])),
                                         {"graph_weight": 1.0, "cross_fitted": False}),
                "RRF-DenseGraph": (_rank_fusion(data, backend, qid, tuned[(backend, qid, "rrf")]["graph_weight"],
                                                 False, int(cfg["rrf_k0"])),
                                   {**tuned[(backend, qid, "rrf")], "cross_fitted": True}),
                "RRF-DenseDeep": (_rank_dense_deep(data, backend, qid, int(cfg["rrf_k0"])),
                                  {"degenerate_deep_view": True, "independent_retriever": False}),
                "RRF-DenseGraphDeep": (_rank_fusion(data, backend, qid, 1.0, True, int(cfg["rrf_k0"])),
                                       {"degenerate_deep_view": True}),
                "CC-DenseGraph": (_score_fusion(data, backend, qid,
                                                 tuned[(backend, qid, "cc")]["dense_weight"],
                                                 tuned[(backend, qid, "cc")]["normalization"]),
                                  {**tuned[(backend, qid, "cc")], "cross_fitted": True}),
                "Global-Linear": (_global_linear(data, backend, qid, False), {"candidate_union": "DenseTop8+Graph"}),
                "Global-Linear-Top100": (_global_linear(data, backend, qid, True), {"candidate_union": "DenseTop100+Graph"}),
                "Oracle-Global-JudgedVisible": (_oracle(data, backend, qid, None), {"nondeployable": True}),
                "Oracle-CTS-Graph-q2-JudgedVisible": (_oracle(data, backend, qid, 2), {"nondeployable": True}),
            }
            for normalization in cfg["cc_normalizations"]:
                systems[f"CC-{normalization}-lambda0.5"] = (
                    _score_fusion(data, backend, qid, 0.5, normalization),
                    {"normalization": normalization, "dense_weight": 0.5, "sensitivity": True})
            for quota in QUOTAS[1:]:
                systems[f"CTS-Graph-Raw-q{quota}"] = (_select_tail(dense, graph, quota, "score"), {"quota": quota})
                systems[f"CTS-Graph-Linear-q{quota}"] = (_select_tail(dense, graph, quota, "linear"), {"quota": quota})
                systems[f"CTS-Deep-Raw-q{quota}"] = (_select_tail(dense, deep, quota, "score"), {"quota": quota})
                systems[f"CTS-Deep-Linear-q{quota}"] = (_select_tail(dense, deep, quota, "linear"), {"quota": quota})
            for system, (ids, metadata) in systems.items():
                rows.append(_system_row(backend, system, qid, ids, data["qrels"], data["qrels_by_query"],
                                        data["qtypes"][qid], metadata))
    return rows, tuning_audit


def aggregate_metrics(rows: list[dict]) -> dict:
    output = {}
    systems = sorted({(row["backend"], row["system"]) for row in rows})
    metric_names = ("mean_utility_at8", "common_idcg_ndcg_at8", "acceptable_count_at8",
                    "useful_count_at8", "high_quality_count_at8", "success_at1", "success_at3",
                    "success_at8", "zero_useful_query")
    for backend, system in systems:
        selected = [row for row in rows if row["backend"] == backend and row["system"] == system]
        record = {"backend": backend, "system": system, "query_n": len(selected),
                  "complete_query_n": sum(row["complete_top8"] for row in selected),
                  "mean_judgment_coverage_at8": statistics.fmean(row["judgment_coverage_at8"] for row in selected),
                  "output_failure_query_n": sum(row["output_count"] != 8 or row["unique_output_count"] != 8 for row in selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            record[metric] = statistics.fmean(values) if values else None
        output[f"{backend}:{system}"] = record
    return output


def _significance(ci: list[float]) -> str:
    if ci[0] > 0:
        return "positive"
    if ci[1] < 0:
        return "negative"
    return "crosses_zero"


def paired_comparisons(rows: list[dict], cfg: dict) -> list[dict]:
    lookup = {(row["backend"], row["system"], row["query_id"]): row for row in rows}
    contrasts = (
        ("CTS-Graph-Linear-q2", "Dense-Top8"),
        ("CTS-Graph-Linear-q2", "RRF-DenseGraph"),
        ("CTS-Graph-Linear-q2", "CC-DenseGraph"),
        ("CTS-Graph-Linear-q2", "Global-Linear"),
        ("CTS-Graph-Linear-q2", "CTS-Graph-Raw-q2"),
        ("CTS-Graph-Linear-q2", "CTS-Deep-Linear-q2"),
        ("CTS-Graph-Linear-q1", "CTS-Graph-Linear-q2"),
        ("CTS-Graph-Linear-q4", "CTS-Graph-Linear-q2"),
        ("Global-Linear", "Dense-Top8"),
        ("Oracle-Global-JudgedVisible", "Global-Linear"),
        ("Oracle-CTS-Graph-q2-JudgedVisible", "CTS-Graph-Linear-q2"),
    )
    metrics = ("mean_utility_at8", "common_idcg_ndcg_at8", "useful_count_at8", "high_quality_count_at8")
    result = []
    qids = sorted({row["query_id"] for row in rows})
    for backend_index, backend in enumerate(BACKENDS):
        for contrast_index, (left, right) in enumerate(contrasts):
            if not all((backend, system, qids[0]) in lookup for system in (left, right)):
                continue
            for metric_index, metric in enumerate(metrics):
                deltas = []
                for qid in qids:
                    a = lookup[(backend, left, qid)][metric]; b = lookup[(backend, right, qid)][metric]
                    if a is not None and b is not None:
                        deltas.append(float(a) - float(b))
                if not deltas:
                    result.append({"backend": backend, "left": left, "right": right, "metric": metric,
                                   "exact_common_n": 0, "mean_delta": None, "bootstrap_95ci": None,
                                   "significance_status": "not_available"})
                    continue
                ci = list(bootstrap_ci(deltas, n_boot=int(cfg["bootstrap_samples"]),
                                       seed=int(cfg["bootstrap_seed"]) + backend_index * 10007
                                       + contrast_index * 101 + metric_index))
                sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
                result.append({"backend": backend, "left": left, "right": right, "metric": metric,
                               "exact_common_n": len(deltas), "mean_delta": statistics.fmean(deltas),
                               "median_delta": statistics.median(deltas), "bootstrap_95ci": ci,
                               "significance_status": _significance(ci),
                               "effect_size": statistics.fmean(deltas) / sd if sd else None,
                               "improved": sum(value > 1e-12 for value in deltas),
                               "tied": sum(abs(value) <= 1e-12 for value in deltas),
                               "degraded": sum(value < -1e-12 for value in deltas)})
    return result


def metric_audit(root: Path, cfg: dict, data: dict, fixed_rows: list[dict]) -> tuple[dict, str]:
    old_paths = {
        "SBERT": root / "out/sbert_graph_minimal_dev100_v1/sbert_graph_minimal_metrics.json",
        "CohereGraph": root / "out/graph_supplementation_dev100_v1/graph_supplementation_metrics.json",
        "Minimal": root / "out/sbert_graph_comparison_dev100_v1/sbert_minimal_system_metrics.json",
    }
    old = {name: _read_json(path) for name, path in old_paths.items()}
    metrics = aggregate_metrics(fixed_rows)
    manifest = {
        "schema": "fusion-authoritative-metric-manifest-v1", "dataset": "dev100-v2",
        "queries": 100, "judgment_registry": str(_resolve(root, cfg["utility_registry"])),
        "judgment_registry_sha256": _sha(_resolve(root, cfg["utility_registry"])),
        "judged_pairs": len(data["qrels"]), "gain": "2^utility-1",
        "discount": "1/log2(rank+1)", "common_idcg": "all complete utility-v2 judgments per query",
        "missing_policy": "metric undefined unless system output is unique complete judged Top-8",
        "thresholds": THRESHOLDS,
        "authoritative_common_idcg_values": {
            key: metrics[key]["common_idcg_ndcg_at8"] for key in
            ("Cohere:Dense-Top8", "SBERT:Dense-Top8", "Cohere:CTS-Graph-Linear-q2", "SBERT:CTS-Graph-Linear-q2")
        },
        "historical_display_values": {
            "Cohere_Dense": old["CohereGraph"]["B0"]["ndcg_at8"],
            "Cohere_Deep2": old["Minimal"]["C2"]["ndcg_at8"],
            "Cohere_Graph2": old["CohereGraph"]["B4"]["ndcg_at8"],
            "Official_HippoRAG2": old["Minimal"]["H0"]["ndcg_at8"],
        },
        "historical_values_directly_comparable": False,
        "external_model_calls": 0, "frozen_test_read": False,
    }
    text = f"""# Fusion metric version audit

The previous Natural unseen table mixed experiment-specific ideal denominators. Historical display values were Cohere Dense `{manifest['historical_display_values']['Cohere_Dense']:.4f}`, Cohere Deep-2 `{manifest['historical_display_values']['Cohere_Deep2']:.4f}`, Cohere Graph-2 `{manifest['historical_display_values']['Cohere_Graph2']:.4f}`, and Official HippoRAG2 `{manifest['historical_display_values']['Official_HippoRAG2']:.4f}`. They are retained only with provenance and are not subtracted across versions.

This experiment recomputes every list against one 4,442-pair complete utility-v2 registry, one query-wise IDCG, exponential gain `2^utility-1`, and log2 discount. Missing judgments are never assigned zero: a system-query metric is undefined unless all eight unique outputs are judged. Complete-case sensitivity is therefore the primary estimand.

Thresholds are now unambiguous: acceptable `>=4.0`, useful `>=4.5`, and high-quality `>=5.0`. Closed structural Recall@5 is reported in a separate Recall column/row and never under nDCG.
"""
    return manifest, text


def residual_report(rows: list[dict], qrels: dict[tuple[str, str], dict]) -> tuple[dict, str]:
    """Report exact unjudged Top-8 pairs without treating them as negatives."""
    deployable = [row for row in rows if not row["system"].startswith("Oracle-")]
    core_names = {
        "Dense-Top8", "RRF-DenseGraph", "CC-DenseGraph", "Global-Linear",
        "Global-Linear-Top100", "CTS-Graph-Raw-q2", "CTS-Graph-Linear-q2",
        "CTS-Deep-Raw-q2", "CTS-Deep-Linear-q2",
    }
    substantive_names = {
        "Dense-Top8", "RRF-DenseGraph-Equal", "RRF-DenseGraph", "CC-DenseGraph",
        "Global-Linear", "Global-Linear-Top100", "CTS-Graph-Raw-q2",
        "CTS-Graph-Linear-q1", "CTS-Graph-Linear-q2", "CTS-Graph-Linear-q4",
        "CTS-Deep-Linear-q2",
    }

    def summarize(selected: list[dict]) -> dict:
        pairs: set[tuple[str, str]] = set()
        sources: dict[tuple[str, str], set[str]] = defaultdict(set)
        by_backend: dict[str, set[tuple[str, str]]] = defaultdict(set)
        by_system: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for row in selected:
            for cid in row["comment_ids"]:
                pair = (row["query_id"], cid)
                if pair in qrels:
                    continue
                pairs.add(pair)
                system_key = f"{row['backend']}:{row['system']}"
                sources[pair].add(system_key)
                by_backend[row["backend"]].add(pair)
                by_system[system_key].add(pair)
        by_query = Counter(qid for qid, _ in pairs)
        all_qids = sorted({row["query_id"] for row in selected})
        distribution = Counter(by_query.values())
        distribution[0] += sum(qid not in by_query for qid in all_qids)
        return {
            "unique_residual_pairs": len(pairs),
            "affected_query_n": len(by_query),
            "per_query_residual_distribution": {str(key): distribution[key] for key in sorted(distribution)},
            "per_query_residual_counts": {qid: by_query.get(qid, 0) for qid in all_qids},
            "by_backend_unique_pairs": {key: len(value) for key, value in sorted(by_backend.items())},
            "by_system_unique_pairs": {key: len(value) for key, value in sorted(by_system.items())},
            "residual_pairs": [
                {"query_id": qid, "comment_id": cid, "introduced_by": sorted(sources[(qid, cid)])}
                for qid, cid in sorted(pairs)
            ],
        }

    per_system = {}
    for backend, system in sorted({(row["backend"], row["system"]) for row in deployable}):
        summary = summarize([row for row in deployable
                             if row["backend"] == backend and row["system"] == system])
        per_system[f"{backend}:{system}"] = {
            key: summary[key] for key in
            ("unique_residual_pairs", "affected_query_n", "per_query_residual_distribution")
        }
    core = summarize([row for row in deployable if row["system"] in core_names])
    substantive = summarize([row for row in deployable if row["system"] in substantive_names])
    nondegenerate = summarize([row for row in deployable
                               if row["system"] not in {"RRF-DenseDeep", "RRF-DenseGraphDeep"}])
    full = summarize(deployable)
    report = {
        "schema": "fusion-stage1-residual-report-v1", "created_at": _now(),
        "judgment_completeness": "canonical complete utility-v2 helper",
        "missing_judgment_policy": "never impute zero",
        "core_system_names": sorted(core_names),
        "substantive_system_names": sorted(substantive_names),
        "core_deployable_union": core,
        "substantive_stage1_union": substantive,
        "complete_nondegenerate_stage1_union": nondegenerate,
        "all_stage1_deployable_union": full,
        "per_system": per_system, "external_model_calls": 0,
    }
    lines = [
        "# Fusion Stage-1 residual report", "",
        "Missing judgments are exact `(query_id, comment_id)` pairs and are never assigned utility 0.", "",
        f"- Core deployable comparison: **{core['unique_residual_pairs']}** pairs across **{core['affected_query_n']}** queries.",
        f"- Recommended substantive Stage-1 matrix: **{substantive['unique_residual_pairs']}** pairs across **{substantive['affected_query_n']}** queries.",
        f"- Complete non-degenerate Stage-1 matrix: **{nondegenerate['unique_residual_pairs']}** pairs across **{nondegenerate['affected_query_n']}** queries.",
        f"- All Stage-1 deployable arms: **{full['unique_residual_pairs']}** pairs across **{full['affected_query_n']}** queries.",
        "- No external model call was made.", "", "## Per-system residuals", "",
        "| System | Residual pairs | Affected queries |", "|---|---:|---:|",
    ]
    for key, value in per_system.items():
        lines.append(f"| {key} | {value['unique_residual_pairs']} | {value['affected_query_n']} |")
    return report, "\n".join(lines) + "\n"


def prepare_judging_payload(root: Path, cfg: dict) -> dict:
    """Freeze a blinded residual+anchor batch; never performs an external call."""
    out = _resolve(root, cfg["output_dir"])
    residual_path = out / "fusion_stage1_residual_report.json"
    if not residual_path.exists():
        raise FileNotFoundError("run fusion-fixed before preparing judging payload")
    report = _read_json(residual_path)
    residual_rows = report["complete_nondegenerate_stage1_union"]["residual_pairs"]
    residual = {(str(row["query_id"]), str(row["comment_id"])) for row in residual_rows}
    queries = _load_queries(_resolve(root, cfg["queries"]))
    corpus = _load_corpus(_resolve(root, cfg["corpus"]))
    anchors_raw = _read_jsonl(_resolve(root, cfg["calibration_anchors"]))
    anchors = {(str(row["query_id"]), str(row["comment_id"])) for row in anchors_raw}
    if len(anchors_raw) != 50 or len(anchors) != 50 or anchors & residual:
        raise ValueError("50-anchor identity/overlap gate failed")
    ordered = sorted(residual) + sorted(anchors)
    blind, admin = [], []
    residual_sources = {(str(row["query_id"]), str(row["comment_id"])): row["introduced_by"]
                        for row in residual_rows}
    for index, (qid, cid) in enumerate(ordered, 1):
        if qid not in queries or cid not in corpus:
            raise ValueError(f"payload pair cannot map to frozen text: {(qid, cid)}")
        rendered = {"query_text": queries[qid], "comment_text": corpus[cid], "facets_json": {}}
        payload_hash = _hash_json(rendered)
        blind.append({"item_index": index,
                      "payload": {"query_text": queries[qid], "comment_text": corpus[cid]},
                      "payload_sha256": payload_hash})
        admin.append({
            "item_index": index, "query_id": qid, "comment_id": cid,
            "item_type": "calibration_anchor" if (qid, cid) in anchors else "residual",
            "experiment_source": ("fixed_utility_v2_anchor" if (qid, cid) in anchors
                                  else "fusion_complete_nondegenerate_stage1_residual"),
            "arm_memberships": residual_sources.get((qid, cid), []),
            "payload_sha256": payload_hash,
        })
    blind_path = out / "fusion_stage1_blinded_payload.jsonl"
    admin_path = out / "fusion_stage1_payload_ADMIN.jsonl"
    _write_jsonl(blind_path, blind); _write_jsonl(admin_path, admin)
    prompt = project_config.prompt_path("evidence_card_judge_v2")
    manifest = {
        "schema": "fusion-stage1-payload-manifest-v1", "created_at": _now(),
        "authorization_status": "PENDING_EXPLICIT_USER_APPROVAL",
        "residual_items": len(residual), "anchor_items": len(anchors), "total_items": len(ordered),
        "involved_residual_queries": len({qid for qid, _ in residual}),
        "contains_potentially_sensitive_full_reddit_query_and_comment": True,
        "judge_visible_fields": ["query_text", "comment_text"],
        "administrative_provenance_visible_to_judge": False,
        "historical_facets_json": {},
        "blinded_payload": {"path": str(blind_path), "sha256": _sha(blind_path)},
        "admin_payload": {"path": str(admin_path), "sha256": _sha(admin_path)},
        "prompt": {"path": str(prompt), "sha256": _sha(prompt)},
        "judge": cfg["judge"], "duplicate_pairs": len(ordered) - len(set(ordered)),
        "residual_anchor_overlap": len(residual & anchors), "bedrock_calls": 0,
        "frozen_test_read": False,
    }
    _write_json(out / "fusion_stage1_payload_manifest.json", manifest)
    return manifest


def _domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def semantic_url_placeholder(raw_url: str, context: str) -> tuple[str, str, str | None]:
    """Return a non-reversible URL category and optional public institution."""
    normalized = raw_url if "://" in raw_url else "https://" + raw_url
    parsed = urlparse(normalized.rstrip(".,;:!?\"'"))
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.lower()
    surrounding = context.lower()

    if (_domain_matches(host, "reddit.com") and "/comments/" in path) or _domain_matches(host, "redd.it"):
        return "[SOURCE_POST]", "SOURCE_POST", None
    if any(token in path for token in ("/user/", "/users/", "/profile/", "/u/")):
        return "[PERSONAL_PROFILE]", "PERSONAL_PROFILE", None
    if any(_domain_matches(host, domain) for domain in
           ("maps.google.com", "google.com", "maps.apple.com", "openstreetmap.org", "mapquest.com")) and (
            "map" in host or "/maps" in path or "location" in surrounding or "directions" in surrounding):
        return "[LOCATION_OR_MAP]", "LOCATION_OR_MAP", None
    if any(_domain_matches(host, domain) for domain in
           ("amazon.com", "amazon.co.uk", "ebay.com", "etsy.com", "walmart.com", "target.com",
            "aliexpress.com", "shopify.com")) or any(token in surrounding for token in
                                                      ("buy this", "product link", "purchase", "shopping link")):
        return "[PRODUCT_LINK]", "PRODUCT_LINK", None
    if any(_domain_matches(host, domain) for domain in RESEARCH_DOMAINS) or any(
            token in path or token in surrounding for token in
            ("/article/", "/paper/", "journal article", "research paper", "study", "doi")):
        return "[RESEARCH_OR_ARTICLE]", "RESEARCH_OR_ARTICLE", None
    for domain, (institution, default_type) in OFFICIAL_HEALTH_DOMAINS.items():
        if not _domain_matches(host, domain):
            continue
        resource_type = default_type
        if any(token in surrounding or token in path for token in ("screening", "questionnaire", "assessment", "tool")):
            resource_type = "SCREENING_OR_ASSESSMENT_TOOL"
        elif any(token in surrounding or token in path for token in ("guideline", "guidance", "guide")):
            resource_type = "GUIDANCE"
        elif any(token in surrounding or token in path for token in ("helpline", "crisis", "support line")):
            resource_type = "SUPPORT_SERVICE"
        return (f"[OFFICIAL_HEALTH_RESOURCE: {institution}]",
                "OFFICIAL_HEALTH_RESOURCE", institution)
    for domain, institution in PUBLIC_INFORMATION_DOMAINS.items():
        if _domain_matches(host, domain):
            return (f"[PUBLIC_INFORMATION_RESOURCE: {institution}]",
                    "PUBLIC_INFORMATION_RESOURCE", institution)
    if any(_domain_matches(host, domain) for domain in
           ("reddit.com", "discord.com", "discord.gg", "facebook.com", "groups.io", "quora.com")) or any(
            token in host for token in ("forum", "community", "adhd")):
        return "[ADHD_COMMUNITY_RESOURCE]", "ADHD_COMMUNITY_RESOURCE", None
    if any(_domain_matches(host, domain) for domain in TOOL_APP_DOMAINS) or any(
            token in surrounding for token in ("app link", "download the app", "use this tool", "online tool")):
        return "[TOOL_OR_APP]", "TOOL_OR_APP", None
    if any(token in path for token in ("/profile/", "/author/", "/member/")):
        return "[PERSONAL_PROFILE]", "PERSONAL_PROFILE", None
    if any(token in surrounding for token in ("my post", "original post", "this thread", "source post")):
        return "[SOURCE_POST]", "SOURCE_POST", None
    if any(token in surrounding for token in ("map link", "location link", "directions", "exact location")):
        return "[LOCATION_OR_MAP]", "LOCATION_OR_MAP", None
    return "[OTHER_URL]", "OTHER_URL", None


def redact_direct_identifiers(text: str) -> tuple[str, dict[str, int]]:
    """Deterministically redact likely direct identifiers, preserving URL semantics."""
    result = text
    counts: Counter[str] = Counter()
    # Email runs first so an address is not absorbed into a broader URL-like span.
    email_label, email_pattern, email_placeholder = DIRECT_IDENTIFIER_PATTERNS[0]
    result, email_count = email_pattern.subn(email_placeholder, result)
    counts[email_label] += email_count

    def replace_url(match: re.Match) -> str:
        local_context = result[max(0, match.start() - 160):min(len(result), match.end() + 160)]
        placeholder, category, institution = semantic_url_placeholder(match.group(0), local_context)
        counts["url"] += 1
        counts[f"url_category:{category}"] += 1
        if institution:
            counts[f"official_institution:{institution}"] += 1
        return placeholder

    result = URL_PATTERN.sub(replace_url, result)
    for label, pattern, placeholder in DIRECT_IDENTIFIER_PATTERNS[1:]:
        result, count = pattern.subn(placeholder, result)
        counts[label] += count
    return result, dict(counts)


def _url_inventory(text: str) -> list[dict[str, str | None]]:
    """Describe URL semantics without retaining a domain, path, or raw URL."""
    records = []
    for match in URL_PATTERN.finditer(text):
        local_context = text[max(0, match.start() - 160):min(len(text), match.end() + 160)]
        placeholder, category, institution = semantic_url_placeholder(match.group(0), local_context)
        records.append({
            "category": category,
            "institution": institution,
            "placeholder": placeholder,
        })
    return records


URL_DEPENDENCY_PATTERN = re.compile(
    r"(?i)\b(?:click|follow|open|visit|check|look at|see|use|try|download|buy|purchase|"
    r"read|watch)\s+(?:this|the|that|it|here)|\b(?:this|the|that)\s+(?:link|url|website|"
    r"page|resource|article|tool|app|product)|\b(?:link|url)\s+(?:here|below|above)\b"
)


def _url_reevaluation_reasons(query_text: str, comment_text: str) -> list[str]:
    """Conservatively flag cases where a URL may carry judgment-relevant meaning."""
    reasons: set[str] = set()
    for field_name, text in (("query", query_text), ("comment", comment_text)):
        url_records = _url_inventory(text)
        if not url_records:
            continue
        without_urls = re.sub(r"\s+", " ", URL_PATTERN.sub(" ", text)).strip()
        word_n = len(re.findall(r"\b\w+\b", without_urls))
        categories = {str(record["category"]) for record in url_records}
        if word_n <= 18:
            reasons.add(f"{field_name}_link_dominant_or_bare")
        if URL_DEPENDENCY_PATTERN.search(text):
            reasons.add(f"{field_name}_explicit_link_dependency")
        if categories & {"TOOL_OR_APP", "PRODUCT_LINK"} and word_n <= 45:
            reasons.add(f"{field_name}_action_depends_on_tool_or_product_target")
        if categories & {
            "OFFICIAL_HEALTH_RESOURCE", "PUBLIC_INFORMATION_RESOURCE",
            "RESEARCH_OR_ARTICLE", "ADHD_COMMUNITY_RESOURCE",
        } and word_n <= 28:
            reasons.add(f"{field_name}_authority_or_resource_type_may_affect_judgment")
        if "OTHER_URL" in categories and word_n <= 28:
            reasons.add(f"{field_name}_opaque_target_may_carry_missing_semantics")
    return sorted(reasons)


def _pair_system_memberships(system_rows: list[dict]) -> dict[tuple[str, str], list[str]]:
    memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in system_rows:
        qid = str(row["query_id"])
        system = f"{row['backend']}:{row['system']}"
        for cid in row["comment_ids"]:
            memberships[(qid, str(cid))].add(system)
    return {pair: sorted(systems) for pair, systems in memberships.items()}


def _make_semantic_blind_rows(
        pairs: list[tuple[str, str]], queries: dict[str, str], corpus: dict[str, str],
        item_types: dict[tuple[str, str], str], memberships: dict[tuple[str, str], list[str]],
        source_name: str) -> tuple[list[dict], list[dict], Counter[str]]:
    blind_rows, admin_rows = [], []
    replacement_counts: Counter[str] = Counter()
    for index, (qid, cid) in enumerate(pairs, 1):
        query_text, query_counts = redact_direct_identifiers(queries[qid])
        comment_text, comment_counts = redact_direct_identifiers(corpus[cid])
        replacement_counts.update(query_counts); replacement_counts.update(comment_counts)
        payload = {"query_text": query_text, "comment_text": comment_text}
        payload_sha = _hash_json({**payload, "facets_json": {}})
        blind_rows.append({"item_index": index, "payload": payload, "payload_sha256": payload_sha})
        admin_rows.append({
            "item_index": index,
            "query_id": qid,
            "comment_id": cid,
            "item_type": item_types[(qid, cid)],
            "experiment_source": source_name,
            "system_memberships": memberships.get((qid, cid), []),
            "payload_sha256": payload_sha,
        })
    return blind_rows, admin_rows, replacement_counts


def run_url_representation_audit(root: Path, cfg: dict) -> dict:
    """Audit URL semantics and prepare local-only rejudgment/final payloads."""
    out = _resolve(root, cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    audit_cfg = dict(cfg)
    audit_cfg["utility_registry"] = cfg.get("url_audit_source_registry", cfg["utility_registry"])
    data = load_inputs(root, audit_cfg)
    queries, corpus = data["queries"], data["corpus"]
    historical = set(data["qrels"])
    residual_report = _read_json(out / "fusion_stage1_residual_report.json")
    residual = _residual_pair_set(residual_report, "substantive_stage1_union")
    anchors_raw = _read_jsonl(_resolve(root, cfg["calibration_anchors"]))
    anchors = {(str(row["query_id"]), str(row["comment_id"])) for row in anchors_raw}
    if len(residual) != 963 or len(anchors) != 50 or residual & anchors:
        raise ValueError("frozen 963 residual + 50 anchor boundary failed")
    if not anchors <= historical:
        raise ValueError("all anchors must be existing complete utility-v2 judgments")

    system_rows = _read_jsonl(out / "fusion_all_system_per_query.jsonl")
    memberships = _pair_system_memberships(system_rows)
    union_pairs = historical | residual | anchors
    audit_queries = dict(queries); audit_corpus = dict(corpus)
    for pair, judgment in data["qrels"].items():
        if pair[0] not in audit_queries and judgment.get("query_text"):
            audit_queries[pair[0]] = str(judgment["query_text"])
        if pair[1] not in audit_corpus and judgment.get("comment_text"):
            audit_corpus[pair[1]] = str(judgment["comment_text"])
    unmapped = sorted(pair for pair in union_pairs
                      if pair[0] not in audit_queries or pair[1] not in audit_corpus)
    if unmapped:
        raise ValueError(f"URL audit cannot map {len(unmapped)} judged/residual pairs to frozen text")

    group_sets = {
        "historical_judged_pool_including_anchors": historical,
        "historical_judged_pool_excluding_anchors": historical - anchors,
        "residual": residual,
        "anchor": anchors,
    }
    pair_inventory = []
    category_pair_counts = {name: Counter() for name in [*group_sets, "unique_union"]}
    category_occurrence_counts = {name: Counter() for name in [*group_sets, "unique_union"]}
    group_url_pairs = Counter()
    group_url_occurrences = Counter()
    url_pairs: set[tuple[str, str]] = set()
    recommended_historical: set[tuple[str, str]] = set()
    for qid, cid in sorted(union_pairs):
        query_urls = _url_inventory(audit_queries[qid]); comment_urls = _url_inventory(audit_corpus[cid])
        occurrences = [{**record, "field": "query_text"} for record in query_urls]
        occurrences += [{**record, "field": "comment_text"} for record in comment_urls]
        if not occurrences:
            continue
        pair = (qid, cid); url_pairs.add(pair)
        groups = [name for name, pairs in group_sets.items() if pair in pairs]
        categories = sorted({str(record["category"]) for record in occurrences})
        institutions = sorted({str(record["institution"]) for record in occurrences if record["institution"]})
        reasons = _url_reevaluation_reasons(audit_queries[qid], audit_corpus[cid]) if pair in historical else []
        recommended = bool(reasons)
        if recommended:
            recommended_historical.add(pair)
        semantic_query, _ = redact_direct_identifiers(audit_queries[qid])
        semantic_comment, _ = redact_direct_identifiers(audit_corpus[cid])
        pair_inventory.append({
            "query_id": qid,
            "comment_id": cid,
            "groups": groups,
            "url_occurrence_n": len(occurrences),
            "url_categories": categories,
            "public_institutions": institutions,
            "field_distribution": dict(Counter(str(record["field"]) for record in occurrences)),
            "semantic_text_sha256": _hash_json({"query_text": semantic_query, "comment_text": semantic_comment}),
            "historical_reevaluation_recommended": recommended,
            "reevaluation_reasons": reasons,
            "system_memberships": memberships.get(pair, []),
        })
        for group in [*groups, "unique_union"]:
            group_url_pairs[group] += 1
            group_url_occurrences[group] += len(occurrences)
            category_pair_counts[group].update(categories)
            category_occurrence_counts[group].update(str(record["category"]) for record in occurrences)

    _write_jsonl(out / "fusion_url_semantic_pair_ADMIN.jsonl", pair_inventory)

    historical_url_pairs = sorted(historical & url_pairs)
    historical_types = {pair: "historical_url_semantic_view" for pair in historical_url_pairs}
    historical_blind, historical_admin, historical_replacements = _make_semantic_blind_rows(
        historical_url_pairs, audit_queries, audit_corpus, historical_types, memberships,
        "historical_utility_v2_url_semantic_view")
    historical_blind_path = out / "fusion_historical_url_semantic_blinded_view.jsonl"
    historical_admin_path = out / "fusion_historical_url_semantic_view_ADMIN.jsonl"
    _write_jsonl(historical_blind_path, historical_blind)
    _write_jsonl(historical_admin_path, historical_admin)
    historical_view_manifest = {
        "schema": "fusion-historical-url-semantic-view-manifest-v1",
        "created_at": _now(),
        "purpose": "non-mutating semantic representation view of all historical URL-containing judged pairs",
        "historical_registry_modified": False,
        "pair_n": len(historical_url_pairs),
        "blinded_view": {"path": str(historical_blind_path), "sha256": _sha(historical_blind_path)},
        "admin_view": {"path": str(historical_admin_path), "sha256": _sha(historical_admin_path)},
        "replacement_counts": dict(historical_replacements),
        "external_model_calls": 0,
    }
    _write_json(out / "fusion_historical_url_semantic_view_manifest.json", historical_view_manifest)

    recommended_nonanchor = recommended_historical - anchors
    recommended_anchor = recommended_historical & anchors
    control_target = max(
        int(cfg.get("url_audit_min_controls", 20)),
        math.ceil((float(cfg.get("url_audit_control_share", 0.20)) /
                   (1.0 - float(cfg.get("url_audit_control_share", 0.20)))) * len(recommended_nonanchor)),
    )
    control_pool = sorted((historical - url_pairs) - anchors)
    rng = np.random.default_rng(int(cfg.get("url_audit_control_seed", 20260721)))
    control_indices = rng.choice(len(control_pool), size=min(control_target, len(control_pool)), replace=False)
    controls = {control_pool[int(index)] for index in control_indices}
    reeval_items = [(pair, "url_reevaluation") for pair in sorted(recommended_nonanchor)]
    reeval_items += [(pair, "old_version_control") for pair in sorted(controls)]
    rng.shuffle(reeval_items)
    reeval_pairs = [pair for pair, _ in reeval_items]
    reeval_types = {pair: item_type for pair, item_type in reeval_items}
    reeval_blind, reeval_admin, reeval_replacements = _make_semantic_blind_rows(
        reeval_pairs, audit_queries, audit_corpus, reeval_types, memberships, "fusion_url_representation_reevaluation")
    for row in reeval_admin:
        pair = (row["query_id"], row["comment_id"])
        old = data["qrels"][pair]
        row["old_utility"] = float(old["utility"])
        row["old_judgment_provenance_retained_admin_only"] = True
        if pair in recommended_nonanchor:
            inventory_row = next(item for item in pair_inventory
                                 if item["query_id"] == pair[0] and item["comment_id"] == pair[1])
            row["url_categories"] = inventory_row["url_categories"]
            row["reevaluation_reasons"] = inventory_row["reevaluation_reasons"]
    reeval_blind_path = out / "fusion_url_reevaluation_blinded_payload.jsonl"
    reeval_admin_path = out / "fusion_url_reevaluation_payload_ADMIN.jsonl"
    _write_jsonl(reeval_blind_path, reeval_blind); _write_jsonl(reeval_admin_path, reeval_admin)
    reeval_control_share = len(controls) / len(reeval_items) if reeval_items else 0.0
    if reeval_items and reeval_control_share + 1e-12 < float(cfg.get("url_audit_control_share", 0.20)):
        raise ValueError("URL rejudgment control share is below 20%")
    prompt = project_config.prompt_path("evidence_card_judge_v2")
    reeval_manifest = {
        "schema": "fusion-url-reevaluation-payload-manifest-v1",
        "created_at": _now(),
        "authorization_status": "PENDING_SEPARATE_EXPLICIT_USER_APPROVAL",
        "reevaluation_items": len(recommended_nonanchor),
        "recommended_anchor_items_deferred_to_final_anchor_batch": len(recommended_anchor),
        "old_version_control_items": len(controls),
        "control_share": reeval_control_share,
        "total_items": len(reeval_items),
        "judge_visible_fields": ["query_text", "comment_text"],
        "judge_hidden_fields": ["query_id", "comment_id", "old_utility", "reevaluation_status",
                                "system", "rank", "source", "url_category"],
        "blinded_payload": {"path": str(reeval_blind_path), "sha256": _sha(reeval_blind_path)},
        "admin_payload": {"path": str(reeval_admin_path), "sha256": _sha(reeval_admin_path)},
        "prompt": {"path": str(prompt), "sha256": _sha(prompt)},
        "judge": cfg["judge"],
        "replacement_counts": dict(reeval_replacements),
        "bedrock_calls": 0,
        "frozen_test_read": False,
    }
    _write_json(out / "fusion_url_reevaluation_payload_manifest.json", reeval_manifest)

    affected_systems = {}
    for system in sorted({system for pair in recommended_historical for system in memberships.get(pair, [])}):
        pairs = {pair for pair in recommended_historical if system in memberships.get(pair, [])}
        positions = 0
        affected_queries = set()
        for row in system_rows:
            if f"{row['backend']}:{row['system']}" != system:
                continue
            qid = str(row["query_id"])
            hits = sum((qid, str(cid)) in pairs for cid in row["comment_ids"])
            positions += hits
            if hits:
                affected_queries.add(qid)
        affected_systems[system] = {
            "affected_top8_position_n": positions,
            "affected_query_n": len(affected_queries),
            "affected_query_ids": sorted(affected_queries),
            "utility_at8_absolute_change_upper_bound": 6.0 * positions / (8.0 * 100.0),
            "threshold_count_at8_mean_absolute_change_upper_bound": positions / 100.0,
            "ndcg_note": "must be recomputed after rejudgment; both DCG and common-IDCG may change",
        }
    impact_plan = {
        "schema": "fusion-url-rejudgment-impact-plan-v1",
        "status": "PENDING_URL_REJUDGMENT",
        "comparison_available_now": False,
        "reason": "No external judging was authorized or called in this audit-only phase.",
        "future_pair_level_comparison": {
            "six_dimension_deltas": "required",
            "composite_utility_delta": "required",
            "threshold_crossings": [4.0, 4.5, 5.0],
        },
        "affected_systems": affected_systems,
        "bedrock_calls": 0,
    }
    _write_json(out / "fusion_url_rejudgment_impact_plan.json", impact_plan)

    final_pairs = sorted(residual) + sorted(anchors)
    final_types = {pair: ("calibration_anchor" if pair in anchors else "residual") for pair in final_pairs}
    final_blind, final_admin, final_replacements = _make_semantic_blind_rows(
        final_pairs, queries, corpus, final_types, memberships, "fusion_substantive_stage1_semantic_redaction")
    raw_url_hits = sum(len(URL_PATTERN.findall(value)) for row in final_blind
                       for value in row["payload"].values())
    remaining_identifier_hits = Counter()
    for row in final_blind:
        for label, pattern, _ in DIRECT_IDENTIFIER_PATTERNS:
            for value in row["payload"].values():
                remaining_identifier_hits[label] += len(pattern.findall(value))
    if raw_url_hits or any(remaining_identifier_hits.values()):
        raise ValueError("semantic final payload retains a raw URL or direct-identifier pattern")
    final_blind_path = out / "fusion_stage1_semantic_redacted_blinded_payload.jsonl"
    final_admin_path = out / "fusion_stage1_semantic_redacted_payload_ADMIN.jsonl"
    _write_jsonl(final_blind_path, final_blind); _write_jsonl(final_admin_path, final_admin)
    old_authorized_sha = "dd84b9dbe54fbb0d4f3be32abe665f2c8bc9d1d9ee7f722c7900505ba5f004bd"
    rules_hash = _hash_json({
        "version": "semantic-url-representation-v1",
        "categories": URL_CATEGORIES,
        "institution_retention": "public institution name only",
        "forbidden_url_components": ["clickable_url", "domain", "path", "query_string", "username", "tracking"],
    })
    final_manifest = {
        "schema": "fusion-stage1-semantic-redacted-payload-manifest-v1",
        "created_at": _now(),
        "authorization_status": "PENDING_NEW_EXPLICIT_USER_APPROVAL",
        "prior_authorized_sha256_invalidated": old_authorized_sha,
        "authorization_applies_only_to_this_sha256": True,
        "residual_items": len(residual), "anchor_items": len(anchors), "total_items": len(final_pairs),
        "url_representation_rules_sha256": rules_hash,
        "url_categories": list(URL_CATEGORIES),
        "judge_visible_fields": ["query_text", "comment_text"],
        "administrative_provenance_visible_to_judge": False,
        "blinded_payload": {"path": str(final_blind_path), "sha256": _sha(final_blind_path)},
        "admin_payload": {"path": str(final_admin_path), "sha256": _sha(final_admin_path)},
        "prompt": {"path": str(prompt), "sha256": _sha(prompt)},
        "judge": cfg["judge"],
        "replacement_counts": dict(final_replacements),
        "remaining_raw_url_hits": raw_url_hits,
        "remaining_direct_identifier_hits": dict(remaining_identifier_hits),
        "duplicate_pairs": len(final_pairs) - len(set(final_pairs)),
        "residual_anchor_overlap": len(residual & anchors),
        "bedrock_calls": 0,
        "frozen_test_read": False,
    }
    final_manifest_path = out / "fusion_stage1_semantic_redacted_payload_manifest.json"
    _write_json(final_manifest_path, final_manifest)

    group_summary = {}
    for group in [*group_sets, "unique_union"]:
        group_summary[group] = {
            "url_pair_n": group_url_pairs[group],
            "url_occurrence_n": group_url_occurrences[group],
            "category_pair_counts": {category: category_pair_counts[group].get(category, 0)
                                     for category in URL_CATEGORIES},
            "category_occurrence_counts": {category: category_occurrence_counts[group].get(category, 0)
                                           for category in URL_CATEGORIES},
        }
    audit = {
        "schema": "fusion-url-representation-audit-v1",
        "created_at": _now(),
        "verdict": "AUDIT_COMPLETE_REJUDGMENT_PENDING",
        "historical_complete_judged_pair_n": len(historical),
        "historical_query_n": len({qid for qid, _ in historical}),
        "historical_pairs_outside_dev100_query_file_n": sum(qid not in queries for qid, _ in historical),
        "historical_queries_outside_dev100_query_file_n": len({qid for qid, _ in historical if qid not in queries}),
        "residual_pair_n": len(residual),
        "anchor_pair_n": len(anchors),
        "unique_scanned_pair_n": len(union_pairs),
        "unique_url_pair_n": len(url_pairs),
        "group_summary": group_summary,
        "historical_reevaluation_recommended_pair_n": len(recommended_historical),
        "historical_nonanchor_reevaluation_payload_pair_n": len(recommended_nonanchor),
        "historical_anchor_reevaluation_deferred_pair_n": len(recommended_anchor),
        "historical_no_reevaluation_pair_n": len((historical & url_pairs) - recommended_historical),
        "reevaluation_involved_query_n": len({qid for qid, _ in recommended_historical}),
        "reevaluation_involved_query_ids": sorted({qid for qid, _ in recommended_historical}),
        "reevaluation_involved_systems": affected_systems,
        "classification_basis": "domain class and local +/-160-character context; no raw URL retained",
        "raw_urls_written_to_audit_outputs": False,
        "historical_registry_modified": False,
        "historical_semantic_view": historical_view_manifest,
        "comparison_status": "PENDING_URL_REJUDGMENT",
        "reevaluation_payload": reeval_manifest,
        "final_payload": final_manifest,
        "bedrock_calls": 0,
        "frozen_test_read": False,
    }
    audit_path = out / "fusion_url_representation_audit.json"
    _write_json(audit_path, audit)

    category_lines = []
    union_category = group_summary["unique_union"]["category_pair_counts"]
    union_category_occurrences = group_summary["unique_union"]["category_occurrence_counts"]
    for category in URL_CATEGORIES:
        category_lines.append(
            f"| {category} | {union_category[category]} | {union_category_occurrences[category]} |")
    group_lines = []
    for group, row in group_summary.items():
        group_lines.append(f"| {group} | {row['url_pair_n']} | {row['url_occurrence_n']} |")
    system_lines = []
    for system, row in affected_systems.items():
        system_lines.append(f"| {system} | {row['affected_query_n']} | {row['affected_top8_position_n']} | "
                            f"{row['utility_at8_absolute_change_upper_bound']:.4f} |")
    audit_md = "\n".join([
        "# Fusion URL representation consistency audit", "",
        "Verdict: **AUDIT_COMPLETE_REJUDGMENT_PENDING**. No Bedrock or other external model call was made.", "",
        f"Scanned **{len(union_pairs):,}** unique pairs: {len(historical):,} complete historical utility-v2 pairs, "
        f"{len(residual)} frozen residual pairs, and {len(anchors)} anchors (anchors are a subset of the historical pool). "
        f"Found **{len(url_pairs)}** unique pairs containing at least one raw URL in the frozen source text.", "",
        f"The historical registry contains {sum(qid not in queries for qid, _ in historical)} complete pairs from "
        f"{len({qid for qid, _ in historical if qid not in queries})} legacy query outside the dev100 query file. "
        "Those pairs were retained in the all-history scan using the frozen text stored in the judgment registry; they "
        "are not part of the 963 residual or 50-anchor final batch.", "",
        "## Distribution", "", "| Group | URL pairs | URL occurrences |", "|---|---:|---:|", *group_lines, "",
        "| Semantic category | Unique union pairs | URL occurrences |", "|---|---:|---:|", *category_lines, "",
        "Pair counts are non-exclusive because one query-comment pair may contain URLs from more than one category; "
        "every individual URL occurrence receives exactly one category.", "",
        "No full URL, domain, path, query string, username, or tracking parameter is reproduced in this report or its "
        "ADMIN inventory. Public institution names are retained only inside approved semantic placeholders.", "",
        f"A non-mutating semantic view was generated for all {len(historical_url_pairs)} historical URL-containing pairs "
        f"at SHA256 `{historical_view_manifest['blinded_view']['sha256']}`. This standardizes representation without "
        "overwriting the authoritative historical registry or its old scores.", "",
        "## Historical re-evaluation recommendation", "",
        f"Recommended: **{len(recommended_historical)}** historical URL pairs across "
        f"**{len({qid for qid, _ in recommended_historical})}** queries. Of these, "
        f"{len(recommended_nonanchor)} non-anchor pairs are placed in the independent blind re-evaluation payload; "
        f"{len(recommended_anchor)} anchor pairs are deferred to the already planned 50-anchor batch to avoid duplicate calls.", "",
        "The deterministic rule flags link-dominant text, explicit 'use/check/click this link' dependence, short "
        "tool/product recommendations, or cases where authority/resource type may affect usefulness, actionability, safety, "
        "or grounding. Descriptive text whose recommendation remains intact after semantic replacement is not flagged.", "",
        "## Systems potentially affected", "", "| System | Queries | Top-8 positions | Utility@8 absolute-change upper bound |",
        "|---|---:|---:|---:|", *system_lines, "",
        "The bound assumes the impossible worst case that every affected position moves across the full 1-7 utility range. "
        "nDCG@8 must be recomputed after rejudgment because both a system's DCG and the query-level common IDCG can change. "
        "Threshold counts at 4, 4.5, and 5 can change only where a rejudged pair crosses the corresponding boundary.", "",
        "## Blind batches", "",
        f"URL re-evaluation payload: {len(recommended_nonanchor)} flagged pairs + {len(controls)} old-version controls; "
        f"control share **{reeval_control_share:.1%}**; SHA256 `{reeval_manifest['blinded_payload']['sha256']}`. "
        "This batch is not authorized for sending.", "",
        f"Final semantic payload: 963 residual + 50 anchors = 1,013 items; SHA256 "
        f"`{final_manifest['blinded_payload']['sha256']}`. Its status is `PENDING_NEW_EXPLICIT_USER_APPROVAL`. "
        f"The former authorized SHA `{old_authorized_sha}` is invalidated and was not sent.", "",
        "## Pending comparison", "",
        "Six-dimension deltas, composite-utility deltas, threshold crossings, and realized system metric changes are "
        "`PENDING_URL_REJUDGMENT`; they cannot be computed without new scores and were not fabricated in this local-only phase.", "",
    ])
    (out / "fusion_url_representation_audit.md").write_text(audit_md, encoding="utf-8")
    impact_md = "\n".join([
        "# URL rejudgment impact plan", "", "Status: **PENDING_URL_REJUDGMENT**.", "",
        "After separately authorized scoring, join by ADMIN `(query_id, comment_id)` and report, without overwriting the "
        "historical truth source: six utility-v2 dimension deltas; composite utility delta; crossings at 4, 4.5, and 5; "
        "then recompute Utility@8, common-IDCG nDCG@8, and threshold counts for every affected frozen system.", "",
        "No comparison is available in this audit-only phase because no new judgment exists and no Bedrock call was made.", "",
    ])
    (out / "fusion_url_rejudgment_impact_plan.md").write_text(impact_md, encoding="utf-8")
    return {
        "verdict": audit["verdict"],
        "unique_scanned_pair_n": len(union_pairs),
        "unique_url_pair_n": len(url_pairs),
        "historical_reevaluation_recommended_pair_n": len(recommended_historical),
        "reevaluation_payload_items": len(reeval_items),
        "reevaluation_payload_sha256": reeval_manifest["blinded_payload"]["sha256"],
        "final_payload_items": len(final_pairs),
        "final_payload_sha256": final_manifest["blinded_payload"]["sha256"],
        "authorization_status": final_manifest["authorization_status"],
        "bedrock_calls": 0,
    }


URL_REEVALUATION_AUTHORIZED_SHA256 = "5bf3c20ad2615a51ac4746ffa5b54e4ad2b392921e09adae33a51d3a0e865d13"
URL_REEVALUATION_BATCH_ID = "utility-v2-url-semantic-reevaluation-dev100-v1-20260721"


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _validate_url_reevaluation_batch(out: Path, cfg: dict) -> tuple[list[dict], list[dict], dict]:
    manifest_path = out / "fusion_url_reevaluation_payload_manifest.json"
    manifest = _read_json(manifest_path)
    blind_path = Path(manifest["blinded_payload"]["path"])
    admin_path = Path(manifest["admin_payload"]["path"])
    if _sha(blind_path) != URL_REEVALUATION_AUTHORIZED_SHA256:
        raise ValueError("authorized URL reevaluation payload SHA256 mismatch")
    if manifest["blinded_payload"]["sha256"] != URL_REEVALUATION_AUTHORIZED_SHA256:
        raise ValueError("URL reevaluation manifest SHA256 mismatch")
    if manifest["reevaluation_items"] != 21 or manifest["old_version_control_items"] != 20:
        raise ValueError("authorized URL reevaluation 21+20 composition mismatch")
    if manifest["total_items"] != 41 or manifest["control_share"] < 0.20:
        raise ValueError("authorized URL reevaluation size/control gate failed")
    expected_model = "bedrock:us.meta.llama3-3-70b-instruct-v1:0"
    if manifest["judge"]["model"] != expected_model or cfg["judge"]["model"] != expected_model:
        raise ValueError("URL reevaluation model differs from authorized Meta Llama 3.3 70B")
    if manifest["judge"]["aws_region"] != "us-east-1" or cfg["judge"]["aws_region"] != "us-east-1":
        raise ValueError("URL reevaluation region differs from authorized us-east-1")
    blind = _read_jsonl(blind_path); admin = _read_jsonl(admin_path)
    if len(blind) != 41 or len(admin) != 41:
        raise ValueError("URL reevaluation blind/ADMIN row count mismatch")
    blind_by_index = {int(row["item_index"]): row for row in blind}
    admin_by_index = {int(row["item_index"]): row for row in admin}
    if set(blind_by_index) != set(range(1, 42)) or set(admin_by_index) != set(blind_by_index):
        raise ValueError("URL reevaluation blind/ADMIN item index mismatch")
    forbidden = {"query_id", "comment_id", "system", "rank", "source", "old_utility",
                 "item_type", "url_category", "reevaluation_status"}
    for index, row in blind_by_index.items():
        if set(row) != {"item_index", "payload", "payload_sha256"}:
            raise ValueError(f"blind row {index} has unexpected fields")
        payload = row["payload"]
        if set(payload) != {"query_text", "comment_text"} or forbidden & set(payload):
            raise ValueError(f"blind row {index} payload allowlist failed")
        if URL_PATTERN.search(payload["query_text"]) or URL_PATTERN.search(payload["comment_text"]):
            raise ValueError(f"blind row {index} retains a raw URL")
        rendered_sha = _hash_json({**payload, "facets_json": {}})
        if rendered_sha != row["payload_sha256"] or rendered_sha != admin_by_index[index]["payload_sha256"]:
            raise ValueError(f"blind/ADMIN payload hash mismatch at row {index}")
    return [blind_by_index[index] for index in sorted(blind_by_index)], [
        admin_by_index[index] for index in sorted(admin_by_index)], manifest


def run_url_reevaluation_judging(root: Path, cfg: dict) -> dict:
    """Run only the explicitly authorized 41-item URL semantic reevaluation batch."""
    from utility_scoring.annotation.run_top3_residual_judging import judge_payload

    out = _resolve(root, cfg["output_dir"])
    blind, admin, manifest = _validate_url_reevaluation_batch(out, cfg)
    prompt_sha = manifest["prompt"]["sha256"]
    if _sha(Path(manifest["prompt"]["path"])) != prompt_sha:
        raise ValueError("utility-v2 prompt changed after URL reevaluation payload freeze")
    checkpoint_dir = out / "fusion_url_reevaluation_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    blind_by_index = {int(row["item_index"]): row for row in blind}
    admin_by_index = {int(row["item_index"]): row for row in admin}
    done = {}
    for path in checkpoint_dir.glob("item_*.json"):
        checkpoint = _read_json(path)
        index = int(checkpoint["item_index"])
        if checkpoint.get("validation_status") != "valid":
            continue
        if checkpoint["payload_sha256"] != blind_by_index[index]["payload_sha256"]:
            raise ValueError(f"checkpoint payload hash mismatch at item {index}")
        done[index] = checkpoint
    pending = [index for index in sorted(blind_by_index) if index not in done]
    run_manifest_path = out / "fusion_url_reevaluation_run_manifest.json"
    run_manifest = {
        "schema": "fusion-url-reevaluation-run-manifest-v1",
        "created_at": _now(),
        "authorization": {
            "authorized_payload_sha256": URL_REEVALUATION_AUTHORIZED_SHA256,
            "authorized_item_n": 41,
            "authorized_reevaluation_n": 21,
            "authorized_control_n": 20,
            "admin_file_authorized_for_external_send": False,
            "other_files_authorized_for_external_send": False,
        },
        "external_destination": "AWS Bedrock us-east-1",
        "model": cfg["judge"]["model"],
        "protocol": "utility-v2",
        "judge_visible_fields": ["query_text", "comment_text"],
        "judge_hidden_fields": ["item_index", "query_id", "comment_id", "item_type", "old_utility",
                                "system_memberships", "rank", "source", "url_categories"],
        "prompt_sha256": prompt_sha,
        "batch_id": URL_REEVALUATION_BATCH_ID,
        "already_complete_before_invocation": len(done),
        "pending_before_invocation": len(pending),
        "status": "RUNNING" if pending else "COMPLETE",
    }
    _atomic_write_json(run_manifest_path, run_manifest)

    def evaluate(index: int) -> tuple[int, dict, dict]:
        blind_row = blind_by_index[index]; admin_row = admin_by_index[index]
        prompt_payload = {**blind_row["payload"], "facets_json": {}}
        raw, valid = judge_payload(
            prompt_payload,
            str(admin_row["item_type"]),
            URL_REEVALUATION_BATCH_ID,
            prompt_sha,
            identity={
                "item_index": index,
                "query_id": str(admin_row["query_id"]),
                "comment_id": str(admin_row["comment_id"]),
            },
        )
        return index, raw, valid

    completed_this_run = 0; failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(cfg["judge"].get("workers", 4)))) as executor:
        futures = {executor.submit(evaluate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                item_index, raw, valid = future.result()
                checkpoint = {
                    "item_index": item_index,
                    "payload_sha256": blind_by_index[item_index]["payload_sha256"],
                    "raw": raw,
                    "validated": valid,
                    "validation_status": "valid",
                    "completed_at": _now(),
                }
                _atomic_write_json(checkpoint_dir / f"item_{item_index:04d}.json", checkpoint)
                done[item_index] = checkpoint
                completed_this_run += 1
            except Exception as exc:
                failures.append({"item_index": index, "error": f"{type(exc).__name__}: {exc}"})
    raw_rows = [done[index]["raw"] for index in sorted(done)]
    validated_rows = [done[index]["validated"] for index in sorted(done)]
    _write_jsonl(out / "fusion_url_reevaluation_judgments_raw.jsonl", raw_rows)
    _write_jsonl(out / "fusion_url_reevaluation_judgments_validated.jsonl", validated_rows)
    _write_jsonl(out / "fusion_url_reevaluation_judgment_failures.jsonl", failures)
    complete = len(done) == 41 and not failures
    run_manifest.update({
        "completed_at": _now(),
        "completed_this_invocation": completed_this_run,
        "validated_total": len(done),
        "failed_this_invocation": len(failures),
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "bedrock_calls_successfully_checkpointed": len(done),
        "main_963_plus_50_payload_sent": False,
    })
    _atomic_write_json(run_manifest_path, run_manifest)
    manifest.update({
        "authorization_status": "AUTHORIZED_AND_COMPLETED" if complete else "AUTHORIZED_INCOMPLETE",
        "authorized_payload_sha256": URL_REEVALUATION_AUTHORIZED_SHA256,
        "validated_items": len(done),
        "failed_items": len(failures),
        "bedrock_calls": len(done),
        "main_batch_sent": False,
        "run_manifest": {"path": str(run_manifest_path), "sha256": _sha(run_manifest_path)},
    })
    _atomic_write_json(out / "fusion_url_reevaluation_payload_manifest.json", manifest)
    if not complete:
        raise RuntimeError(f"URL reevaluation incomplete: valid={len(done)}, failures={len(failures)}")
    return {
        "status": "COMPLETE",
        "validated_items": len(done),
        "completed_this_invocation": completed_this_run,
        "payload_sha256": URL_REEVALUATION_AUTHORIZED_SHA256,
        "main_batch_sent": False,
    }


def _url_control_stability(old_registry: dict[tuple[str, str], dict], controls: list[dict]) -> dict:
    from utility_scoring.annotation.run_top3_residual_judging import DIMS_V2

    if len(controls) != 20:
        raise ValueError(f"expected 20 URL old-version controls, got {len(controls)}")
    dimensions = {}; severe = []
    all_old, all_new = [], []
    for dim_index, dim in enumerate(DIMS_V2):
        old = [int(old_registry[(str(row["query_id"]), str(row["comment_id"]))][f"label_{dim}"])
               for row in controls]
        new = [int(row["validated_scores"][dim]) for row in controls]
        diffs = [right - left for left, right in zip(old, new)]
        absdiff = [abs(value) for value in diffs]
        kappa = (float(cohen_kappa_score(old, new, weights="quadratic"))
                 if len(set(old + new)) > 1 else None)
        row = {
            "n": len(old),
            "exact_agreement": statistics.fmean(left == right for left, right in zip(old, new)),
            "weighted_cohen_kappa_quadratic": kappa,
            "mean_absolute_difference": statistics.fmean(absdiff),
            "two_or_more_grade_flip_rate": statistics.fmean(value >= 2 for value in absdiff),
            "extreme_flip_rate_abs_ge_4": statistics.fmean(value >= 4 for value in absdiff),
            "old_mean": statistics.fmean(old), "new_mean": statistics.fmean(new),
            "mean_shift_new_minus_old": statistics.fmean(diffs),
            "paired_difference_bootstrap_95ci": list(bootstrap_ci(
                diffs, n_boot=5000, seed=20260721 + dim_index)),
        }
        warnings = []
        if row["extreme_flip_rate_abs_ge_4"] > .05:
            warnings.append("extreme_flip_rate_gt_5pct")
        if row["two_or_more_grade_flip_rate"] > .15:
            warnings.append("two_or_more_flip_rate_gt_15pct")
        if kappa is None or kappa < .60:
            warnings.append("weighted_kappa_lt_0.60")
        if abs(row["mean_shift_new_minus_old"]) > .25:
            warnings.append("absolute_mean_drift_gt_0.25")
        row["severe_warnings"] = warnings
        severe.extend(f"{dim}:{warning}" for warning in warnings)
        dimensions[dim] = row
        all_old.extend(old); all_new.extend(new)
    old_utility = [float(old_registry[(str(row["query_id"]), str(row["comment_id"]))]["utility"])
                   for row in controls]
    new_utility = [float(row["utility"]) for row in controls]
    utility_diffs = [right - left for left, right in zip(old_utility, new_utility)]
    overall_exact = statistics.fmean(left == right for left, right in zip(all_old, all_new))
    utility_shift = statistics.fmean(utility_diffs)
    if severe:
        verdict = "MATERIAL_BATCH_DRIFT"
    elif overall_exact >= .75 and abs(utility_shift) <= .10:
        verdict = "STABLE"
    else:
        verdict = "STABLE_WITH_MINOR_DRIFT"
    return {
        "verdict": verdict,
        "valid_control_pairs": len(controls),
        "dimensions": dimensions,
        "overall_dimension_exact_agreement": overall_exact,
        "utility": {
            "old_mean": statistics.fmean(old_utility), "new_mean": statistics.fmean(new_utility),
            "mean_shift_new_minus_old": utility_shift,
            "mean_absolute_difference": statistics.fmean(abs(value) for value in utility_diffs),
            "paired_difference_bootstrap_95ci": list(bootstrap_ci(
                utility_diffs, n_boot=5000, seed=20260721)),
        },
        "severe_warnings": severe,
        "thresholds": {"extreme_flip_rate": .05, "two_or_more_flip_rate": .15,
                       "weighted_kappa": .60, "absolute_mean_drift": .25,
                       "overall_exact": .75, "absolute_utility_shift": .10},
        "human_gold": False, "silver_repeatability_only": True,
    }


def analyze_url_reevaluation(root: Path, cfg: dict) -> dict:
    """Compare URL rejudgments, freeze a new registry, and recompute affected systems."""
    from utility_scoring.annotation.run_top3_residual_judging import DIMS_V2

    out = _resolve(root, cfg["output_dir"])
    blind, admin, payload_manifest = _validate_url_reevaluation_batch(out, cfg)
    validated = _read_jsonl(out / "fusion_url_reevaluation_judgments_validated.jsonl")
    if len(validated) != 41 or any(row.get("validation_status") != "valid" for row in validated):
        raise ValueError("URL reevaluation requires 41 validated judgments")
    validated_by_index = {int(row["item_index"]): row for row in validated}
    admin_by_index = {int(row["item_index"]): row for row in admin}
    if set(validated_by_index) != set(admin_by_index):
        raise ValueError("URL validated/ADMIN item identities differ")

    source_registry_path = _resolve(root, cfg.get("url_audit_source_registry", cfg["utility_registry"]))
    source_rows = _read_jsonl(source_registry_path)
    _, old_registry = complete_utility_v2_rows(source_rows)
    controls = [validated_by_index[index] for index in sorted(validated_by_index)
                if admin_by_index[index]["item_type"] == "old_version_control"]
    rejudged = [validated_by_index[index] for index in sorted(validated_by_index)
                if admin_by_index[index]["item_type"] == "url_reevaluation"]
    if len(rejudged) != 21 or len(controls) != 20:
        raise ValueError("URL validated 21 reevaluation + 20 control composition mismatch")
    stability = _url_control_stability(old_registry, controls)
    _write_json(out / "fusion_url_control_stability.json", stability)
    stability_lines = [
        "# URL semantic reevaluation control stability", "",
        f"Verdict: **{stability['verdict']}**. Controls: 20/20.", "",
        "| Dimension | Exact | Weighted kappa | Mean shift | >=2 flip | Extreme flip |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dim, row in stability["dimensions"].items():
        kappa = "NA" if row["weighted_cohen_kappa_quadratic"] is None else f"{row['weighted_cohen_kappa_quadratic']:.3f}"
        stability_lines.append(
            f"| {dim} | {row['exact_agreement']:.3f} | {kappa} | "
            f"{row['mean_shift_new_minus_old']:+.3f} | {row['two_or_more_grade_flip_rate']:.3f} | "
            f"{row['extreme_flip_rate_abs_ge_4']:.3f} |")
    stability_lines += ["", f"Utility shift: {stability['utility']['mean_shift_new_minus_old']:+.4f}, "
                        f"95% CI {stability['utility']['paired_difference_bootstrap_95ci']}.", "",
                        "These controls measure LLM-silver batch repeatability, not human inter-rater reliability.", ""]
    (out / "fusion_url_control_stability.md").write_text("\n".join(stability_lines), encoding="utf-8")
    if stability["verdict"] == "MATERIAL_BATCH_DRIFT":
        verdict = {"verdict": "URL_REEVALUATION_INCONCLUSIVE_BATCH_DRIFT",
                   "registry_updated": False, "main_batch_sent": False}
        _write_json(out / "fusion_url_reevaluation_final_verdict.json", verdict)
        return verdict

    changes = []
    rejudged_by_pair = {}
    for row in rejudged:
        pair = (str(row["query_id"]), str(row["comment_id"]))
        old = old_registry[pair]
        dimension_deltas = {dim: int(row["validated_scores"][dim]) - int(old[f"label_{dim}"])
                            for dim in DIMS_V2}
        old_utility = float(old["utility"]); new_utility = float(row["utility"])
        crossings = {}
        for threshold in (4.0, 4.5, 5.0):
            before, after = old_utility >= threshold, new_utility >= threshold
            crossings[str(threshold)] = "up" if after and not before else "down" if before and not after else "none"
        changes.append({
            "query_id": pair[0], "comment_id": pair[1],
            "old_scores": {dim: int(old[f"label_{dim}"]) for dim in DIMS_V2},
            "new_scores": {dim: int(row["validated_scores"][dim]) for dim in DIMS_V2},
            "dimension_deltas_new_minus_old": dimension_deltas,
            "old_utility": old_utility, "new_utility": new_utility,
            "utility_delta_new_minus_old": new_utility - old_utility,
            "threshold_crossings": crossings,
        })
        rejudged_by_pair[pair] = row
    _write_jsonl(out / "fusion_url_reevaluation_pairwise_changes.jsonl", changes)

    historical_blind = _read_jsonl(out / "fusion_historical_url_semantic_blinded_view.jsonl")
    historical_admin = _read_jsonl(out / "fusion_historical_url_semantic_view_ADMIN.jsonl")
    historical_payload_by_pair = {
        (str(admin_row["query_id"]), str(admin_row["comment_id"])): blind_row["payload"]
        for blind_row, admin_row in zip(historical_blind, historical_admin)
        if int(blind_row["item_index"]) == int(admin_row["item_index"])
        and blind_row["payload_sha256"] == admin_row["payload_sha256"]
    }
    if len(historical_payload_by_pair) != 116:
        raise ValueError("historical URL semantic view must contain 116 unique pairs")
    source_sha_before = _sha(source_registry_path)
    augmented_rows = []
    score_update_n = text_update_n = 0
    for source_row in source_rows:
        pair = (str(source_row["query_id"]), str(source_row["comment_id"]))
        updated = dict(source_row)
        if pair in historical_payload_by_pair:
            updated["query_text"] = historical_payload_by_pair[pair]["query_text"]
            updated["comment_text"] = historical_payload_by_pair[pair]["comment_text"]
            updated["url_semantic_representation_version"] = "semantic-url-representation-v1"
            text_update_n += 1
        if pair in rejudged_by_pair:
            judged = rejudged_by_pair[pair]
            updated.update({f"label_{dim}": int(judged["validated_scores"][dim]) for dim in DIMS_V2})
            updated.update({
                "utility": float(judged["utility"]), "rationale": judged.get("rationale", ""),
                "judge_model": judged["model"], "judge_id": "utility-v2",
                "judge_version": judged["judge_version"], "batch_id": URL_REEVALUATION_BATCH_ID,
                "judgment_source": "url_semantic_representation_reevaluation",
                "previous_judgment_sha256": _hash_json(source_row),
                "url_reevaluation_payload_sha256": URL_REEVALUATION_AUTHORIZED_SHA256,
                "validation_status": "valid",
            })
            score_update_n += 1
        augmented_rows.append(updated)
    if score_update_n != 21 or text_update_n != 116:
        raise ValueError(f"URL registry update count mismatch: score={score_update_n}, text={text_update_n}")
    complete_rows, augmented_registry = complete_utility_v2_rows(augmented_rows)
    if len(complete_rows) != len(source_rows) or len(augmented_registry) != 4442:
        raise ValueError("URL semantic augmented registry completeness failed")
    registry_path = out / "fusion_url_semantic_augmented_registry.jsonl"
    _write_jsonl(registry_path, augmented_rows)
    if _sha(source_registry_path) != source_sha_before:
        raise ValueError("authoritative source registry changed during URL reevaluation merge")
    registry_manifest = {
        "schema": "fusion-url-semantic-augmented-registry-manifest-v1",
        "created_at": _now(), "status": "FROZEN",
        "source_registry": {"path": str(source_registry_path), "sha256": source_sha_before,
                            "modified": False, "pair_n": len(source_rows)},
        "augmented_registry": {"path": str(registry_path), "sha256": _sha(registry_path),
                               "pair_n": len(augmented_rows)},
        "historical_url_text_pairs_semantically_normalized": text_update_n,
        "historical_url_pairs_rejudged": score_update_n,
        "control_scores_written_to_registry": 0,
        "old_rows_deleted": 0, "unjudged_assigned_zero": False,
        "control_stability_verdict": stability["verdict"],
        "main_batch_sent": False,
    }
    _write_json(out / "fusion_url_semantic_augmented_registry_manifest.json", registry_manifest)

    old_qrels_by_query: dict[str, dict[str, float]] = defaultdict(dict)
    new_qrels_by_query: dict[str, dict[str, float]] = defaultdict(dict)
    for (qid, cid), row in old_registry.items():
        old_qrels_by_query[qid][cid] = float(row["utility"])
    for (qid, cid), row in augmented_registry.items():
        new_qrels_by_query[qid][cid] = float(row["utility"])
    data = load_inputs(root, cfg)
    frozen_system_rows = _read_jsonl(out / "fusion_all_system_per_query.jsonl")
    old_system_rows, new_system_rows = [], []
    for row in frozen_system_rows:
        qid = str(row["query_id"]); ids = [str(cid) for cid in row["comment_ids"]]
        old_system_rows.append(_system_row(row["backend"], row["system"], qid, ids, old_registry,
                                           dict(old_qrels_by_query), data["qtypes"][qid], row.get("metadata")))
        new_system_rows.append(_system_row(row["backend"], row["system"], qid, ids, augmented_registry,
                                           dict(new_qrels_by_query), data["qtypes"][qid], row.get("metadata")))
    old_metrics = aggregate_metrics(old_system_rows); new_metrics = aggregate_metrics(new_system_rows)
    _write_jsonl(out / "fusion_url_updated_system_per_query.jsonl", new_system_rows)
    _write_json(out / "fusion_url_updated_system_metrics.json", new_metrics)
    metric_names = ("mean_utility_at8", "common_idcg_ndcg_at8", "acceptable_count_at8",
                    "useful_count_at8", "high_quality_count_at8")
    system_impacts = {}
    for system in sorted(new_metrics):
        system_impacts[system] = {
            "complete_query_n": new_metrics[system]["complete_query_n"],
            "judgment_coverage_at8": new_metrics[system]["mean_judgment_coverage_at8"],
            **{metric: {
                "old": old_metrics[system][metric], "new": new_metrics[system][metric],
                "delta": (None if old_metrics[system][metric] is None or new_metrics[system][metric] is None
                          else new_metrics[system][metric] - old_metrics[system][metric]),
            } for metric in metric_names},
        }
    impact = {
        "schema": "fusion-url-reevaluation-realized-system-impact-v1",
        "registry_sha256": registry_manifest["augmented_registry"]["sha256"],
        "systems": system_impacts,
        "coverage_note": "current pre-main-batch complete-case coverage retained; no missing judgment was set to zero",
        "main_batch_sent": False,
    }
    _write_json(out / "fusion_url_reevaluation_realized_system_impact.json", impact)

    dimension_summary = {}
    for dim in DIMS_V2:
        values = [row["dimension_deltas_new_minus_old"][dim] for row in changes]
        dimension_summary[dim] = {
            "mean_delta": statistics.fmean(values), "median_delta": statistics.median(values),
            "exact_n": sum(value == 0 for value in values),
            "up_n": sum(value > 0 for value in values), "down_n": sum(value < 0 for value in values),
            "two_or_more_change_n": sum(abs(value) >= 2 for value in values),
        }
    utility_deltas = [row["utility_delta_new_minus_old"] for row in changes]
    threshold_counts = {str(threshold): {
        "up": sum(row["threshold_crossings"][str(threshold)] == "up" for row in changes),
        "down": sum(row["threshold_crossings"][str(threshold)] == "down" for row in changes),
        "none": sum(row["threshold_crossings"][str(threshold)] == "none" for row in changes),
    } for threshold in (4.0, 4.5, 5.0)}
    changed_systems = {
        system: row for system, row in system_impacts.items()
        if any(abs(row[metric]["delta"] or 0.0) > 1e-12 for metric in metric_names)
    }
    summary = {
        "schema": "fusion-url-reevaluation-change-summary-v1",
        "rejudged_pair_n": len(changes), "query_n": len({row["query_id"] for row in changes}),
        "dimensions": dimension_summary,
        "utility": {
            "mean_delta": statistics.fmean(utility_deltas), "median_delta": statistics.median(utility_deltas),
            "mean_absolute_delta": statistics.fmean(abs(value) for value in utility_deltas),
            "bootstrap_95ci": list(bootstrap_ci(utility_deltas, n_boot=5000, seed=20260721)),
            "increased_n": sum(value > 1e-12 for value in utility_deltas),
            "tied_n": sum(abs(value) <= 1e-12 for value in utility_deltas),
            "decreased_n": sum(value < -1e-12 for value in utility_deltas),
        },
        "threshold_crossings": threshold_counts,
        "control_stability": stability,
        "realized_changed_system_n": len(changed_systems),
        "registry": registry_manifest,
        "main_batch_sent": False,
    }
    _write_json(out / "fusion_url_reevaluation_change_summary.json", summary)
    important_systems = [
        "SBERT:Dense-Top8", "Cohere:Dense-Top8", "SBERT:CTS-Deep-Linear-q2",
        "SBERT:CTS-Graph-Linear-q2", "Cohere:CTS-Deep-Linear-q2", "Cohere:CTS-Graph-Linear-q2",
    ]
    result_lines = [
        "# URL semantic representation reevaluation results", "",
        f"Control verdict: **{stability['verdict']}**. The 21 URL-dependent historical pairs were updated in a new "
        "frozen registry; the source registry was not overwritten. The main 963+50 batch was not sent.", "",
        "## Rejudged pair changes", "", "| Dimension | Mean delta | Up | Exact | Down | |delta|>=2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dim, row in dimension_summary.items():
        result_lines.append(f"| {dim} | {row['mean_delta']:+.3f} | {row['up_n']} | {row['exact_n']} | "
                            f"{row['down_n']} | {row['two_or_more_change_n']} |")
    result_lines += ["", f"Mean utility delta: **{summary['utility']['mean_delta']:+.4f}** "
                     f"(95% CI {summary['utility']['bootstrap_95ci']}); increased/tied/decreased "
                     f"= {summary['utility']['increased_n']}/{summary['utility']['tied_n']}/{summary['utility']['decreased_n']}.", "",
                     f"Threshold crossings: {threshold_counts}.", "", "## Realized main-system impact", "",
                     "| System | Complete n | Coverage | Delta Utility@8 | Delta nDCG@8 | Delta >=4 | Delta >=4.5 | Delta >=5 |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|" ]
    for system in important_systems:
        row = system_impacts[system]
        result_lines.append(
            f"| {system} | {row['complete_query_n']} | {row['judgment_coverage_at8']:.3f} | "
            f"{row['mean_utility_at8']['delta'] or 0.0:+.4f} | {row['common_idcg_ndcg_at8']['delta'] or 0.0:+.4f} | "
            f"{row['acceptable_count_at8']['delta'] or 0.0:+.4f} | {row['useful_count_at8']['delta'] or 0.0:+.4f} | "
            f"{row['high_quality_count_at8']['delta'] or 0.0:+.4f} |")
    result_lines += ["", f"Frozen augmented registry SHA256: `{registry_manifest['augmented_registry']['sha256']}`.", "",
                     "All system metrics use current exact complete-query coverage. Missing main-batch judgments remain "
                     "unjudged and were not assigned zero.", ""]
    (out / "fusion_url_reevaluation_results.md").write_text("\n".join(result_lines), encoding="utf-8")
    verdict = {
        "verdict": "URL_SEMANTIC_REEVALUATION_COMPLETE",
        "control_stability": stability["verdict"], "registry_updated": True,
        "registry_sha256": registry_manifest["augmented_registry"]["sha256"],
        "rejudged_pairs": 21, "controls": 20, "main_batch_sent": False,
    }
    _write_json(out / "fusion_url_reevaluation_final_verdict.json", verdict)
    payload_manifest.update({
        "authorization_status": "AUTHORIZED_COMPLETED_ANALYZED",
        "analysis_completed_at": _now(), "control_stability_verdict": stability["verdict"],
        "augmented_registry": registry_manifest["augmented_registry"],
        "main_batch_sent": False,
    })
    _atomic_write_json(out / "fusion_url_reevaluation_payload_manifest.json", payload_manifest)
    return verdict


FUSION_STAGE1_AUTHORIZED_SHA256 = "a5f5f78c483109461a1798ec26bd6db272509350ea451326c8e3c95e7cd12778"


def _validate_fusion_stage1_main_batch(root: Path, cfg: dict) -> tuple[list[dict], list[dict], dict]:
    out = _resolve(root, cfg["output_dir"])
    manifest = _read_json(out / "fusion_stage1_semantic_redacted_payload_manifest.json")
    blind_path = Path(manifest["blinded_payload"]["path"])
    admin_path = Path(manifest["admin_payload"]["path"])
    if _sha(blind_path) != FUSION_STAGE1_AUTHORIZED_SHA256:
        raise ValueError("authorized fusion Stage-1 payload SHA256 mismatch")
    if manifest["blinded_payload"]["sha256"] != FUSION_STAGE1_AUTHORIZED_SHA256:
        raise ValueError("fusion Stage-1 manifest SHA256 mismatch")
    if (manifest["residual_items"], manifest["anchor_items"], manifest["total_items"]) != (963, 50, 1013):
        raise ValueError("fusion Stage-1 963+50 composition mismatch")
    if manifest["remaining_raw_url_hits"] != 0 or any(manifest["remaining_direct_identifier_hits"].values()):
        raise ValueError("fusion Stage-1 semantic redaction gate failed")
    if manifest.get("url_representation_rules_sha256") != _hash_json({
        "version": "semantic-url-representation-v1",
        "categories": URL_CATEGORIES,
        "institution_retention": "public institution name only",
        "forbidden_url_components": ["clickable_url", "domain", "path", "query_string", "username", "tracking"],
    }):
        raise ValueError("fusion Stage-1 URL semantic rule hash mismatch")
    expected_model = "bedrock:us.meta.llama3-3-70b-instruct-v1:0"
    if manifest["judge"]["model"] != expected_model or cfg["judge"]["model"] != expected_model:
        raise ValueError("fusion Stage-1 model differs from frozen Meta Llama 3.3 70B")
    if manifest["judge"]["aws_region"] != "us-east-1" or cfg["judge"]["aws_region"] != "us-east-1":
        raise ValueError("fusion Stage-1 region differs from frozen us-east-1")
    merge_base = _resolve(root, cfg["fusion_stage1_merge_base_registry"])
    expected_base_sha = "353fcb5edce96bd22594d822691476810142616fd47092ff011dcd2fadb8f74a"
    if _sha(merge_base) != expected_base_sha:
        raise ValueError("fusion Stage-1 URL-semantic merge-base SHA mismatch")
    blind = _read_jsonl(blind_path); admin = _read_jsonl(admin_path)
    if len(blind) != 1013 or len(admin) != 1013:
        raise ValueError("fusion Stage-1 blind/ADMIN row count mismatch")
    blind_by_index = {int(row["item_index"]): row for row in blind}
    admin_by_index = {int(row["item_index"]): row for row in admin}
    if set(blind_by_index) != set(range(1, 1014)) or set(admin_by_index) != set(blind_by_index):
        raise ValueError("fusion Stage-1 blind/ADMIN index mismatch")
    item_types = Counter(str(row["item_type"]) for row in admin)
    if item_types != Counter({"residual": 963, "calibration_anchor": 50}):
        raise ValueError(f"fusion Stage-1 ADMIN composition mismatch: {item_types}")
    forbidden = {"query_id", "comment_id", "system", "rank", "source", "old_utility",
                 "item_type", "url_category", "anchor"}
    for index, row in blind_by_index.items():
        if set(row) != {"item_index", "payload", "payload_sha256"}:
            raise ValueError(f"fusion Stage-1 blind row {index} has unexpected fields")
        payload = row["payload"]
        if set(payload) != {"query_text", "comment_text"} or forbidden & set(payload):
            raise ValueError(f"fusion Stage-1 blind row {index} payload allowlist failed")
        if URL_PATTERN.search(payload["query_text"]) or URL_PATTERN.search(payload["comment_text"]):
            raise ValueError(f"fusion Stage-1 blind row {index} retains raw URL")
        rendered_sha = _hash_json({**payload, "facets_json": {}})
        if rendered_sha != row["payload_sha256"] or rendered_sha != admin_by_index[index]["payload_sha256"]:
            raise ValueError(f"fusion Stage-1 payload hash mismatch at item {index}")
    manifest["merge_base_registry"] = {"path": str(merge_base), "sha256": expected_base_sha,
                                        "pair_n": 4442, "url_semantic_representation": True}
    manifest["url_semantic_placeholder_occurrences"] = int(manifest["replacement_counts"].get("url", 0))
    return [blind_by_index[index] for index in sorted(blind_by_index)], [
        admin_by_index[index] for index in sorted(admin_by_index)], manifest


def _execute_blind_judging(
        blind: list[dict], admin: list[dict], cfg: dict, prompt_sha: str, batch_id: str,
        checkpoint_dir: Path) -> tuple[list[dict], list[dict], list[dict], int]:
    from utility_scoring.annotation.run_top3_residual_judging import judge_payload

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    blind_by_index = {int(row["item_index"]): row for row in blind}
    admin_by_index = {int(row["item_index"]): row for row in admin}
    done = {}
    for path in checkpoint_dir.glob("item_*.json"):
        checkpoint = _read_json(path); index = int(checkpoint["item_index"])
        if checkpoint.get("validation_status") != "valid":
            continue
        if checkpoint["payload_sha256"] != blind_by_index[index]["payload_sha256"]:
            raise ValueError(f"checkpoint payload hash mismatch at item {index}")
        done[index] = checkpoint
    pending = [index for index in sorted(blind_by_index) if index not in done]

    def evaluate(index: int) -> tuple[int, dict, dict]:
        blind_row = blind_by_index[index]; admin_row = admin_by_index[index]
        raw, valid = judge_payload(
            {**blind_row["payload"], "facets_json": {}}, str(admin_row["item_type"]),
            batch_id, prompt_sha,
            identity={"item_index": index, "query_id": str(admin_row["query_id"]),
                      "comment_id": str(admin_row["comment_id"])},
        )
        return index, raw, valid

    failures = []; completed_this_run = 0
    with ThreadPoolExecutor(max_workers=max(1, int(cfg["judge"].get("workers", 4)))) as executor:
        futures = {executor.submit(evaluate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                item_index, raw, valid = future.result()
                checkpoint = {"item_index": item_index,
                              "payload_sha256": blind_by_index[item_index]["payload_sha256"],
                              "raw": raw, "validated": valid, "validation_status": "valid",
                              "completed_at": _now()}
                _atomic_write_json(checkpoint_dir / f"item_{item_index:04d}.json", checkpoint)
                done[item_index] = checkpoint; completed_this_run += 1
            except Exception as exc:
                failures.append({"item_index": index, "error": f"{type(exc).__name__}: {exc}"})
    raw_rows = [done[index]["raw"] for index in sorted(done)]
    valid_rows = [done[index]["validated"] for index in sorted(done)]
    return raw_rows, valid_rows, failures, completed_this_run


def run_fusion_stage1_judging(root: Path, cfg: dict) -> dict:
    """Run only the authorized semantic-redacted 963 residual + 50 anchor batch."""
    out = _resolve(root, cfg["output_dir"])
    blind, admin, manifest = _validate_fusion_stage1_main_batch(root, cfg)
    prompt_path = Path(manifest["prompt"]["path"]); prompt_sha = manifest["prompt"]["sha256"]
    if _sha(prompt_path) != prompt_sha:
        raise ValueError("utility-v2 prompt changed after fusion Stage-1 payload freeze")
    run_manifest_path = out / "fusion_stage1_semantic_run_manifest.json"
    checkpoint_dir = out / "fusion_stage1_semantic_judging_checkpoints"
    already = len(list(checkpoint_dir.glob("item_*.json"))) if checkpoint_dir.exists() else 0
    run_manifest = {
        "schema": "fusion-stage1-semantic-run-manifest-v1", "created_at": _now(),
        "status": "RUNNING", "authorized_payload_sha256": FUSION_STAGE1_AUTHORIZED_SHA256,
        "authorized_items": 1013, "residual_items": 963, "anchor_items": 50,
        "external_destination": "AWS Bedrock us-east-1", "model": cfg["judge"]["model"],
        "protocol": "utility-v2", "prompt_sha256": prompt_sha,
        "judge_visible_fields": ["query_text", "comment_text"],
        "admin_file_authorized_for_external_send": False,
        "url_semantic_representation": True,
        "merge_base_registry": manifest["merge_base_registry"],
        "already_checkpointed": already, "main_batch_only": True,
    }
    _atomic_write_json(run_manifest_path, run_manifest)
    raw_rows, valid_rows, failures, completed_this_run = _execute_blind_judging(
        blind, admin, cfg, prompt_sha, str(cfg["judge"]["batch_id"]), checkpoint_dir)
    _write_jsonl(out / "fusion_stage1_semantic_judgments_raw.jsonl", raw_rows)
    _write_jsonl(out / "fusion_stage1_semantic_judgments_validated.jsonl", valid_rows)
    _write_jsonl(out / "fusion_stage1_semantic_judgment_failures.jsonl", failures)
    complete = len(valid_rows) == 1013 and not failures
    run_manifest.update({"completed_at": _now(), "status": "COMPLETE" if complete else "INCOMPLETE",
                         "completed_this_invocation": completed_this_run,
                         "validated_total": len(valid_rows), "failed_this_invocation": len(failures),
                         "bedrock_calls_successfully_checkpointed": len(valid_rows)})
    _atomic_write_json(run_manifest_path, run_manifest)
    manifest.update({"authorization_status": "AUTHORIZED_AND_COMPLETED" if complete else "AUTHORIZED_INCOMPLETE",
                     "authorized_payload_sha256": FUSION_STAGE1_AUTHORIZED_SHA256,
                     "validated_items": len(valid_rows), "failed_items": len(failures),
                     "bedrock_calls": len(valid_rows),
                     "run_manifest": {"path": str(run_manifest_path), "sha256": _sha(run_manifest_path)}})
    _atomic_write_json(out / "fusion_stage1_semantic_redacted_payload_manifest.json", manifest)
    if not complete:
        raise RuntimeError(f"fusion Stage-1 judging incomplete: valid={len(valid_rows)}, failures={len(failures)}")
    return {"status": "COMPLETE", "validated_items": len(valid_rows),
            "completed_this_invocation": completed_this_run,
            "payload_sha256": FUSION_STAGE1_AUTHORIZED_SHA256}


def analyze_fusion_stage1(root: Path, cfg: dict) -> dict:
    """Stability-gate, merge 963 residuals, and compute complete Stage-1 comparisons."""
    from utility_scoring.annotation.run_top3_residual_judging import DIMS_V2, anchor_report

    out = _resolve(root, cfg["output_dir"])
    blind, admin, manifest = _validate_fusion_stage1_main_batch(root, cfg)
    validated = _read_jsonl(out / "fusion_stage1_semantic_judgments_validated.jsonl")
    if len(validated) != 1013 or any(row.get("validation_status") != "valid" for row in validated):
        raise ValueError("fusion Stage-1 analysis requires 1,013 valid judgments")
    valid_by_index = {int(row["item_index"]): row for row in validated}
    blind_by_index = {int(row["item_index"]): row for row in blind}
    admin_by_index = {int(row["item_index"]): row for row in admin}
    if set(valid_by_index) != set(admin_by_index):
        raise ValueError("fusion Stage-1 validated/ADMIN item mismatch")
    base_path = _resolve(root, cfg["fusion_stage1_merge_base_registry"])
    base_rows = _read_jsonl(base_path); _, base_registry = complete_utility_v2_rows(base_rows)
    anchors = [valid_by_index[index] for index in sorted(valid_by_index)
               if admin_by_index[index]["item_type"] == "calibration_anchor"]
    residuals = [valid_by_index[index] for index in sorted(valid_by_index)
                 if admin_by_index[index]["item_type"] == "residual"]
    if len(anchors) != 50 or len(residuals) != 963:
        raise ValueError("fusion Stage-1 validated 963+50 composition mismatch")
    stability = anchor_report(base_registry, anchors)
    _write_json(out / "fusion_stage1_semantic_anchor_stability.json", stability)
    if stability["verdict"] not in {"STABLE", "STABLE_WITH_MINOR_DRIFT"}:
        verdict = {"verdict": "LABEL_DRIFT_INCONCLUSIVE", "anchor_verdict": stability["verdict"],
                   "registry_merged": False}
        _write_json(out / "fusion_stage1_semantic_final_verdict.json", verdict)
        return verdict
    residual_by_pair = {(str(row["query_id"]), str(row["comment_id"])): row for row in residuals}
    if len(residual_by_pair) != 963 or set(residual_by_pair) & set(base_registry):
        raise ValueError("fusion Stage-1 residual identity/overlap gate failed")
    new_rows = []
    for index, admin_row in admin_by_index.items():
        if admin_row["item_type"] != "residual":
            continue
        pair = (str(admin_row["query_id"]), str(admin_row["comment_id"]))
        judged = valid_by_index[index]; payload = blind_by_index[index]["payload"]
        new_rows.append({
            "query_id": pair[0], "comment_id": pair[1],
            "query_text": payload["query_text"], "comment_text": payload["comment_text"],
            **{f"label_{dim}": int(judged["validated_scores"][dim]) for dim in DIMS_V2},
            "utility": float(judged["utility"]), "rationale": judged.get("rationale", ""),
            "judge_model": judged["model"], "judge_id": "utility-v2",
            "judge_version": judged["judge_version"], "batch_id": cfg["judge"]["batch_id"],
            "judgment_source": "fusion_stage1_semantic_residual",
            "label_role": "LLM simulated-user silver; not human gold",
            "validation_status": "valid", "url_semantic_representation_version": "semantic-url-representation-v1",
            "payload_sha256": admin_row["payload_sha256"],
        })
    augmented_rows = base_rows + sorted(new_rows, key=lambda row: (row["query_id"], row["comment_id"]))
    complete_rows, augmented_registry = complete_utility_v2_rows(augmented_rows)
    if len(complete_rows) != 5405 or len(augmented_registry) != 5405:
        raise ValueError("fusion Stage-1 augmented registry completeness failed")
    registry_path = out / "fusion_stage1_semantic_augmented_registry.jsonl"
    _write_jsonl(registry_path, augmented_rows)
    registry_manifest = {
        "schema": "fusion-stage1-semantic-augmented-registry-manifest-v1", "created_at": _now(),
        "status": "FROZEN", "base_registry": {"path": str(base_path), "sha256": _sha(base_path),
                                                      "pair_n": len(base_rows)},
        "augmented_registry": {"path": str(registry_path), "sha256": _sha(registry_path), "pair_n": 5405},
        "new_residual_pairs": 963, "anchor_scores_written": 0, "old_rows_overwritten": 0,
        "unjudged_assigned_zero": False, "anchor_stability": stability["verdict"],
        "url_semantic_representation": True,
    }
    _write_json(out / "fusion_stage1_semantic_augmented_registry_manifest.json", registry_manifest)
    qrels_by_query: dict[str, dict[str, float]] = defaultdict(dict)
    for (qid, cid), row in augmented_registry.items():
        qrels_by_query[qid][cid] = float(row["utility"])
    data = load_inputs(root, cfg)
    frozen_rows = _read_jsonl(out / "fusion_all_system_per_query.jsonl")
    scored_rows = []
    for row in frozen_rows:
        qid = str(row["query_id"]); ids = [str(cid) for cid in row["comment_ids"]]
        scored_rows.append(_system_row(row["backend"], row["system"], qid, ids,
                                       augmented_registry, dict(qrels_by_query), data["qtypes"][qid],
                                       row.get("metadata")))
    metrics = aggregate_metrics(scored_rows); comparisons = paired_comparisons(scored_rows, cfg)
    _write_jsonl(out / "fusion_stage1_semantic_system_per_query.jsonl", scored_rows)
    _write_json(out / "fusion_stage1_semantic_system_metrics.json", metrics)
    _write_json(out / "fusion_stage1_semantic_paired_comparisons.json", comparisons)
    first_eight = {(left, right) for left, right in (
        ("CTS-Graph-Linear-q2", "Dense-Top8"), ("CTS-Graph-Linear-q2", "RRF-DenseGraph"),
        ("CTS-Graph-Linear-q2", "CC-DenseGraph"), ("CTS-Graph-Linear-q2", "Global-Linear"),
        ("CTS-Graph-Linear-q2", "CTS-Graph-Raw-q2"),
        ("CTS-Graph-Linear-q2", "CTS-Deep-Linear-q2"),
        ("CTS-Graph-Linear-q1", "CTS-Graph-Linear-q2"),
        ("CTS-Graph-Linear-q4", "CTS-Graph-Linear-q2"))}
    primary_rows = [row for row in comparisons if (row["left"], row["right"]) in first_eight]
    if any(row["exact_common_n"] != 100 for row in primary_rows):
        raise ValueError("fusion Stage-1 primary contrast did not reach exact paired n=100")
    summary = {"anchor_stability": stability, "metrics": metrics,
               "primary_comparisons": primary_rows, "registry": registry_manifest,
               "development_only": True, "llm_silver": True, "frozen_test_read": False}
    _write_json(out / "fusion_stage1_semantic_results.json", summary)
    lines = ["# Fusion Stage-1 semantic-redacted final results", "",
             f"Anchor verdict: **{stability['verdict']}**. Registry: 4,442 + 963 = 5,405 complete pairs.", "",
             "All eight preregistered primary contrasts have exact paired query n=100. URL semantic placeholders "
             "were used consistently for historical and new judgments; no ADMIN field was Judge-visible.", "",
             "Detailed system metrics and paired bootstrap intervals are authoritative in the JSON artifacts. "
             "Development/validation LLM-silver only; frozen test was not read.", ""]
    (out / "fusion_stage1_semantic_experiment_results.md").write_text("\n".join(lines), encoding="utf-8")
    verdict = {"verdict": "FUSION_STAGE1_COMPLETE", "anchor_verdict": stability["verdict"],
               "registry_merged": True, "registry_sha256": registry_manifest["augmented_registry"]["sha256"],
               "primary_contrasts_exact_n": 100, "frozen_test_read": False}
    _write_json(out / "fusion_stage1_semantic_final_verdict.json", verdict)
    manifest.update({"authorization_status": "AUTHORIZED_COMPLETED_ANALYZED",
                     "analysis_completed_at": _now(), "anchor_stability_verdict": stability["verdict"],
                     "augmented_registry": registry_manifest["augmented_registry"]})
    _atomic_write_json(out / "fusion_stage1_semantic_redacted_payload_manifest.json", manifest)
    return verdict


def _residual_pair_set(report: dict, scope: str) -> set[tuple[str, str]]:
    return {(str(row["query_id"]), str(row["comment_id"]))
            for row in report[scope]["residual_pairs"]}


def _expected_coverage(rows: list[dict], judged: set[tuple[str, str]],
                       contrasts: Iterable[tuple[str, str, str]] = PRIMARY_CONTRASTS) -> list[dict]:
    lookup = {(row["backend"], row["system"], row["query_id"]): row for row in rows}
    qids = sorted({row["query_id"] for row in rows})
    records = []
    for backend in BACKENDS:
        for contrast, left, right in contrasts:
            left_complete = right_complete = paired = 0
            left_slots = right_slots = joint_slots = 0
            missing_left_only = missing_right_only = missing_both = output_failure = 0
            for qid in qids:
                left_row = lookup[(backend, left, qid)]; right_row = lookup[(backend, right, qid)]
                left_ok_output = len(left_row["comment_ids"]) == 8 and len(set(left_row["comment_ids"])) == 8
                right_ok_output = len(right_row["comment_ids"]) == 8 and len(set(right_row["comment_ids"])) == 8
                if not left_ok_output or not right_ok_output:
                    output_failure += 1
                left_judged = sum((qid, cid) in judged for cid in left_row["comment_ids"])
                right_judged = sum((qid, cid) in judged for cid in right_row["comment_ids"])
                left_slots += left_judged; right_slots += right_judged
                joint_slots += left_judged + right_judged
                left_ok = left_ok_output and left_judged == 8
                right_ok = right_ok_output and right_judged == 8
                left_complete += left_ok; right_complete += right_ok; paired += left_ok and right_ok
                if not left_ok and right_ok:
                    missing_left_only += 1
                elif left_ok and not right_ok:
                    missing_right_only += 1
                elif not left_ok and not right_ok:
                    missing_both += 1
            records.append({
                "backend": backend, "contrast": contrast, "left": left, "right": right,
                "query_n": len(qids), "left_complete_query_n": left_complete,
                "right_complete_query_n": right_complete, "exact_paired_query_n": paired,
                "left_mean_judged_top8_coverage": left_slots / (8 * len(qids)),
                "right_mean_judged_top8_coverage": right_slots / (8 * len(qids)),
                "joint_mean_judged_top8_coverage": joint_slots / (16 * len(qids)),
                "paired_complete_rate": paired / len(qids),
                "incomplete_reason_counts": {
                    "left_only_incomplete": missing_left_only,
                    "right_only_incomplete": missing_right_only,
                    "both_incomplete": missing_both,
                    "output_failure": output_failure,
                },
            })
    return records


def _system_complete_counts(rows: list[dict], judged: set[tuple[str, str]]) -> dict[str, int]:
    result = Counter()
    for row in rows:
        if len(row["comment_ids"]) == 8 and len(set(row["comment_ids"])) == 8 and all(
                (row["query_id"], cid) in judged for cid in row["comment_ids"]):
            result[f"{row['backend']}:{row['system']}"] += 1
    return dict(result)


def _direct_identifier_field_audit(rows: list[dict]) -> dict:
    forbidden = {"author", "url", "permalink", "post_id", "comment_id", "timestamp",
                 "created", "created_utc", "source", "source_metadata"}
    found = Counter()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in forbidden:
                    found[str(key).lower()] += 1
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(rows)
    return {"forbidden_field_occurrences": dict(found), "verdict": "PASS" if not found else "FAIL"}


def run_batch_audit(root: Path, cfg: dict) -> dict:
    """Audit label-budget marginal value and create a new redacted batch only."""
    out = _resolve(root, cfg["output_dir"])
    report = _read_json(out / "fusion_stage1_residual_report.json")
    rows = _read_jsonl(out / "fusion_all_system_per_query.jsonl")
    data = load_inputs(root, cfg)
    current = set(data["qrels"])
    scopes = {
        "current": set(),
        "add_901_core": _residual_pair_set(report, "core_deployable_union"),
        "add_963_substantive": _residual_pair_set(report, "substantive_stage1_union"),
        "add_1238_complete_nondegenerate": _residual_pair_set(report, "complete_nondegenerate_stage1_union"),
    }
    if not (scopes["add_901_core"] <= scopes["add_1238_complete_nondegenerate"]
            and scopes["add_963_substantive"] <= scopes["add_1238_complete_nondegenerate"]):
        raise ValueError("901/963 residual scopes must both be subsets of 1,238")
    extra_275 = scopes["add_1238_complete_nondegenerate"] - scopes["add_963_substantive"]
    if len(extra_275) != 275:
        raise ValueError(f"expected 275-pair increment, got {len(extra_275)}")

    complete_source = {(str(row["query_id"]), str(row["comment_id"])): row["introduced_by"]
                       for row in report["complete_nondegenerate_stage1_union"]["residual_pairs"]}
    system_contributions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for pair in extra_275:
        for system in complete_source[pair]:
            system_contributions[system].add(pair)
    system_contribution_rows = []
    for system, pairs in sorted(system_contributions.items()):
        suffix = system.split(":", 1)[1]
        if suffix.startswith("CC-"):
            direct = [f"{suffix} vs CC-DenseGraph", "normalization sensitivity"]
            question = "CC score normalization robustness"
        elif "Raw" in suffix and "q4" in suffix:
            direct = [suffix.replace("Raw", "Raw") + " vs corresponding Linear-q4",
                      suffix + " vs corresponding Raw-q2"]
            question = "raw-selector quota and source sensitivity"
        elif "Linear-q4" in suffix:
            direct = [suffix + " vs corresponding Linear-q2", "Deep-q4 vs Graph-q4"]
            question = "quota/source sensitivity beyond the primary graph arm"
        else:
            direct = [suffix + " vs matched selector/source control"]
            question = "secondary source-by-selector completeness"
        system_contribution_rows.append({
            "system": system, "additional_unique_pairs_served": len(pairs),
            "affected_query_n": len({qid for qid, _ in pairs}),
            "direct_contrasts": direct, "research_question": question,
        })

    category_pairs = {"q4": set(), "Global-Linear": set(), "CC-normalization": set(), "other": set()}
    for pair in extra_275:
        memberships = complete_source[pair]
        if any("q4" in name for name in memberships):
            category_pairs["q4"].add(pair)
        elif any("Global-Linear" in name for name in memberships):
            category_pairs["Global-Linear"].add(pair)
        elif any(":CC-" in name for name in memberships):
            category_pairs["CC-normalization"].add(pair)
        else:
            category_pairs["other"].add(pair)

    coverage = {
        "schema": "fusion-stage1-expected-pairwise-coverage-v1", "created_at": _now(),
        "assumption": "listed residual pairs receive valid complete utility-v2 judgments; no missing value is imputed",
        "scenarios": {}, "external_model_calls": 0, "frozen_test_read": False,
    }
    for name, additions in scopes.items():
        coverage["scenarios"][name] = {
            "added_residual_pairs": len(additions),
            "expected_total_complete_judged_pairs": len(current | additions),
            "contrasts": _expected_coverage(rows, current | additions),
        }
    _write_json(out / "fusion_stage1_expected_pairwise_coverage.json", coverage)

    first_six = {name for name, _, _ in PRIMARY_CONTRASTS[:6]}
    first_six_963 = [row for row in coverage["scenarios"]["add_963_substantive"]["contrasts"]
                     if row["contrast"] in first_six]
    substantive_sufficient = all(row["exact_paired_query_n"] >= 95 for row in first_six_963)
    all_primary_963 = coverage["scenarios"]["add_963_substantive"]["contrasts"]
    complete_963 = all(row["exact_paired_query_n"] == 100 for row in all_primary_963)

    counts_963 = _system_complete_counts(rows, current | scopes["add_963_substantive"])
    counts_1238 = _system_complete_counts(rows, current | scopes["add_1238_complete_nondegenerate"])
    marginal_system_queries = [
        {"system": key, "complete_queries_at_963": counts_963.get(key, 0),
         "complete_queries_at_1238": counts_1238.get(key, 0),
         "new_complete_queries": counts_1238.get(key, 0) - counts_963.get(key, 0)}
        for key in sorted(set(counts_963) | set(counts_1238))
        if counts_1238.get(key, 0) > counts_963.get(key, 0)
        and key.split(":", 1)[1] not in {"RRF-DenseDeep", "RRF-DenseGraphDeep"}
    ]
    secondary_contrasts = (
        ("CC-minmax-lambda0.5_vs_CC-DenseGraph", "CC-minmax-lambda0.5", "CC-DenseGraph"),
        ("CC-zscore-lambda0.5_vs_CC-DenseGraph", "CC-zscore-lambda0.5", "CC-DenseGraph"),
        ("CC-rank-percentile-lambda0.5_vs_CC-DenseGraph", "CC-rank_percentile-lambda0.5", "CC-DenseGraph"),
        ("CTS-Graph-Raw-q4_vs_Linear-q4", "CTS-Graph-Raw-q4", "CTS-Graph-Linear-q4"),
        ("CTS-Deep-Raw-q1_vs_Linear-q1", "CTS-Deep-Raw-q1", "CTS-Deep-Linear-q1"),
        ("CTS-Deep-Raw-q2_vs_Linear-q2", "CTS-Deep-Raw-q2", "CTS-Deep-Linear-q2"),
        ("CTS-Deep-Raw-q4_vs_Linear-q4", "CTS-Deep-Raw-q4", "CTS-Deep-Linear-q4"),
        ("CTS-Deep-Raw-q4_vs_Graph-Raw-q4", "CTS-Deep-Raw-q4", "CTS-Graph-Raw-q4"),
        ("CTS-Deep-Linear-q4_vs_Graph-Linear-q4", "CTS-Deep-Linear-q4", "CTS-Graph-Linear-q4"),
        ("CTS-Deep-Raw-q4_vs_Raw-q2", "CTS-Deep-Raw-q4", "CTS-Deep-Raw-q2"),
    )
    secondary_963 = _expected_coverage(rows, current | scopes["add_963_substantive"], secondary_contrasts)
    secondary_1238 = _expected_coverage(rows, current | scopes["add_1238_complete_nondegenerate"], secondary_contrasts)
    secondary_963_lookup = {(row["backend"], row["contrast"]): row for row in secondary_963}
    secondary_marginal = []
    for row in secondary_1238:
        old = secondary_963_lookup[(row["backend"], row["contrast"])]
        secondary_marginal.append({
            "backend": row["backend"], "contrast": row["contrast"],
            "paired_n_at_963": old["exact_paired_query_n"],
            "paired_n_at_1238": row["exact_paired_query_n"],
            "new_paired_queries": row["exact_paired_query_n"] - old["exact_paired_query_n"],
            "newly_reaches_95": old["exact_paired_query_n"] < 95 <= row["exact_paired_query_n"],
            "newly_complete_100": old["exact_paired_query_n"] < 100 == row["exact_paired_query_n"],
        })
    recommendation = ("AUTHORISE_963_SUBSTANTIVE" if substantive_sufficient and complete_963
                      else "AUTHORISE_1238_COMPLETE")

    old_manifest_path = out / "fusion_stage1_payload_manifest.json"
    old_manifest = _read_json(old_manifest_path)
    old_payload_path = Path(old_manifest["blinded_payload"]["path"])
    old_admin_path = Path(old_manifest["admin_payload"]["path"])
    old_hash_before = _sha(old_payload_path); old_admin_hash_before = _sha(old_admin_path)
    if old_hash_before != old_manifest["blinded_payload"]["sha256"]:
        raise ValueError("current unredacted payload already differs from its manifest")

    chosen = scopes["add_963_substantive"] if recommendation == "AUTHORISE_963_SUBSTANTIVE" else scopes[
        "add_1238_complete_nondegenerate"]
    chosen_scope = ("substantive_stage1_union" if recommendation == "AUTHORISE_963_SUBSTANTIVE"
                    else "complete_nondegenerate_stage1_union")
    source_rows = report[chosen_scope]["residual_pairs"]
    source_memberships = {(str(row["query_id"]), str(row["comment_id"])): row["introduced_by"]
                          for row in source_rows}
    anchors_raw = _read_jsonl(_resolve(root, cfg["calibration_anchors"]))
    anchors = {(str(row["query_id"]), str(row["comment_id"])) for row in anchors_raw}
    if len(anchors_raw) != 50 or len(anchors) != 50 or anchors & chosen:
        raise ValueError("anchor count/duplicate/residual overlap gate failed")
    queries = data["queries"]; corpus = data["corpus"]
    ordered = sorted(chosen) + sorted(anchors)
    redacted_rows, admin_rows = [], []
    replacement_counts: Counter[str] = Counter()
    replaced_item_n = 0
    sensitivity = {category: {"items": 0, "residual_items": 0, "anchor_items": 0,
                              "query_ids": set()} for category in SENSITIVITY_PATTERNS}
    total_characters_before = total_characters_after = 0
    for index, (qid, cid) in enumerate(ordered, 1):
        item_type = "calibration_anchor" if (qid, cid) in anchors else "residual"
        original_texts = {"query_text": queries[qid], "comment_text": corpus[cid]}
        for category, pattern in SENSITIVITY_PATTERNS.items():
            if any(pattern.search(text) for text in original_texts.values()):
                sensitivity[category]["items"] += 1
                sensitivity[category][f"{'anchor' if item_type == 'calibration_anchor' else 'residual'}_items"] += 1
                sensitivity[category]["query_ids"].add(qid)
        redacted_payload = {}; item_replaced = False
        for field, text in original_texts.items():
            redacted, field_counts = redact_direct_identifiers(text)
            total_characters_before += len(text); total_characters_after += len(redacted)
            if any(field_counts.values()):
                item_replaced = True
            replacement_counts.update(field_counts)
            redacted_payload[field] = redacted
        replaced_item_n += item_replaced
        payload_hash = _hash_json({**redacted_payload, "facets_json": {}})
        redacted_rows.append({"item_index": index, "payload": redacted_payload,
                              "payload_sha256": payload_hash})
        admin_rows.append({
            "item_index": index, "query_id": qid, "comment_id": cid, "item_type": item_type,
            "experiment_source": ("fixed_utility_v2_anchor" if item_type == "calibration_anchor"
                                  else f"fusion_{chosen_scope}_redacted"),
            "arm_memberships": source_memberships.get((qid, cid), []),
            "payload_sha256": payload_hash,
        })
    field_audit = _direct_identifier_field_audit(redacted_rows)
    residual_identifier_hits = Counter()
    for row in redacted_rows:
        for label, pattern, _ in DIRECT_IDENTIFIER_PATTERNS:
            for value in row["payload"].values():
                residual_identifier_hits[label] += len(pattern.findall(value))
    if field_audit["verdict"] != "PASS" or any(residual_identifier_hits.values()):
        raise ValueError("redacted payload still contains forbidden fields or identifier-pattern hits")

    redacted_path = out / "fusion_stage1_redacted_blinded_payload.jsonl"
    redacted_admin_path = out / "fusion_stage1_redacted_payload_ADMIN.jsonl"
    _write_jsonl(redacted_path, redacted_rows); _write_jsonl(redacted_admin_path, admin_rows)
    qtype_counts = Counter(data["qtypes"][qid] for qid, _ in anchors)
    anchor_values = [float(data["qrels"][(qid, cid)]["utility"]) for qid, cid in anchors]
    utility_bins = Counter("low_[1,3)" if value < 3 else "mid_[3,5)" if value < 5 else "high_[5,7]"
                           for value in anchor_values)
    anchors_existing = sum(pair in current for pair in anchors)
    sensitivity_public = {
        category: {"item_n": value["items"], "residual_item_n": value["residual_items"],
                   "anchor_item_n": value["anchor_items"], "unique_query_n": len(value["query_ids"])}
        for category, value in sensitivity.items()
    }
    minimisation = {
        "source_payload_unchanged": True, "source_payload_sha256": old_hash_before,
        "recommended_scope": chosen_scope, "residual_items": len(chosen), "anchor_items": len(anchors),
        "total_items": len(ordered), "field_audit": field_audit,
        "replacement_counts": dict(replacement_counts), "items_with_replacement": replaced_item_n,
        "remaining_identifier_pattern_hits": dict(residual_identifier_hits),
        "character_count_before": total_characters_before, "character_count_after": total_characters_after,
        "health_and_life_context_policy": "preserved; only deterministic likely direct-identifier spans replaced",
        "sensitivity_category_counts_only": sensitivity_public,
        "anchor_audit": {
            "existing_judgment_n": anchors_existing, "anchor_n": len(anchors),
            "utility_bins": dict(utility_bins), "utility_min": min(anchor_values),
            "utility_max": max(anchor_values), "utility_mean": statistics.fmean(anchor_values),
            "query_type_counts": dict(qtype_counts), "residual_overlap": len(anchors & chosen),
            "identity_in_blinded_payload": False, "identity_in_admin_only": True,
        },
    }
    prompt = project_config.prompt_path("evidence_card_judge_v2")
    redacted_manifest = {
        "schema": "fusion-stage1-redacted-payload-manifest-v1", "created_at": _now(),
        "recommendation": recommendation, "authorization_status": "PENDING_NEW_EXPLICIT_USER_APPROVAL",
        "authorization_applies_only_to_this_sha256": True,
        "residual_items": len(chosen), "anchor_items": len(anchors), "total_items": len(ordered),
        "blinded_payload": {"path": str(redacted_path), "sha256": _sha(redacted_path)},
        "admin_payload": {"path": str(redacted_admin_path), "sha256": _sha(redacted_admin_path)},
        "prompt": {"path": str(prompt), "sha256": _sha(prompt)}, "judge": cfg["judge"],
        "data_minimisation": minimisation, "duplicate_pairs": len(ordered) - len(set(ordered)),
        "bedrock_calls": 0, "frozen_test_read": False,
        "supersedes_for_future_authorisation_only": {
            "old_payload_path": str(old_payload_path), "old_payload_sha256": old_hash_before,
            "old_payload_modified": False,
        },
    }
    _write_json(out / "fusion_stage1_redacted_payload_manifest.json", redacted_manifest)

    extra_lines = [
        "# Fusion Stage-1 batch increment analysis", "",
        f"Recommendation: **`{recommendation}`**. No Bedrock call was made.", "",
        "## Nested batch sizes", "",
        "| Batch | Residual pairs | Role |", "|---|---:|---|",
        "| Core | 901 | First six primary contrasts; q4 remains short for 2 SBERT and 4 Cohere queries |",
        "| Substantive | 963 | All eight primary contrasts reach 100/100 on both backends |",
        "| Complete non-degenerate | 1,238 | Adds secondary normalization/source×selector completeness |", "",
        f"The 901 and 963 sets are not strictly nested: overlap={len(scopes['add_901_core'] & scopes['add_963_substantive'])}, "
        f"901-only={len(scopes['add_901_core'] - scopes['add_963_substantive'])}, and "
        f"963-only={len(scopes['add_963_substantive'] - scopes['add_901_core'])}. The net size difference is 62, "
        "but moving from the 901 definition to the 963 definition means adding 76 primary-scope pairs and dropping "
        "14 secondary Deep-Raw-q2 pairs.", "",
        "## The extra 275 pairs: 1,238 minus 963", "",
        "| System | Extra unique pairs | Queries | Direct use | Research question |", "|---|---:|---:|---|---|",
    ]
    for row in system_contribution_rows:
        extra_lines.append(f"| {row['system']} | {row['additional_unique_pairs_served']} | "
                           f"{row['affected_query_n']} | {'; '.join(row['direct_contrasts'])} | "
                           f"{row['research_question']} |")
    extra_lines += ["", "Pair-level marginal source (mutually exclusive priority): " + ", ".join(
        f"{key}={len(value)}" for key, value in category_pairs.items()) + ".", "",
        "The 275 pairs add **no new primary contrast** and do not change primary-contrast identifiability: "
        "all eight are already 100/100 paired under the 963 batch. Their value is secondary robustness and "
        "completion of source×selector×quota cells.", "", "## Secondary contrasts gained at 1,238", "",
        "| Backend | Secondary contrast | Paired n at 963 | Paired n at 1,238 | Added | Reaches 95? | Complete 100? |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in secondary_marginal:
        if row["new_paired_queries"] <= 0:
            continue
        extra_lines.append(f"| {row['backend']} | {row['contrast']} | {row['paired_n_at_963']} | "
                           f"{row['paired_n_at_1238']} | {row['new_paired_queries']} | "
                           f"{'yes' if row['newly_reaches_95'] else 'no'} | "
                           f"{'yes' if row['newly_complete_100'] else 'no'} |")
    extra_lines += ["", "No primary contrast is newly enabled. The rows above are secondary robustness contrasts; "
        "the 1,238 batch mainly converts their partial coverage into full 100-query coverage.", "",
        "## Complete system-query cells gained at 1,238", "",
        "| System | Complete at 963 | Complete at 1,238 | Added |", "|---|---:|---:|---:|",
    ]
    for row in marginal_system_queries:
        extra_lines.append(f"| {row['system']} | {row['complete_queries_at_963']} | "
                           f"{row['complete_queries_at_1238']} | {row['new_complete_queries']} |")
    extra_lines += ["", "## Decision", "",
                    "The 901 batch is adequate for the first six contrasts, but it leaves the preregistered q4-vs-q2 "
                    "comparison at 98/100 SBERT and 96/100 Cohere. Relative to the 901 definition, the 963 definition "
                    "adds 76 primary-scope pairs and omits 14 Deep-Raw-q2 secondary pairs (net +62); this closes both "
                    "q4 comparisons to 100/100 and retains the equal-weight RRF diagnostic. The next 275 pairs do not improve "
                    "any primary contrast. Therefore the marginal-cost/identifiability choice is "
                    f"`{recommendation}`.", ""]
    (out / "fusion_stage1_batch_increment_analysis.md").write_text("\n".join(extra_lines), encoding="utf-8")

    replacement_lines = [f"{key}: {value}" for key, value in sorted(replacement_counts.items())]
    sensitivity_lines = [
        f"| {key} | {value['item_n']} | {value['residual_item_n']} | {value['anchor_item_n']} | {value['unique_query_n']} |"
        for key, value in sensitivity_public.items()
    ]
    privacy_md = "\n".join([
        "# Fusion Stage-1 data minimisation audit", "",
        "Verdict: **PASS_WITH_DETERMINISTIC_REDACTION**. No external model or Bedrock call was used.", "",
        f"The new blind file contains {len(chosen)} residuals plus 50 anchors. It has no author, URL, permalink, "
        "post/comment ID, timestamp, or source-metadata field. Pair identity and anchor status exist only in the "
        "separate ADMIN file.", "", "## Direct-identifier replacement", "",
        f"Items with at least one replacement: **{replaced_item_n}**. Replacement counts:", "",
        *[f"- {line}" for line in replacement_lines], "",
        "Examples are intentionally schematic and reproduce no source text: `name@domain` -> `[EMAIL]`; "
        "`https://…` -> `[URL]`; `@handle`/`u/handle` -> `[USERNAME]`; digit groups -> `[PHONE]`; "
        "street-level location -> `[STREET_ADDRESS]`/`[POSTCODE]`.", "",
        "Health, medication, disability, relationship, work, education, financial and other life context was retained. "
        "Only spans matching the frozen direct-identifier patterns were replaced.", "",
        "## Extremely sensitive content: category counts only", "",
        "These are conservative keyword flags, not clinical labels. No matched text is reproduced.", "",
        "| Category | Items | Residual items | Anchor items | Unique queries |", "|---|---:|---:|---:|---:|",
        *sensitivity_lines, "", "## Anchor audit", "",
        f"- Existing complete judgments: {anchors_existing}/50.",
        f"- Utility bins: {dict(utility_bins)}; range {min(anchor_values):.2f}-{max(anchor_values):.2f}; "
        f"mean {statistics.fmean(anchor_values):.3f}.",
        f"- Query types: {dict(qtype_counts)}.",
        "- Residual overlap: 0; duplicate anchors: 0.",
        "- Anchor identity is absent from the blind file and retained only in ADMIN.", "",
        "## Immutability", "",
        f"The old payload remains at SHA256 `{old_hash_before}` and was not overwritten. The new redacted payload "
        f"has SHA256 `{redacted_manifest['blinded_payload']['sha256']}`. Any future authorization applies only to "
        "the new redacted hash.", "",
    ])
    (out / "fusion_stage1_data_minimisation_audit.md").write_text(privacy_md, encoding="utf-8")

    if _sha(old_payload_path) != old_hash_before or _sha(old_admin_path) != old_admin_hash_before:
        raise ValueError("old payload or ADMIN file changed during audit")
    return {
        "recommendation": recommendation, "core_residual": 901, "substantive_residual": 963,
        "complete_nondegenerate_residual": 1238, "increment_275": len(extra_275),
        "redacted_total_items": len(ordered),
        "redacted_payload_sha256": redacted_manifest["blinded_payload"]["sha256"],
        "old_payload_unchanged": True, "bedrock_calls": 0,
    }


def run_fixed(root: Path, cfg: dict) -> dict:
    out = _resolve(root, cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    data = load_inputs(root, cfg)
    candidate_rows = build_candidate_registry(data)
    _write_jsonl(out / "fusion_common_candidate_registry.jsonl", candidate_rows)
    _write_jsonl(out / "fusion_common_idcg_registry.jsonl", common_idcg_rows(data))
    fixed_rows, tuning = fixed_systems(data, cfg)
    _write_jsonl(out / "fusion_all_system_per_query.jsonl", fixed_rows)
    metrics = aggregate_metrics(fixed_rows); _write_json(out / "fusion_all_system_metrics.json", metrics)
    comparisons = paired_comparisons(fixed_rows, cfg); _write_json(out / "fusion_paired_comparisons.json", comparisons)
    _write_json(out / "fusion_hyperparameter_selection_by_fold.json", tuning)
    residuals, residuals_md = residual_report(fixed_rows, data["qrels"])
    _write_json(out / "fusion_stage1_residual_report.json", residuals)
    (out / "fusion_stage1_residual_report.md").write_text(residuals_md, encoding="utf-8")
    audit, audit_md = metric_audit(root, cfg, data, fixed_rows)
    _write_json(out / "fusion_authoritative_metric_manifest.json", audit)
    (out / "fusion_metric_version_audit.md").write_text(audit_md, encoding="utf-8")
    fold_manifest = {"source": str(_resolve(root, cfg["split_manifest"])),
                     "source_sha256": _sha(_resolve(root, cfg["split_manifest"])),
                     "split_unit": "query_id", "repeats": data["split"]["repeats"],
                     "folds": data["split"]["folds"], "outer_rows": len(data["split"]["rows"]),
                     "inner_folds_per_outer": 4, "query_overlap_violations": 0,
                     "pair_level_random_split": False}
    _write_json(out / "fusion_fold_manifest.json", fold_manifest)
    coverage = {key: {"complete_query_n": value["complete_query_n"],
                      "mean_judgment_coverage_at8": value["mean_judgment_coverage_at8"]}
                for key, value in metrics.items()}
    leakage = f"""# Fusion leakage audit

- Dataset: dev100-v2, 100 development queries; frozen test read: **false**.
- Candidate pool, texts, 4,442 judgments, query summaries, OOF predictions and 5x5 query folds were frozen before this ablation.
- Every tuned RRF/CC choice was selected only inside its outer training-query set using four inner validation buckets.
- Linear scores are the existing 5x5 query-grouped OOF predictions; utility is not an inference feature.
- Community replies were not used by fixed strategy construction or hyperparameter selection.
- Missing judgments were not set to zero. Exact complete-query coverage is stored in `fusion_all_system_metrics.json`.
"""
    (out / "fusion_leakage_audit.md").write_text(leakage, encoding="utf-8")
    manifest = {"schema": "fusion-strategy-stage1-manifest-v1", "created_at": _now(),
                "candidate_rows": len(candidate_rows), "system_query_rows": len(fixed_rows),
                "systems": len(metrics), "paired_comparisons": len(comparisons),
                "core_residual_pairs": residuals["core_deployable_union"]["unique_residual_pairs"],
                "substantive_stage1_residual_pairs": residuals["substantive_stage1_union"]["unique_residual_pairs"],
                "complete_nondegenerate_stage1_residual_pairs": residuals["complete_nondegenerate_stage1_union"]["unique_residual_pairs"],
                "all_stage1_residual_pairs": residuals["all_stage1_deployable_union"]["unique_residual_pairs"],
                "coverage": coverage, "external_model_calls": 0, "frozen_test_read": False}
    _write_json(out / "fusion_stage1_manifest.json", manifest)
    return manifest
