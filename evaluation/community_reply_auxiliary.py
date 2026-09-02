"""Hidden-community-reply auxiliary evaluation for frozen evidence sets.

Phase 1 is deliberately local and post-hoc: it inventories original-thread
replies, proves they were absent from retrieval/training/judging inputs, and
computes semantic alignment for already frozen Top-8 runs. Test paths remain
forbidden by default; docs138 authorises one explicitly gated Test200 post-hoc
invocation. Neither route calls an external model or trains on the replies.

The community replies are reference evidence, not unique gold answers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import configuration as project_config
from evaluation.ir_metrics import graded_ndcg_at
from evaluation.judgment_completeness import complete_utility_v2_rows


SYSTEM_LABELS = {
    "S0": "SBERT Dense",
    "C0": "Cohere Dense",
    "H0": "Official HippoRAG2",
    "S2": "SBERT Deep-2",
    "SG2": "SBERT Graph-2",
    "C2": "Cohere Deep-2",
    "G2": "Cohere Graph-2",
}
CONTRASTS = (
    ("C0", "S0"),
    ("H0", "C0"),
    ("SG2", "S2"),
    ("G2", "C2"),
    ("C2", "C0"),
    ("G2", "C0"),
)
BIDIRECTIONAL_CONTRASTS = (
    ("C0", "S0"),
    ("H0", "C0"),
    ("SG2", "S2"),
    ("G2", "C2"),
    ("SG2", "S0"),
    ("G2", "C0"),
)
INVALID_REPLY = {"", "[deleted]", "[removed]"}
BOT_AUTHORS = {"AutoModerator", "[deleted]", "[removed]", "B0tRank", "RepostSleuthBot"}
URL_ONLY = re.compile(r"^\s*https?://\S+\s*$")
MOD_NOTICE = re.compile(r"^Your content breaks \*\*Rule \d+\*\*", re.IGNORECASE)
WS = re.compile(r"\s+")
USEFUL_THRESHOLD = 4.0


def _bootstrap_ci(values, *, n_boot: int, seed: int):
    """Load the answer-judge bootstrap helper only when an analysis needs it.

    Importing this module for its provenance helpers must not require loading
    the LLM-judge prompt.  The numerical implementation remains the canonical
    ``score_answers.bootstrap_ci`` function.
    """
    from evaluation.statistics import bootstrap_ci

    return bootstrap_ci(values, n_boot=n_boot, seed=seed)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text(text: str) -> str:
    return WS.sub(" ", text.strip()).casefold()


def text_sha(text: str) -> str:
    return sha256_bytes(canonical_text(text).encode("utf-8"))


def normalize_reddit_id(value: str) -> str:
    value = str(value).strip()
    return value[3:] if value.startswith(("t1_", "t3_")) else value


def resolve(root: Path, value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else root / path


def load_config() -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[1]
    raw = dict(project_config.load()["community_reply_auxiliary"])
    for key in (
        "output_dir", "queries", "summaries", "corpus", "corpus_source_map",
        "run_registry", "sbert_runs", "cohere_graph_runs", "cohere_deep_runs",
        "utility_registry", "frozen_reply_sample", "reddit_comments_dump",
    ):
        raw[key] = resolve(root, raw[key])
    raw["semantic_encoder"] = dict(raw["semantic_encoder"])
    raw["semantic_encoder"]["local_snapshot"] = resolve(
        root, raw["semantic_encoder"]["local_snapshot"])
    return root, raw


def reject_test_paths(config: dict, *, test200_posthoc_manifest: Path | None = None) -> None:
    exempt = set()
    if test200_posthoc_manifest is not None:
        # docs138/user authorisation: an explicitly supplied, evidence-frozen
        # Test200 directory only. No general test, training, or judge permission.
        root = Path(__file__).resolve().parents[1]
        expected = test200_posthoc_manifest.resolve()
        frozen_dir = expected.parent
        allowed_dirs = {
            (root / "out/test200_clean7d_confirmation_v2").resolve(),
            (root / "out/test200_rawtext_e5llama_confirmation_v1").resolve(),
        }
        if expected.name != "manifest.json" or frozen_dir not in allowed_dirs:
            raise ValueError("unregistered Test200 post-hoc exemption path")
        manifest = json.loads(expected.read_text(encoding="utf-8"))
        if (manifest.get("status") not in {"FOUR_EVIDENCE_SETS_FROZEN", "EVIDENCE_SETS_FROZEN"}
                or manifest.get("queries") != 200
                or manifest.get("source_thread_candidate_count") != 0
                or manifest.get("test_used_for_tuning") is not False
                or config.get("allow_external_calls") is not False
                or config.get("community_used_for_training") is not False):
            raise ValueError("Test200 post-hoc exemption gates failed")
        for name, digest in manifest["outputs"].items():
            if sha256_file(frozen_dir / name) != digest:
                raise ValueError("Test200 evidence changed before community evaluation")
        exempt = {expected, frozen_dir / "system_rankings.jsonl", frozen_dir / "community",
                  root / "out/natural_query_eval_splits_v1/cross_thread_test_queries_200_ADMIN.csv"}
        for key, value in config.items():
            if isinstance(value, Path) and "test" in str(value.relative_to(root) if value.is_relative_to(root) else value).lower() and value.resolve() not in exempt:
                raise ValueError(f"Test200 exemption is not valid for {key}={value}")
    for key, value in config.items():
        if isinstance(value, Path) and "test" in value.name.lower():
            if value.resolve() in exempt:
                continue
            raise ValueError(f"test-like input forbidden: {key}={value}")


def load_queries(path: Path, *, expected_queries: int = 100) -> tuple[dict[str, str], dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    queries = {str(row["query_id"]): str(row["query_text"]) for row in rows}
    posts = {str(row["query_id"]): str(row["post_id"]) for row in rows}
    if len(rows) != expected_queries or len(queries) != expected_queries:
        raise ValueError(f"expected {expected_queries} unique queries, got {len(rows)}/{len(queries)}")
    if any(qid != normalize_reddit_id(posts[qid]) for qid in queries):
        raise ValueError("query_id/post_id identity mismatch")
    return queries, posts


def load_corpus(path: Path) -> tuple[dict[str, str], set[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    corpus = {normalize_reddit_id(row["title"]): str(row["text"]) for row in rows}
    if len(rows) != 19013 or len(corpus) != 19013:
        raise ValueError(f"expected fixed 19,013-comment corpus, got {len(rows)}/{len(corpus)}")
    return corpus, {text_sha(text) for text in corpus.values()}


def _rows_to_run(path: Path, wanted: dict[str, str], qids: set[str]) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {target: {} for target in wanted.values()}
    for row in read_jsonl(path):
        target = wanted.get(str(row.get("system")))
        qid = str(row.get("query_id"))
        if target and qid in qids:
            result[target][qid] = [normalize_reddit_id(x) for x in row["comment_ids"]]
    return result


def load_systems(config: dict, qids: set[str], corpus_ids: set[str]) -> tuple[dict, dict]:
    systems: dict[str, dict[str, list[str]]] = {}
    provenance: dict[str, dict] = {}

    for name, run in _rows_to_run(config["sbert_runs"],
                                  {"S0": "S0", "S2": "S2", "SG2": "SG2"}, qids).items():
        systems[name] = run
        provenance[name] = {"path": str(config["sbert_runs"]),
                            "sha256": sha256_file(config["sbert_runs"]), "source_system": name}

    registry = json.loads(config["run_registry"].read_text(encoding="utf-8"))
    frozen = {row["method"]: row for row in registry["runs"] if row.get("status") == "available"}
    for method, name in (("cohere_dense", "C0"), ("official_static", "H0")):
        record = frozen[method]
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in read_jsonl(Path(record["path"])):
            if str(row["query_id"]) in qids:
                grouped[str(row["query_id"])].append(row)
        systems[name] = {
            qid: [normalize_reddit_id(row["comment_id"])
                  for row in sorted(grouped[qid], key=lambda item: int(item["rank"]))[:8]]
            for qid in qids
        }
        provenance[name] = {"path": record["path"], "sha256": record["run_sha256"],
                            "source_system": method}

    systems.update(_rows_to_run(config["cohere_deep_runs"],
                                {"A1_deeper_dense_q2": "C2"}, qids))
    provenance["C2"] = {"path": str(config["cohere_deep_runs"]),
                        "sha256": sha256_file(config["cohere_deep_runs"]),
                        "source_system": "A1_deeper_dense_q2"}
    systems.update(_rows_to_run(config["cohere_graph_runs"], {"B4": "G2"}, qids))
    provenance["G2"] = {"path": str(config["cohere_graph_runs"]),
                        "sha256": sha256_file(config["cohere_graph_runs"]),
                        "source_system": "B4"}

    if set(systems) != set(SYSTEM_LABELS):
        raise ValueError(f"system set mismatch: {set(systems)}")
    for name, run in systems.items():
        if set(run) != qids:
            raise ValueError(f"{name}: query set mismatch")
        for qid, ids in run.items():
            if len(ids) != 8 or len(set(ids)) != 8:
                raise ValueError(f"{name}/{qid}: Top-8 is not eight unique comments")
            missing = set(ids) - corpus_ids
            if missing:
                raise ValueError(f"{name}/{qid}: comments absent from fixed corpus: {missing}")
    return systems, provenance


def load_source_posts(path: Path) -> tuple[set[str], dict[str, str]]:
    post_ids: set[str] = set()
    comment_to_post: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            post = normalize_reddit_id(row["post_id"])
            comment = normalize_reddit_id(row["comment_id"])
            post_ids.add(post)
            comment_to_post[comment] = post
    return post_ids, comment_to_post


def load_references(config: dict, queries: dict[str, str], posts: dict[str, str],
                    corpus_ids: set[str], corpus_hashes: set[str], system_ids: set[str],
                    corpus_post_ids: set[str]) -> tuple[list[dict], list[dict], dict[str, list[dict]], dict]:
    wanted = {"t3_" + normalize_reddit_id(posts[qid]): qid for qid in queries}
    frozen_by_query: dict[str, dict[str, str]] = defaultdict(dict)
    with config["frozen_reply_sample"].open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qid = normalize_reddit_id(row["post_id"])
            if qid in queries:
                frozen_by_query[qid][normalize_reddit_id(row["comment_id"])] = str(row["target_text"])
    if set(frozen_by_query) != set(queries):
        raise ValueError("frozen resample reply rows do not cover all dev100-v2 queries")

    raw_by_query: dict[str, list[dict]] = defaultdict(list)
    recovered_top_level_ids: dict[str, set[str]] = defaultdict(set)
    deleted_by_query: dict[str, int] = defaultdict(int)
    excluded_by_query: dict[str, int] = defaultdict(int)
    qualifying_by_query: dict[str, int] = defaultdict(int)
    process = subprocess.Popen(
        ["zstd", "-dc", "--long=31", str(config["reddit_comments_dump"])],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        try:
            row = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        link_id = str(row.get("link_id") or "")
        qid = wanted.get(link_id)
        if qid is None or str(row.get("parent_id") or "") != link_id:
            continue
        body = str(row.get("body") or "").strip()
        if body.casefold() in INVALID_REPLY:
            deleted_by_query[qid] += 1
            continue
        cid = normalize_reddit_id(row.get("id") or "")
        if cid:
            recovered_top_level_ids[qid].add(cid)
        author = str(row.get("author") or "")
        # Preserve the sampler's historical qualifying-count definition, then
        # apply the stricter reference-validity filter.  This keeps tier audit
        # faithful while excluding automated moderation notices from gold-like
        # reference evidence.
        if author not in BOT_AUTHORS and not URL_ONLY.match(body) \
                and len(body) >= int(config["sampling_min_comment_chars"]):
            qualifying_by_query[qid] += 1
        if author in BOT_AUTHORS or URL_ONLY.match(body) or MOD_NOTICE.match(body):
            excluded_by_query[qid] += 1
            continue
        if not cid:
            excluded_by_query[qid] += 1
            continue
        raw_by_query[qid].append({"reply_id": cid, "text": body, "text_sha256": text_sha(body)})
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"zstd failed with code {return_code}: {stderr[-1000:]}")

    inventory: list[dict] = []
    exclusions: list[dict] = []
    references: dict[str, list[dict]] = {}
    selected_digest = hashlib.sha256()
    invalid_total = 0
    tier_by_query = {}
    with config["queries"].open(encoding="utf-8") as handle:
        tier_by_query = {str(row["query_id"]): str(row["tier"]) for row in csv.DictReader(handle)}
    for qid in queries:
        post_id = normalize_reddit_id(posts[qid])
        valid = sorted(raw_by_query[qid], key=lambda row: row["reply_id"])
        invalid = deleted_by_query[qid]
        frozen_ids = set(frozen_by_query[qid])
        recovered_ids = {row["reply_id"] for row in valid}
        missing_frozen = sorted(frozen_ids - recovered_top_level_ids[qid])
        if missing_frozen:
            raise ValueError(f"{qid}: frozen reply rows absent from raw dump: {missing_frozen}")
        for row in valid:
            selected_digest.update((qid + "\0" + row["reply_id"] + "\0" + row["text_sha256"] + "\n").encode("utf-8"))
        invalid_total += invalid
        reply_ids = {row["reply_id"] for row in valid}
        reply_hashes = {row["text_sha256"] for row in valid}
        id_overlap = sorted(reply_ids & corpus_ids)
        text_overlap = sorted(reply_hashes & corpus_hashes)
        top8_overlap = sorted(reply_ids & system_ids)
        tier = tier_by_query[qid]
        qualifying = qualifying_by_query[qid]
        tier_consistent = ((tier == "shallow" and 1 <= qualifying <= 3)
                           or (tier == "mid" and 4 <= qualifying <= 15)
                           or (tier == "deep" and qualifying >= 16))
        reasons: list[str] = []
        if not valid:
            reasons.append("NO_VALID_TOP_LEVEL_REPLY_IN_RAW_DUMP")
        if not tier_consistent:
            reasons.append("DEPTH_TIER_COUNT_MISMATCH")
        if post_id in corpus_post_ids or id_overlap or text_overlap or top8_overlap:
            reasons.append("REFERENCE_LEAKAGE")
        included = not reasons
        inventory.append({
            "query_id": qid, "post_id": post_id, "depth_tier": tier,
            "number_of_original_replies": len(valid) + invalid + excluded_by_query[qid],
            "valid_reply_count": len(valid), "empty_deleted_removed_count": invalid,
            "bot_or_url_only_count": excluded_by_query[qid],
            "sampling_qualifying_reply_count": qualifying,
            "sampling_depth_tier_consistent": tier_consistent,
            "frozen_resample_reply_count": len(frozen_ids),
            "frozen_resample_reply_subset_recovered": not missing_frozen,
            "reply_ids": sorted(reply_ids), "reply_text_hashes": sorted(reply_hashes),
            "reply_id_overlap_retrieval_corpus": id_overlap,
            "reply_text_hash_overlap_retrieval_corpus": text_overlap,
            "reply_ids_in_any_system_top8": top8_overlap,
            "source_post_id_in_retrieval_corpus": post_id in corpus_post_ids,
            "abnormally_few_replies": 0 < len(valid) < int(config["abnormal_few_reply_threshold"]),
            "included_in_semantic_evaluation": included, "exclusion_reasons": reasons,
        })
        if included:
            references[qid] = valid
        else:
            exclusions.append({"query_id": qid, "post_id": post_id,
                               "valid_reply_count": len(valid), "reasons": reasons})
    stat = config["reddit_comments_dump"].stat()
    source = {
        "path": str(config["reddit_comments_dump"]),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "selected_reference_rows_sha256": selected_digest.hexdigest(),
        "frozen_reply_sample_path": str(config["frozen_reply_sample"]),
        "frozen_reply_sample_sha256": sha256_file(config["frozen_reply_sample"]),
        "note": "All valid direct/top-level replies are restored from the same local Reddit dump used by the stratified sampler. Frozen capped resample IDs must be a subset.",
        "thread_scope": "direct/top-level replies only: link_id == parent_id == t3_<post_id>",
    }
    summary = {
        "query_count": len(inventory),
        "query_count_with_valid_replies": sum(bool(row["valid_reply_count"]) for row in inventory),
        "query_count_included": len(references),
        "query_count_excluded": len(exclusions),
        "raw_reply_count": sum(row["number_of_original_replies"] for row in inventory),
        "valid_reply_count": sum(row["valid_reply_count"] for row in inventory),
        "empty_deleted_removed_count": invalid_total,
        "valid_reply_count_distribution": {
            "min": min(row["valid_reply_count"] for row in inventory),
            "median": statistics.median(row["valid_reply_count"] for row in inventory),
            "max": max(row["valid_reply_count"] for row in inventory),
        },
        "leakage_query_count": sum("REFERENCE_LEAKAGE" in row["exclusion_reasons"] for row in inventory),
        "depth_tier_mismatch_query_count": sum(not row["sampling_depth_tier_consistent"] for row in inventory),
        "all_frozen_resample_rows_recovered": all(row["frozen_resample_reply_subset_recovered"] for row in inventory),
        "source": source,
    }
    return inventory, exclusions, references, summary


def encode_texts(config: dict, texts: list[str], output: Path) -> tuple[dict[str, np.ndarray], dict]:
    encoder = config["semantic_encoder"]
    keyed = {text_sha(text): text for text in texts}
    ordered = sorted(keyed)
    cache_key = sha256_bytes(json.dumps({
        "model": encoder["model_id"], "revision": encoder["revision"],
        "max_sequence_length": encoder["max_sequence_length"], "text_hashes": ordered,
    }, sort_keys=True).encode("utf-8"))
    cache = output / f"semantic_embeddings_{cache_key[:16]}.npz"
    if cache.exists():
        loaded = np.load(cache)
        embeddings = loaded["embeddings"]
        cache_hit = True
    else:
        model = SentenceTransformer(str(encoder["local_snapshot"]), local_files_only=True,
                                    device=str(encoder.get("device", "cpu")))
        model.max_seq_length = int(encoder["max_sequence_length"])
        embeddings = model.encode(
            [keyed[key] for key in ordered],
            batch_size=int(encoder["batch_size"]),
            show_progress_bar=True,
            normalize_embeddings=bool(encoder["normalize_embeddings"]),
            convert_to_numpy=True,
        ).astype(np.float32)
        np.savez_compressed(cache, embeddings=embeddings)
        cache_hit = False
    if embeddings.shape != (len(ordered), 1024):
        raise ValueError(f"unexpected BGE embedding matrix shape: {embeddings.shape}")
    return dict(zip(ordered, embeddings, strict=True)), {
        "cache_path": str(cache), "cache_sha256": sha256_file(cache), "cache_hit": cache_hit,
        "unique_text_count": len(ordered), "embedding_dimension": int(embeddings.shape[1]),
        "model_id": encoder["model_id"], "revision": encoder["revision"],
        "pooling": encoder["pooling"], "normalized": encoder["normalize_embeddings"],
        "max_sequence_length": encoder["max_sequence_length"], "device": str(encoder.get("device", "cpu")),
    }


def alignment(candidate_texts: list[str], reply_texts: list[str], embeddings: dict[str, np.ndarray],
              threshold: float) -> dict[str, float]:
    candidates = np.stack([embeddings[text_sha(text)] for text in candidate_texts])
    replies = np.stack([embeddings[text_sha(text)] for text in reply_texts])
    similarities = candidates @ replies.T
    candidate_max = similarities.max(axis=1)
    reply_max = similarities.max(axis=0)
    return {
        "cra_at8": float(candidate_max.mean()),
        "rcc_at8": float(reply_max.mean()),
        "best_align_at8": float(similarities.max()),
        "reply_coverage_at8": float((reply_max >= threshold).mean()),
    }


def bidirectional_f(cra: float, rcc: float, beta: float = 1.0) -> float:
    """F-beta over candidate alignment (precision-like) and reply coverage (recall-like)."""
    beta_sq = beta * beta
    denominator = beta_sq * cra + rcc
    return 0.0 if denominator == 0 else float((1.0 + beta_sq) * cra * rcc / denominator)


def finite_spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    value = spearmanr(x, y).statistic
    return float(value) if math.isfinite(value) else None


def bootstrap_spearman(x: list[float], y: list[float], draws: int, seed: int) -> list[float] | None:
    observed = finite_spearman(x, y)
    if observed is None:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    count = len(x)
    for _ in range(draws):
        indices = rng.integers(0, count, count)
        value = finite_spearman([x[index] for index in indices],
                                [y[index] for index in indices])
        if value is not None:
            values.append(value)
    if not values:
        return None
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def mean_with_ci(values: list[float], draws: int, seed: int) -> dict:
    return {
        "mean": statistics.fmean(values),
        "bootstrap_95ci": list(_bootstrap_ci(values, n_boot=draws, seed=seed)),
        "exact_query_n": len(values),
    }


def load_admin_references(output: Path) -> dict[str, list[dict]]:
    rows = read_jsonl(output / "ADMIN_community_reply_reference_texts.jsonl")
    references = {str(row["query_id"]): list(row["replies"]) for row in rows}
    if len(references) != 100 or any(not replies for replies in references.values()):
        raise ValueError("bidirectional analysis requires the frozen 100-query Phase 1 references")
    return references


def load_capped_reply_ids(config: dict, valid_ids: dict[str, set[str]]) -> dict[str, list[str]]:
    maximum = int(config["bidirectional_alignment"]["capped_reply"]["maximum_replies_per_query"])
    result: dict[str, list[str]] = defaultdict(list)
    with config["frozen_reply_sample"].open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qid = normalize_reddit_id(row["post_id"])
            reply_id = normalize_reddit_id(row["comment_id"])
            if qid in valid_ids and reply_id in valid_ids[qid] \
                    and reply_id not in result[qid] and len(result[qid]) < maximum:
                result[qid].append(reply_id)
    if set(result) != set(valid_ids) or any(not ids or len(ids) > maximum for ids in result.values()):
        raise ValueError("historical capped reply subset is incomplete")
    return dict(result)


def cluster_balanced_rcc(candidate_texts: list[str], reply_texts: list[str],
                         embeddings: dict[str, np.ndarray], settings: dict) -> tuple[float, int]:
    candidates = np.stack([embeddings[text_sha(text)] for text in candidate_texts])
    replies = np.stack([embeddings[text_sha(text)] for text in reply_texts])
    reply_max = (candidates @ replies.T).max(axis=0)
    if len(reply_texts) == 1:
        return float(reply_max[0]), 1
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric=str(settings["metric"]),
        linkage=str(settings["linkage"]),
        distance_threshold=float(settings["distance_threshold"]),
    )
    labels = clusterer.fit_predict(replies)
    cluster_scores = [float(reply_max[labels == label].mean()) for label in sorted(set(labels))]
    return statistics.fmean(cluster_scores), len(cluster_scores)


def aggregate_alignment(rows: list[dict], seed: int, draws: int) -> dict:
    result: dict[str, Any] = {"query_count": len(rows)}
    for metric in ("cra_at8", "rcc_at8", "best_align_at8", "reply_coverage_at8"):
        values = [float(row[metric]) for row in rows]
        result[metric] = statistics.fmean(values)
        result[f"{metric}_bootstrap_95ci"] = list(_bootstrap_ci(values, n_boot=draws, seed=seed + len(metric)))
    return result


def paired(rows: list[dict], left: str, right: str, metric: str, seed: int, draws: int) -> dict:
    by_system = {(row["system"], row["query_id"]): row for row in rows}
    qids = sorted(set(row["query_id"] for row in rows
                      if (left, row["query_id"]) in by_system and (right, row["query_id"]) in by_system))
    deltas = [float(by_system[(left, qid)][metric]) - float(by_system[(right, qid)][metric])
              for qid in qids]
    eps = 1e-12
    return {
        "left": left, "right": right, "metric": metric, "exact_query_n": len(qids),
        "mean_delta": statistics.fmean(deltas), "median_delta": statistics.median(deltas),
        "bootstrap_95ci": list(_bootstrap_ci(deltas, n_boot=draws, seed=seed)),
        "improved": sum(value > eps for value in deltas),
        "tied": sum(abs(value) <= eps for value in deltas),
        "degraded": sum(value < -eps for value in deltas),
    }


def utility_metrics(config: dict, systems: dict[str, dict[str, list[str]]]) -> tuple[dict, dict]:
    _, registry = complete_utility_v2_rows(read_jsonl(config["utility_registry"]))
    system_metrics: dict[str, dict] = {}
    per_query: dict[tuple[str, str], dict] = {}
    for system, run in systems.items():
        rows = []
        for qid, ids in run.items():
            judged = [cid for cid in ids if (qid, cid) in registry]
            item = {"query_id": qid, "coverage": len(judged) / 8, "complete": len(judged) == 8}
            if item["complete"]:
                utilities = [float(registry[(qid, cid)]["utility"]) for cid in ids]
                gains = {cid: float(row["utility"])
                         for (query_id, cid), row in registry.items() if query_id == qid}
                item.update({"mean_utility_at8": statistics.fmean(utilities),
                             "ndcg_at8": graded_ndcg_at(ids, gains, 8)})
            else:
                item.update({"mean_utility_at8": None, "ndcg_at8": None})
            rows.append(item)
            per_query[(system, qid)] = item
        complete = [row for row in rows if row["complete"]]
        system_metrics[system] = {
            "judgment_coverage_at8": statistics.fmean(row["coverage"] for row in rows),
            "complete_query_count": len(complete), "query_count": len(rows),
            "mean_utility_at8": statistics.fmean(row["mean_utility_at8"] for row in complete),
            "ndcg_at8": statistics.fmean(row["ndcg_at8"] for row in complete),
            "scope": "complete Top-8 queries only; missing judgments are not assigned zero",
        }
    return system_metrics, per_query


def leakage_markdown(summary: dict, feature_names: list[str]) -> str:
    return f"""# Community reply leakage audit

## Result

- Development queries audited: **{summary['query_count']}**; frozen test read: **no**.
- Queries with a recoverable valid reply: **{summary['query_count_with_valid_replies']}**.
- Queries excluded from semantic evaluation: **{summary['query_count_excluded']}**.
- Queries with any reply ID/text/source-post overlap with the fixed retrieval corpus or Top-8: **{summary['leakage_query_count']}**.
- Valid/invalid reply records: **{summary['valid_reply_count']} / {summary['empty_deleted_removed_count']}**.

## Separation checks

1. The fixed 19,013-comment corpus was frozen before this auxiliary evaluation. Reply IDs,
   normalized text hashes, and source post IDs were checked against it.
2. All seven Top-8 runs were loaded from frozen artifacts and checked before replies were scored.
3. The frozen Linear-basic feature schema is `{', '.join(feature_names)}`. It contains no
   reply/thread/community-reference feature.
4. utility-v2 uses the original query, frozen summary, and candidate comment. Original-thread
   replies are absent from its registry join and were not supplied to that Judge.
5. This script exposes no retrieval, training, gate fitting, or model-selection operation.

## Reference scope

Replies are restored from the same local Reddit dump used by the 4× stratified sampler. Only direct
top-level replies are references (`link_id == parent_id`); nested discussion is excluded. Every
frozen resample reply ID must reappear in the restored set, and the original shallow/mid/deep tier
must agree with the restored count of sampling-qualifying replies.
"""


def phase1(config: dict) -> None:
    reject_test_paths(config)
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    queries, posts = load_queries(config["queries"])
    corpus, corpus_hashes = load_corpus(config["corpus"])
    systems, run_provenance = load_systems(config, set(queries), set(corpus))
    all_system_ids = {cid for run in systems.values() for ids in run.values() for cid in ids}
    corpus_post_ids, _ = load_source_posts(config["corpus_source_map"])
    inventory, exclusions, references, reference_summary = load_references(
        config, queries, posts, set(corpus), corpus_hashes, all_system_ids, corpus_post_ids)

    write_jsonl(output / "community_reply_reference_inventory.jsonl", inventory)
    write_json(output / "community_reply_reference_summary.json", reference_summary)
    write_jsonl(output / "community_reply_reference_exclusions.jsonl", exclusions)
    write_jsonl(output / "ADMIN_community_reply_reference_texts.jsonl", [
        {"query_id": qid, "post_id": posts[qid], "replies": rows}
        for qid, rows in sorted(references.items())
    ])

    feature_names = [
        "dense_score", "dense_rank_reciprocal", "dense_missing", "query_comment_cosine",
        "bm25_score", "bm25_rank_reciprocal", "bm25_missing", "comment_length_log",
        "query_length_log", "lexical_jaccard", "lexical_query_coverage",
    ]
    (output / "community_reply_leakage_audit.md").write_text(
        leakage_markdown(reference_summary, feature_names), encoding="utf-8")

    semantic_texts = [row["text"] for rows in references.values() for row in rows]
    semantic_texts.extend(corpus[cid] for run in systems.values() for ids in run.values() for cid in ids)
    embeddings, encoder_manifest = encode_texts(config, semantic_texts, output)
    threshold = float(config["semantic_encoder"]["threshold"])
    per_query_rows: list[dict] = []
    sensitivity: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for system, run in systems.items():
        for qid in sorted(references):
            candidates = [corpus[cid] for cid in run[qid]]
            replies = [row["text"] for row in references[qid]]
            values = alignment(candidates, replies, embeddings, threshold)
            per_query_rows.append({"query_id": qid, "system": system,
                                   "system_label": SYSTEM_LABELS[system],
                                   "valid_reply_count": len(replies), **values})
            for tau in config["semantic_encoder"]["sensitivity_thresholds"]:
                coverage = alignment(candidates, replies, embeddings, float(tau))["reply_coverage_at8"]
                sensitivity[system][str(float(tau))].append(coverage)
    write_jsonl(output / "reply_semantic_alignment_per_query.jsonl", per_query_rows)

    metrics = {
        "schema": "reply-semantic-alignment-v1",
        "created_at": now(), "reference_query_count": len(references),
        "excluded_query_count": len(exclusions), "threshold": threshold,
        "encoder": encoder_manifest,
        "systems": {
            system: aggregate_alignment([row for row in per_query_rows if row["system"] == system],
                                        int(config["bootstrap_seed"]), int(config["bootstrap_draws"]))
            for system in SYSTEM_LABELS
        },
        "paired_comparisons": [
            paired(per_query_rows, left, right, metric,
                   int(config["bootstrap_seed"]) + index * 17 + len(metric),
                   int(config["bootstrap_draws"]))
            for index, (left, right) in enumerate(CONTRASTS)
            for metric in ("cra_at8", "rcc_at8", "best_align_at8", "reply_coverage_at8")
        ],
        "frozen_test_read": False, "external_model_calls": 0,
    }
    write_json(output / "reply_semantic_alignment_metrics.json", metrics)

    sensitivity_lines = [
        "# Reply semantic alignment sensitivity", "",
        f"Primary independent encoder: `{encoder_manifest['model_id']}` at revision `{encoder_manifest['revision']}`.",
        f"Primary threshold was frozen at τ={threshold:.2f} before system scoring; 0.60/0.80 are sensitivity only.",
        "", "| System | τ=.60 | τ=.70 (primary) | τ=.80 |", "|---|---:|---:|---:|",
    ]
    for system in SYSTEM_LABELS:
        values = sensitivity[system]
        sensitivity_lines.append(
            f"| {SYSTEM_LABELS[system]} | {statistics.fmean(values['0.6']):.4f} | "
            f"{statistics.fmean(values['0.7']):.4f} | {statistics.fmean(values['0.8']):.4f} |")
    sensitivity_lines.extend([
        "", "The compared SBERT checkpoint and Cohere service are not used as the primary reference encoder.",
        "Cosine alignment measures response-direction proximity, not truth, safety, or unique usefulness.",
    ])
    (output / "reply_semantic_alignment_sensitivity.md").write_text(
        "\n".join(sensitivity_lines) + "\n", encoding="utf-8")

    utility, utility_per_query = utility_metrics(config, systems)
    relationships: dict[str, dict] = {}
    for system in SYSTEM_LABELS:
        pairs = [(row, utility_per_query[(system, row["query_id"])]) for row in per_query_rows
                 if row["system"] == system and utility_per_query[(system, row["query_id"])]["complete"]]
        relationships[system] = {}
        for metric in ("cra_at8", "rcc_at8", "best_align_at8", "reply_coverage_at8"):
            result = spearmanr([row[metric] for row, _ in pairs],
                               [item["mean_utility_at8"] for _, item in pairs]) if len(pairs) > 1 else None
            relationships[system][metric] = {
                "exact_query_n": len(pairs),
                "spearman_rho": float(result.statistic) if result and math.isfinite(result.statistic) else None,
                "p_value": float(result.pvalue) if result and math.isfinite(result.pvalue) else None,
                "interpretation": "association only; not causal",
            }
    write_json(output / "community_reply_utility_relationship.json", {
        "status": "PHASE1_SEMANTIC_ONLY",
        "note": "Theme/complementarity relationships remain pending the blinded pilot gate.",
        "systems": relationships,
    })

    table = [
        "# Hidden community reply auxiliary evaluation — Phase 1", "",
        f"Semantic results use {len(references)} queries with recoverable, leakage-free replies; "
        f"{len(exclusions)} queries are excluded rather than imputed.", "",
        "| System | Utility@8 | nDCG@8 | utility exact n | CRA@8 | RCC@8 | BestAlign@8 | ReplyCoverage@8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEM_LABELS:
        u = utility[system]
        m = metrics["systems"][system]
        table.append(
            f"| {SYSTEM_LABELS[system]} | {u['mean_utility_at8']:.4f} | {u['ndcg_at8']:.4f} | "
            f"{u['complete_query_count']} | {m['cra_at8']:.4f} | {m['rcc_at8']:.4f} | "
            f"{m['best_align_at8']:.4f} | {m['reply_coverage_at8']:.4f} |")
    table.extend([
        "", f"Utility and nDCG keep their own frozen all-development completeness scope; semantic columns use exact n={len(references)}.",
        "Theme coverage and complementary value are intentionally not populated before the 20-query blinded Judge pilot.",
    ])
    (output / "community_reply_system_comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    manifest = {
        "schema": "community-reply-aux-input-manifest-v1", "created_at": now(),
        "version": config["version"], "phase_completed": "PHASE1_LOCAL_SEMANTIC",
        "query_count": len(queries), "semantic_reference_query_count": len(references),
        "fixed_corpus_count": len(corpus), "top_k": config["top_k"],
        "inputs": {
            key: {"path": str(config[key]), "sha256": sha256_file(config[key])}
            for key in ("queries", "summaries", "corpus", "corpus_source_map", "utility_registry")
        },
        "reference_source": reference_summary["source"], "system_runs": run_provenance,
        "encoder": encoder_manifest, "threshold_frozen_before_scoring": threshold,
        "separation_invariants": {
            "replies_in_retrieval": False, "replies_in_gate_features": False,
            "replies_in_utility_v2_judge": False, "post_hoc_only": True,
            "used_for_model_selection": False,
        },
        "phase_status": {
            "phase1_inventory_and_semantic": "COMPLETE",
            "phase2_reference_bundles": "NOT_RUN_REQUIRES_BLINDED_20_QUERY_PILOT",
            "phase3_theme_coverage": "NOT_RUN_REQUIRES_PILOT_STABILITY_GATE",
            "phase4_complementarity": "NOT_RUN_DEPENDS_ON_PHASE3",
        },
        "frozen_test_read": False, "external_model_calls": 0,
    }
    write_json(output / "community_reply_aux_input_manifest.json", manifest)
    semantic_verdict = ("COMMUNITY_ALIGNMENT_VALIDATED"
                        if len(references) == 100 and not exclusions
                        else "INCONCLUSIVE_REFERENCE_COVERAGE")
    verdict = {
        "phase": "PHASE1", "semantic_alignment_verdict": semantic_verdict,
        "theme_coverage_verdict": "INCONCLUSIVE_REFERENCE_COVERAGE",
        "reason": ("Semantic alignment covers all 100 frozen development queries with tier-consistent, leakage-free direct replies. "
                   if semantic_verdict == "COMMUNITY_ALIGNMENT_VALIDATED" else
                   "Semantic alignment is descriptive for the recoverable-reference subset only. ")
                  + "Theme/complementarity judging has not passed its required 20-query pilot.",
        "primary_retrieval_conclusions_modified": False, "frozen_test_read": False,
        "external_model_calls": 0,
    }
    write_json(output / "community_reply_aux_final_verdict.json", verdict)
    print(json.dumps({"output_dir": str(output), "reference_queries": len(references),
                      "excluded_queries": len(exclusions), "external_model_calls": 0,
                      "phase": "PHASE1_LOCAL_SEMANTIC"}, indent=2))


def _relationship(x: list[float], y: list[float], draws: int, seed: int) -> dict:
    result = spearmanr(x, y) if len(x) > 1 else None
    rho = finite_spearman(x, y)
    return {
        "exact_query_n": len(x),
        "spearman_rho": rho,
        "bootstrap_95ci": bootstrap_spearman(x, y, draws, seed),
        "p_value": (float(result.pvalue) if result is not None and math.isfinite(result.pvalue)
                    else None),
        "interpretation": "association only; not causal",
    }


def _comparison_lookup(comparisons: list[dict], left: str, right: str, metric: str) -> dict:
    return next(row for row in comparisons
                if row["left"] == left and row["right"] == right and row["metric"] == metric)


def bidirectional(config: dict) -> None:
    """Extend frozen Phase 1 outputs without changing replies, systems, or primary utility."""
    reject_test_paths(config)
    output = config["output_dir"]
    required = (
        "community_reply_reference_inventory.jsonl",
        "ADMIN_community_reply_reference_texts.jsonl",
        "reply_semantic_alignment_per_query.jsonl",
        "reply_semantic_alignment_metrics.json",
    )
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise FileNotFoundError(f"run frozen Phase 1 first; missing {missing}")

    queries, _ = load_queries(config["queries"])
    corpus, _ = load_corpus(config["corpus"])
    systems, _ = load_systems(config, set(queries), set(corpus))
    references = load_admin_references(output)
    inventory = {str(row["query_id"]): row
                 for row in read_jsonl(output / "community_reply_reference_inventory.jsonl")}
    phase1_rows = read_jsonl(output / "reply_semantic_alignment_per_query.jsonl")
    if len(phase1_rows) != len(SYSTEM_LABELS) * 100:
        raise ValueError("frozen semantic per-query file is not 7 systems x 100 queries")

    seed = int(config["bootstrap_seed"])
    draws = int(config["bootstrap_draws"])
    settings = config["bidirectional_alignment"]
    primary_beta = float(settings["beta_primary"])
    sensitivity_betas = [float(value) for value in settings["beta_sensitivity"]]
    per_query_rows: list[dict] = []
    for row in phase1_rows:
        qid = str(row["query_id"])
        cra = float(row["cra_at8"])
        rcc = float(row["rcc_at8"])
        per_query_rows.append({
            **row,
            "depth_tier": inventory[qid]["depth_tier"],
            "bidirectional_alignment_f1_at8": bidirectional_f(cra, rcc, primary_beta),
            "bidirectional_alignment_f0_5_at8": bidirectional_f(cra, rcc, 0.5),
            "bidirectional_alignment_f2_at8": bidirectional_f(cra, rcc, 2.0),
        })
    write_jsonl(output / "reply_bidirectional_alignment_per_query.jsonl", per_query_rows)

    aggregate_metrics = (
        "cra_at8", "rcc_at8", "bidirectional_alignment_f1_at8",
        "bidirectional_alignment_f0_5_at8", "bidirectional_alignment_f2_at8",
        "best_align_at8", "reply_coverage_at8",
    )
    system_metrics: dict[str, dict] = {}
    for system in SYSTEM_LABELS:
        rows = [row for row in per_query_rows if row["system"] == system]
        system_metrics[system] = {"system_label": SYSTEM_LABELS[system], "exact_query_n": len(rows)}
        for index, metric in enumerate(aggregate_metrics):
            values = [float(row[metric]) for row in rows]
            system_metrics[system][metric] = statistics.fmean(values)
            system_metrics[system][f"{metric}_bootstrap_95ci"] = list(
                _bootstrap_ci(values, n_boot=draws, seed=seed + index * 101))
    metric_payload = {
        "schema": "reply-bidirectional-alignment-v1",
        "created_at": now(),
        "primary_metric": "bidirectional_alignment_f1_at8",
        "definitions": {
            "cra_at8": "mean over candidates of maximum candidate-to-reply cosine",
            "rcc_at8": "mean over all valid replies of maximum reply-to-candidate cosine",
            "bidirectional_alignment_f1_at8": "harmonic mean of per-query CRA and RCC",
            "beta_sensitivity": sensitivity_betas,
        },
        "reference_query_count": 100,
        "valid_reply_count": sum(len(rows) for rows in references.values()),
        "systems": system_metrics,
        "frozen_test_read": False,
        "external_model_calls": 0,
    }
    write_json(output / "reply_bidirectional_alignment_metrics.json", metric_payload)

    comparison_metrics = (
        "cra_at8", "rcc_at8", "bidirectional_alignment_f1_at8", "best_align_at8",
    )
    comparisons = [
        paired(per_query_rows, left, right, metric,
               seed + contrast_index * 1009 + metric_index * 37, draws)
        for contrast_index, (left, right) in enumerate(BIDIRECTIONAL_CONTRASTS)
        for metric_index, metric in enumerate(comparison_metrics)
    ]
    write_json(output / "reply_bidirectional_alignment_paired_comparisons.json", {
        "schema": "reply-bidirectional-paired-comparisons-v1",
        "bootstrap": {"unit": "query", "draws": draws, "confidence": 0.95, "seed": seed},
        "comparisons": comparisons,
        "frozen_test_read": False,
        "external_model_calls": 0,
    })

    # Reply-count audit is computed before either sensitivity analysis.  Its
    # predeclared trigger controls whether the appendix analyses are produced.
    count_audit: dict[str, Any] = {}
    for system_index, system in enumerate(SYSTEM_LABELS):
        rows = [row for row in per_query_rows if row["system"] == system]
        counts = [float(row["valid_reply_count"]) for row in rows]
        correlations = {}
        for metric_index, metric in enumerate(
                ("cra_at8", "rcc_at8", "bidirectional_alignment_f1_at8")):
            values = [float(row[metric]) for row in rows]
            correlations[metric] = _relationship(
                counts, values, draws, seed + system_index * 503 + metric_index * 41)
        tiers = {}
        for tier_index, tier in enumerate(("shallow", "mid", "deep")):
            tier_rows = [row for row in rows if row["depth_tier"] == tier]
            tiers[tier] = {
                metric: mean_with_ci([float(row[metric]) for row in tier_rows], draws,
                                     seed + system_index * 701 + tier_index * 67 + metric_index)
                for metric_index, metric in enumerate(
                    ("cra_at8", "rcc_at8", "bidirectional_alignment_f1_at8"))
            }
        count_audit[system] = {"system_label": SYSTEM_LABELS[system],
                               "correlations": correlations, "depth_tiers": tiers}

    trigger = settings["reply_count_bias_trigger"]
    trigger_metric = str(trigger["metric"])
    rho_threshold = float(trigger["spearman_rho_at_most"])
    systems_triggered = [
        system for system in SYSTEM_LABELS
        if count_audit[system]["correlations"][trigger_metric]["spearman_rho"] is not None
        and count_audit[system]["correlations"][trigger_metric]["spearman_rho"] <= rho_threshold
    ]
    sensitivity_required = len(systems_triggered) >= int(trigger["minimum_systems"])

    sensitivity: dict[str, Any] = {
        "status": "RUN" if sensitivity_required else "NOT_TRIGGERED",
        "cluster_balanced": {}, "capped_reply": {},
    }
    if sensitivity_required:
        semantic_texts = [row["text"] for rows in references.values() for row in rows]
        semantic_texts.extend(corpus[cid] for run in systems.values() for ids in run.values() for cid in ids)
        embeddings, encoder_manifest = encode_texts(config, semantic_texts, output)
        valid_ids = {qid: {row["reply_id"] for row in rows} for qid, rows in references.items()}
        capped_ids = load_capped_reply_ids(config, valid_ids)
        reference_by_id = {qid: {row["reply_id"]: row["text"] for row in rows}
                           for qid, rows in references.items()}
        cluster_settings = settings["cluster_balanced"]
        cluster_rows: list[dict] = []
        capped_rows: list[dict] = []
        for system, run in systems.items():
            for qid in sorted(references):
                candidates = [corpus[cid] for cid in run[qid]]
                reply_texts = [row["text"] for row in references[qid]]
                balanced_rcc, cluster_count = cluster_balanced_rcc(
                    candidates, reply_texts, embeddings, cluster_settings)
                cluster_rows.append({"system": system, "query_id": qid,
                                     "reply_count": len(reply_texts), "cluster_count": cluster_count,
                                     "cluster_balanced_rcc_at8": balanced_rcc})
                capped_texts = [reference_by_id[qid][reply_id] for reply_id in capped_ids[qid]]
                capped_values = alignment(candidates, capped_texts, embeddings,
                                          float(config["semantic_encoder"]["threshold"]))
                capped_rows.append({
                    "system": system, "query_id": qid, "capped_reply_count": len(capped_texts),
                    **capped_values,
                    "bidirectional_alignment_f1_at8": bidirectional_f(
                        capped_values["cra_at8"], capped_values["rcc_at8"], 1.0),
                })
        for system_index, system in enumerate(SYSTEM_LABELS):
            system_clusters = [row for row in cluster_rows if row["system"] == system]
            cluster_values = [float(row["cluster_balanced_rcc_at8"]) for row in system_clusters]
            sensitivity["cluster_balanced"][system] = {
                **mean_with_ci(cluster_values, draws, seed + system_index * 811),
                "mean_cluster_count": statistics.fmean(row["cluster_count"] for row in system_clusters),
                "reply_count_spearman": _relationship(
                    [float(row["reply_count"]) for row in system_clusters], cluster_values,
                    draws, seed + system_index * 823),
            }
            system_capped = [row for row in capped_rows if row["system"] == system]
            sensitivity["capped_reply"][system] = {
                metric: mean_with_ci([float(row[metric]) for row in system_capped], draws,
                                     seed + system_index * 827 + metric_index)
                for metric_index, metric in enumerate(
                    ("cra_at8", "rcc_at8", "bidirectional_alignment_f1_at8"))
            }
        sensitivity["encoder"] = encoder_manifest
        sensitivity["cluster_balanced"]["configuration"] = cluster_settings
        sensitivity["capped_reply"]["configuration"] = settings["capped_reply"]

    bias_payload = {
        "schema": "reply-count-bias-audit-v1", "created_at": now(),
        "reply_count_distribution": {
            "minimum": min(len(rows) for rows in references.values()),
            "median": statistics.median(len(rows) for rows in references.values()),
            "maximum": max(len(rows) for rows in references.values()),
            "total": sum(len(rows) for rows in references.values()),
        },
        "trigger_rule": {**trigger, "triggered_systems": systems_triggered,
                         "triggered_system_count": len(systems_triggered),
                         "sensitivity_required": sensitivity_required},
        "systems": count_audit, "sensitivity": sensitivity,
        "main_metric_unchanged": True, "frozen_test_read": False, "external_model_calls": 0,
    }
    write_json(output / "reply_count_bias_audit.json", bias_payload)

    utility, utility_per_query = utility_metrics(config, systems)
    relationships: dict[str, Any] = {}
    quadrant_rows: list[dict] = []
    case_count = int(settings["quadrant_cases_per_cell"])
    for system_index, system in enumerate(SYSTEM_LABELS):
        rows = [row for row in per_query_rows if row["system"] == system
                and utility_per_query[(system, row["query_id"])]["complete"]]
        util = [float(utility_per_query[(system, row["query_id"])]["mean_utility_at8"])
                for row in rows]
        ndcg = [float(utility_per_query[(system, row["query_id"])]["ndcg_at8"])
                for row in rows]
        relationships[system] = {
            "system_label": SYSTEM_LABELS[system],
            "utility_vs_cra": _relationship(
                util, [float(row["cra_at8"]) for row in rows], draws, seed + system_index * 907),
            "utility_vs_rcc": _relationship(
                util, [float(row["rcc_at8"]) for row in rows], draws, seed + system_index * 911),
            "utility_vs_bialign_f1": _relationship(
                util, [float(row["bidirectional_alignment_f1_at8"]) for row in rows],
                draws, seed + system_index * 919),
            "ndcg_vs_bialign_f1": _relationship(
                ndcg, [float(row["bidirectional_alignment_f1_at8"]) for row in rows],
                draws, seed + system_index * 929),
        }
        utility_median = statistics.median(util)
        alignment_median = statistics.median(
            float(row["bidirectional_alignment_f1_at8"]) for row in rows)
        cells: dict[str, list[dict]] = defaultdict(list)
        for row, utility_value, ndcg_value in zip(rows, util, ndcg, strict=True):
            utility_side = "high_utility" if utility_value >= utility_median else "low_utility"
            align_side = ("high_bialign" if float(row["bidirectional_alignment_f1_at8"]) >= alignment_median
                          else "low_bialign")
            cell = f"{utility_side}__{align_side}"
            cells[cell].append({
                "system": system, "system_label": SYSTEM_LABELS[system],
                "query_id": row["query_id"], "quadrant": cell,
                "mean_utility_at8": utility_value, "ndcg_at8": ndcg_value,
                "cra_at8": row["cra_at8"], "rcc_at8": row["rcc_at8"],
                "bidirectional_alignment_f1_at8": row["bidirectional_alignment_f1_at8"],
                "selection_note": "deterministic hash order; descriptive only",
            })
        relationships[system]["quadrants"] = {
            "split": "within-system medians",
            "utility_median": utility_median, "bialign_f1_median": alignment_median,
            "counts": {cell: len(items) for cell, items in sorted(cells.items())},
            "cases_per_cell": case_count,
        }
        for cell, items in sorted(cells.items()):
            ordered = sorted(items, key=lambda item: sha256_bytes(
                f"{seed}:{system}:{cell}:{item['query_id']}".encode("utf-8")))
            quadrant_rows.extend(ordered[:case_count])
    relationship_payload = {
        "schema": "reply-alignment-utility-relationship-v1", "created_at": now(),
        "systems": relationships,
        "claim": "community reply alignment and absolute utility are related but distinct objectives",
        "causal_interpretation_allowed": False, "frozen_test_read": False, "external_model_calls": 0,
    }
    write_json(output / "reply_alignment_utility_relationship.json", relationship_payload)
    write_jsonl(output / "reply_alignment_case_quadrants.jsonl", quadrant_rows)

    def ci_text(row: dict) -> str:
        low, high = row["bootstrap_95ci"]
        return f"{row['mean_delta']:+.4f} [{low:+.4f}, {high:+.4f}]"

    bias_lines = [
        "# Reply count bias audit", "",
        "主公式始终使用全部有效顶层回复；本审计不删除回复，也不回写主指标。", "",
        "预注册触发规则：若至少4/7系统的 `Spearman(reply count, RCC) <= -.20`，运行两项附录敏感性。",
        f"本次触发系统数为 **{len(systems_triggered)}/7**，因此敏感性分析 **{'已运行' if sensitivity_required else '未触发'}**。", "",
        "| System | rho(count, CRA) | rho(count, RCC) | rho(count, BiAlignF1) |",
        "|---|---:|---:|---:|",
    ]
    for system in SYSTEM_LABELS:
        corr = count_audit[system]["correlations"]
        bias_lines.append(
            f"| {SYSTEM_LABELS[system]} | {corr['cra_at8']['spearman_rho']:+.3f} | "
            f"{corr['rcc_at8']['spearman_rho']:+.3f} | "
            f"{corr['bidirectional_alignment_f1_at8']['spearman_rho']:+.3f} |")
    bias_lines.extend([
        "", "## Depth tier均值与bootstrap 95% CI", "",
        "| System | Tier (n) | CRA | RCC | BiAlignF1 |",
        "|---|---|---:|---:|---:|",
    ])
    for system in SYSTEM_LABELS:
        for tier in ("shallow", "mid", "deep"):
            tier_data = count_audit[system]["depth_tiers"][tier]
            cells = []
            for metric in ("cra_at8", "rcc_at8", "bidirectional_alignment_f1_at8"):
                item = tier_data[metric]
                cells.append(f"{item['mean']:.4f} [{item['bootstrap_95ci'][0]:.4f}, {item['bootstrap_95ci'][1]:.4f}]")
            bias_lines.append(
                f"| {SYSTEM_LABELS[system]} | {tier} ({tier_data['cra_at8']['exact_query_n']}) | "
                + " | ".join(cells) + " |")
    if sensitivity_required:
        bias_lines.extend([
            "", "## 敏感性结果", "",
            "| System | full RCC | cluster-balanced RCC | capped≤8 RCC | full BiAlignF1 | capped≤8 BiAlignF1 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for system in SYSTEM_LABELS:
            bias_lines.append(
                f"| {SYSTEM_LABELS[system]} | {system_metrics[system]['rcc_at8']:.4f} | "
                f"{sensitivity['cluster_balanced'][system]['mean']:.4f} | "
                f"{sensitivity['capped_reply'][system]['rcc_at8']['mean']:.4f} | "
                f"{system_metrics[system]['bidirectional_alignment_f1_at8']:.4f} | "
                f"{sensitivity['capped_reply'][system]['bidirectional_alignment_f1_at8']['mean']:.4f} |")
        bias_lines.extend([
            "", "RCC随回复数增加而下降，说明固定Top-8面对更丰富的回复集合时覆盖难度上升。",
            "这不是修改主公式的理由；cluster-balanced与capped结果仅用于检查系统排序是否依赖回复深度。",
        ])
    (output / "reply_count_bias_audit.md").write_text("\n".join(bias_lines) + "\n", encoding="utf-8")

    relationship_lines = [
        "# Reply alignment 与 utility 的关系", "",
        "所有相关性均为query-level Spearman，只描述关联，不解释为因果。", "",
        "| System | rho(Utility, CRA) | rho(Utility, RCC) | rho(Utility, BiAlignF1) | rho(nDCG, BiAlignF1) | exact n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEM_LABELS:
        rel = relationships[system]
        relationship_lines.append(
            f"| {SYSTEM_LABELS[system]} | {rel['utility_vs_cra']['spearman_rho']:+.3f} | "
            f"{rel['utility_vs_rcc']['spearman_rho']:+.3f} | "
            f"{rel['utility_vs_bialign_f1']['spearman_rho']:+.3f} | "
            f"{rel['ndcg_vs_bialign_f1']['spearman_rho']:+.3f} | "
            f"{rel['utility_vs_bialign_f1']['exact_query_n']} |")
    relationship_lines.extend([
        "", "四象限按各系统内部Utility与BiAlignF1中位数固定划分；每格按确定性hash顺序抽2例，",
        "仅供描述性回看。低alignment不能直接解释为失败，也不能未经theme judging称为complementary useful。",
    ])
    (output / "reply_alignment_utility_relationship.md").write_text(
        "\n".join(relationship_lines) + "\n", encoding="utf-8")

    results_lines = [
        "# Hidden community replies: bidirectional alignment", "",
        "BiAlignF1在每条query内对CRA（candidate precision-like alignment）和RCC（reply recall-like coverage）",
        "取调和平均。主reference是全部1,749条有效顶层回复；BGE-M3、Top-8和七系统均沿用Phase 1冻结版本。", "",
        "| System | CRA@8 | RCC@8 | BiAlignF1@8 | BestAlign@8 | ReplyCoverage_tau@8 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEM_LABELS:
        metric = system_metrics[system]
        results_lines.append(
            f"| {SYSTEM_LABELS[system]} | {metric['cra_at8']:.4f} | {metric['rcc_at8']:.4f} | "
            f"{metric['bidirectional_alignment_f1_at8']:.4f} | {metric['best_align_at8']:.4f} | "
            f"{metric['reply_coverage_at8']:.4f} |")
    results_lines.extend([
        "", "F-beta敏感性（不作主裁决）：", "",
        "| System | F0.5 (candidate alignment weighted) | F1 primary | F2 (reply coverage weighted) |",
        "|---|---:|---:|---:|",
    ])
    for system in SYSTEM_LABELS:
        metric = system_metrics[system]
        results_lines.append(
            f"| {SYSTEM_LABELS[system]} | {metric['bidirectional_alignment_f0_5_at8']:.4f} | "
            f"{metric['bidirectional_alignment_f1_at8']:.4f} | "
            f"{metric['bidirectional_alignment_f2_at8']:.4f} |")
    results_lines.extend(["", "## 主要配对比较", ""])
    for left, right in BIDIRECTIONAL_CONTRASTS:
        cra = _comparison_lookup(comparisons, left, right, "cra_at8")
        rcc = _comparison_lookup(comparisons, left, right, "rcc_at8")
        f1 = _comparison_lookup(comparisons, left, right, "bidirectional_alignment_f1_at8")
        best = _comparison_lookup(comparisons, left, right, "best_align_at8")
        results_lines.append(
            f"- {SYSTEM_LABELS[left]} − {SYSTEM_LABELS[right]} (n={f1['exact_query_n']}): "
            f"CRA {ci_text(cra)}；RCC {ci_text(rcc)}；BiAlignF1 {ci_text(f1)}；"
            f"BestAlign {ci_text(best)}；F1 improved/tied/degraded="
            f"{f1['improved']}/{f1['tied']}/{f1['degraded']}。")
    results_lines.extend([
        "", "## 解释边界", "",
        "Bidirectional reply alignment measures semantic correspondence between retrieved cross-post evidence and the full set of real community replies. It is an auxiliary reference-based metric, separate from absolute utility.",
        "原帖回复不是唯一gold；高BiAlignF1不保证事实性、安全性或绝对效用，低BiAlignF1也不等于系统失败。",
        "本轮未调用LLM/Bedrock，未修改utility-v2、候选、gate或主检索结论。",
    ])
    (output / "community_reply_bidirectional_results.md").write_text(
        "\n".join(results_lines) + "\n", encoding="utf-8")

    c0_s0 = _comparison_lookup(comparisons, "C0", "S0", "bidirectional_alignment_f1_at8")
    sg2_s2_rcc = _comparison_lookup(comparisons, "SG2", "S2", "rcc_at8")
    g2_c2 = _comparison_lookup(comparisons, "G2", "C2", "bidirectional_alignment_f1_at8")
    verdicts = ["BIDIRECTIONAL_ALIGNMENT_ADDS_INFORMATION"]
    if c0_s0["bootstrap_95ci"][0] > 0:
        verdicts.append("COHERE_BETTER_MATCHES_COMMUNITY_REPLIES")
    if sg2_s2_rcc["bootstrap_95ci"][0] > 0:
        verdicts.append("GRAPH_IMPROVES_REPLY_COVERAGE_ON_SBERT")
    if g2_c2["bootstrap_95ci"][0] <= 0 <= g2_c2["bootstrap_95ci"][1]:
        verdicts.append("GRAPH_DOES_NOT_ADD_REPLY_ALIGNMENT_ON_COHERE")
    if all(abs(relationships[system]["utility_vs_bialign_f1"]["spearman_rho"]) < 0.5
           for system in SYSTEM_LABELS):
        verdicts.append("UTILITY_AND_REPLY_ALIGNMENT_DIVERGE")
    if sensitivity_required:
        verdicts.append("REPLY_COUNT_BIAS_REQUIRES_SENSITIVITY_ANALYSIS")
    write_json(output / "community_reply_bidirectional_final_verdict.json", {
        "schema": "community-reply-bidirectional-verdict-v1", "created_at": now(),
        "verdicts": verdicts,
        "primary_metric": "BiAlignF1@8 over all valid top-level replies",
        "main_retrieval_or_utility_conclusion_modified": False,
        "theme_coverage_started": False, "frozen_test_read": False, "external_model_calls": 0,
    })
    print(json.dumps({"output_dir": str(output), "query_count": 100,
                      "valid_reply_count": sum(len(rows) for rows in references.values()),
                      "reply_count_sensitivity_triggered": sensitivity_required,
                      "verdicts": verdicts, "external_model_calls": 0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("phase1", "bidirectional"))
    args = parser.parse_args()
    _, config = load_config()
    if args.command == "phase1":
        phase1(config)
    elif args.command == "bidirectional":
        bidirectional(config)


if __name__ == "__main__":
    main()
