#!/usr/bin/env python3
"""Cross-backend candidate-depth frontier preflight and gated analysis entry.

The current experiment configuration deliberately has both external gates
closed.  This entry therefore performs the complete local Phase 0--3
preflight:

* reproduce the frozen MiniLM D100 exactly;
* create a pinned E5-base-v2 D100 using the model-card pooling recipe;
* freeze utility-blind depth pools and backend overlaps;
* join the union against the authoritative utility-v2 registry;
* audit existing BGE-M3 embedding coverage without using replies to construct
  candidates; and
* emit a blinded, content-addressed residual judging package.

If any formal curve pair lacks utility-v2, the script stops normally with
``READY_FOR_EXPLICIT_EXTERNAL_JUDGING_AUTHORIZATION``.  It never calls an
external judge and rejects frozen-test-looking paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

try:
    from evaluation.judgment_completeness import (
        DIMS_V2,
        complete_utility_v2_rows,
    )
    from evaluation import community_reply_auxiliary as community
    from utility_scoring.annotation.run_coverage_complete_residual_judging import (
        hash_json,
        protocol_lock,
        provider_payload,
        token_estimate,
    )
    from candidate_pool.analyze_strict_sbert_graph_oracle import (
        _round_robin_graph_head,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.judgment_completeness import (
        DIMS_V2,
        complete_utility_v2_rows,
    )
    from evaluation import community_reply_auxiliary as community
    from utility_scoring.annotation.run_coverage_complete_residual_judging import (
        hash_json,
        protocol_lock,
        provider_payload,
        token_estimate,
    )
    from candidate_pool.analyze_strict_sbert_graph_oracle import (
        _round_robin_graph_head,
    )


CONFIG_KEY = "depth_graph_utility_community_frontier"
M50_CONFIG_KEY = "depth_graph_utility_community_frontier_m50"
DEPTHS = (8, 12, 20, 50, 100)
M50_DEPTHS = (8, 12, 20, 50)
RANK_BINS = (
    (1, 8, "1-8"),
    (9, 12, "9-12"),
    (13, 20, "13-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
)
UTILITY_WEIGHTS = {
    "relevance": 0.25,
    "usefulness": 0.30,
    "actionability": 0.15,
    "novelty": 0.10,
    "resonance": 0.10,
    "safety": 0.10,
}
TEST_PATH = re.compile(
    r"(^|[/_.-])(?:frozen[_-]?)?test(?:200)?($|[/_.-])",
    re.IGNORECASE,
)
READY_VERDICT = "READY_FOR_EXPLICIT_EXTERNAL_JUDGING_AUTHORIZATION"
M50_READY_VERDICT = "READY_FOR_EXPLICIT_M50_JUDGING_AUTHORIZATION"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to replace non-identical artefact: {path}")
        return
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to replace non-identical artefact: {path}")
        return
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    rendered = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != rendered.encode("utf-8"):
            raise FileExistsError(f"refusing to replace non-identical artefact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(rendered)


def reject_test_path(path: Path) -> None:
    if TEST_PATH.search(str(path.resolve())):
        raise ValueError(f"frozen-test-looking path rejected: {path}")


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_config(
    root: Path,
    path: Path,
    config_key: str = CONFIG_KEY,
) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config_key not in raw:
        raise KeyError(f"missing config block: {config_key}")
    cfg = dict(raw[config_key])
    cfg["_root"] = root
    cfg["_config_key"] = config_key
    path_keys = (
        "output_dir",
        "queries",
        "query_admin",
        "corpus",
        "minilm_rankings",
        "minilm_corpus_embeddings",
        "minilm_query_embeddings",
        "utility_registry",
        "strict_graph_pool",
        "strict_graph_manifest",
        "fixed_graph_pool",
        "fixed_graph_manifest",
        "split_manifest",
        "community_manifest",
        "community_reference_texts",
        "community_embedding_cache",
        "anchors",
        "prompt",
        "pricing_precheck",
        "source_preflight_manifest",
        "source_minilm_rankings",
        "source_e5_rankings",
        "source_minilm_manifest",
        "source_e5_manifest",
        "selector_contract_report",
    )
    for key in path_keys:
        if key not in cfg:
            continue
        cfg[key] = resolve(root, cfg[key])
        reject_test_path(cfg[key])
    for mapping_key in (
        "graph_runs",
        "graph_entry_traces",
        "graph_manifests",
    ):
        if mapping_key not in cfg:
            continue
        cfg[mapping_key] = {
            name: resolve(root, value)
            for name, value in dict(cfg[mapping_key]).items()
        }
        for candidate in cfg[mapping_key].values():
            reject_test_path(candidate)
    if "e5" in cfg:
        cfg["e5"] = dict(cfg["e5"])
        if "local_snapshot" in cfg["e5"]:
            cfg["e5"]["local_snapshot"] = resolve(
                root, cfg["e5"]["local_snapshot"]
            )
            reject_test_path(cfg["e5"]["local_snapshot"])
    if cfg.get("allow_frozen_test") is not False:
        raise ValueError("ALLOW_FROZEN_TEST must remain false")
    return cfg


def rank_bin(rank: int) -> str:
    for low, high, label in RANK_BINS:
        if low <= rank <= high:
            return label
    raise ValueError(f"rank outside frozen D100: {rank}")


def load_inputs(cfg: dict) -> tuple[list[dict], list[dict], dict, dict]:
    queries = json.loads(cfg["queries"].read_text(encoding="utf-8"))
    corpus = json.loads(cfg["corpus"].read_text(encoding="utf-8"))
    if len(queries) != 100 or len({str(row["id"]) for row in queries}) != 100:
        raise ValueError("expected exactly 100 unique development queries")
    if len(corpus) != 19013 or len({str(row["title"]) for row in corpus}) != 19013:
        raise ValueError("expected exactly 19,013 unique corpus comments")
    query_text = {str(row["id"]): str(row["question"]) for row in queries}
    corpus_text = {str(row["title"]): str(row["text"]) for row in corpus}
    if any(not text.strip() for text in query_text.values()):
        raise ValueError("empty development query text")
    if any(not text.strip() for text in corpus_text.values()):
        raise ValueError("empty corpus comment text")
    return queries, corpus, query_text, corpus_text


def load_flat_rankings(path: Path, qids: set[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        qid = str(row["query_id"])
        if qid in qids:
            grouped[qid].append(row)
    if set(grouped) != qids:
        raise ValueError("MiniLM D100 query identity mismatch")
    for qid, rows in grouped.items():
        rows.sort(key=lambda item: (int(item["rank"]), str(item["comment_id"])))
        ranks = [int(row["rank"]) for row in rows]
        ids = [str(row["comment_id"]) for row in rows]
        if ranks != list(range(1, 101)) or len(ids) != len(set(ids)):
            raise ValueError(f"{qid}: MiniLM D100 rank/duplicate failure")
    return grouped


def stable_top100(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: np.ndarray,
    qids: list[str],
    *,
    backend: str,
    model_id: str,
    revision: str,
) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    matrix = np.asarray(corpus_embeddings, dtype=np.float32)
    for index, qid in enumerate(qids):
        scores = matrix @ np.asarray(query_embeddings[index], dtype=np.float32)
        order = np.lexsort((corpus_ids, -scores))[:100]
        output[qid] = [
            {
                "backend": backend,
                "model_id": model_id,
                "revision": revision,
                "query_id": qid,
                "comment_id": str(corpus_ids[corpus_index]),
                "rank": rank,
                "score": float(scores[corpus_index]),
                "score_type": "cosine",
                "tie_break": "score_desc_then_comment_id_asc",
            }
            for rank, corpus_index in enumerate(order, start=1)
        ]
    return output


def reproduce_minilm(
    cfg: dict,
    corpus: list[dict],
    qids: list[str],
) -> tuple[dict[str, list[dict]], dict]:
    corpus_ids = np.asarray([str(row["title"]) for row in corpus], dtype=object)
    corpus_embeddings = np.load(cfg["minilm_corpus_embeddings"], mmap_mode="r")
    query_embeddings = np.load(cfg["minilm_query_embeddings"], mmap_mode="r")
    if corpus_embeddings.shape != (19013, 384):
        raise ValueError(f"unexpected MiniLM corpus shape: {corpus_embeddings.shape}")
    if query_embeddings.shape != (100, 384):
        raise ValueError(f"unexpected MiniLM query shape: {query_embeddings.shape}")
    computed = stable_top100(
        query_embeddings,
        corpus_embeddings,
        corpus_ids,
        qids,
        backend="minilm",
        model_id=cfg["minilm"]["model_id"],
        revision=cfg["minilm"]["revision"],
    )
    authoritative = load_flat_rankings(cfg["minilm_rankings"], set(qids))
    mismatches = []
    max_score_error = 0.0
    for qid in qids:
        expected_ids = [str(row["comment_id"]) for row in authoritative[qid]]
        computed_ids = [str(row["comment_id"]) for row in computed[qid]]
        if computed_ids != expected_ids:
            first = next(
                index
                for index, pair in enumerate(zip(computed_ids, expected_ids), start=1)
                if pair[0] != pair[1]
            )
            mismatches.append({"query_id": qid, "first_rank": first})
        for got, expected in zip(computed[qid], authoritative[qid], strict=True):
            max_score_error = max(
                max_score_error,
                abs(float(got["score"]) - float(expected["score"])),
            )
    if mismatches:
        raise AssertionError(f"MiniLM authoritative D100 identity failed: {mismatches[:3]}")
    return computed, {
        "identity_verdict": "EXACT_D100_IDENTITY",
        "query_count": 100,
        "rows": 10000,
        "candidate_id_rank_mismatches": 0,
        "max_absolute_score_error": max_score_error,
        "corpus_embeddings_sha256": sha256(cfg["minilm_corpus_embeddings"]),
        "query_embeddings_sha256": sha256(cfg["minilm_query_embeddings"]),
        "authoritative_rankings_sha256": sha256(cfg["minilm_rankings"]),
    }


def model_file_hashes(snapshot: Path) -> dict[str, str]:
    wanted = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "sentence_bert_config.json",
        "modules.json",
        "1_Pooling/config.json",
    )
    output = {}
    for relative in wanted:
        path = snapshot / relative
        if not path.exists():
            raise FileNotFoundError(f"pinned E5 file missing: {path}")
        output[relative] = sha256(path)
    return output


def encode_e5(
    texts: list[str],
    *,
    prefix: str,
    snapshot: Path,
    output: Path,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, dict]:
    if output.exists():
        values = np.load(output, mmap_mode="r")
        if values.shape != (len(texts), 768):
            raise ValueError(f"cached E5 shape mismatch: {output}={values.shape}")
        norms = np.linalg.norm(np.asarray(values), axis=1)
        if not np.allclose(norms, 1.0, atol=2e-5):
            raise ValueError(f"cached E5 embeddings are not L2-normalised: {output}")
        return values, {"cache_hit": True, "sha256": sha256(output)}

    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True)
    model.to(device)
    model.eval()

    # Length sorting affects compute only.  Embeddings are restored to the
    # immutable source order before they are saved.
    compute_order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    values = np.empty((len(texts), 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(compute_order), batch_size):
            indices = compute_order[start : start + batch_size]
            batch = [prefix + texts[index] for index in indices]
            tokens = tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"]
            hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
            pooled = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
            normalised = functional.normalize(pooled, p=2, dim=1)
            batch_values = normalised.cpu().to(torch.float32).numpy()
            values[np.asarray(indices)] = batch_values
            if start == 0 or (start // batch_size + 1) % 50 == 0:
                print(
                    f"E5 {output.name}: {min(start + batch_size, len(texts))}/"
                    f"{len(texts)}",
                    flush=True,
                )
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, output)
    return np.load(output, mmap_mode="r"), {
        "cache_hit": False,
        "sha256": sha256(output),
        "device": str(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def build_e5(
    cfg: dict,
    queries: list[dict],
    corpus: list[dict],
    out_dir: Path,
) -> tuple[dict[str, list[dict]], dict]:
    e5 = cfg["e5"]
    snapshot = e5["local_snapshot"]
    if snapshot.name != e5["revision"]:
        raise ValueError("E5 snapshot directory does not match frozen revision")
    backend_dir = out_dir / "backend_manifests"
    backend_dir.mkdir(parents=True, exist_ok=True)
    corpus_output = backend_dir / "e5_corpus_embeddings.npy"
    query_output = backend_dir / "e5_query_embeddings.npy"
    corpus_texts = [str(row["text"]) for row in corpus]
    query_rows = sorted(queries, key=lambda row: str(row["id"]))
    query_texts = [str(row["question"]) for row in query_rows]
    corpus_embeddings, corpus_cache = encode_e5(
        corpus_texts,
        prefix="passage: ",
        snapshot=snapshot,
        output=corpus_output,
        batch_size=int(e5["batch_size"]),
        max_length=int(e5["max_sequence_length"]),
    )
    query_embeddings, query_cache = encode_e5(
        query_texts,
        prefix="query: ",
        snapshot=snapshot,
        output=query_output,
        batch_size=int(e5["batch_size"]),
        max_length=int(e5["max_sequence_length"]),
    )
    qids = [str(row["id"]) for row in query_rows]
    corpus_ids = np.asarray([str(row["title"]) for row in corpus], dtype=object)
    rankings = stable_top100(
        query_embeddings,
        corpus_embeddings,
        corpus_ids,
        qids,
        backend="e5",
        model_id=e5["model_id"],
        revision=e5["revision"],
    )
    try:
        import sentence_transformers

        sentence_transformers_version = sentence_transformers.__version__
    except Exception as exc:  # pragma: no cover - environment audit only
        sentence_transformers_version = f"IMPORT_FAILED:{type(exc).__name__}"
    manifest = {
        "model_id": e5["model_id"],
        "revision": e5["revision"],
        "local_snapshot": str(snapshot),
        "model_files_sha256": model_file_hashes(snapshot),
        "input_prefixes": {"query": "query: ", "passage": "passage: "},
        "pooling": "attention-mask-aware average pooling",
        "normalisation": "L2",
        "similarity": "normalised inner product (cosine equivalent)",
        "dtype": "float32",
        "device": "cpu",
        "batch_size": int(e5["batch_size"]),
        "max_sequence_length": int(e5["max_sequence_length"]),
        "compute_order": "text-length sorted; output restored to frozen source order",
        "transformers_version": __import__("transformers").__version__,
        "sentence_transformers_version": sentence_transformers_version,
        "corpus_order_sha256": sha256_text(
            "\n".join(str(row["title"]) for row in corpus)
        ),
        "query_order_sha256": sha256_text("\n".join(qids)),
        "corpus_cache": corpus_cache,
        "query_cache": query_cache,
    }
    frozen_manifest_path = backend_dir / "e5_manifest.json"
    if frozen_manifest_path.exists():
        frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
        invariant_keys = (
            "model_id",
            "revision",
            "model_files_sha256",
            "input_prefixes",
            "pooling",
            "normalisation",
            "similarity",
            "dtype",
            "batch_size",
            "max_sequence_length",
            "corpus_order_sha256",
            "query_order_sha256",
        )
        if any(frozen.get(key) != manifest.get(key) for key in invariant_keys):
            raise ValueError("existing E5 manifest identity differs during resume")
        if (
            frozen["corpus_cache"]["sha256"] != corpus_cache["sha256"]
            or frozen["query_cache"]["sha256"] != query_cache["sha256"]
        ):
            raise ValueError("existing E5 embedding cache hash differs during resume")
        manifest = frozen
    return rankings, manifest


def flatten_rankings(rankings: dict[str, list[dict]]) -> list[dict]:
    return [
        row
        for qid in sorted(rankings)
        for row in rankings[qid]
    ]


def build_depth_pools(
    all_rankings: dict[str, dict[str, list[dict]]],
    depths: tuple[int, ...] = DEPTHS,
) -> tuple[list[dict], list[dict]]:
    maximum_depth = max(depths)
    rows = []
    overlap_rows = []
    for backend, rankings in all_rankings.items():
        for qid in sorted(rankings):
            for row in rankings[qid]:
                rank = int(row["rank"])
                if rank > maximum_depth:
                    continue
                rows.append({
                    **row,
                    "rank_bin": rank_bin(rank),
                    "depth_memberships": [
                        depth for depth in depths if rank <= depth
                    ],
                    "construction_used_utility": False,
                    "construction_used_community_replies": False,
                })
    left, right = all_rankings["minilm"], all_rankings["e5"]
    for depth in depths:
        per_query = []
        intersections = []
        for qid in sorted(left):
            left_ids = {str(row["comment_id"]) for row in left[qid][:depth]}
            right_ids = {str(row["comment_id"]) for row in right[qid][:depth]}
            intersection = len(left_ids & right_ids)
            intersections.append(intersection)
            per_query.append(intersection / len(left_ids | right_ids))
        overlap_rows.append({
            "depth": depth,
            "mean_intersection_count": float(np.mean(intersections)),
            "min_intersection_count": min(intersections),
            "max_intersection_count": max(intersections),
            "mean_jaccard": float(np.mean(per_query)),
            "query_count": len(per_query),
        })
    return rows, overlap_rows


def utility_formula_audit(registry: dict[tuple[str, str], dict]) -> dict:
    mismatches = []
    maximum_error = 0.0
    for (qid, cid), row in registry.items():
        linear = sum(
            weight * float(row[f"label_{dimension}"])
            for dimension, weight in UTILITY_WEIGHTS.items()
        )
        expected = min(linear, 2.0) if float(row["label_safety"]) <= 2 else linear
        error = abs(float(row["utility"]) - expected)
        maximum_error = max(maximum_error, error)
        if error > 1e-6:
            mismatches.append({
                "query_id": qid,
                "comment_id": cid,
                "stored": float(row["utility"]),
                "expected": expected,
            })
    return {
        "formula": "0.25R+0.30H+0.15A+0.10N+0.10E+0.10S",
        "safety_gate": "if S<=2, utility=min(linear,2.0)",
        "complete_pairs_checked": len(registry),
        "mismatch_count": len(mismatches),
        "max_absolute_error": maximum_error,
        "verdict": "PASS" if not mismatches else "FAIL",
        "mismatch_examples": mismatches[:10],
    }


def existing_community_cache_keys(cfg: dict, corpus_text: dict[str, str]) -> set[str]:
    # Reconstruct the exact text-key set used by the historical BGE-M3 cache.
    # This occurs only after candidate pools have been frozen.
    _, community_cfg = community.load_config()
    qids = set(
        str(row["query_id"])
        for row in read_jsonl(cfg["community_reference_texts"])
    )
    systems, _ = community.load_systems(
        community_cfg,
        qids,
        set(corpus_text),
    )
    texts = [
        str(reply["text"])
        for row in read_jsonl(cfg["community_reference_texts"])
        for reply in row["replies"]
    ]
    texts.extend(
        corpus_text[cid]
        for run in systems.values()
        for ids in run.values()
        for cid in ids
    )
    keys = {community.text_sha(text) for text in texts}
    with np.load(cfg["community_embedding_cache"], allow_pickle=False) as cache:
        count = int(cache["embeddings"].shape[0])
    if len(keys) != count:
        raise ValueError(
            f"reconstructed BGE-M3 cache identity mismatch: {len(keys)} != {count}"
        )
    return keys


def coverage_report(
    cfg: dict,
    depth_rows: list[dict],
    registry: dict[tuple[str, str], dict],
    query_text: dict[str, str],
    corpus_text: dict[str, str],
) -> tuple[dict, list[dict], list[dict]]:
    cache_keys = existing_community_cache_keys(cfg, corpus_text)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    pair_membership: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"backends": set(), "backend_ranks": {}}
    )
    for row in depth_rows:
        backend = str(row["backend"])
        qid = str(row["query_id"])
        cid = str(row["comment_id"])
        rank = int(row["rank"])
        pair_membership[(qid, cid)]["backends"].add(backend)
        pair_membership[(qid, cid)]["backend_ranks"][backend] = rank
        for depth in DEPTHS:
            if rank <= depth:
                grouped[(backend, depth)].append(row)

    by_pool = []
    for (backend, depth), rows in sorted(grouped.items()):
        pairs = [(str(row["query_id"]), str(row["comment_id"])) for row in rows]
        judged = sum(pair in registry for pair in pairs)
        embedded = sum(
            community.text_sha(corpus_text[pair[1]]) in cache_keys for pair in pairs
        )
        by_pool.append({
            "backend": backend,
            "depth": depth,
            "pair_rows": len(pairs),
            "unique_pairs": len(set(pairs)),
            "judged_pairs": judged,
            "missing_utility_pairs": len(pairs) - judged,
            "utility_coverage_rate": judged / len(pairs),
            "existing_bge_m3_candidate_embeddings": embedded,
            "missing_bge_m3_candidate_embeddings": len(pairs) - embedded,
            "existing_bge_m3_coverage_rate": embedded / len(pairs),
        })

    residual = []
    for (qid, cid), membership in sorted(pair_membership.items()):
        if (qid, cid) in registry:
            continue
        payload = provider_payload(
            query_text[qid],
            corpus_text[cid],
        )
        residual.append({
            "payload_index": len(residual),
            "query_id": qid,
            "comment_id": cid,
            "backends": sorted(membership["backends"]),
            "backend_ranks": dict(sorted(membership["backend_ranks"].items())),
            "depth_memberships": {
                backend: [
                    depth
                    for depth in DEPTHS
                    if rank <= depth
                ]
                for backend, rank in sorted(membership["backend_ranks"].items())
            },
            "provider_payload_sha256": hash_json(payload),
            "external_call_authorised": False,
            "test_split": False,
        })
    blind = [
        provider_payload(
            query_text[item["query_id"]],
            corpus_text[item["comment_id"]],
        )
        for item in residual
    ]
    union_pairs = set(pair_membership)
    report = {
        "authoritative_registry_complete_pairs": len(registry),
        "formal_union_unique_pairs": len(union_pairs),
        "formal_union_judged_pairs": len(union_pairs & set(registry)),
        "formal_union_missing_pairs": len(residual),
        "formal_union_utility_coverage_rate": (
            len(union_pairs & set(registry)) / len(union_pairs)
        ),
        "existing_bge_m3_cache_unique_texts": len(cache_keys),
        "by_backend_depth": by_pool,
        "community_embedding_coverage_note": (
            "This reports coverage in the already frozen BGE-M3 cache only. "
            "Missing candidate embeddings are not generated before the utility gate."
        ),
    }
    return report, residual, blind


def build_anchor_payloads(
    cfg: dict,
    query_text: dict[str, str],
    corpus_text: dict[str, str],
    registry: dict[tuple[str, str], dict],
) -> tuple[list[dict], list[dict]]:
    anchors = read_jsonl(cfg["anchors"])
    if len(anchors) != 50:
        raise ValueError(f"expected 50 calibration anchors, got {len(anchors)}")
    provider_rows = []
    admin_rows = []
    for index, row in enumerate(anchors):
        qid = str(row["query_id"])
        cid = str(row["comment_id"])
        if (qid, cid) not in registry:
            raise ValueError(f"anchor absent from authoritative registry: {(qid, cid)}")
        payload = provider_payload(query_text[qid], corpus_text[cid])
        provider_rows.append(payload)
        admin_rows.append({
            "payload_index": index,
            "query_id": qid,
            "comment_id": cid,
            "historical_utility": float(registry[(qid, cid)]["utility"]),
            "historical_scores": {
                dimension: int(registry[(qid, cid)][f"label_{dimension}"])
                for dimension in DIMS_V2
            },
            "provider_payload_sha256": hash_json(payload),
        })
    return provider_rows, admin_rows


def cost_estimate(
    cfg: dict,
    residual_payloads: list[dict],
    anchor_payloads: list[dict],
) -> dict:
    residual = token_estimate(residual_payloads)
    anchors = token_estimate(anchor_payloads)
    total = token_estimate(residual_payloads + anchor_payloads)
    pricing = json.loads(cfg["pricing_precheck"].read_text(encoding="utf-8"))
    price = pricing["official_pricing"]
    input_rate = float(price["input_usd_per_1k_tokens"])
    output_rate = float(price["output_usd_per_1k_tokens"])
    first_pass = (
        total["estimated_input_tokens"] / 1000 * input_rate
        + total["estimated_output_tokens"] / 1000 * output_rate
    )
    return {
        "model": pricing["model_access"]["inference_profile_id"],
        "temperature": 0.0,
        "residual": residual,
        "anchors": anchors,
        "total": total,
        "official_pricing_reused_from_frozen_precheck": {
            "source_path": str(cfg["pricing_precheck"]),
            "source_sha256": sha256(cfg["pricing_precheck"]),
            "input_usd_per_1k_tokens": input_rate,
            "output_usd_per_1k_tokens": output_rate,
            "effective_date": price["effective_date"],
            "lookup_date": price["lookup_date"],
        },
        "estimated_first_pass_usd": first_pass,
        "external_requests_executed": 0,
    }


def verify_frozen_d100_source(
    cfg: dict,
    qids: set[str],
) -> tuple[dict[str, dict[str, list[dict]]], dict]:
    manifest = json.loads(
        cfg["source_preflight_manifest"].read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != READY_VERDICT
        or manifest.get("frozen_test_read") is not False
        or int(manifest.get("external_judging_calls") or 0) != 0
    ):
        raise ValueError("Report 94 source preflight boundary changed")
    paths = {
        "minilm_rankings": cfg["source_minilm_rankings"],
        "e5_rankings": cfg["source_e5_rankings"],
        "minilm_manifest": cfg["source_minilm_manifest"],
        "e5_manifest": cfg["source_e5_manifest"],
    }
    source_root = cfg["source_preflight_manifest"].parent
    for name, path in paths.items():
        relative = str(path.relative_to(source_root))
        expected = manifest["output_hashes"].get(relative)
        if expected is None or sha256(path) != expected:
            raise ValueError(f"frozen D100 source hash changed: {name}")
    minilm_manifest = json.loads(
        cfg["source_minilm_manifest"].read_text(encoding="utf-8")
    )
    e5_manifest = json.loads(
        cfg["source_e5_manifest"].read_text(encoding="utf-8")
    )
    if (
        minilm_manifest.get("identity_verdict") != "EXACT_D100_IDENTITY"
        or int(minilm_manifest.get("candidate_id_rank_mismatches", -1)) != 0
    ):
        raise ValueError("MiniLM D100 identity is no longer exact")
    rankings = {
        "minilm": load_flat_rankings(cfg["source_minilm_rankings"], qids),
        "e5": load_flat_rankings(cfg["source_e5_rankings"], qids),
    }
    return rankings, {
        "status": "PASS",
        "source_report": "docs_v2/94",
        "source_manifest_sha256": sha256(cfg["source_preflight_manifest"]),
        "source_status": manifest["status"],
        "d100_retained_as_retrieval_artefact_only": True,
        "d100_enters_judging_union": False,
        "d100_enters_selector_or_curves": False,
        "minilm": {
            "rank_identity": minilm_manifest["identity_verdict"],
            "rankings_sha256": sha256(cfg["source_minilm_rankings"]),
            "manifest_sha256": sha256(cfg["source_minilm_manifest"]),
        },
        "e5": {
            "model_id": e5_manifest["model_id"],
            "revision": e5_manifest["revision"],
            "rankings_sha256": sha256(cfg["source_e5_rankings"]),
            "manifest_sha256": sha256(cfg["source_e5_manifest"]),
        },
    }


def load_native_graph_routes(
    cfg: dict,
    qids: set[str],
    corpus_ids: set[str],
) -> tuple[dict[str, dict[str, dict]], dict]:
    routes: dict[str, dict[str, dict]] = {}
    route_audit = {}
    expected_routes = set(cfg["graph_runs"])
    if (
        set(cfg["graph_entry_traces"]) != expected_routes
        or set(cfg["graph_manifests"]) != expected_routes
    ):
        raise ValueError("Graph route/run/trace/manifest names differ")
    for route in sorted(expected_routes):
        run_path = cfg["graph_runs"][route]
        trace_path = cfg["graph_entry_traces"][route]
        manifest_path = cfg["graph_manifests"][route]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            bool(manifest.get("test_split_used"))
            or int(manifest.get("top_k") or 0) < 100
            or not bool(
                (manifest.get("retrieval_ablation") or {}).get("ppr_enabled")
            )
        ):
            raise ValueError(f"{route}: frozen native-PPR manifest failed")
        run_rows = {
            str(row["query_id"]): row
            for row in read_jsonl(run_path)
            if str(row["query_id"]) in qids
        }
        trace_rows = {
            str(row["query_id"]): row
            for row in read_jsonl(trace_path)
            if str(row["query_id"]) in qids
        }
        if set(run_rows) != qids or set(trace_rows) != qids:
            raise ValueError(f"{route}: development100 query identity mismatch")
        native_rows: dict[str, dict] = {}
        for qid in sorted(qids):
            if int(trace_rows[qid].get("selected_fact_count") or 0) <= 0:
                raise ValueError(f"{route}/{qid}: route entered Dense fallback")
            ids = [str(value) for value in run_rows[qid]["retrieved_titles"][:100]]
            scores = [
                float(value)
                for value in run_rows[qid]["retrieved_scores"][:100]
            ]
            if (
                len(ids) != 100
                or len(scores) != 100
                or len(ids) != len(set(ids))
                or not set(ids) <= corpus_ids
            ):
                raise ValueError(f"{route}/{qid}: Graph Top-100 identity failed")
            if not all(math.isfinite(score) and score > 0 for score in scores):
                raise ValueError(f"{route}/{qid}: non-positive native PPR score")
            native_rows[qid] = {
                "ids": ids,
                "scores": scores,
                "selected_fact_count": int(
                    trace_rows[qid]["selected_fact_count"]
                ),
            }
        routes[route] = native_rows
        route_audit[route] = {
            "queries": len(native_rows),
            "top100_rows": len(native_rows) * 100,
            "native_ppr_queries": sum(
                row["selected_fact_count"] > 0
                for row in native_rows.values()
            ),
            "fallback_queries": 0,
            "callback_candidates": 0,
            "padding_candidates": 0,
            "run_sha256": sha256(run_path),
            "trace_sha256": sha256(trace_path),
            "manifest_sha256": sha256(manifest_path),
        }
    return routes, {
        "status": "PASS",
        "routes": route_audit,
        "candidate_construction_used_utility": False,
        "candidate_construction_used_community_replies": False,
    }


def graph_route_provenance(
    routes: dict[str, dict[str, dict]],
    qid: str,
    candidate_id: str,
) -> dict:
    route_ranks = {}
    route_scores = {}
    for route, by_query in routes.items():
        ids = by_query[qid]["ids"]
        if candidate_id not in ids:
            continue
        rank = ids.index(candidate_id) + 1
        route_ranks[route] = rank
        route_scores[route] = by_query[qid]["scores"][rank - 1]
    if not route_ranks:
        raise ValueError(f"{qid}/{candidate_id}: no native Graph route")
    if not all(
        math.isfinite(float(score)) and float(score) > 0
        for score in route_scores.values()
    ):
        raise ValueError(f"{qid}/{candidate_id}: invalid native Graph score")
    return {
        "graph_routes": sorted(route_ranks),
        "graph_pre_fallback_rank": dict(sorted(route_ranks.items())),
        "native_graph_score": dict(sorted(route_scores.items())),
        "native_graph": True,
        "fallback_used": False,
        "callback_used": False,
        "padding_used": False,
    }


def build_m50_graph_views(
    *,
    rankings: dict[str, dict[str, list[dict]]],
    routes: dict[str, dict[str, dict]],
    strict_graph_rows: list[dict],
    graph_budget: int,
    fixed_graph_source: str | None = None,
) -> tuple[list[dict], dict]:
    qids = set(rankings["minilm"])
    fixed_by_query: dict[str, list[dict]] = defaultdict(list)
    for row in strict_graph_rows:
        if (
            fixed_graph_source is not None
            and str(row.get("source_pool")) != fixed_graph_source
        ):
            continue
        qid = str(row["query_id"])
        if qid in qids:
            fixed_by_query[qid].append(row)
    views = []
    for qid in sorted(qids):
        fixed = sorted(
            fixed_by_query[qid],
            key=lambda row: int(
                row.get("reported_source_rank", row.get("source_rank"))
            ),
        )
        if len(fixed) != graph_budget:
            raise ValueError(f"{qid}: fixed Graph4 size changed")
        for expected_rank, row in enumerate(fixed, start=1):
            if not (
                row.get("native_graph") is True
                and not bool(row.get("fallback_used"))
                and not bool(row.get("callback_used"))
                and not bool(row.get("padding_used"))
            ):
                raise ValueError(f"{qid}: fixed Graph4 provenance failed")
            candidate_id = str(row["candidate_id"])
            provenance = graph_route_provenance(routes, qid, candidate_id)
            views.append({
                "query_id": qid,
                "comment_id": candidate_id,
                "view_type": "fixed_graph4",
                "backend": None,
                "depth": None,
                "graph_view_rank": expected_rank,
                "construction_used_utility": False,
                "construction_used_community_replies": False,
                **provenance,
            })
    fixed_pairs = {
        (str(row["query_id"]), str(row["comment_id"])) for row in views
    }
    for backend in ("minilm", "e5"):
        for depth in M50_DEPTHS:
            for qid in sorted(qids):
                dense_ids = {
                    str(row["comment_id"])
                    for row in rankings[backend][qid][:depth]
                }
                route_ids = {
                    route: rows[qid]["ids"]
                    for route, rows in routes.items()
                }
                selected = _round_robin_graph_head(
                    route_ids,
                    dense_ids,
                    graph_budget,
                )
                if len(selected) != graph_budget:
                    raise ValueError(
                        f"{backend}/D{depth}/{qid}: residual Graph4 shortfall"
                    )
                for graph_rank, candidate_id in enumerate(selected, start=1):
                    provenance = graph_route_provenance(
                        routes, qid, candidate_id
                    )
                    views.append({
                        "query_id": qid,
                        "comment_id": candidate_id,
                        "view_type": "residual_graph4",
                        "backend": backend,
                        "depth": depth,
                        "graph_view_rank": graph_rank,
                        "outside_dense_depth": True,
                        "also_fixed_graph4": (qid, candidate_id) in fixed_pairs,
                        "construction_used_utility": False,
                        "construction_used_community_replies": False,
                        **provenance,
                    })
    residual_rows = [
        row for row in views if row["view_type"] == "residual_graph4"
    ]
    return views, {
        "status": "PASS",
        "fixed_graph4_rows": len(fixed_pairs),
        "residual_graph4_membership_rows": len(residual_rows),
        "residual_graph4_unique_pairs": len({
            (str(row["query_id"]), str(row["comment_id"]))
            for row in residual_rows
        }),
        "fixed_candidates_per_query": graph_budget,
        "residual_candidates_per_backend_depth_query": graph_budget,
        "strict_native_candidates": len(views),
        "fallback_candidates": 0,
        "callback_candidates": 0,
        "padding_candidates": 0,
    }


def build_m50_union(
    depth_rows: list[dict],
    graph_views: list[dict],
) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    membership: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "dense": {},
            "fixed_graph4": False,
            "residual_graph4": [],
        }
    )
    for row in depth_rows:
        pair = (str(row["query_id"]), str(row["comment_id"]))
        membership[pair]["dense"][str(row["backend"])] = {
            "rank": int(row["rank"]),
            "rank_bin": str(row["rank_bin"]),
            "depth_memberships": list(row["depth_memberships"]),
        }
    for row in graph_views:
        pair = (str(row["query_id"]), str(row["comment_id"]))
        if row["view_type"] == "fixed_graph4":
            membership[pair]["fixed_graph4"] = True
        else:
            membership[pair]["residual_graph4"].append({
                "backend": str(row["backend"]),
                "depth": int(row["depth"]),
                "graph_view_rank": int(row["graph_view_rank"]),
            })
    rows = []
    for (qid, cid), sources in sorted(membership.items()):
        rows.append({
            "query_id": qid,
            "comment_id": cid,
            "dense_memberships": dict(sorted(sources["dense"].items())),
            "fixed_graph4": bool(sources["fixed_graph4"]),
            "residual_graph4_memberships": sorted(
                sources["residual_graph4"],
                key=lambda row: (
                    row["backend"],
                    row["depth"],
                    row["graph_view_rank"],
                ),
            ),
            "construction_used_utility": False,
            "construction_used_community_replies": False,
            "maximum_dense_rank_in_scope": 50,
            "m100_in_scope": False,
        })
    return rows, membership


def m50_coverage_report(
    *,
    depth_rows: list[dict],
    graph_views: list[dict],
    formal_rows: list[dict],
    membership: dict[tuple[str, str], dict],
    registry: dict[tuple[str, str], dict],
) -> tuple[dict, list[dict], list[dict], dict]:
    registry_pairs = set(registry)
    formal_pairs = set(membership)
    residual_pairs = formal_pairs - registry_pairs
    backend_required: dict[str, set[tuple[str, str]]] = {}
    dense_by_backend = {
        backend: {
            (str(row["query_id"]), str(row["comment_id"]))
            for row in depth_rows
            if str(row["backend"]) == backend
        }
        for backend in ("minilm", "e5")
    }
    fixed_pairs = {
        (str(row["query_id"]), str(row["comment_id"]))
        for row in graph_views
        if row["view_type"] == "fixed_graph4"
    }
    for backend in ("minilm", "e5"):
        residual_graph_pairs = {
            (str(row["query_id"]), str(row["comment_id"]))
            for row in graph_views
            if row["view_type"] == "residual_graph4"
            and str(row["backend"]) == backend
        }
        backend_required[backend] = (
            dense_by_backend[backend] | fixed_pairs | residual_graph_pairs
        )
    backend_residual = {
        backend: pairs & residual_pairs
        for backend, pairs in backend_required.items()
    }
    dense_residual = {
        backend: pairs & residual_pairs
        for backend, pairs in dense_by_backend.items()
    }
    overlap = {
        "backend_required_pool_residual_overlap": len(
            backend_residual["minilm"] & backend_residual["e5"]
        ),
        "dense_d50_residual_overlap": len(
            dense_residual["minilm"] & dense_residual["e5"]
        ),
        "minilm_required_residual_pairs": len(backend_residual["minilm"]),
        "e5_required_residual_pairs": len(backend_residual["e5"]),
        "minilm_dense_d50_residual_pairs": len(dense_residual["minilm"]),
        "e5_dense_d50_residual_pairs": len(dense_residual["e5"]),
    }
    rank_rows = []
    for backend in ("minilm", "e5"):
        for label in ("1-8", "9-12", "13-20", "21-50"):
            pairs = {
                (str(row["query_id"]), str(row["comment_id"]))
                for row in depth_rows
                if str(row["backend"]) == backend
                and str(row["rank_bin"]) == label
            }
            rank_rows.append({
                "backend": backend,
                "source_view": "dense",
                "rank_bin_or_depth": label,
                "candidate_pair_memberships": len(pairs),
                "residual_unique_pairs": len(pairs & residual_pairs),
            })
        rank_rows.append({
            "backend": backend,
            "source_view": "fixed_graph4",
            "rank_bin_or_depth": "fixed4",
            "candidate_pair_memberships": len(fixed_pairs),
            "residual_unique_pairs": len(fixed_pairs & residual_pairs),
        })
        for depth in M50_DEPTHS:
            pairs = {
                (str(row["query_id"]), str(row["comment_id"]))
                for row in graph_views
                if row["view_type"] == "residual_graph4"
                and str(row["backend"]) == backend
                and int(row["depth"]) == depth
            }
            rank_rows.append({
                "backend": backend,
                "source_view": "residual_graph4",
                "rank_bin_or_depth": f"D{depth}",
                "candidate_pair_memberships": len(pairs),
                "residual_unique_pairs": len(pairs & residual_pairs),
            })
    by_backend_depth = []
    for backend in ("minilm", "e5"):
        backend_rows = [
            row for row in depth_rows if str(row["backend"]) == backend
        ]
        for depth in M50_DEPTHS:
            pairs = {
                (str(row["query_id"]), str(row["comment_id"]))
                for row in backend_rows
                if int(row["rank"]) <= depth
            }
            judged = len(pairs & registry_pairs)
            by_backend_depth.append({
                "backend": backend,
                "depth": depth,
                "unique_pairs": len(pairs),
                "judged_pairs": judged,
                "missing_utility_pairs": len(pairs) - judged,
                "utility_coverage_rate": judged / len(pairs),
            })
    residual_rows = []
    formal_by_pair = {
        (str(row["query_id"]), str(row["comment_id"])): row
        for row in formal_rows
    }
    for payload_index, pair in enumerate(sorted(residual_pairs)):
        row = formal_by_pair[pair]
        residual_rows.append({
            "payload_index": payload_index,
            **row,
            "external_call_authorised": False,
            "test_split": False,
        })
    report = {
        "authoritative_registry_complete_pairs": len(registry),
        "formal_union_definition": (
            "MiniLM D50 union E5 D50 union fixed Graph4 union every "
            "backend/depth-specific strict residual Graph4(M), M<=50"
        ),
        "formal_union_unique_pairs": len(formal_pairs),
        "formal_union_judged_pairs": len(formal_pairs & registry_pairs),
        "formal_union_missing_pairs": len(residual_pairs),
        "formal_union_utility_coverage_rate": (
            len(formal_pairs & registry_pairs) / len(formal_pairs)
        ),
        "by_backend_depth_dense_only": by_backend_depth,
        "residual_overlap": overlap,
        "m100_candidate_pairs_in_formal_union": 0,
        "no_additional_exploratory_candidates": True,
    }
    return report, residual_rows, [], {
        "rank_rows": rank_rows,
        "residual_pairs": residual_pairs,
        "overlap": overlap,
    }


def m50_community_workload(
    cfg: dict,
    formal_rows: list[dict],
    corpus_text: dict[str, str],
) -> tuple[dict, list[dict]]:
    cache_keys = existing_community_cache_keys(cfg, corpus_text)
    pairs_by_hash: dict[str, set[tuple[str, str]]] = defaultdict(set)
    ids_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in formal_rows:
        cid = str(row["comment_id"])
        key = community.text_sha(corpus_text[cid])
        pairs_by_hash[key].add((str(row["query_id"]), cid))
        ids_by_hash[key].add(cid)
    missing_keys = sorted(set(pairs_by_hash) - cache_keys)
    rows = [
        {
            "text_sha256": key,
            "candidate_ids": sorted(ids_by_hash[key]),
            "query_candidate_pairs": len(pairs_by_hash[key]),
            "local_encoder": "frozen BGE-M3 reference encoder",
            "external_api_required": False,
        }
        for key in missing_keys
    ]
    return {
        "formal_union_unique_candidate_texts": len(pairs_by_hash),
        "already_cached_unique_candidate_texts": len(
            set(pairs_by_hash) & cache_keys
        ),
        "missing_unique_candidate_texts_to_encode_locally": len(missing_keys),
        "missing_unique_candidate_ids": len({
            candidate_id
            for key in missing_keys
            for candidate_id in ids_by_hash[key]
        }),
        "formal_query_candidate_pairs_using_missing_text": sum(
            len(pairs_by_hash[key]) for key in missing_keys
        ),
        "existing_reference_cache_unique_texts": len(cache_keys),
        "encoding_deferred_until_candidate_and_oof_outputs_are_frozen": True,
        "community_metrics_used_for_selection": False,
        "external_api_required": False,
    }, rows


def build_m50_freeze_documents(cfg: dict) -> tuple[dict, dict, dict]:
    selector_report = json.loads(
        cfg["selector_contract_report"].read_text(encoding="utf-8")
    )
    selector_contract = dict(selector_report["selector_contract"])
    source_policy = yaml.safe_load(
        (cfg["_root"] / "configuration" / "params.yaml").read_text(encoding="utf-8")
    )["dense_semantic_drift_rescue_audit_v3"]
    freeze = {
        "status": "PRE_REGISTERED_WAITING_FOR_COMPLETE_M50_UTILITY",
        "dataset": "existing development100 only",
        "final_evidence_budget_k": 8,
        "candidate_depths": list(M50_DEPTHS),
        "backends": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "intfloat/e5-base-v2",
        ],
        "selector_family": "Direct Huber utility-oriented fixed-budget selection",
        "selector_contract": selector_contract,
        "hyperparameters": {
            "direct_delta": source_policy["direct_delta"],
            "kappa_options": source_policy["kappa_options"],
            "threshold_quantiles": source_policy["threshold_quantiles"],
            "inner_folds": source_policy["inner_folds"],
            "bootstrap_samples": source_policy["bootstrap_samples"],
            "bootstrap_seed": source_policy["bootstrap_seed"],
        },
        "multi_action_rule": {
            "method": "deterministic sequential application of one frozen predictor",
            "maximum_final_items": 8,
            "candidate_reuse": False,
            "NO_OP_or_stop_available_each_step": True,
            "gold_utility_used_to_stop": False,
            "one_swap_sensitivity_retained": True,
        },
        "same_family_features_splits_and_grids_for_every_backend_depth": True,
        "manual_per_depth_optimisation": False,
        "community_metrics_enter_selector_or_depth_selection": False,
        "m100_excluded": True,
        "analysis_outputs_intentionally_absent_until_coverage_complete": [
            "m50_depth_metrics.csv",
            "m50_selection_burden.csv",
            "m50_similarity_utility_actions.csv",
            "m50_graph_absorption.csv",
            "m50_graph_residual_value.csv",
            "m50_utility_community_metrics.csv",
            "m50_action_utility_alignment.csv",
            "frozen_figures_and_source_csvs",
        ],
    }
    continuation = {
        "status": "NOT_EVALUATED_M50_UTILITY_INCOMPLETE",
        "automatic_m100_continuation": False,
        "m100_may_enter_only_if_all_conditions_hold": [
            "Oracle marginal headroom from M=20 to M=50 is positive",
            "realised Utility from M=20 to M=50 is positive",
            "M=50 belongs to the one-standard-error eligible set",
            "conversion@50 is at least 50% of conversion@20",
            "neither Dense realised nor Graph marginal-value curve shows a clear plateau by M=50",
        ],
        "failure_verdict": "TOP100_NOT_REQUIRED_BY_PRE_REGISTERED_RULE",
        "success_verdict": "TOP100_CONTINUATION_PREFLIGHT_REQUIRED",
        "utility_or_community_results_used_to_define_rule": False,
    }
    replication = {
        "status": "NOT_PREPARED_BEFORE_DEV100_COHORT_DECISION",
        "remaining_development298_accessed": False,
        "frozen_test_accessed": False,
        "trigger_required": "READY_FOR_REPLICATION100_FROZEN_TRANSFER",
        "planned_only": {
            "sample_size": 100,
            "source": "remaining development298 after exact dev100 exclusion",
            "sampling": (
                "deterministic stratified hash ordering within the already frozen "
                "single/multi need strata; exact seed/hash function to be frozen "
                "only after trigger A without reading test200"
            ),
            "execution_authorised": False,
        },
        "possible_dev100_decisions": [
            "READY_FOR_REPLICATION100_FROZEN_TRANSFER",
            "ADDITIONAL_DEV100_DEPTH_EVIDENCE_REQUIRED",
            "TOP100_CONTINUATION_PREFLIGHT_REQUIRED",
        ],
    }
    return freeze, continuation, replication


def run_m50(cfg: dict) -> dict:
    out_dir = cfg["output_dir"]
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {out_dir}")
    preflight_dir = out_dir / "m50_residual_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    queries, corpus, query_text, corpus_text = load_inputs(cfg)
    qids = set(query_text)
    corpus_ids = set(corpus_text)
    rankings, source_audit = verify_frozen_d100_source(cfg, qids)
    depth_rows, overlap_rows = build_depth_pools(rankings, M50_DEPTHS)
    if (
        len(depth_rows) != 10000
        or any(int(row["rank"]) > 50 for row in depth_rows)
        or any(100 in row["depth_memberships"] for row in depth_rows)
    ):
        raise AssertionError("M50 Dense pool boundary failed")

    routes, route_audit = load_native_graph_routes(cfg, qids, corpus_ids)
    graph_views, graph_audit = build_m50_graph_views(
        rankings=rankings,
        routes=routes,
        strict_graph_rows=read_jsonl(cfg["fixed_graph_pool"]),
        graph_budget=int(cfg["graph_budget"]),
        fixed_graph_source=str(cfg["fixed_graph_source"]),
    )
    formal_rows, membership = build_m50_union(depth_rows, graph_views)

    registry_rows = read_jsonl(cfg["utility_registry"])
    complete_rows, registry = complete_utility_v2_rows(registry_rows)
    if len(complete_rows) != 14290:
        raise ValueError(
            f"authoritative registry changed: expected 14,290, got {len(complete_rows)}"
        )
    formula_audit = utility_formula_audit(registry)
    if formula_audit["verdict"] != "PASS":
        raise AssertionError("authoritative utility-v2 formula audit failed")
    coverage, residual, _, coverage_admin = m50_coverage_report(
        depth_rows=depth_rows,
        graph_views=graph_views,
        formal_rows=formal_rows,
        membership=membership,
        registry=registry,
    )
    blind = []
    for row in residual:
        pair = (str(row["query_id"]), str(row["comment_id"]))
        payload = provider_payload(query_text[pair[0]], corpus_text[pair[1]])
        row["provider_payload_sha256"] = hash_json(payload)
        blind.append(payload)
    if len(blind) != coverage["formal_union_missing_pairs"]:
        raise AssertionError("M50 residual payload count mismatch")
    if any(
        int(details["rank"]) > 50
        for row in residual
        for details in row["dense_memberships"].values()
    ):
        raise AssertionError("M100 candidate leaked into residual")

    anchors, anchor_admin = build_anchor_payloads(
        cfg, query_text, corpus_text, registry
    )
    workload, workload_rows = m50_community_workload(
        cfg, formal_rows, corpus_text
    )
    freeze, continuation, replication = build_m50_freeze_documents(cfg)
    protocol = protocol_lock(cfg["prompt"], registry_rows)
    protocol["experiment_gate"] = {
        "allow_external_judging": False,
        "allow_frozen_test": False,
        "external_requests_executed": 0,
        "scope": "development100 M<=50 formal union only",
        "m100_excluded": True,
    }

    write_jsonl(preflight_dir / "dense_m50_memberships.jsonl", depth_rows)
    write_csv(preflight_dir / "backend_overlap_m50.csv", overlap_rows)
    write_jsonl(preflight_dir / "graph_candidate_views.jsonl", graph_views)
    write_jsonl(preflight_dir / "formal_union_manifest.jsonl", formal_rows)
    write_json(preflight_dir / "judgment_coverage.json", coverage)
    write_json(
        preflight_dir / "residual_overlap.json",
        coverage_admin["overlap"],
    )
    write_csv(
        preflight_dir / "residual_by_backend_rank_bin.csv",
        coverage_admin["rank_rows"],
    )
    write_jsonl(preflight_dir / "residual_union_manifest.jsonl", residual)
    write_jsonl(preflight_dir / "blinded_judging_payload.jsonl", blind)
    write_jsonl(preflight_dir / "anchor_payload.jsonl", anchors)
    write_jsonl(preflight_dir / "anchor_payload_admin.jsonl", anchor_admin)
    write_json(preflight_dir / "judge_protocol_lock.json", protocol)
    costs = cost_estimate(cfg, blind, anchors)
    write_json(preflight_dir / "calls_token_cost_estimate.json", costs)
    write_json(preflight_dir / "utility_formula_audit.json", formula_audit)
    write_json(preflight_dir / "frozen_d100_source_audit.json", source_audit)
    write_json(preflight_dir / "strict_graph_route_audit.json", route_audit)
    write_json(preflight_dir / "graph_view_audit.json", graph_audit)
    write_json(preflight_dir / "local_bge_m3_workload.json", workload)
    write_jsonl(
        preflight_dir / "local_bge_m3_workload_manifest.jsonl",
        workload_rows,
    )
    write_json(out_dir / "m50_depth_selection_freeze.json", freeze)
    write_json(out_dir / "m100_continuation_decision.json", continuation)
    write_json(out_dir / "replication100_preflight_plan.json", replication)

    continuation_text = (
        "External judging is intentionally disabled. A future explicit "
        "authorisation must name exactly the frozen M<=50 residual count and "
        "50 anchors in this version. Run anchors first; continue only after "
        "STABLE or STABLE_WITH_MINOR_DRIFT. Do not send or analyse D100 and "
        "do not read frozen test200.\n"
    )
    continuation_path = preflight_dir / "continuation_boundary.txt"
    if continuation_path.exists():
        if continuation_path.read_text(encoding="utf-8") != continuation_text:
            raise FileExistsError(continuation_path)
    else:
        continuation_path.write_text(continuation_text, encoding="utf-8")

    input_paths = [
        cfg["queries"],
        cfg["query_admin"],
        cfg["corpus"],
        cfg["source_preflight_manifest"],
        cfg["source_minilm_rankings"],
        cfg["source_e5_rankings"],
        cfg["source_minilm_manifest"],
        cfg["source_e5_manifest"],
        cfg["utility_registry"],
        cfg["fixed_graph_pool"],
        cfg["fixed_graph_manifest"],
        cfg["split_manifest"],
        cfg["selector_contract_report"],
        cfg["community_manifest"],
        cfg["community_reference_texts"],
        cfg["community_embedding_cache"],
        cfg["anchors"],
        cfg["prompt"],
        cfg["pricing_precheck"],
        *cfg["graph_runs"].values(),
        *cfg["graph_entry_traces"].values(),
        *cfg["graph_manifests"].values(),
    ]
    input_hashes = {
        (
            str(path.relative_to(cfg["_root"]))
            if path.is_relative_to(cfg["_root"])
            else str(path)
        ): sha256(path)
        for path in input_paths
    }
    preliminary_outputs = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file()
        and path.name not in {"manifest.json", "sha256_manifest.json"}
    )
    sha_manifest = {
        str(path.relative_to(out_dir)): sha256(path)
        for path in preliminary_outputs
    }
    write_json(out_dir / "sha256_manifest.json", sha_manifest)
    output_paths = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": "dev100-m50-depth-graph-utility-community-preflight-v1",
        "version": cfg["version"],
        "created_at": utc_now(),
        "status": M50_READY_VERDICT,
        "phase_completed": "M50_RESIDUAL_PREFLIGHT",
        "blocked_before_analysis": True,
        "reason": (
            f"{len(residual)} formal M<=50 query-candidate pairs lack utility-v2"
        ),
        "development_queries": 100,
        "remaining_development298_accessed": False,
        "fixed_corpus_comments": 19013,
        "final_evidence_budget_k": 8,
        "depths": list(M50_DEPTHS),
        "m100_source_rankings_retained": True,
        "m100_enters_judging_or_analysis": False,
        "backends": ["minilm", "e5"],
        "formal_union_unique_pairs": coverage["formal_union_unique_pairs"],
        "formal_union_judged_pairs": coverage["formal_union_judged_pairs"],
        "missing_utility_pairs": len(residual),
        "anchors": len(anchors),
        "first_pass_calls": len(residual) + len(anchors),
        "estimated_input_tokens": costs["total"]["estimated_input_tokens"],
        "estimated_output_tokens": costs["total"]["estimated_output_tokens"],
        "estimated_first_pass_usd": costs["estimated_first_pass_usd"],
        "missing_local_bge_m3_unique_texts": workload[
            "missing_unique_candidate_texts_to_encode_locally"
        ],
        "external_model_calls": 0,
        "external_judging_calls": 0,
        "frozen_test_read": False,
        "community_used_for_candidate_construction": False,
        "utility_used_for_candidate_construction": False,
        "later_analysis_outputs_intentionally_absent": True,
        "input_hashes": input_hashes,
        "output_hashes": {
            str(path.relative_to(out_dir)): sha256(path)
            for path in output_paths
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(M50_READY_VERDICT)
    return manifest


def run(cfg: dict) -> dict:
    out_dir = cfg["output_dir"]
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    backend_dir = out_dir / "backend_manifests"
    backend_dir.mkdir(parents=True, exist_ok=True)

    queries, corpus, query_text, corpus_text = load_inputs(cfg)
    qids = sorted(query_text)
    minilm_rankings, minilm_identity = reproduce_minilm(cfg, corpus, qids)
    e5_rankings, e5_manifest = build_e5(cfg, queries, corpus, out_dir)
    all_rankings = {"minilm": minilm_rankings, "e5": e5_rankings}

    # Phase 2 candidate pools are frozen before any community reply artefact is
    # opened by coverage_report.
    depth_rows, overlap_rows = build_depth_pools(all_rankings)
    write_jsonl(backend_dir / "minilm_d100.jsonl", flatten_rankings(minilm_rankings))
    write_jsonl(backend_dir / "e5_d100.jsonl", flatten_rankings(e5_rankings))
    write_json(backend_dir / "minilm_manifest.json", {
        "model_id": cfg["minilm"]["model_id"],
        "revision": cfg["minilm"]["revision"],
        **minilm_identity,
    })
    write_json(backend_dir / "e5_manifest.json", e5_manifest)
    write_jsonl(out_dir / "candidate_depth_pools.jsonl", depth_rows)
    write_csv(out_dir / "backend_overlap.csv", overlap_rows)

    registry_rows = read_jsonl(cfg["utility_registry"])
    complete_rows, registry = complete_utility_v2_rows(registry_rows)
    if len(complete_rows) != 14290:
        raise ValueError(
            f"authoritative registry changed: expected 14,290, got {len(complete_rows)}"
        )
    formula_audit = utility_formula_audit(registry)
    if formula_audit["verdict"] != "PASS":
        raise AssertionError("authoritative utility-v2 formula audit failed")

    coverage, residual, blind = coverage_report(
        cfg, depth_rows, registry, query_text, corpus_text
    )
    write_json(out_dir / "judgment_coverage.json", coverage)
    write_jsonl(out_dir / "utility_residual_manifest.jsonl", residual)
    write_jsonl(out_dir / "residual_union_manifest.jsonl", residual)
    write_jsonl(out_dir / "blinded_judging_payload.jsonl", blind)

    anchors, anchor_admin = build_anchor_payloads(
        cfg, query_text, corpus_text, registry
    )
    write_jsonl(out_dir / "anchor_payload.jsonl", anchors)
    write_jsonl(out_dir / "anchor_payload_admin.jsonl", anchor_admin)
    protocol = protocol_lock(cfg["prompt"], registry_rows)
    protocol["experiment_gate"] = {
        "allow_external_judging": cfg.get("allow_external_judging") is True,
        "allow_frozen_test": False,
        "external_requests_executed": 0,
    }
    write_json(out_dir / "judge_protocol_lock.json", protocol)
    write_json(
        out_dir / "calls_token_cost_estimate.json",
        cost_estimate(cfg, blind, anchors),
    )
    write_json(out_dir / "utility_formula_audit.json", formula_audit)

    strict_rows = read_jsonl(cfg["strict_graph_pool"])
    strict_checks = {
        "rows": len(strict_rows),
        "queries": len({str(row["query_id"]) for row in strict_rows}),
        "all_native_graph": all(row.get("native_graph") is True for row in strict_rows),
        "all_positive_finite_graph_score": all(
            any(
                math.isfinite(float(score)) and float(score) > 0
                for score in (row.get("native_graph_score") or {}).values()
            )
            for row in strict_rows
        ),
        "fallback_count": sum(bool(row.get("fallback_used")) for row in strict_rows),
        "callback_count": sum(bool(row.get("callback_used")) for row in strict_rows),
        "padding_count": sum(bool(row.get("padding_used")) for row in strict_rows),
    }
    if strict_checks != {
        "rows": 400,
        "queries": 100,
        "all_native_graph": True,
        "all_positive_finite_graph_score": True,
        "fallback_count": 0,
        "callback_count": 0,
        "padding_count": 0,
    }:
        raise AssertionError(f"strict Graph4 identity failed: {strict_checks}")
    write_json(out_dir / "strict_graph_identity_audit.json", strict_checks)

    if not residual:
        status = "UTILITY_COVERAGE_COMPLETE_READY_FOR_PHASE4"
    else:
        status = READY_VERDICT
    if cfg.get("allow_external_judging") is not False:
        raise ValueError(
            "this invocation is local-only; external judging gate must be false"
        )
    continuation = (
        "External judging is intentionally not executable under the current gate.\n"
        "After a new explicit authorisation, run the canonical utility-v2 "
        "50-anchor stability gate and judge only residual_union_manifest.jsonl; "
        "then rerun this same entry against a new non-destructive versioned "
        "coverage-complete registry. Do not edit or overwrite the 14,290-pair "
        "historical registry.\n"
    )
    (out_dir / "exact_continuation_command.txt").write_text(
        continuation,
        encoding="utf-8",
    )

    inputs = {
        str(path.relative_to(cfg["_root"])) if path.is_relative_to(cfg["_root"])
        else str(path): sha256(path)
        for path in (
            cfg["queries"],
            cfg["query_admin"],
            cfg["corpus"],
            cfg["minilm_rankings"],
            cfg["minilm_corpus_embeddings"],
            cfg["minilm_query_embeddings"],
            cfg["utility_registry"],
            cfg["strict_graph_pool"],
            cfg["strict_graph_manifest"],
            cfg["split_manifest"],
            cfg["community_manifest"],
            cfg["community_reference_texts"],
            cfg["community_embedding_cache"],
            cfg["anchors"],
            cfg["prompt"],
            cfg["pricing_precheck"],
        )
    }
    output_paths = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": "depth-graph-utility-community-frontier-preflight-v1",
        "version": cfg["version"],
        "created_at": utc_now(),
        "status": status,
        "phase_completed": "PHASE_0_TO_3_LOCAL_PREFLIGHT",
        "blocked_before_phase": 4 if residual else None,
        "reason": (
            f"{len(residual)} formal depth-pool pairs lack utility-v2"
            if residual else None
        ),
        "development_queries": 100,
        "fixed_corpus_comments": 19013,
        "depths": list(DEPTHS),
        "final_evidence_budget_k": 8,
        "backends": ["minilm", "e5"],
        "formal_union_unique_pairs": coverage["formal_union_unique_pairs"],
        "missing_utility_pairs": len(residual),
        "external_model_calls": 0,
        "external_judging_calls": 0,
        "frozen_test_read": False,
        "community_used_for_candidate_construction": False,
        "utility_used_for_candidate_construction": False,
        "later_analysis_outputs_intentionally_absent": bool(residual),
        "input_hashes": inputs,
        "output_hashes": {
            str(path.relative_to(out_dir)): sha256(path)
            for path in output_paths
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(status)
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configuration" / "params.yaml",
    )
    parser.add_argument(
        "--config-key",
        default=M50_CONFIG_KEY,
        choices=(CONFIG_KEY, M50_CONFIG_KEY),
    )
    args = parser.parse_args()
    cfg = load_config(root, args.config.resolve(), args.config_key)
    if args.config_key == M50_CONFIG_KEY:
        run_m50(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
