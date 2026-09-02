#!/usr/bin/env python3
"""Add the frozen E5 Dense backend to closed600 and audit LOPO judging cost.

This is a local-only, label-blind retrieval/preflight runner.  It reuses the
canonical E5 encoder and ranking implementation from the depth-frontier study,
the canonical IR metrics, and the historical closed600 rankings.  It never
calls an external model and never reads a test split.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from evaluation.ir_metrics import eval_full, mean_metrics
    from evaluation.same_post_shortcut_audit import (
        _bare,
        filter_same_post,
    )
    from data_preparation.sampling.create_mixed_query_splits import _balanced_take
    from shared.io_utils import read_csv_rows, write_csv_rows
    from fusion.run_depth_graph_utility_community_frontier import (
        encode_e5,
        sha256,
        stable_top100,
        write_json,
        write_jsonl,
    )
    from data_preparation.sampling.sample_human_annotation_candidates import _load_cards
    from evaluation.score_multihop_retrieval import (
        _load_pred,
        paired_bootstrap_delta,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.ir_metrics import eval_full, mean_metrics
    from evaluation.same_post_shortcut_audit import (
        _bare,
        filter_same_post,
    )
    from data_preparation.sampling.create_mixed_query_splits import _balanced_take
    from shared.io_utils import read_csv_rows, write_csv_rows
    from fusion.run_depth_graph_utility_community_frontier import (
        encode_e5,
        sha256,
        stable_top100,
        write_json,
        write_jsonl,
    )
    from data_preparation.sampling.sample_human_annotation_candidates import _load_cards
    from evaluation.score_multihop_retrieval import (
        _load_pred,
        paired_bootstrap_delta,
    )


CONFIG_KEY = "closed600_e5_lopo_preflight"


def load_closed600_e5_lopo_config(root: Path, path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = dict(raw[CONFIG_KEY])
    cfg["_root"] = root
    path_keys = (
        "output_dir",
        "queries",
        "heldout",
        "corpus",
        "corpus_map",
        "e5_corpus_embeddings",
        "e5_source_manifest",
        "e5_snapshot",
        "existing_utility_db",
    )
    for key in path_keys:
        value = Path(cfg[key]).expanduser()
        cfg[key] = value if value.is_absolute() else root / value
    if cfg.get("e5_query_embeddings_cache"):
        value = Path(cfg["e5_query_embeddings_cache"]).expanduser()
        cfg["e5_query_embeddings_cache"] = (
            value if value.is_absolute() else root / value
        )
    cfg["retrieval_runs"] = {
        name: (Path(value).expanduser() if Path(value).expanduser().is_absolute()
               else root / value)
        for name, value in dict(cfg["retrieval_runs"]).items()
    }
    if cfg.get("allow_external_judging") is not False:
        raise ValueError("allow_external_judging must remain false")
    if cfg.get("allow_frozen_test") is not False:
        raise ValueError("allow_frozen_test must remain false")
    guarded = [cfg[key] for key in path_keys] + list(cfg["retrieval_runs"].values())
    if cfg.get("e5_query_embeddings_cache"):
        guarded.append(cfg["e5_query_embeddings_cache"])
    if any("test" in str(path).lower() for path in guarded):
        raise ValueError("test-looking input path rejected")
    return cfg


def structural_profile(
    rankings: dict[str, list[str]],
    gold: dict[str, set[str]],
    comment_to_post: dict[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    rows: list[dict[str, float]] = []
    per_query: dict[str, dict[str, float]] = {}
    same_post_at = {5: [], 8: [], 10: []}
    for qid in sorted(gold):
        values = eval_full(
            rankings[qid], gold[qid], recall_ks=(5, 10, 20)
        )
        rows.append(values)
        per_query[qid] = {key: float(value) for key, value in values.items()}
        for depth in same_post_at:
            top = rankings[qid][:depth]
            same_post_at[depth].append(
                sum(comment_to_post.get(cid) == qid for cid in top) / depth
            )
    return {
        "queries": len(rows),
        "metrics": mean_metrics(rows, ndigits=12),
        "same_post_share": {
            f"@{depth}": float(statistics.fmean(values))
            for depth, values in same_post_at.items()
        },
    }, per_query


def candidate_union(
    rankings: dict[str, dict[str, list[str]]],
    query_ids: list[str],
    comment_to_post: dict[str, str],
    existing_labels: dict[tuple[str, str], float],
    *,
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    rows: list[dict[str, Any]] = []
    per_query: dict[str, dict[str, int]] = {}
    for qid in query_ids:
        lopo = {
            name: filter_same_post(run[qid], qid, comment_to_post)[:k]
            for name, run in rankings.items()
        }
        union = sorted(set().union(*(set(values) for values in lopo.values())))
        existing = 0
        for cid in union:
            sources = sorted(name for name, values in lopo.items() if cid in values)
            key = (qid, cid)
            judged = key in existing_labels
            existing += int(judged)
            rows.append({
                "query_id": qid,
                "candidate_id": cid,
                "lopo_top8_sources": sources,
                "source_count": len(sources),
                "existing_utility_judgment": judged,
                "existing_utility": existing_labels.get(key),
            })
        per_query[qid] = {
            "union_candidates": len(union),
            "already_judged": existing,
            "residual": len(union) - existing,
        }
    return rows, per_query


def paired_raw_lopo_candidate_union(
    rankings: dict[str, dict[str, list[str]]],
    query_ids: list[str],
    comment_to_post: dict[str, str],
    existing_labels: dict[tuple[str, str], float],
    *,
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Build the complete candidate union needed for paired raw/LOPO U@K."""
    rows: list[dict[str, Any]] = []
    per_query: dict[str, dict[str, int]] = {}
    for qid in query_ids:
        raw = {name: run[qid][:k] for name, run in rankings.items()}
        lopo = {
            name: filter_same_post(run[qid], qid, comment_to_post)[:k]
            for name, run in rankings.items()
        }
        union = sorted(
            set().union(
                *(set(values) for values in raw.values()),
                *(set(values) for values in lopo.values()),
            )
        )
        existing = 0
        for cid in union:
            raw_sources = sorted(
                name for name, values in raw.items() if cid in values
            )
            lopo_sources = sorted(
                name for name, values in lopo.items() if cid in values
            )
            sources = sorted(set(raw_sources) | set(lopo_sources))
            key = (qid, cid)
            judged = key in existing_labels
            existing += int(judged)
            rows.append({
                "query_id": qid,
                "candidate_id": cid,
                "raw_top8_sources": raw_sources,
                "lopo_top8_sources": lopo_sources,
                "top8_sources": sources,
                "same_thread": comment_to_post.get(cid) == qid,
                "existing_utility_judgment": judged,
                "existing_utility": existing_labels.get(key),
            })
        per_query[qid] = {
            "union_candidates": len(union),
            "already_judged": existing,
            "residual": len(union) - existing,
        }
    return rows, per_query


def describe_candidate_scope(
    query_ids: list[str],
    per_query: dict[str, dict[str, int]],
) -> dict[str, Any]:
    fields = ("union_candidates", "already_judged", "residual")
    output: dict[str, Any] = {
        "queries": len(query_ids),
        "candidate_pairs": sum(per_query[qid]["union_candidates"] for qid in query_ids),
        "already_judged_pairs": sum(per_query[qid]["already_judged"] for qid in query_ids),
        "residual_pairs": sum(per_query[qid]["residual"] for qid in query_ids),
        "queries_with_any_existing_judgment": sum(
            per_query[qid]["already_judged"] > 0 for qid in query_ids
        ),
    }
    for field in fields:
        values = [per_query[qid][field] for qid in query_ids]
        output[f"per_query_{field}"] = {
            "min": min(values),
            "mean": float(statistics.fmean(values)),
            "max": max(values),
        }
    return output


def describe_candidate_scope_by_sources(
    rows: list[dict[str, Any]],
    query_ids: list[str],
    source_names: list[str],
    *,
    source_field: str = "lopo_top8_sources",
) -> dict[str, Any]:
    """Describe a source-restricted union without inspecting label values."""
    selected_queries = set(query_ids)
    selected_sources = set(source_names)
    per_query = {
        qid: {"union_candidates": 0, "already_judged": 0, "residual": 0}
        for qid in query_ids
    }
    for row in rows:
        qid = str(row["query_id"])
        if qid not in selected_queries:
            continue
        if not selected_sources.intersection(row[source_field]):
            continue
        per_query[qid]["union_candidates"] += 1
        if bool(row["existing_utility_judgment"]):
            per_query[qid]["already_judged"] += 1
        else:
            per_query[qid]["residual"] += 1
    output = describe_candidate_scope(query_ids, per_query)
    output["sources"] = sorted(selected_sources)
    return output


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    output_dir: Path = cfg["output_dir"]
    if output_dir.exists():
        raise FileExistsError(f"versioned output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="closed600_e5_lopo_", dir=output_dir.parent))
    try:
        query_rows = read_csv_rows(cfg["queries"])
        heldout_rows = read_csv_rows(cfg["heldout"])
        if len(query_rows) != 600 or len(heldout_rows) != 600:
            raise ValueError("expected exactly 600 closed-corpus queries")
        query_ids = [str(row["query_id"]) for row in query_rows]
        heldout_ids = [str(row["post_id"]) for row in heldout_rows]
        if query_ids != heldout_ids or len(set(query_ids)) != 600:
            raise ValueError("query identity/order mismatch between metadata and heldout")

        corpus = json.loads(cfg["corpus"].read_text(encoding="utf-8"))
        if len(corpus) != 19013:
            raise ValueError("expected the frozen 19,013-comment corpus")
        corpus_ids = np.asarray([str(row["title"]) for row in corpus], dtype=object)
        corpus_text = {str(row["title"]): str(row["text"]) for row in corpus}
        mapping_rows = read_csv_rows(cfg["corpus_map"])
        comment_to_post = {
            str(row["comment_id"]): str(row["post_id"]) for row in mapping_rows
        }
        if set(corpus_ids) - set(comment_to_post):
            raise ValueError("corpus comment-to-post mapping is incomplete")

        source_e5 = json.loads(cfg["e5_source_manifest"].read_text(encoding="utf-8"))
        if source_e5["model_id"] != cfg["e5_model_id"]:
            raise ValueError("E5 model identity mismatch")
        if source_e5["revision"] != cfg["e5_revision"]:
            raise ValueError("E5 revision mismatch")
        if cfg["e5_snapshot"].name != cfg["e5_revision"]:
            raise ValueError("E5 snapshot directory does not match frozen revision")
        corpus_embeddings = np.load(cfg["e5_corpus_embeddings"], mmap_mode="r")
        if corpus_embeddings.shape != (19013, 768):
            raise ValueError(f"unexpected E5 corpus shape: {corpus_embeddings.shape}")
        backend_dir = staging / "backend"
        backend_dir.mkdir(parents=True)
        query_embedding_output = backend_dir / "e5_query_embeddings.npy"
        if cfg.get("e5_query_embeddings_cache"):
            shutil.copyfile(
                cfg["e5_query_embeddings_cache"], query_embedding_output
            )
        query_embeddings, query_cache = encode_e5(
            [str(row["query_text"]) for row in query_rows],
            prefix="query: ",
            snapshot=cfg["e5_snapshot"],
            output=query_embedding_output,
            batch_size=int(cfg["embedding_batch_size"]),
            max_length=int(cfg["max_sequence_length"]),
        )
        e5_ranked = stable_top100(
            query_embeddings,
            corpus_embeddings,
            corpus_ids,
            query_ids,
            backend="e5",
            model_id=cfg["e5_model_id"],
            revision=cfg["e5_revision"],
        )
        write_jsonl(
            backend_dir / "e5_d100.jsonl",
            [row for qid in query_ids for row in e5_ranked[qid]],
        )
        e5_query_rows = [
            {
                "id": qid,
                "retrieved_titles": [row["comment_id"] for row in e5_ranked[qid]],
            }
            for qid in query_ids
        ]
        write_jsonl(staging / "e5_dense_closed600.jsonl", e5_query_rows)

        rankings = {
            name: {
                _bare(qid): values for qid, values in _load_pred(path).items()
            }
            for name, path in cfg["retrieval_runs"].items()
        }
        rankings["e5_dense"] = {
            row["id"]: list(row["retrieved_titles"]) for row in e5_query_rows
        }
        if any(set(run) != set(query_ids) for run in rankings.values()):
            raise ValueError("retrieval run query identity mismatch")

        gold = {
            str(row["post_id"]): {
                cid for cid in str(row["gold_comment_ids"]).split("|") if cid
            }
            for row in heldout_rows
        }
        structural: dict[str, Any] = {}
        per_system: dict[str, dict[str, dict[str, float]]] = {}
        for name, ranking in rankings.items():
            structural[name], per_system[name] = structural_profile(
                ranking, gold, comment_to_post
            )
        e5_paired = []
        for other in cfg["retrieval_runs"]:
            for metric in ("nDCG@10", "Recall@5", "MRR", "Hit@10"):
                delta = paired_bootstrap_delta(
                    {qid: values[metric] for qid, values in per_system["e5_dense"].items()},
                    {qid: values[metric] for qid, values in per_system[other].items()},
                    n_boot=int(cfg["bootstrap_samples"]),
                    seed=int(cfg["bootstrap_seed"]),
                )
                e5_paired.append({
                    "left": "e5_dense",
                    "right": other,
                    "metric": metric,
                    **(delta or {}),
                })

        existing_cards = _load_cards(cfg["existing_utility_db"], cfg["heldout"])
        existing_labels = {
            (_bare(str(row["query_id"])), str(row["comment_id"])): float(row["utility"])
            for row in existing_cards
        }
        union_rows, per_query = candidate_union(
            rankings, query_ids, comment_to_post, existing_labels,
            k=int(cfg["top_k"]),
        )
        write_jsonl(staging / "lopo_top8_candidate_union.jsonl", union_rows)
        paired_union_rows, paired_per_query = paired_raw_lopo_candidate_union(
            rankings, query_ids, comment_to_post, existing_labels,
            k=int(cfg["top_k"]),
        )
        write_jsonl(
            staging / "raw_lopo_top8_candidate_union.jsonl", paired_union_rows
        )

        sample_rows = _balanced_take(
            [dict(row) for row in query_rows],
            int(cfg["sample_size"]),
            random.Random(int(cfg["sample_seed"])),
            str(cfg["sample_balance"]),
        )
        sample_ids = [str(row["query_id"]) for row in sample_rows]
        write_csv_rows(
            staging / "proposed_balanced200_queries.csv",
            sample_rows,
            fieldnames=sorted({key for row in sample_rows for key in row}),
        )

        candidate_scope = {
            "lopo_only": {
                "full600": describe_candidate_scope(query_ids, per_query),
                "proposed_balanced200": describe_candidate_scope(
                    sample_ids, per_query
                ),
            },
            "raw_lopo_comparison": {
                "full600": describe_candidate_scope(
                    query_ids, paired_per_query
                ),
                "proposed_balanced200": describe_candidate_scope(
                    sample_ids, paired_per_query
                ),
            },
            "source_groups": {},
            "sample_selection": {
                "size": len(sample_ids),
                "seed": int(cfg["sample_seed"]),
                "balance": str(cfg["sample_balance"]),
                "selection_reads_retrieval_or_utility": False,
                "source": dict(sorted(Counter(row["source"] for row in sample_rows).items())),
                "scenario": dict(sorted(Counter(row["scenario"] for row in sample_rows).items())),
                "tier": dict(sorted(Counter(row["tier"] for row in sample_rows).items())),
            },
        }
        source_groups = {
            str(name): [str(source) for source in sources]
            for name, sources in dict(cfg["candidate_scope_groups"]).items()
        }
        available_sources = set(rankings)
        for group_name, sources in source_groups.items():
            unknown = set(sources) - available_sources
            if unknown:
                raise ValueError(
                    f"candidate scope group {group_name} has unknown sources: "
                    f"{sorted(unknown)}"
                )
            candidate_scope["source_groups"][group_name] = {
                "lopo_only": {
                    "full600": describe_candidate_scope_by_sources(
                        union_rows, query_ids, sources
                    ),
                    "proposed_balanced200": describe_candidate_scope_by_sources(
                        union_rows, sample_ids, sources
                    ),
                },
                "raw_lopo_comparison": {
                    "full600": describe_candidate_scope_by_sources(
                        paired_union_rows, query_ids, sources,
                        source_field="top8_sources",
                    ),
                    "proposed_balanced200": describe_candidate_scope_by_sources(
                        paired_union_rows, sample_ids, sources,
                        source_field="top8_sources",
                    ),
                },
            }
        manifest = {
            "protocol": cfg["version"],
            "status": "LOCAL_RETRIEVAL_COMPLETE_UTILITY_JUDGING_NOT_AUTHORIZED",
            "external_calls": 0,
            "test_split_used": False,
            "query_count": len(query_ids),
            "corpus_comments": len(corpus),
            "top_k": int(cfg["top_k"]),
            "e5": {
                "model_id": cfg["e5_model_id"],
                "revision": cfg["e5_revision"],
                "recipe": source_e5["input_prefixes"] | {
                    "pooling": source_e5["pooling"],
                    "normalisation": source_e5["normalisation"],
                    "similarity": source_e5["similarity"],
                },
                "source_manifest_sha256": sha256(cfg["e5_source_manifest"]),
                "corpus_embeddings_sha256": sha256(cfg["e5_corpus_embeddings"]),
                "query_embeddings": query_cache,
            },
            "structural": structural,
            "e5_paired_comparisons": e5_paired,
            "candidate_scope": candidate_scope,
            "existing_utility_registry": {
                "path": str(cfg["existing_utility_db"]),
                "pairs": len(existing_labels),
                "queries": len({qid for qid, _ in existing_labels}),
            },
            "claim_boundary": (
                "Closed structural metrics evaluate native same-thread replies. "
                "After same-thread removal those structural qrels are empty; "
                "the candidate audit only estimates a future paired raw/LOPO "
                "evidence-utility judging requirement and does not report a "
                "LOPO utility effect."
            ),
            "inputs": {
                "queries": str(cfg["queries"]),
                "heldout": str(cfg["heldout"]),
                "corpus": str(cfg["corpus"]),
                "corpus_map": str(cfg["corpus_map"]),
                "retrieval_runs": {name: str(path) for name, path in cfg["retrieval_runs"].items()},
            },
        }
        write_json(staging / "manifest.json", manifest)
        write_json(staging / "structural_report.json", {
            "systems": structural,
            "e5_paired_comparisons": e5_paired,
        })
        write_json(staging / "lopo_judging_preflight.json", candidate_scope)
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configuration/params.yaml"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    cfg = load_closed600_e5_lopo_config(
        args.project.resolve(), args.config.resolve()
    )
    result = run(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
