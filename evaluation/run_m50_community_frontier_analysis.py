#!/usr/bin/env python3
"""Post-hoc BGE-M3 community correspondence for frozen M<=50 OOF sets."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    from evaluation import community_reply_auxiliary as community
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from evaluation.statistics import bootstrap_ci
    from candidate_pool.run_m50_dense_frontier_analysis import (
        BACKENDS,
        DEPTHS,
        TOP_K,
        write_csv,
    )
    from utility_scoring.annotation.run_top3_residual_judging import (
        read_jsonl,
        sha256,
        utc_now,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation import community_reply_auxiliary as community
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from evaluation.statistics import bootstrap_ci
    from candidate_pool.run_m50_dense_frontier_analysis import (
        BACKENDS,
        DEPTHS,
        TOP_K,
        write_csv,
    )
    from utility_scoring.annotation.run_top3_residual_judging import (
        read_jsonl,
        sha256,
        utc_now,
        write_json,
        write_jsonl,
    )


def final_ids(baseline: list[str], row: dict) -> list[str]:
    selected = list(baseline)
    if bool(row["acted"]):
        replaced = str(row["replaced_candidate_id"])
        candidate = str(row["selected_candidate_id"])
        selected[selected.index(replaced)] = candidate
    if len(selected) != TOP_K or len(set(selected)) != TOP_K:
        raise ValueError("OOF final set is not eight unique candidates")
    return selected


def alignment_for_ids(
    ids: list[str],
    replies: list[str],
    corpus: dict[str, str],
    embeddings: dict,
    threshold: float,
) -> dict:
    result = community.alignment(
        [corpus[cid] for cid in ids],
        replies,
        embeddings,
        threshold,
    )
    result["bialign_f1_at8"] = community.bidirectional_f(
        result["cra_at8"], result["rcc_at8"]
    )
    return result


def aggregate(rows: list[dict], seed: int) -> dict:
    output = {"query_repeat_rows": len(rows)}
    for field in (
        "utility_at8",
        "cra_at8",
        "rcc_at8",
        "bialign_f1_at8",
        "best_align_at8",
        "reply_coverage_at8",
    ):
        by_query: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_query[str(row["query_id"])].append(float(row[field]))
        values = [statistics.fmean(items) for items in by_query.values()]
        lo, hi = bootstrap_ci(values, n_boot=5000, seed=seed + len(field))
        output[field] = statistics.fmean(values)
        output[f"{field}_bootstrap_95ci_lo"] = lo
        output[f"{field}_bootstrap_95ci_hi"] = hi
    return output


def encode_with_verified_prior_cache(
    *,
    config: dict,
    texts: list[str],
    prior_texts: list[str],
    prior_output: Path,
    output: Path,
) -> tuple[dict, dict]:
    """Reuse prior BGE-M3 rows by canonical text hash; encode only missing rows."""
    encoder = config["semantic_encoder"]
    keyed = {community.text_sha(text): text for text in texts}
    ordered = sorted(keyed)
    cache_payload = {
        "model": encoder["model_id"],
        "revision": encoder["revision"],
        "max_sequence_length": encoder["max_sequence_length"],
        "text_hashes": ordered,
    }
    cache_key = community.sha256_bytes(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    )
    cache = output / f"semantic_embeddings_{cache_key[:16]}.npz"

    prior_keyed = {community.text_sha(text): text for text in prior_texts}
    prior_ordered = sorted(prior_keyed)
    prior_payload = {
        "model": encoder["model_id"],
        "revision": encoder["revision"],
        "max_sequence_length": encoder["max_sequence_length"],
        "text_hashes": prior_ordered,
    }
    prior_key = community.sha256_bytes(
        json.dumps(prior_payload, sort_keys=True).encode("utf-8")
    )
    prior_cache = prior_output / f"semantic_embeddings_{prior_key[:16]}.npz"
    if not prior_cache.exists():
        raise FileNotFoundError(f"verified prior BGE cache absent: {prior_cache}")
    prior_matrix = community.np.load(prior_cache)["embeddings"]
    if prior_matrix.shape != (len(prior_ordered), 1024):
        raise ValueError(
            f"prior BGE cache identity mismatch: {prior_matrix.shape} "
            f"vs {(len(prior_ordered), 1024)}"
        )
    embeddings = dict(zip(prior_ordered, prior_matrix, strict=True))
    reused = [key for key in ordered if key in embeddings]
    missing = [key for key in ordered if key not in embeddings]
    incremental_manifest = None
    if missing:
        incremental_output = output / "incremental_bge_cache"
        incremental_output.mkdir(exist_ok=True)
        incremental, incremental_manifest = community.encode_texts(
            config,
            [keyed[key] for key in missing],
            incremental_output,
        )
        embeddings.update(incremental)

    matrix = community.np.stack([embeddings[key] for key in ordered]).astype(
        community.np.float32
    )
    community.np.savez_compressed(cache, embeddings=matrix)
    return dict(zip(ordered, matrix, strict=True)), {
        "cache_path": str(cache),
        "cache_sha256": community.sha256_file(cache),
        "cache_hit": not missing,
        "cache_mode": "verified_text_hash_reuse_plus_incremental",
        "prior_cache_path": str(prior_cache),
        "prior_cache_sha256": community.sha256_file(prior_cache),
        "reused_text_count": len(reused),
        "incrementally_encoded_text_count": len(missing),
        "unique_text_count": len(ordered),
        "embedding_dimension": int(matrix.shape[1]),
        "model_id": encoder["model_id"],
        "revision": encoder["revision"],
        "pooling": encoder["pooling"],
        "normalized": encoder["normalize_embeddings"],
        "max_sequence_length": encoder["max_sequence_length"],
        "batch_size": encoder["batch_size"],
        "device": str(encoder.get("device", "cpu")),
        "incremental_manifest": incremental_manifest,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "out/depth_graph_utility_community_frontier_dev100_m50_community_v2",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)

    _, config = community.load_config()
    threshold = float(config["semantic_encoder"]["threshold"])
    reference_dir = root / "out/community_reply_aux_dev100_v1"
    references = community.load_admin_references(reference_dir)
    corpus, _ = community.load_corpus(
        root / "out/hipporag_official_adapter/adhd_peer_support_validation_corpus.json"
    )
    registry_path = (
        root
        / "out/depth_graph_utility_community_frontier_dev100_m50_judging_v1"
        / "complete/utility_registry_coverage_complete.jsonl"
    )
    complete, registry = complete_utility_v2_rows(read_jsonl(registry_path))
    if len(complete) != 19813:
        raise ValueError("coverage-complete registry identity changed")

    preflight = root / "out/depth_graph_utility_community_frontier_dev100_m50_v2"
    memberships = read_jsonl(
        preflight / "m50_residual_preflight/dense_m50_memberships.jsonl"
    )
    rankings: dict[str, dict[str, list[dict]]] = {
        backend: defaultdict(list) for backend in BACKENDS
    }
    for row in memberships:
        rankings[str(row["backend"])][str(row["query_id"])].append(row)
    for backend in BACKENDS:
        rankings[backend] = dict(rankings[backend])
        for query_id in rankings[backend]:
            rankings[backend][query_id].sort(key=lambda row: int(row["rank"]))

    dense_oof_path = (
        root
        / "out/depth_graph_utility_community_frontier_dev100_m50_dense_v1"
        / "m50_dense_oof_actions.jsonl"
    )
    graph_oof_path = (
        root
        / "out/depth_graph_utility_community_frontier_dev100_m50_graph_v2"
        / "m50_graph_oof_actions.jsonl"
    )
    matched_oof_path = (
        root
        / "out/depth_graph_utility_community_frontier_dev100_m50_matched_v1"
        / "m50_matched_budget_oof_actions.jsonl"
    )
    dense_oof = read_jsonl(dense_oof_path)
    graph_oof = read_jsonl(graph_oof_path)
    matched_oof = read_jsonl(matched_oof_path)

    candidate_ids: set[str] = set()
    for backend in BACKENDS:
        for query_id in rankings[backend]:
            candidate_ids.update(
                str(row["comment_id"])
                for row in rankings[backend][query_id][:TOP_K]
            )
    for row in dense_oof + graph_oof + matched_oof:
        if bool(row["acted"]):
            candidate_ids.add(str(row["selected_candidate_id"]))
    texts = [
        reply["text"]
        for items in references.values()
        for reply in items
    ]
    texts.extend(corpus[cid] for cid in sorted(candidate_ids))
    prior_systems, _ = community.load_systems(
        config, set(references), set(corpus)
    )
    prior_texts = [
        reply["text"]
        for items in references.values()
        for reply in items
    ]
    prior_texts.extend(
        corpus[cid]
        for run in prior_systems.values()
        for ids in run.values()
        for cid in ids
    )
    embeddings, encoder_manifest = encode_with_verified_prior_cache(
        config=config,
        texts=texts,
        prior_texts=prior_texts,
        prior_output=reference_dir,
        output=out_dir,
    )

    system_rows, action_rows = [], []
    for source, rows in (
        ("dense", dense_oof),
        ("graph_union", graph_oof),
        ("matched_graph", matched_oof),
    ):
        for row in rows:
            backend = str(row["backend"])
            depth = int(row["depth"])
            query_id = str(row["query_id"])
            baseline = [
                str(item["comment_id"])
                for item in rankings[backend][query_id][:TOP_K]
            ]
            selected = final_ids(baseline, row)
            reply_texts = [reply["text"] for reply in references[query_id]]
            aligned = alignment_for_ids(
                selected,
                reply_texts,
                corpus,
                embeddings,
                threshold,
            )
            baseline_aligned = alignment_for_ids(
                baseline,
                reply_texts,
                corpus,
                embeddings,
                threshold,
            )
            utility = statistics.fmean(
                float(registry[(query_id, cid)]["utility"]) for cid in selected
            )
            record = {
                "source": source,
                "backend": backend,
                "depth": depth,
                "graph_view": (
                    str(row["pool"]).rsplit("_plus_", 1)[-1]
                    if source == "graph_union"
                    else (
                        "matched_fixed_graph4"
                        if source == "matched_graph"
                        else None
                    )
                ),
                "repeat": int(row["repeat"]),
                "query_id": query_id,
                "acted": bool(row["acted"]),
                "utility_at8": utility,
                **aligned,
            }
            system_rows.append(record)
            if bool(row["acted"]):
                action_rows.append({
                    **record,
                    "selected_candidate_id": row["selected_candidate_id"],
                    "replaced_candidate_id": row["replaced_candidate_id"],
                    "utility_delta": float(row["raw_reward"]),
                    "cra_delta": aligned["cra_at8"] - baseline_aligned["cra_at8"],
                    "rcc_delta": aligned["rcc_at8"] - baseline_aligned["rcc_at8"],
                    "bialign_f1_delta": (
                        aligned["bialign_f1_at8"]
                        - baseline_aligned["bialign_f1_at8"]
                    ),
                    "utility_positive": float(row["raw_reward"]) > 0,
                    "community_positive": (
                        aligned["bialign_f1_at8"]
                        - baseline_aligned["bialign_f1_at8"]
                    ) > 0,
                })

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in system_rows:
        key = (
            row["source"],
            row["backend"],
            row["depth"],
            row["graph_view"],
        )
        grouped[key].append(row)
    metric_rows = []
    for index, (key, rows) in enumerate(sorted(grouped.items(), key=str)):
        source, backend, depth, graph_view = key
        metric_rows.append({
            "source": source,
            "backend": backend,
            "depth": depth,
            "graph_view": graph_view,
            **aggregate(rows, 20260730 + index * 100),
        })

    quadrant_rows = []
    for row in action_rows:
        quadrant = (
            "utility_up_community_up"
            if row["utility_positive"] and row["community_positive"]
            else (
                "utility_up_community_down"
                if row["utility_positive"] else (
                    "utility_down_community_up"
                    if row["community_positive"]
                    else "utility_down_community_down"
                )
            )
        )
        quadrant_rows.append({**row, "quadrant": quadrant})

    write_csv(out_dir / "m50_utility_community_metrics.csv", metric_rows)
    write_csv(out_dir / "m50_action_utility_alignment.csv", quadrant_rows)
    write_jsonl(out_dir / "m50_utility_community_per_query_repeat.jsonl", system_rows)
    write_json(out_dir / "encoder_manifest.json", encoder_manifest)
    outputs = sorted(path for path in out_dir.iterdir() if path.is_file())
    write_json(out_dir / "manifest.json", {
        "schema": "dev100-m50-community-frontier-analysis-v2",
        "created_utc": utc_now(),
        "status": "COMMUNITY_M50_FRONTIER_COMPLETE",
        "development_queries": 100,
        "encoder": encoder_manifest,
        "threshold": threshold,
        "community_used_in_selector": False,
        "community_used_in_depth_selection": False,
        "community_used_in_graph_quota": False,
        "candidate_and_oof_outputs_frozen_before_encoding": True,
        "frozen_test_read": False,
        "m100_analysed": False,
        "remaining_development298_accessed": False,
        "input_hashes": {
            str(registry_path.relative_to(root)): sha256(registry_path),
            str(dense_oof_path.relative_to(root)): sha256(dense_oof_path),
            str(graph_oof_path.relative_to(root)): sha256(graph_oof_path),
            str(matched_oof_path.relative_to(root)): sha256(matched_oof_path),
            str((reference_dir / "ADMIN_community_reply_reference_texts.jsonl").relative_to(root)): sha256(
                reference_dir / "ADMIN_community_reply_reference_texts.jsonl"
            ),
        },
        "output_hashes": {path.name: sha256(path) for path in outputs},
    })


if __name__ == "__main__":
    main()
