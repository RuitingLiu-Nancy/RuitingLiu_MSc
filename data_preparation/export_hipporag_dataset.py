#!/usr/bin/env python3
"""Export the frozen validation corpus/query adapter for official HippoRAG.

The adapter performs no retrieval and no LLM calls.  It serializes the same
comment nodes already indexed by this project's graph into HippoRAG's custom
dataset format.  Test files are deliberately rejected.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_nodes(graph_dir: Path):
    import pandas as pd

    parquet = graph_dir / "graph_nodes.parquet"
    csv_path = graph_dir / "graph_nodes.csv"
    return pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(csv_path)


def _read_text_lookup(text_csv: Path) -> dict:
    """Canonical comment texts (unified extraction input, 19,013 rows).

    graph_nodes.text may be empty *or an LLM summary*.  Graph nodes decide
    WHICH comments are in scope; this file is the canonical source for WHAT
    they say.  When supplied, it therefore overrides every matching node text
    rather than filling empty values only."""
    import pandas as pd

    df = pd.read_csv(text_csv, engine="python", usecols=["comment_id", "target_text"])
    df["comment_id"] = df["comment_id"].astype(str)
    return dict(zip(df["comment_id"], df["target_text"].fillna("")))


def export(graph_dir: Path, validation_csv: Path, out_dir: Path, dataset: str,
           text_csv: Path | None = None) -> dict:
    if "test" in validation_csv.name.lower():
        raise ValueError("test split is frozen; export a validation CSV only")
    nodes = _read_nodes(graph_dir)
    text_lookup = _read_text_lookup(text_csv) if text_csv else {}
    comments = nodes[nodes["node_type"].astype(str) == "comment"].copy()
    comments = comments.sort_values("node_id")
    corpus = []
    comment_by_id = {}
    texts_overridden_from_lookup = 0
    texts_filled_from_lookup = 0
    for idx, row in enumerate(comments.to_dict("records")):
        cid = str(row["node_id"]).removeprefix("CMT::")
        node_text = str(row.get("text") or "").strip()
        text = str(text_lookup[cid]).strip() if cid in text_lookup else node_text
        if cid in text_lookup and text != node_text:
            texts_overridden_from_lookup += 1
        if cid in text_lookup and not node_text and text:
            texts_filled_from_lookup += 1
        record = {"title": cid, "text": text, "idx": idx}
        corpus.append(record)
        comment_by_id[cid] = record
    n_empty = sum(1 for r in corpus if not r["text"])
    if n_empty:
        raise ValueError(
            f"{n_empty} comments still empty after text join; pass --text-csv "
            "pointing at the unified extraction input (canonical text source)")

    queries = []
    with validation_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gold_ids = [x for x in str(row.get("gold_comment_ids") or "").split("|") if x]
            paragraphs = [
                {**comment_by_id[cid], "is_supporting": True}
                for cid in gold_ids if cid in comment_by_id
            ]
            queries.append({
                "id": f"{dataset}/{row['post_id']}.json",
                "question": row["query_text"],
                "answer": [],
                "answerable": bool(paragraphs),
                "paragraphs": paragraphs,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / f"{dataset}_corpus.json"
    query_path = out_dir / f"{dataset}.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    query_path.write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "protocol": "official HippoRAG custom-dataset adapter",
        "test_split_used": False,
        "dataset": dataset,
        "graph_dir": str(graph_dir),
        "validation_csv": str(validation_csv),
        "corpus_comments": len(corpus),
        "text_csv": str(text_csv) if text_csv else None,
        "text_policy": "canonical_raw_overlay" if text_lookup else "graph_nodes_text",
        "texts_overridden_from_lookup": texts_overridden_from_lookup,
        "texts_filled_from_lookup": texts_filled_from_lookup,
        "empty_texts": n_empty,
        "validation_queries": len(queries),
        "queries_with_mapped_gold": sum(bool(row["paragraphs"]) for row in queries),
        "outputs": {"corpus": str(corpus_path), "queries": str(query_path)},
        "guard": "Adapter only; official HippoRAG must rebuild OpenIE/embeddings/index separately.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", type=Path, required=True)
    ap.add_argument("--validation-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dataset", default="adhd_peer_support_validation")
    ap.add_argument("--text-csv", type=Path, default=None,
                    help="canonical comment texts (unified_extract_input_1618.csv)")
    args = ap.parse_args()
    print(json.dumps(export(
        args.graph_dir, args.validation_csv, args.out_dir, args.dataset,
        text_csv=args.text_csv),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
