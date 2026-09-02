#!/usr/bin/env python3
"""GLOBAL relation grounding — the "resolve/fusion" stage of an
extract-then-resolve KG construction pipeline.

Background (method & literature)
--------------------------------
Standard LLM knowledge-graph construction is a three-stage pipeline:
  1. GENERATE  : per text-unit, extract entities + relations, keep ALL instances.
  2. AGGREGATE : pool extracted entities/relations across all documents.
  3. RESOLVE   : cluster mentions of the same entity into CANONICAL nodes and
                 attach relations to those canonical nodes (global entity
                 resolution / canonicalisation).
Discarding a relation in stage 1 just because an endpoint was not also extracted
as an entity in the SAME comment causes the well-documented "fragmented graph"
failure mode (Microsoft GraphRAG; Graphusion, arXiv:2407.10794; KGGen,
arXiv:2502.09956). The correct place to ground a relation's endpoints is AFTER
canonicalisation, globally — which is what this script does.

What it does
------------
Input:
  --relations  : the extraction JSONL (now retains ALL relations + grounded_local)
  --canon-dir  : a canonicalize_open_entity_clusters.py output dir, which has
                 entity_mentions_clustered.csv (canonical_raw -> canonical_id).
For each extracted relation (source, relation, target) — both lowercase canonical
strings — we map source/target to canonical_id via the global mention map. A
relation becomes a GRAPH EDGE iff BOTH endpoints map to a canonical node. We
report how many relations are now globally groundable that were NOT locally
grounded (i.e. recovered by the global step), quantifying the fragmentation the
old per-comment discard caused.

Output (in --out-dir):
  canonical_relation_edges.csv  : source_id, relation, target_id, n_support, ...
  relation_grounding_report.json: counts (local vs global grounded, recovered)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_mention_map(canon_dir: Path) -> dict[str, str]:
    """canonical_raw (lowercased) -> canonical_id, from the clustered mentions."""
    f = canon_dir / "entity_mentions_clustered.csv"
    m: dict[str, str] = {}
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            raw = (r.get("canonical_raw") or "").strip().lower()
            cid = (r.get("canonical_id") or "").strip()
            if raw and cid:
                m.setdefault(raw, cid)   # first wins; mentions of same raw share id
    return m


def load_relations(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for rel in rec.get("relations", []):
                yield rec.get("comment_id", ""), rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relations", required=True, type=Path,
                    help="extraction JSONL (open_entities_*.jsonl)")
    ap.add_argument("--canon-dir", required=True, type=Path,
                    help="canonicalize output dir with entity_mentions_clustered.csv")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    mention_map = load_mention_map(args.canon_dir)
    print(f"[ground] canonical mention map: {len(mention_map)} raw->id entries")

    edges = defaultdict(lambda: {"n_support": 0, "evidence": []})
    n_total = n_local = n_global = n_recovered = n_unmapped = 0

    for cid, rel in load_relations(args.relations):
        n_total += 1
        s = (rel.get("source") or "").strip().lower()
        t = (rel.get("target") or "").strip().lower()
        relation = (rel.get("relation") or "related_to").strip()
        was_local = bool(rel.get("grounded_local"))
        if was_local:
            n_local += 1

        sid = mention_map.get(s)
        tid = mention_map.get(t)
        if sid and tid and sid != tid:
            n_global += 1
            if not was_local:
                n_recovered += 1   # relation the OLD per-comment discard would have lost
            key = (sid, relation, tid)
            edges[key]["n_support"] += 1
            ev = rel.get("evidence") or ""
            if ev and len(edges[key]["evidence"]) < 3:
                edges[key]["evidence"].append(ev[:160])
        else:
            n_unmapped += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "canonical_relation_edges.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_id", "relation", "target_id", "n_support", "evidence_examples"])
        for (sid, relation, tid), v in sorted(edges.items(), key=lambda x: -x[1]["n_support"]):
            w.writerow([sid, relation, tid, v["n_support"], " || ".join(v["evidence"])])

    report = {
        "relations_total": n_total,
        "locally_grounded": n_local,
        "globally_grounded": n_global,
        "recovered_by_global_step": n_recovered,
        "unmapped_endpoints": n_unmapped,
        "distinct_canonical_edges": len(edges),
        "note": "recovered_by_global_step = relations the old per-comment discard "
                "would have dropped, now correctly attached to canonical nodes.",
    }
    (args.out_dir / "relation_grounding_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
