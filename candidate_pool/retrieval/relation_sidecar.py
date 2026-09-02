"""Relation provenance sidecar for a frozen Official HippoRAG2 graph.

HippoRAG2 persists only a numeric ``weight`` edge attribute.  This module maps
existing OpenIE triples back to existing graph edge ids without adding nodes,
edges or attributes to the graph.  The sidecar is a read-only query-time index.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _text_processing(text: object) -> str:
    """Mirror upstream HippoRAG ``text_processing`` exactly."""
    return re.sub(r"[^A-Za-z0-9 ]", " ", str(text).lower()).strip()


def _entity_key(text: object) -> str:
    value = _text_processing(text)
    return "entity-" + hashlib.md5(value.encode()).hexdigest()


@dataclass(frozen=True)
class RelationSidecar:
    relation_labels: tuple[str, ...]
    edge_ids: np.ndarray
    relation_ids: np.ndarray
    graph_edges: int
    openie_triples: int
    relation_pairs: int
    mapped_pairs: int
    edges_with_relation: int
    provenance_sample: tuple[dict, ...] = ()

    @property
    def coverage(self) -> dict:
        return {
            "graph_edges": int(self.graph_edges),
            "openie_triples": int(self.openie_triples),
            "unique_relation_labels": len(self.relation_labels),
            "relation_pairs": int(self.relation_pairs),
            "mapped_pairs": int(self.mapped_pairs),
            "pair_coverage": (
                self.mapped_pairs / self.relation_pairs if self.relation_pairs else 0.0),
            "edges_with_relation": int(self.edges_with_relation),
            "edge_coverage": (
                self.edges_with_relation / self.graph_edges if self.graph_edges else 0.0),
            "edge_relation_associations": int(len(self.edge_ids)),
        }

    def aggregate(self, relation_scores: np.ndarray, mode: str = "max") -> np.ndarray:
        """Return one score per graph edge; edges without relations use ``NaN``."""
        values = np.asarray(relation_scores, dtype=np.float64)
        if len(values) != len(self.relation_labels):
            raise ValueError("relation score vector is not aligned with sidecar labels")
        result = np.full(self.graph_edges, np.nan, dtype=np.float64)
        if not len(self.edge_ids):
            return result
        scores = values[self.relation_ids]
        if mode == "max":
            tmp = np.full(self.graph_edges, -np.inf, dtype=np.float64)
            np.maximum.at(tmp, self.edge_ids, scores)
            result[np.isfinite(tmp)] = tmp[np.isfinite(tmp)]
        elif mode == "mean":
            total = np.zeros(self.graph_edges, dtype=np.float64)
            count = np.zeros(self.graph_edges, dtype=np.int64)
            np.add.at(total, self.edge_ids, scores)
            np.add.at(count, self.edge_ids, 1)
            mask = count > 0
            result[mask] = total[mask] / count[mask]
        else:
            raise ValueError("relation aggregation must be 'max' or 'mean'")
        return result


def build_relation_sidecar(graph, openie_payload: dict, sample_size: int = 25) -> RelationSidecar:
    docs = openie_payload.get("docs") or []
    pair_relations: dict[tuple[str, str], set[str]] = {}
    triple_count = 0
    samples = []
    names = set(graph.vs["name"])
    for doc in docs:
        for triple in doc.get("extracted_triples") or []:
            if len(triple) < 3:
                continue
            triple_count += 1
            src, relation, dst = _entity_key(triple[0]), _text_processing(triple[1]), _entity_key(triple[2])
            if not relation:
                continue
            pair = tuple(sorted((src, dst)))
            pair_relations.setdefault(pair, set()).add(relation)
            if len(samples) < sample_size:
                samples.append({
                    "source_text": _text_processing(triple[0]),
                    "target_text": _text_processing(triple[2]),
                    "source_node": src,
                    "target_node": dst,
                    "relation": relation,
                    "endpoints_exist": src in names and dst in names,
                })

    labels = tuple(sorted({r for values in pair_relations.values() for r in values}))
    relation_to_id = {value: idx for idx, value in enumerate(labels)}
    edge_ids, relation_ids, mapped_pairs = [], [], set()
    vertex_names = graph.vs["name"]
    for edge_id, (src_idx, dst_idx) in enumerate(graph.get_edgelist()):
        pair = tuple(sorted((vertex_names[src_idx], vertex_names[dst_idx])))
        relations = pair_relations.get(pair)
        if not relations:
            continue
        mapped_pairs.add(pair)
        for relation in sorted(relations):
            edge_ids.append(edge_id)
            relation_ids.append(relation_to_id[relation])
    edge_array = np.asarray(edge_ids, dtype=np.int64)
    return RelationSidecar(
        relation_labels=labels,
        edge_ids=edge_array,
        relation_ids=np.asarray(relation_ids, dtype=np.int32),
        graph_edges=graph.ecount(),
        openie_triples=triple_count,
        relation_pairs=len(pair_relations),
        mapped_pairs=len(mapped_pairs),
        edges_with_relation=len(np.unique(edge_array)) if len(edge_array) else 0,
        provenance_sample=tuple(samples),
    )


def save_relation_sidecar(sidecar: RelationSidecar, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        relation_labels=np.asarray(sidecar.relation_labels),
        edge_ids=sidecar.edge_ids,
        relation_ids=sidecar.relation_ids,
        counts=np.asarray([
            sidecar.graph_edges, sidecar.openie_triples, sidecar.relation_pairs,
            sidecar.mapped_pairs, sidecar.edges_with_relation], dtype=np.int64),
    )
    path.with_suffix(".manifest.json").write_text(json.dumps({
        **sidecar.coverage,
        "provenance_sample": list(sidecar.provenance_sample),
        "source": "existing Official HippoRAG2 OpenIE cache",
        "graph_mutated": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_relation_sidecar(path: Path) -> RelationSidecar:
    payload = np.load(path, allow_pickle=False)
    counts = payload["counts"].astype(np.int64).tolist()
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return RelationSidecar(
        relation_labels=tuple(str(x) for x in payload["relation_labels"].tolist()),
        edge_ids=payload["edge_ids"].astype(np.int64),
        relation_ids=payload["relation_ids"].astype(np.int32),
        graph_edges=int(counts[0]), openie_triples=int(counts[1]),
        relation_pairs=int(counts[2]), mapped_pairs=int(counts[3]),
        edges_with_relation=int(counts[4]),
        provenance_sample=tuple(manifest.get("provenance_sample") or []),
    )


def labels_fingerprint(labels: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for label in labels:
        digest.update(label.encode()); digest.update(b"\0")
    return digest.hexdigest()


def load_or_encode_relation_embeddings(
    sidecar: RelationSidecar,
    embedding_model,
    cache_path: Path,
    *,
    instruction: str,
) -> tuple[np.ndarray, dict]:
    """Cache relation-label embeddings outside the immutable graph directory."""
    fingerprint = labels_fingerprint(sidecar.relation_labels)
    manifest_path = cache_path.with_suffix(".manifest.json")
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("labels_sha256") == fingerprint:
            matrix = np.load(cache_path, allow_pickle=False)
            if len(matrix) == len(sidecar.relation_labels):
                return np.asarray(matrix, dtype=np.float32), {**manifest, "cache_hit": True}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = embedding_model.batch_encode(
        list(sidecar.relation_labels), instruction=instruction, norm=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    np.save(cache_path, matrix)
    manifest = {
        "labels": len(sidecar.relation_labels),
        "labels_sha256": fingerprint,
        "shape": list(matrix.shape),
        "instruction": instruction,
        "cache_hit": False,
        "graph_mutated": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return matrix, manifest
