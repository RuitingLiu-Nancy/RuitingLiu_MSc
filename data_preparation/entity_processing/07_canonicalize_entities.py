#!/usr/bin/env python3
"""Canonicalise open LLM entities with blocking + local NLP similarity.

This is the next step after llm_extract_open_entities.py. It is stricter than
the raw `canonical` strings but lighter than a hand-built ontology:

  raw mentions -> unique phrase records -> within-kind clustering -> graph table

The script deliberately does not call an LLM. It uses cheap local NLP signals:
normalisation, blocking by loose kind, TF-IDF char/word similarity, and
agglomerative clustering. The output is reviewable before graph rebuild.

Outputs:
  entity_mentions_clustered.csv
  entity_cluster_nodes.csv
  entity_cluster_review.md
  entities_canonical.csv              # graph-builder compatible table
  canonicalization_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.io_utils import write_table

WS = re.compile(r"\s+")
NONWORD = re.compile(r"[^a-z0-9 ]+")

KIND_TO_TYPE = {
    "difficulty_or_barrier": "OPEN_DIFFICULTY",
    "life_task_or_goal": "OPEN_LIFE_TASK",
    "strategy_or_method": "OPEN_STRATEGY",
    "tool_or_artifact": "OPEN_TOOL",
    "medication_or_treatment": "OPEN_MED_TREATMENT",
    "context_or_situation": "OPEN_CONTEXT",
    "emotion_or_self_state": "OPEN_AFFECT_STATE",
    "community_phrase": "OPEN_COMMUNITY_TERM",
    "resource_or_reference": "OPEN_RESOURCE",
    "other_specific_concept": "OPEN_CONCEPT",
}

TYPE_PREFIX = {
    "OPEN_DIFFICULTY": "odiff",
    "OPEN_LIFE_TASK": "otask",
    "OPEN_STRATEGY": "ostrategy",
    "OPEN_TOOL": "otool",
    "OPEN_MED_TREATMENT": "omed",
    "OPEN_CONTEXT": "ocontext",
    "OPEN_AFFECT_STATE": "oaffect",
    "OPEN_COMMUNITY_TERM": "oterm",
    "OPEN_RESOURCE": "oresource",
    "OPEN_CONCEPT": "oconcept",
}

GENERIC = {
    "thing", "things", "problem", "problems", "advice", "people", "person",
    "stuff", "way", "ways", "point", "points", "time", "times", "life",
    "work", "help", "support", "experience", "experiences", "strategy",
    "method", "task", "tasks", "issue", "issues",
}

STOP = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "without", "by", "from", "as", "at", "my", "your", "their", "our",
}


def norm(text: object) -> str:
    s = str(text or "").lower().strip()
    s = s.replace("adhd-specific", "adhd")
    s = NONWORD.sub(" ", s)
    s = WS.sub(" ", s).strip()
    words = []
    for w in s.split():
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("ses"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def slug(text: str, max_words: int = 7) -> str:
    words = norm(text).split()
    return "_".join(words[:max_words]) if words else "empty"


def is_bad_phrase(text: str) -> bool:
    n = norm(text)
    if not n or len(n) < 3:
        return True
    if n in GENERIC:
        return True
    # Single generic adjectives/verbs do not make useful graph nodes.
    if len(n.split()) == 1 and n in {
        "better", "hard", "easy", "easier", "different", "important",
        "useful", "helpful", "start", "finish", "remember", "focus",
    }:
        return True
    return False


def load_mentions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            for ent in rec.get("entities", []) or []:
                canonical = str(ent.get("canonical") or ent.get("surface") or "").strip().lower()
                surface = str(ent.get("surface") or canonical).strip()
                loose_kind = str(ent.get("loose_kind") or "other_specific_concept").strip().lower()
                open_type = KIND_TO_TYPE.get(loose_kind, "OPEN_CONCEPT")
                if is_bad_phrase(canonical):
                    continue
                rows.append({
                    "unit_id": rec.get("unit_id", ""),
                    "post_id": rec.get("post_id", ""),
                    "comment_id": rec.get("comment_id", ""),
                    "unit_type": rec.get("unit_type", "comment"),
                    "scenario": rec.get("scenario", ""),
                    "tier": rec.get("tier", ""),
                    "support_functions": rec.get("support_functions", ""),
                    "strategy_domains": rec.get("strategy_domains", ""),
                    "ef_mechanisms": rec.get("ef_mechanisms", ""),
                    "surface": surface,
                    "canonical_raw": canonical,
                    "canonical_norm": norm(canonical),
                    "loose_kind": loose_kind,
                    "type": open_type,
                    "evidence_source": ent.get("evidence_source", ""),
                    "evidence": ent.get("evidence", ""),
                    "why_useful": ent.get("why_useful", ""),
                })
    return pd.DataFrame(rows)


def compact_counts(values: pd.Series, n: int = 8) -> str:
    counts = Counter(v for v in values.astype(str) if v and v.lower() != "nan")
    return " | ".join(f"{k}:{v}" for k, v in counts.most_common(n))


def phrase_table(mentions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (open_type, canonical_norm), grp in mentions.groupby(["type", "canonical_norm"]):
        text_counts = Counter(grp["canonical_raw"])
        surface_counts = Counter(grp["surface"])
        rows.append({
            "type": open_type,
            "canonical_norm": canonical_norm,
            "display": text_counts.most_common(1)[0][0],
            "surface_example": surface_counts.most_common(1)[0][0],
            "n_mentions": int(len(grp)),
            "n_comments": int(grp["comment_id"].nunique()),
            "scenarios": compact_counts(grp["scenario"], 5),
            "strategy_domains": compact_counts(grp["strategy_domains"], 5),
            "why_useful": " | ".join([x for x in grp["why_useful"].astype(str).head(3) if x]),
            "evidence_examples": " || ".join([x[:180] for x in grp["evidence"].astype(str).head(3) if x]),
        })
    return pd.DataFrame(rows)


SYN_BLOCKS = {
    "time_cue": {
        "timer", "timers", "alarm", "alarms", "pomodoro", "tomato", "countdown",
        "reminder", "reminders", "calendar", "schedule", "scheduled",
    },
    "writing": {
        "write", "writing", "essay", "essays", "draft", "drafting", "outline",
        "outlining", "dictation", "dictate", "typed", "typing", "journal",
    },
    "study_notes": {
        "study", "studying", "lecture", "lectures", "note", "notes",
        "notetaking", "college", "class", "classes", "school",
    },
    "body_doubling": {
        "body", "doubling", "accountability", "coworking", "friend", "partner",
        "peer",
    },
    "environment": {
        "environment", "coffee", "shop", "library", "room", "noise", "noisy",
        "headphones", "music", "background",
    },
    "medication": {
        "medication", "medications", "meds", "stimulant", "stimulants",
        "adderall", "vyvanse", "ritalin", "concerta", "pharmacy", "shortage",
    },
    "task_initiation": {
        "initiation", "start", "starting", "begin", "paralysis", "freeze",
        "activation", "motivation",
    },
    "cleaning_home": {
        "clean", "cleaning", "dishes", "dishwasher", "laundry", "chores",
        "room", "house", "home",
    },
    "email_admin": {
        "email", "emails", "mail", "paperless", "ocr", "scanner", "scan",
        "document", "documents", "bill", "bills",
    },
    "sleep": {
        "sleep", "bed", "bedtime", "wake", "waking", "morning", "night",
        "routine",
    },
    "job_resume": {
        "job", "jobs", "resume", "resumes", "career", "workplace", "work",
        "interview", "bullet", "bullets",
    },
}


def block_key(text: str) -> str:
    toks = norm(text).split()
    tokset = set(toks)
    for name, words in SYN_BLOCKS.items():
        if tokset & words:
            return name
    content = [t for t in toks if t not in STOP and t not in GENERIC]
    if not content:
        return "misc"
    return "_".join(content[:2])


def vectorize(texts: list[str]):
    # Character ngrams catch tomato timer vs timers; word ngrams keep meaning.
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    cmat = char.fit_transform(texts)
    wmat = word.fit_transform(texts)
    from scipy.sparse import hstack
    return normalize(hstack([0.55 * cmat, 0.45 * wmat]))


# lazily-built shared semantic encoder (sentence-transformer; None if unavailable)
_SEM_MODEL = {"m": None, "failed": False}


def _semantic_sim(texts: list[str]):
    """Cosine similarity matrix from sentence-transformer embeddings of
    'name + description' texts (CESI/hybrid canonicalization). Returns None if
    sentence-transformers is unavailable (caller falls back to pure TF-IDF)."""
    if _SEM_MODEL["failed"]:
        return None
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        if _SEM_MODEL["m"] is None:
            from sentence_transformers import SentenceTransformer
            import os as _os
            name = _os.environ.get("EVIDENCE_PIPELINE_BERT_MODEL", "all-MiniLM-L6-v2")
            _SEM_MODEL["m"] = SentenceTransformer(name)
        emb = _SEM_MODEL["m"].encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False)
        return cosine_similarity(np.asarray(emb))
    except Exception:
        _SEM_MODEL["failed"] = True
        return None


def cluster_group(group: pd.DataFrame, threshold: float, min_cluster_size: int,
                  semantic_weight: float = 0.0) -> pd.Series:
    if len(group) == 1:
        return pd.Series([0], index=group.index)
    mat = vectorize(group["cluster_text"].tolist())
    dense = mat.toarray()
    kwargs = {
        "n_clusters": None,
        "distance_threshold": 1.0 - threshold,
        "linkage": "average",
    }
    # ---- hybrid (CESI-style): TF-IDF char/word + optional semantic, weighted ----
    # semantic_weight=0 (default) => identical to the original pure-TF-IDF path.
    if semantic_weight > 0:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf_sim = cosine_similarity(dense)
        sem_sim = _semantic_sim(group["sem_text"].tolist()
                                if "sem_text" in group else
                                group["cluster_text"].tolist())
        if sem_sim is not None:
            w = float(semantic_weight)
            combined = (1.0 - w) * tfidf_sim + w * sem_sim
            dist = np.clip(1.0 - combined, 0.0, None)
            np.fill_diagonal(dist, 0.0)
            try:
                model = AgglomerativeClustering(metric="precomputed", **kwargs)
            except TypeError:
                model = AgglomerativeClustering(affinity="precomputed", **kwargs)
            labels = model.fit_predict(dist)
            return _post_small_clusters(labels, group, min_cluster_size)
        # semantic unavailable -> fall through to pure TF-IDF (no-harm)
    try:
        model = AgglomerativeClustering(metric="cosine", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="cosine", **kwargs)
    labels = model.fit_predict(dense)

    return _post_small_clusters(labels, group, min_cluster_size)


def _post_small_clusters(labels, group, min_cluster_size: int) -> pd.Series:
    """Small clusters are often accidental: split sub-min clusters back into
    singleton-ish phrase nodes. Shared by the TF-IDF and hybrid paths."""
    counts = Counter(labels)
    if min_cluster_size > 1:
        next_label = max(labels) + 1
        labels = labels.copy()
        for i, lab in enumerate(labels):
            if counts[lab] < min_cluster_size:
                labels[i] = next_label
                next_label += 1
    return pd.Series(labels, index=group.index)


def choose_name(group: pd.DataFrame) -> str:
    # Prefer frequent, not overlong phrase. This is reviewable; no LLM naming.
    candidates = group.sort_values(
        ["n_mentions", "n_comments"], ascending=False
    )["display"].astype(str).tolist()
    candidates = sorted(candidates, key=lambda x: (len(norm(x).split()) > 7, len(x)))
    return candidates[0] if candidates else "unnamed"


def make_cluster_id(open_type: str, name: str, existing: set[str]) -> str:
    prefix = TYPE_PREFIX.get(open_type, "oent")
    base = f"{prefix}_{slug(name)}"
    cid = base
    i = 2
    while cid in existing:
        cid = f"{base}_{i}"
        i += 1
    existing.add(cid)
    return cid


def run(input_jsonl: Path, out_dir: Path, threshold: float,
        min_cluster_size: int, min_mentions: int,
        semantic_weight: float = 0.0) -> None:
    mentions = load_mentions(input_jsonl)
    if mentions.empty:
        raise SystemExit("No usable entity mentions found.")
    phrases = phrase_table(mentions)
    phrases["cluster_text"] = (
        phrases["display"].fillna("") + " | " +
        phrases["canonical_norm"].fillna("") + " | " +
        phrases["why_useful"].fillna("")
    )
    # sem_text = name + short description, encoded by the semantic model in the
    # hybrid path (CESI: names lexical via TF-IDF, descriptions semantic).
    phrases["sem_text"] = (
        phrases["display"].fillna("") + ". " +
        phrases["why_useful"].fillna("")
    ).str.slice(0, 220)
    phrases["block_key"] = phrases["canonical_norm"].map(block_key)
    if semantic_weight > 0:
        print(f"[canon] hybrid clustering: semantic_weight={semantic_weight} "
              f"(TF-IDF {1-semantic_weight:.2f} + semantic {semantic_weight:.2f})")

    all_labels = []
    # Blocking keeps clustering local and interpretable. It also avoids the
    # O(n^2) cost of agglomerative clustering over thousands of phrases.
    for (open_type, blk), grp in phrases.groupby(["type", "block_key"]):
        labs = cluster_group(grp, threshold, min_cluster_size, semantic_weight)
        all_labels.append(labs.map(lambda x, t=open_type, b=blk: f"{t}::{b}::{x}"))
    phrases["local_cluster"] = pd.concat(all_labels).sort_index()

    existing_ids: set[str] = set()
    cluster_rows = []
    local_to_id = {}
    for local, grp in phrases.groupby("local_cluster"):
        open_type = grp["type"].iloc[0]
        name = choose_name(grp)
        cid = make_cluster_id(open_type, name, existing_ids)
        local_to_id[local] = cid
        if grp["n_mentions"].sum() < min_mentions:
            continue
        cluster_rows.append({
            "canonical_id": cid,
            "cluster_name": name,
            "type": open_type,
            "n_phrases": int(len(grp)),
            "n_mentions": int(grp["n_mentions"].sum()),
            "n_comments": int(grp["n_comments"].sum()),
            "top_phrases": compact_counts(grp["display"], 12),
            "scenarios": compact_counts(grp["scenarios"], 6),
            "strategy_domains": compact_counts(grp["strategy_domains"], 6),
            "evidence_examples": " || ".join([x for x in grp["evidence_examples"].head(4) if x]),
        })

    clusters = pd.DataFrame(cluster_rows).sort_values(
        ["n_mentions", "n_comments", "n_phrases"], ascending=False)
    kept = set(clusters["canonical_id"]) if not clusters.empty else set()

    phrases["canonical_id_clustered"] = phrases["local_cluster"].map(local_to_id)
    phrase_map = dict(zip(
        zip(phrases["type"], phrases["canonical_norm"]),
        phrases["canonical_id_clustered"],
    ))
    mentions["canonical_id"] = [
        phrase_map.get((t, n), f"{TYPE_PREFIX.get(t, 'oent')}_{slug(n)}")
        for t, n in zip(mentions["type"], mentions["canonical_norm"])
    ]
    if min_mentions > 1:
        mentions = mentions[mentions["canonical_id"].isin(kept)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    mentions.to_csv(out_dir / "entity_mentions_clustered.csv", index=False)
    phrases.drop(columns=["cluster_text"]).to_csv(out_dir / "entity_phrase_records.csv", index=False)
    clusters.to_csv(out_dir / "entity_cluster_nodes.csv", index=False)

    graph_table = mentions[[
        "post_id", "comment_id", "unit_type", "type", "surface", "canonical_id"
    ]].rename(columns={"surface": "text"}).copy()
    graph_table["canonical_mesh_id"] = ""
    write_table(
        graph_table[["post_id", "comment_id", "unit_type", "type", "text",
                     "canonical_id", "canonical_mesh_id"]],
        out_dir,
        "entities_canonical",
    )

    write_review(clusters, out_dir / "entity_cluster_review.md")
    summary = {
        "input": str(input_jsonl),
        "mentions": int(len(mentions)),
        "unique_phrase_records": int(len(phrases)),
        "clusters": int(len(clusters)),
        "threshold": threshold,
        "min_cluster_size": min_cluster_size,
        "min_mentions": min_mentions,
        "type_counts": Counter(mentions["type"]).most_common(),
    }
    (out_dir / "canonicalization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[canonical-cluster] wrote -> {out_dir}")


def write_review(clusters: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Canonical Open Entity Cluster Review", ""]
    for _, row in clusters.head(250).iterrows():
        lines.append(
            f"## {row['canonical_id']} · {row['cluster_name']} "
            f"(mentions={row['n_mentions']}, phrases={row['n_phrases']})"
        )
        lines.append(f"- type: {row['type']}")
        lines.append(f"- top phrases: {row['top_phrases']}")
        lines.append(f"- scenarios: {row['scenarios']}")
        lines.append(f"- strategy domains: {row['strategy_domains']}")
        lines.append("- evidence:")
        for ex in str(row["evidence_examples"]).split(" || ")[:4]:
            if ex:
                lines.append(f"  - {ex}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.72,
                    help="Cosine similarity threshold; lower merges more.")
    ap.add_argument("--min-cluster-size", type=int, default=1,
                    help="Clusters smaller than this are kept as singleton concepts.")
    ap.add_argument("--min-mentions", type=int, default=1,
                    help="Drop final clusters with fewer than this many mentions.")
    ap.add_argument("--semantic-weight", type=float, default=0.0,
                    help="0 = pure TF-IDF (default, original behaviour). "
                         ">0 = hybrid CESI-style: combined_sim = (1-w)*TF-IDF + "
                         "w*semantic(name+description). Needs sentence-transformers; "
                         "falls back to TF-IDF if unavailable. Try 0.5 for A/B.")
    args = ap.parse_args()
    run(args.input, args.out_dir, args.threshold, args.min_cluster_size,
        args.min_mentions, semantic_weight=args.semantic_weight)


if __name__ == "__main__":
    main()
