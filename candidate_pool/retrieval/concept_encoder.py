"""Shared bi-encoder for query ↔ graph-target matching (the common base).

Both retrieval paradigms reduce to ONE problem: match a query text to graph
targets (entities, or ontology concept nodes) in a SHARED vector space.

  - Paradigm A (dense / open):     query → nearest ENTITIES        (multihop seeds)
  - Paradigm B (label-constrained): query → nearest ONTOLOGY CONCEPTS (SUMMARY_OF entry)

This module is the single encoder both reuse, so any improvement (e.g. encoding
"label + description" instead of a bare label — shown in the entity-linking
literature to beat bare-label matching) helps BOTH paradigms at once.

Key feature — the ablation switch you asked for:
    target_mode = "bare"      -> encode the bare label only        (baseline)
    target_mode = "described" -> encode "label: description"       (improved)
So you can run the SAME retrieval with both and report the delta.

Backend:
  - sentence-transformer if available (paper numbers);
  - TF-IDF char/word fallback otherwise (sandbox-runnable, no model download).
Both paths expose the same API, so callers never branch on backend.

API (stable — callers depend on this):
    enc = ConceptEncoder(targets, target_mode="described", model_name=None)
        targets: list[Target]  (id, label, description, kind, payload)
    enc.score(query) -> {target_id: cosine}              # all targets
    enc.top(query, k=8, min_sim=0.0, kinds=None) -> [(target_id, sim), ...]
Convenience builders for the two paradigms:
    ConceptEncoder.from_entities(retr, ...)   # paradigm A targets
    ConceptEncoder.from_ontology(retr, ...)   # paradigm B targets
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Target: any graph node we want to match a query against.
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    id: str                          # node id, e.g. "ENT::odiff_procrastination" or "DOM::planning"
    label: str                       # human label, e.g. "procrastination"
    description: str = ""            # optional gloss / evidence span / definition
    kind: str = ""                  # e.g. "canonical_entity" | "strategy_domain" | "support_function"
    payload: dict = field(default_factory=dict)   # anything the caller wants back


_STOP = {"the", "a", "an", "to", "of", "and", "or", "for", "in", "on", "with",
         "is", "are", "be", "as", "at", "it", "this", "that", "i", "my", "me"}


def _tokens(text: str) -> list[str]:
    text = re.sub(r"\b(o?concept|o?diff|o?term|o?task|o?affect|o?tool|o?context|"
                  r"o?strategy|o?med|o?resource|ent)_", " ", str(text).lower())
    return [w for w in re.findall(r"[a-z]{3,}", text) if w not in _STOP]


def _clean_label(label: str) -> str:
    """Strip o<type>_ prefix and underscores: 'odiff_time_blindness' -> 'time blindness'."""
    return re.sub(r"^o?\w+?_", "", str(label)).replace("_", " ").strip()


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class ConceptEncoder:
    """Bi-encoder over a fixed target set. Lazy-builds the target matrix once."""

    def __init__(self, targets: list[Target], *, target_mode: str = "described",
                 model_name: str | None = None):
        assert target_mode in ("bare", "described"), target_mode
        self.targets = list(targets)
        self.target_mode = target_mode
        self.model_name = model_name or self._default_model_name()
        self.ids = [t.id for t in self.targets]
        self.idx = {t.id: i for i, t in enumerate(self.targets)}
        self._texts = [self._target_text(t) for t in self.targets]
        # backend state (filled lazily)
        self._ready = False
        self._failed = False
        self._backend = None          # "bert" | "tfidf"
        self._np = None
        self._model = None
        self._emb = None              # bert: (N, d) normalized matrix
        self._tfidf = None            # tfidf fallback object
        self._qcache = {}

    # ---- text composition: the bare-vs-described switch lives here ----
    def _target_text(self, t: Target) -> str:
        lab = _clean_label(t.label)
        if self.target_mode == "bare" or not t.description:
            return lab
        # "described": label + a short gloss (literature: name+description > bare)
        desc = str(t.description).strip().replace("\n", " ")
        return f"{lab}: {desc[:200]}" if desc else lab

    @staticmethod
    def _default_model_name() -> str:
        try:
            import configuration as config
            return os.environ.get("EVIDENCE_PIPELINE_BERT_MODEL",
                                  config.params("models", "semantic_backend",
                                                default="all-MiniLM-L6-v2"))
        except Exception:
            return os.environ.get("EVIDENCE_PIPELINE_BERT_MODEL", "all-MiniLM-L6-v2")

    # ---- backend build (bert preferred, tfidf fallback) ----
    def _ensure(self) -> bool:
        if self._ready:
            return True
        if self._failed:
            return False
        # try sentence-transformer
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._np = np
            local_only = os.environ.get("EVIDENCE_PIPELINE_HF_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}
            self._model = SentenceTransformer(
                self.model_name, local_files_only=local_only)
            self._emb = np.asarray(self._model.encode(
                self._texts, normalize_embeddings=True, show_progress_bar=False))
            self._backend = "bert"
            self._ready = True
            return True
        except Exception:
            pass
        # tfidf fallback (sandbox; no model download)
        try:
            import numpy as np
            from ..retrieval.backends import TfidfCosine, toks  # reuse project tfidf
            self._np = np
            self._tfidf = TfidfCosine([toks(t) for t in self._texts])
            self._backend = "tfidf"
            self._ready = True
            return True
        except Exception:
            self._failed = True
            return False

    def backend(self) -> str | None:
        self._ensure()
        return self._backend

    # ---- query encoding ----
    def _query_vec(self, query: str):
        if query in self._qcache:
            return self._qcache[query]
        v = self._np.asarray(self._model.encode([query], normalize_embeddings=True))[0]
        self._qcache[query] = v
        return v

    # ---- main API ----
    def score(self, query: str) -> dict[str, float]:
        """Cosine of query against every target. {} if no backend available."""
        if not query or not self._ensure():
            return {}
        if self._backend == "bert":
            qv = self._query_vec(query)
            sims = self._emb @ qv               # normalized → dot == cosine
            return {self.ids[i]: float(sims[i]) for i in range(len(self.ids))}
        # tfidf fallback: TfidfCosine.scores returns {doc_idx: score} for query tokens
        from ..retrieval.backends import toks
        raw = self._tfidf.scores(toks(query))   # {idx: cosine-ish}
        return {self.ids[i]: float(raw.get(i, 0.0)) for i in range(len(self.ids))}

    def top(self, query: str, k: int = 8, min_sim: float = 0.0,
            kinds: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-k (target_id, sim), optionally restricted to certain target kinds."""
        sc = self.score(query)
        if kinds:
            keep = {t.id for t in self.targets if t.kind in kinds}
            sc = {i: s for i, s in sc.items() if i in keep}
        ranked = sorted(((i, s) for i, s in sc.items() if s >= min_sim),
                        key=lambda x: -x[1])
        return ranked[:k]

    def target(self, target_id: str) -> Target | None:
        i = self.idx.get(target_id)
        return self.targets[i] if i is not None else None

    # ----------------------------------------------------------------- #
    # Convenience builders for the two paradigms (so callers don't
    # hand-assemble Target lists).
    # ----------------------------------------------------------------- #
    @classmethod
    def from_entities(cls, retr, *, target_mode: str = "described",
                      descriptions: dict[str, str] | None = None,
                      model_name: str | None = None) -> "ConceptEncoder":
        """Paradigm A targets = canonical entities.

        `descriptions`: optional {ENT::id -> gloss/evidence span}. If absent,
        falls back to bare label even in 'described' mode (no harm).
        """
        descriptions = descriptions or {}
        targets = []
        nodes = getattr(retr, "nodes", {}) or {}
        for nid, row in nodes.items():
            if str(row.get("node_type")) != "canonical_entity":
                continue
            lab = str(row.get("label") or nid)
            targets.append(Target(id=nid, label=lab,
                                  description=descriptions.get(nid, ""),
                                  kind="canonical_entity"))
        return cls(targets, target_mode=target_mode, model_name=model_name)

    @classmethod
    def from_ontology(cls, retr, *, kinds=("support_function", "strategy_domain",
                                           "ef_mechanism"),
                      target_mode: str = "described",
                      descriptions: dict[str, str] | None = None,
                      model_name: str | None = None) -> "ConceptEncoder":
        """Paradigm B targets = ontology concept nodes (T1/T2).

        These are the SUMMARY_OF entry points. `descriptions` can supply a gloss
        per concept (e.g. from schema definitions) to make 'described' meaningful.
        """
        descriptions = descriptions or {}
        kinds = set(kinds)
        targets = []
        nodes = getattr(retr, "nodes", {}) or {}
        for nid, row in nodes.items():
            nt = str(row.get("node_type"))
            if nt not in kinds:
                continue
            lab = str(row.get("label") or nid)
            targets.append(Target(id=nid, label=lab,
                                  description=descriptions.get(nid, ""),
                                  kind=nt))
        return cls(targets, target_mode=target_mode, model_name=model_name)
