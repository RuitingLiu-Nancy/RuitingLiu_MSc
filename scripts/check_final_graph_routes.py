#!/usr/bin/env python3
"""Offline checks of the final graph wrapper; no model download or inference.

Use --reference with an archived runner to compare retained route behavior.
This exercises the wrapper contract with a deterministic fake upstream, not
full-corpus PPR or model quality.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from candidate_pool import run_official_hipporag_bedrock as current


def load_reference(path):
    spec = importlib.util.spec_from_file_location("archived_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FactStore:
    def get_rows(self, ids):
        return {key: {"content": repr((key, "rel", "object"))} for key in ids}


def profile_result(module, profile, scores, top_k):
    h = SimpleNamespace(global_config=SimpleNamespace(
        linking_top_k=top_k, passage_node_weight=0.05),
        fact_node_keys=[str(i) for i in range(len(scores))],
        fact_embedding_store=FactStore())
    original_scores = list(scores)
    manifest = module._install_retrieval_profile(h, profile)
    result = h.rerank_facts("query", scores)
    assert scores == original_scores, "retained profiles must preserve fact scores"
    assert not manifest["recognition_filter_enabled"]
    assert h.global_config.passage_node_weight == (
        0.0 if profile == "fact_only_no_recognition" else 0.05)
    return result, manifest, h.global_config.passage_node_weight


class FakeHippoRAG:
    instances = []

    def __init__(self, **kwargs):
        self.global_config = SimpleNamespace(linking_top_k=2, passage_node_weight=0.05)
        self.fact_node_keys = ["f0", "f1", "f2"]
        self.fact_embedding_store = FactStore()
        self.llm_model = SimpleNamespace(infer=lambda *a, **k: None,
                                         batch_infer=lambda *a, **k: None)
        self.openie = SimpleNamespace(llm_model=self.llm_model)
        self.embedding_model = SimpleNamespace(encode=lambda x: x,
                                                batch_encode=lambda x: x)
        self.__class__.instances.append(self)

    def index(self, docs):
        self.docs = list(docs)

    def retrieve(self, queries, num_to_retrieve):
        ids, _, _ = self.rerank_facts("query", [0.3, 0.8, 0.8])
        return [SimpleNamespace(docs=[self.docs[i] for i in ids][:num_to_retrieve],
                                doc_scores=[0.8, 0.8][:num_to_retrieve]) for q in queries]


def run_fixture(module, folder, profile):
    corpus = folder / "corpus.json"
    queries = folder / "validation.json"
    corpus.write_text(json.dumps([{"title": str(i), "text": f"raw comment {i}"}
                                  for i in range(3)]))
    queries.write_text(json.dumps([{"id": "q1", "question": "query", "paragraphs": []}]))
    modules = {name: ModuleType(name) for name in (
        "hipporag", "hipporag.HippoRAG", "hipporag.information_extraction",
        "hipporag.information_extraction.openie_openai")}
    modules["hipporag.HippoRAG"].HippoRAG = FakeHippoRAG
    modules["hipporag.information_extraction.openie_openai"].ThreadPoolExecutor = ThreadPoolExecutor
    out = folder / "result.jsonl"
    with patch.dict(sys.modules, modules), patch.dict(os.environ):
        manifest = module.run(corpus, queries, folder / "cache", out, 3, 1,
                              "fixture-llm", "fixture-embedding", 2, 1,
                              retrieval_profile=profile, forbid_llm_inference=True)
        instance = FakeHippoRAG.instances[-1]
        assert instance.docs == [f"raw comment {i}" for i in range(3)]
        try:
            instance.llm_model.infer("must fail")
        except RuntimeError:
            pass
        else:
            raise AssertionError("inference blocker did not reject a call")
    assert manifest["llm_inference_audit"]["enabled"]
    assert manifest["indexed_documents"] == 3
    assert manifest["retrieved_queries"] == 1
    return out.read_text(), manifest["retrieval_ablation"]


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected fail-closed rejection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reference = load_reference(args.reference) if args.reference else None
    cases = [[], [0.4], [0.2, 0.9, 0.9, -0.2], [0.0, 0.0, 0.0], [-0.2, -0.1]]
    for profile in current.RETRIEVAL_PROFILES:
        for scores in cases:
            for top_k in (1, 3, 8):
                result = profile_result(current, profile, list(scores), top_k)
                expected = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
                assert result[0][0] == expected
                if reference:
                    assert result == profile_result(reference, profile, list(scores), top_k)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            actual = run_fixture(current, folder, profile)
            if reference:
                assert actual == run_fixture(reference, folder, profile)
            test_file = folder / "test.json"
            test_file.write_text("[]")
            expect_value_error(lambda: current._read_validation_adapter(test_file))
            expect_value_error(lambda: current._validate_query_contract(
                test_file, test_split_used=True, expected_query_sha256=None,
                expected_query_count=None))
            expect_value_error(lambda: current._validate_query_contract(
                test_file, test_split_used=True, expected_query_sha256="wrong",
                expected_query_count=0))
            current._validate_query_contract(test_file, test_split_used=True,
                expected_query_sha256=current._sha256_file(test_file), expected_query_count=0)
    expect_value_error(lambda: current._install_retrieval_profile(None, "ontology_bridge_no_recognition"))
    print("PASSED: 30 fact-ranking cases, both wrapper routes, inference blocking and frozen-query gates"
          + ("; retained behavior matches archive" if reference else ""))


if __name__ == "__main__":
    main()
