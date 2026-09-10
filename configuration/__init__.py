# 【文件 001】声明 configuration 包，并提供中央配置读取接口
# 【流程位置】公共配置；所属包：configuration
# 【主要函数】load, params, fusion_weights, prompt_path, prompt, judge_criteria
# 【输入接口】path 等函数参数；返回值及写出操作见对应函数
# 【调用方】data_preparation/entity_processing/open_entity_extraction.py
# 【调用方】candidate_pool/retrieval/concept_encoder.py
# 【调用方】candidate_pool/retrieval/hierarchy.py
# 【调用方】candidate_pool/retrieval/iterative_retrieval.py
# 【调用方】candidate_pool/retrieval/multihop.py
# 【调用方】candidate_pool/retrieval/query_rewrite.py
# 【调用方】candidate_pool/run_dense_semantic_drift_rescue_audit.py
# 【调用方】candidate_pool/run_official_hipporag_bedrock.py
# 【调用方】fusion/analyze_rq2a_graph_budget_sweep.py
# 【调用方】utility_scoring/annotation/run_top3_residual_judging.py
# 【调用方】utility_scoring/build_stage2_redesign_features.py
# 【调用方】utility_scoring/build_stage2_redesign_features_rrf2pool.py
# 【调用方】utility_scoring/learned_diffusion/config.py
# 【调用方】utility_scoring/learned_diffusion/reranker_validation.py
# 【调用方】utility_scoring/run_lightweight_scorer_search_dev300.py
# 【调用方】utility_scoring/run_rq2b_scorer_family_oof_dev300.py
# 【调用方】utility_scoring/run_stage2_redesign_crossencoder.py
# 【调用方】evidence_selection/run_rq2b_symmetric_hyperparameter_selection.py
# 【调用方】evidence_selection/run_selection_action_space_repair.py
# 【调用方】evidence_selection/run_set_aware_selection_ablation.py
# 【调用方】evidence_selection/run_strict_native_graph_conservative_policy.py
# 【调用方】evidence_selection/run_strict_sbert_mixed_selector.py
# 【调用方】evaluation/analyze_rq2b_set_correspondence.py
# 【调用方】evaluation/community_reply_auxiliary.py
# 【调用方】evaluation/confirmatory_test200_rq2b.py
# 【调用方】evaluation/external_fusion_utility_rerank.py
# 【调用方】evaluation/fusion_strategy_ablation.py
# 【调用方】evaluation/quality_diversity_rerank.py
# 【调用方】evaluation/run_evidence_signal_triangulation.py
# 【调用方】evaluation/run_stage2_community_dev300_complete.py
# 【调用方】evaluation/two_graph_compare.py
# 【调用方】evaluation/utility.py
# 【调用方】shared/llm_client.py

"""Single source of truth for experiment parameters and external prompts.

Usage:
    from configuration import load, params, prompt
    cfg = load()                      # whole dict (cached)
    fw  = params("fusion", "weights") # nested get with path
    sysp = prompt("generate_answer")  # read <external-prompt-dir>/<name>.txt

Override the yaml path with env EVIDENCE_PIPELINE_PARAMS (e.g. for an experiment variant).
All scripts read from here so a single edit propagates everywhere (no drift).
"""
from __future__ import annotations

import os
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_DEFAULT_YAML = _DIR / "params.yaml"
_PROMPT_DIR = None  # prompt material is supplied outside this release

_CACHE = None
_CACHE_PATH = None


def _parse_scalar(value: str):
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x.strip()) for x in inner.split(",")]
    if ((value.startswith('"') and value.endswith('"')) or
            (value.startswith("'") and value.endswith("'"))):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict:
    """Tiny fallback parser for this repo's simple params.yaml.

    It handles nested indentation, scalars, booleans, and inline lists. It is
    not a general YAML implementation; PyYAML is still preferred when present.
    """
    root: dict = {}
    stack = [(-1, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


# 【函数 001.03】load：按显式路径、EVIDENCE_PIPELINE_PARAMS 环境变量、默认 params.yaml 的优先级载入 YAML，并缓存结果
# 【输入】path: str | None=None, force: bool=False
# 【实现】通过上下文管理器管理资源；调用 Path, os.environ.get, open, yaml.safe_load, _load_simple_yaml
# 【返回】_CACHE；cfg
def load(path: str | None = None, force: bool = False) -> dict:
    """Load params.yaml (cached). Env EVIDENCE_PIPELINE_PARAMS overrides the path."""
    global _CACHE, _CACHE_PATH
    yaml_path = Path(path or os.environ.get("EVIDENCE_PIPELINE_PARAMS", _DEFAULT_YAML))
    if _CACHE is not None and not force and _CACHE_PATH == yaml_path:
        return _CACHE
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except ImportError:
        cfg = _load_simple_yaml(yaml_path)
    _CACHE, _CACHE_PATH = cfg, yaml_path
    return cfg


# 【函数 001.04】params：沿传入的键序列逐层取配置
# 【输入】*keys, default=None
# 【实现】遍历或迭代输入；调用 load
# 【返回】default；cur
def params(*keys, default=None):
    """Nested get: params('fusion','weights') -> {...}. Missing -> default."""
    cur = load()
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def fusion_weights() -> dict:
    """The single fusion-weight source (was duplicated across 5 scripts)."""
    return dict(params("fusion", "weights", default={}))


# 【函数 001.06】prompt_path：定位外部提示目录中的提示文件
# 【输入】name: str
# 【实现】对不满足条件的输入抛出异常；调用 os.environ.get, RuntimeError, Path(root).expanduser().resolve,
# Path(root).expanduser, Path, p.is_file, FileNotFoundError
# 【返回】p
def prompt_path(name: str) -> Path:
    """Resolve a named prompt in the separately governed prompt directory."""
    root = os.environ.get("EVIDENCE_PIPELINE_PROMPT_DIR")
    if not root:
        raise RuntimeError(
            "EVIDENCE_PIPELINE_PROMPT_DIR must point to the separately supplied prompt directory"
        )
    p = Path(root).expanduser().resolve() / f"{name}.txt"
    if not p.is_file():
        raise FileNotFoundError(f"external prompt not found: {p}")
    return p


def prompt(name: str) -> str:
    """Load a named prompt from the external runtime prompt directory.

    Prompt text and detailed LLM scoring rubrics are controlled experiment
    materials and are intentionally absent from this source-only release.
    """
    return prompt_path(name).read_text(encoding="utf-8").rstrip("\n")


def judge_criteria(groups=("benchmarkqed", "domain")) -> list[dict]:
    """LLM-judge criteria (BenchmarkQED verbatim + ADHD domain additions).
    Returns [{name, description}, ...] in the requested groups' order."""
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("pyyaml not installed") from e
    rubric_file = os.environ.get("EVIDENCE_PIPELINE_RUBRIC_FILE")
    if not rubric_file:
        raise RuntimeError(
            "EVIDENCE_PIPELINE_RUBRIC_FILE must point to the separately supplied criteria file"
        )
    data = yaml.safe_load(Path(rubric_file).read_text(encoding="utf-8")) or {}
    out = []
    for g in groups:
        for c in data.get(g, []):
            out.append({"name": c["name"], "description": " ".join(c["description"].split())})
    return out
