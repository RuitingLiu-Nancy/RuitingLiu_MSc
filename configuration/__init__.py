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
