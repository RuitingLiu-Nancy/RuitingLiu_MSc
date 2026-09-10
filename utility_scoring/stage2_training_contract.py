# 【文件 075】从冻结分区读取 Dev300 训练契约，不强迫已物化 RRF2 特征池另加载 Graph 标签
# 【流程位置】特征构造、效用评分与训练；所属包：utility_scoring
# 【主要函数】load_direct_training_contract
# 【输入接口】source 等函数参数；返回值及写出操作见对应函数
# 【依赖文件】evaluation/judgment_completeness.py
# 【调用方】utility_scoring/run_lightweight_scorer_search_dev300.py
# 【调用方】utility_scoring/run_stage2_redesign_crossencoder.py

"""Load the frozen Development300 split contract without a Graph pool.

The legacy action-space loader validates Dense plus Graph candidates because
its own models train on that union.  The Stage-2 RRF2 models train on an
already materialised feature parquet, so requiring labels for a separate G50
pool would incorrectly block an otherwise fully labelled training universe.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.judgment_completeness import complete_utility_v2_rows


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_direct_training_contract(
    root: Path, source: dict[str, Any], feature_qids: set[str]
) -> dict[str, Any]:
    """Return labels and frozen query-grouped splits for a feature universe."""
    registry_path = _resolve(root, source["utility_registry"])
    split_path = _resolve(root, source["split_manifest"])
    for path in (registry_path, split_path):
        if "test" in str(path).lower():
            raise ValueError(f"training contract path resembles test scope: {path}")
        if not path.exists():
            raise FileNotFoundError(path)

    rows = [json.loads(line) for line in registry_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    complete, registry = complete_utility_v2_rows(rows)
    expected_registry = int(source["expected_complete_registry_rows"])
    if len(complete) != expected_registry or len(registry) != expected_registry:
        raise ValueError(
            f"utility registry identity changed: {len(complete)}/{len(registry)} "
            f"!= {expected_registry}")

    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_read")) or bool(
            (manifest.get("audit") or {}).get("test_read")):
        raise ValueError("frozen split manifest reports a test read")
    splits = list(manifest["rows"])
    expected_splits = int(source["outer_repeats"]) * int(source["outer_folds"])
    if len(splits) != expected_splits:
        raise ValueError(f"split count changed: {len(splits)} != {expected_splits}")

    validation_counts = {qid: 0 for qid in feature_qids}
    for split in splits:
        train = set(map(str, split["train_query_ids"]))
        valid = set(map(str, split["validation_query_ids"]))
        if train & valid or train | valid != feature_qids:
            raise ValueError("a frozen split overlaps or differs from the feature cohort")
        for qid in valid:
            validation_counts[qid] += 1
    if set(validation_counts.values()) != {int(source["outer_repeats"])}:
        raise ValueError("each query must be validated once per frozen repeat")
    return {"qids": sorted(feature_qids), "registry": registry, "splits": splits}
