# 【文件 100】提供不依赖模型服务的统计工具，支持配对和不确定性分析
# 【流程位置】检索、效用与社区对应评价；所属包：evaluation
# 【主要函数】bootstrap_ci
# 【调用方】candidate_pool/run_dense_semantic_drift_rescue_audit.py
# 【调用方】candidate_pool/run_m50_dense_frontier_analysis.py
# 【调用方】utility_scoring/annotation/run_top3_residual_judging.py
# 【调用方】utility_scoring/learned_diffusion/reranker_validation.py
# 【调用方】evidence_selection/run_strict_native_graph_conservative_policy.py
# 【调用方】evidence_selection/run_strict_sbert_mixed_selector.py
# 【调用方】evaluation/community_reply_auxiliary.py
# 【调用方】evaluation/external_fusion_utility_rerank.py
# 【调用方】evaluation/fusion_strategy_ablation.py
# 【调用方】evaluation/quality_diversity_rerank.py
# 【调用方】evaluation/run_m50_community_frontier_analysis.py

"""Provider-independent statistical helpers used by evaluation pipelines."""
from __future__ import annotations

import random


def bootstrap_ci(
    values: list[float], n_boot: int = 1000, seed: int = 17
) -> tuple[float | None, float | None]:
    """Return the 2.5th and 97.5th percentiles of bootstrap means."""
    clean = [value for value in values if value is not None]
    if not clean:
        return None, None
    rng = random.Random(seed)
    size = len(clean)
    means = [
        sum(clean[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(n_boot)
    ]
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
