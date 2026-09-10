# 【文件 058】声明 utility_scoring/learned_diffusion 包
# 【流程位置】特征构造、效用评分与训练；所属包：utility_scoring/learned_diffusion
# 【主要函数】包声明及共享定义
# 【依赖文件】utility_scoring/learned_diffusion/config.py
# 【调用方】candidate_pool/run_dense_semantic_drift_rescue_audit.py
# 【调用方】candidate_pool/run_m50_dense_frontier_analysis.py
# 【调用方】utility_scoring/run_lightweight_scorer_search_dev300.py
# 【调用方】utility_scoring/run_rq2b_scorer_family_oof_dev300.py
# 【调用方】evidence_selection/run_selection_action_space_repair.py
# 【调用方】evidence_selection/run_strict_native_graph_conservative_policy.py
# 【调用方】evidence_selection/run_strict_sbert_mixed_selector.py

"""LUAD: development-only utility-supervised graph diffusion experiments."""

from .config import LUADConfig, load_config

__all__ = ["LUADConfig", "load_config"]
