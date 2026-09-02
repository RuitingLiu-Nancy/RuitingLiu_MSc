# Principal reproduction runners

The directories below follow the order of the dissertation pipeline. Shared
loaders and metrics are imported from their owning modules rather than copied
between stages.

## Data preparation

- `data_preparation/sampling/01_base_sample.py`
- `data_preparation/sampling/02_classify_scenarios.py`
- `data_preparation/sampling/03_resample_stratified.py`
- `data_preparation/sampling/create_mixed_query_splits.py`
- `data_preparation/sampling/freeze_research_data_partitions.py`
- `data_preparation/entity_processing/05_extract_open_entities.py`
- `data_preparation/entity_processing/06_extract_entities_batch.py`
- `data_preparation/entity_processing/07_canonicalize_entities.py`
- `data_preparation/entity_processing/08_ground_relations_global.py`

## Candidate-pool construction

- `candidate_pool/run_official_hipporag_bedrock.py`
- `candidate_pool/run_m50_dense_frontier_analysis.py`
- `candidate_pool/run_m50_graph_frontier_analysis.py`
- `candidate_pool/retrieval/` contains the reusable route implementations.
- `candidate_pool/graph_construction/` contains graph assembly and community detection.

## Fusion and candidate access

- `fusion/ranking.py` implements reciprocal-rank and convex score fusion.
- `fusion/candidate_pool.py` constructs fused candidate pools.
- `fusion/analyze_rq2a_graph_budget_sweep.py`
- `fusion/run_depth_graph_utility_community_frontier.py`

## Utility-aware scoring

- `utility_scoring/build_stage2_redesign_features.py`
- `utility_scoring/build_stage2_redesign_features_rrf2pool.py`
- `utility_scoring/run_rq2b_scorer_family_oof_dev300.py`
- `utility_scoring/run_lightweight_scorer_search_dev300.py`
- `utility_scoring/run_stage2_redesign_crossencoder.py`
- `utility_scoring/fit_lambdamart_transfer.py`

## Evidence-set selection

- `evidence_selection/run_rq2b_symmetric_hyperparameter_selection.py`
- `evidence_selection/run_selection_action_space_repair.py`
- `evidence_selection/run_set_aware_selection_ablation.py`

## Evaluation and reporting

- `evaluation/run_evidence_signal_triangulation.py`
- `evaluation/analyze_rq2b_set_correspondence.py`
- `evaluation/run_stage2_community_dev300_complete.py`
- `evaluation/confirmatory_test200_rq2b.py`
- `figures/` contains the scripts for the reported plots.

The principal runners read parameters from `configuration/params.yaml` or from
the file named by `EVIDENCE_PIPELINE_PARAMS`. Relative paths resolve from the
repository root.
