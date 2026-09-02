# Utility-Aware Cross-Thread Evidence Retrieval and Selection for RAG in Online ADHD Communities

This repository contains the complete code release for the MSc project
**Utility-Aware Cross-Thread Evidence Retrieval and Selection for RAG in
Online ADHD Communities**. It studies how candidate depth, complementary
retrieval routes, fusion, utility supervision, and final-set selection affect
the evidence supplied to a retrieval-augmented support system.

The experimental pipeline has two stages. Stage 1 constructs a candidate pool
from dense, lexical, and graph-assisted access routes. Stage 2 learns utility
scores and selects eight comments for the final evidence set. The code is
organised in that methodological order, so each top-level directory corresponds
to a distinct part of the dissertation rather than to one retrieval family.

## Data source

The raw Reddit data were obtained from the historical archive
[*Reddit comments/submissions 2005-06 to 2025-12*](https://academictorrents.com/details/3d426c47c767d40f82c7ef0f47c3acacedd2bf44),
distributed through Academic Torrents by `stuck_in_the_matrix`, `Watchful1`,
and `RaiderBDev` (info hash:
`3d426c47c767d40f82c7ef0f47c3acacedd2bf44`). The archive contains Reddit
submissions and comments collected within the historical Pushshift archive
lineage. From this snapshot, the study extracts submissions and direct
top-level comments from **r/ADHD** and restricts the experimental corpus to
records posted between **January 2023 and December 2025**.

The repository includes the preprocessing code and expected schemas but does
not redistribute Reddit post or comment text. After downloading the archive,
set the local input and output paths in a copy of
`configuration/params.yaml`.

## Code architecture

```text
RuitingLiu_MSc/
├── data_preparation/
│   ├── sampling/                 eligibility, scenario stratification and splits
│   └── entity_processing/        OpenIE extraction, canonicalisation and grounding
├── candidate_pool/
│   ├── retrieval/                dense, BM25 and graph-assisted retrieval components
│   ├── graph_construction/       ontology graph, densification and communities
│   └── *.py                      candidate-access and depth-analysis runners
├── fusion/                       candidate-pool assembly, RRF, CC and RQ2a analyses
├── utility_scoring/
│   ├── learned_diffusion/        reusable model-training and validation components
│   ├── annotation/               runtime adapters for utility annotation
│   └── *.py                      features, lightweight models and cross-encoder training
├── evidence_selection/           Direct, replacement and residual-prior strategies
├── evaluation/                   IR, utility, community and held-out evaluation
├── figures/                      scripts for the reported dissertation figures
├── configuration/                parameters, ontology and dependency specifications
├── shared/                       common file and hosted-model adapters
├── models/
│   ├── primary/                  selected CatBoost and 256-token cross-encoder
│   └── supplementary/            512-token cross-encoder comparison
├── scripts/verify_release.py     release and internal-import verification
├── external_assets.example.env  runtime configuration template
└── pyproject.toml                package metadata
```

Graph construction appears inside `candidate_pool/` alongside dense and
lexical retrieval, following its role in candidate access.

## Correspondence with the dissertation

| Dissertation component | Code |
|---|---|
| Chapter 3: corpus construction, eligibility and partitions | `data_preparation/` |
| Chapter 4: candidate access and graph construction | `candidate_pool/` |
| RQ1: semantic similarity, utility and community correspondence | `evaluation/run_evidence_signal_triangulation.py` |
| RQ2a: depth, graph variants and fusion | `fusion/analyze_rq2a_graph_budget_sweep.py`, `fusion/run_depth_graph_utility_community_frontier.py` |
| RQ2b: utility-aware scorer training | `utility_scoring/` |
| RQ2b: evidence-set strategies | `evidence_selection/` |
| Held-out confirmation | `evaluation/confirmatory_test200_rq2b.py` |
| Chapter 5 figures | `figures/` |

## Installation

Use Python 3.12. Model weights are tracked with Git LFS.

```bash
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r configuration/requirements-reranker-reproduction-py312.txt
python -m pip install "git+https://github.com/OSU-NLP-Group/HippoRAG.git@ad30fc3e2062202d9e975e32cd28212424a56ccb"
python -m pip install -e .
```

The reranker requirements include the graph-community reproduction
environment. The HippoRAG revision used in the study is recorded in
`configuration/hipporag2_official_reproduction.json`.

Copy `external_assets.example.env` and set `EVIDENCE_PIPELINE_PARAMS` to the
experiment configuration. LLM-backed extraction and utility annotation load
their separately governed templates through `EVIDENCE_PIPELINE_PROMPT_DIR` and
`EVIDENCE_PIPELINE_RUBRIC_FILE`.

## Reproduction order

1. Filter the r/ADHD archive and create the research cohorts with
   `data_preparation/sampling/`.
2. Extract and canonicalise entities with
   `data_preparation/entity_processing/`, then construct the graph with
   `candidate_pool/graph_construction/`.
3. Run dense, lexical, and graph-assisted candidate access with
   `candidate_pool/` and `candidate_pool/retrieval/`.
4. Assemble and compare candidate pools with `fusion/`.
5. Build Stage 2 features and fit the utility-aware model families with
   `utility_scoring/`.
6. Apply the final-set strategies in `evidence_selection/`.
7. Reproduce the reported metrics and plots with `evaluation/` and `figures/`.

`PIPELINE.md` lists the principal runners within each stage. Relative input and
output paths resolve from the repository root.

## RQ2b model coverage

The release covers every scorer family reported in the dissertation:

- pointwise regression: Huber, Ridge, ElasticNet, HistGradientBoosting,
  XGBoost regression, CatBoost regression, and a small MLP;
- query-aware ranking: RankNet, XGBoost pairwise ranking, XGBoost LambdaMART,
  LightGBM LambdaRank, and CatBoost YetiRank;
- cross-validated selection over the lightweight model families;
- zero-shot and utility-trained MiniLM cross-encoders.

Feature-based models are reproduced by
`utility_scoring/run_lightweight_scorer_search_dev300.py` and
`utility_scoring/run_rq2b_scorer_family_oof_dev300.py`. The matched
cross-encoder is reproduced by
`utility_scoring/run_stage2_redesign_crossencoder.py`.

## Released checkpoints

- `models/primary/catboost_yetirank`: fitted CatBoost YetiRank model and scaler.
- `models/primary/utility_crossencoder_256`: primary utility-trained MiniLM
  cross-encoder.
- `models/supplementary/utility_crossencoder_512`: supplementary 512-token
  checkpoint.

Each checkpoint directory contains its runtime configuration. The MiniLM
checkpoints retain the upstream Apache-2.0 notice under
`models/THIRD_PARTY_LICENSES/`.

## Verification

```bash
python scripts/verify_release.py
```
