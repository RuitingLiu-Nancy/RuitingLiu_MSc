# Utility-Aware Cross-Thread Evidence Retrieval and Selection for RAG in Online ADHD Communities

This repository contains the final thesis pipeline code release for the MSc project
**Utility-Aware Cross-Thread Evidence Retrieval and Selection for RAG in
Online ADHD Communities**. It studies how candidate depth, complementary
retrieval routes, fusion, utility supervision, and final-set selection affect
the evidence supplied to a retrieval-augmented support system.

The experimental pipeline has two stages. Stage 1 constructs a candidate pool
from dense, lexical, and graph-assisted access routes. Stage 2 learns utility
scores and selects eight comments for the final evidence set. The code is
organised in that methodological order, so each top-level directory corresponds
to a distinct part of the dissertation rather than to one retrieval family.

## Branches

- **main**: final thesis pipeline, its shared dependencies, reported model
  comparisons, selection strategies and evaluation controls.
- **codex/development-archive**: the public development tree before cleanup,
  preserved at commit `841c63963d1ede24afbff76187d49e9fc22fed14`.
  Earlier ontology, EF/empathy schema, custom multi-hop, PCST, IRCoT,
  spreading-activation and learned-diffusion experiments remain there.

The archive is historical code, not an alternative reproduction recipe for
the final results. Main retains all reported scorer families, not only the
selected checkpoints. Some shared helpers keep historical function names to
preserve the experiment interfaces.

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
├── data_preparation/             sampling, partitions and raw-text adapter export
├── candidate_pool/               pinned HippoRAG2 wrapper and access/depth analyses
├── fusion/                       primary graph merge, RRF, CC and RQ2a comparisons
├── utility_scoring/              shared validation, features and reported scorer families
├── evidence_selection/           Direct, replacement and residual-prior strategies
├── evaluation/                   IR, utility, community and held-out evaluation
├── figures/                      reported dissertation figures
├── configuration/                experiment parameters and pinned dependency metadata
├── shared/                       file and hosted-model adapters
├── models/primary/               CatBoost and 256-token cross-encoder
├── models/supplementary/         512-token cross-encoder comparison
└── scripts/                      release verification and offline graph-route regression
```

## Which code actually builds the graph?

`candidate_pool/run_official_hipporag_bedrock.py` calls the pinned upstream
`HippoRAG.index(docs=texts)`. Upstream HippoRAG2 performs OpenIE extraction,
entity/fact indexing and graph construction. The final routes reuse this
common graph and cached extraction; they do not construct the earlier
handwritten EF/empathy ontology.

| Final graph route | Implementation |
|---|---|
| Passage-restart Graph | `no_recognition`: fact seeds plus direct passage restart |
| Fact-only Graph | `fact_only_no_recognition`: same fact seeds, passage restart weight zero |
| Primary Graph | Alternate the two native rankings, deduplicate, and apply the frozen original MiniLM DenseTop8 exclusion |

The Primary Graph merge is `_round_robin_graph_head` in
`candidate_pool/analyze_strict_sbert_graph_oracle.py`, reused by
`fusion/run_depth_graph_utility_community_frontier.py`. All three routes
bypass recognition. The native graph rankings have depth 100; the reported
candidate-budget comparisons use prefixes of those rankings.

The module name `openie_openai` refers to an OpenAI-compatible interface;
it does not establish which provider or model produced a saved extraction.
Use the extraction cache provenance and runtime manifest for that identity.
The wrapper's historical filename is retained, but `--llm-model`,
`--embedding-model` and `--retrieval-profile` must now be explicit.

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

1. Establish eligible posts, stratified cohorts and frozen partitions with
   `data_preparation/sampling/`.
2. Export the frozen comment scope and query adapter with
   `data_preparation/export_hipporag_dataset.py`. Supply `--text-csv` for the
   canonical raw comment text. Its graph-node input identifies corpus
   membership; it does not supply the ontology graph used by retrieval.
3. Build or reuse the pinned HippoRAG2 index with
   `candidate_pool/run_official_hipporag_bedrock.py`, then obtain the two
   recognition-free rankings and construct Primary Graph.
4. Assemble dense, lexical and graph candidate rankings and compare fusion
   with `fusion/` and the candidate-depth analysis runners.
5. Build Stage 2 features and fit the reported model families with
   `utility_scoring/`.
6. Apply final-set strategies in `evidence_selection/`.
7. Reproduce metrics and plots with `evaluation/` and `figures/`.

Exact replay needs the matching frozen corpus-membership files, raw text,
query splits, extraction caches, judgments and external templates. These are
not redistributed here. The repository is a code/checkpoint release, not a
self-contained data package; the archive branch preserves the earlier
construction code for tracing historical scope artifacts.

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
python scripts/check_final_graph_routes.py
python scripts/verify_release.py
```
