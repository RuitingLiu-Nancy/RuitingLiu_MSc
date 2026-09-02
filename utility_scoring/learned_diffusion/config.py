"""Configuration for the validation-only LUAD seed pilot.

All defaults live in ``configuration/params.yaml``.  The dataclass only validates and
resolves those values; it is not a second parameter source.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import configuration as project_config


@dataclass(frozen=True)
class LUADConfig:
    project_root: Path
    version: str
    graph_pickle: Path
    embedding_dir: Path
    corpus: Path
    query_split: Path
    query_registry: Path
    query_summaries: Path
    run_registry: Path
    judgments: Path
    original_entry_trace: Path
    original_checkpoints: Path
    output_dir: Path
    seed_top_m: int
    hops: int
    neighbors_per_node: int
    max_path_length: int
    damping: float
    full_graph_max_steps: int
    local_steps: int
    tolerance: float
    reproduction_min_mean_top100_jaccard: float
    reproduction_min_spearman: float
    folds: int
    repeats: int
    seeds: tuple[int, ...]
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    pair_margin: float
    max_pairs_per_query: int
    lambda_regression: float

    def validate(self) -> None:
        for path in (
            self.graph_pickle, self.embedding_dir, self.corpus, self.query_split,
            self.query_registry, self.query_summaries,
            self.run_registry, self.judgments, self.original_entry_trace,
            self.original_checkpoints,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        if "test" in self.query_split.name.lower() or "test" in str(self.query_split.parent).lower():
            raise ValueError("LUAD refuses a path that appears to be frozen test")
        if not (0.0 < self.damping < 1.0):
            raise ValueError("damping must be in (0,1)")
        if self.folds < 2 or self.repeats < 1 or len(self.seeds) != self.repeats:
            raise ValueError("invalid grouped-CV configuration")
        if min(self.seed_top_m, self.hops, self.neighbors_per_node,
               self.full_graph_max_steps, self.local_steps, self.epochs) < 1:
            raise ValueError("LUAD integer controls must be positive")


def load_config(project_root: Path | None = None) -> LUADConfig:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    raw = project_config.load()["learned_diffusion"]
    local = raw["local_graph"]
    prop = raw["propagation"]
    train = raw["training"]

    def path(value: str) -> Path:
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    cfg = LUADConfig(
        project_root=root,
        version=str(raw["version"]),
        graph_pickle=path(raw["graph_pickle"]),
        embedding_dir=path(raw["embedding_dir"]),
        corpus=path(raw["corpus"]),
        query_split=path(raw["query_split"]), query_registry=path(raw["query_registry"]),
        query_summaries=path(raw["query_summaries"]),
        run_registry=path(raw["run_registry"]),
        judgments=path(raw["judgments"]),
        original_entry_trace=path(raw["original_entry_trace"]),
        original_checkpoints=path(raw["original_checkpoints"]),
        output_dir=path(raw["output_dir"]),
        seed_top_m=int(local["seed_top_m"]), hops=int(local["hops"]),
        neighbors_per_node=int(local["neighbors_per_node"]),
        max_path_length=int(local["max_path_length"]), damping=float(prop["damping"]),
        full_graph_max_steps=int(prop["full_graph_max_steps"]),
        local_steps=int(prop["local_steps"]), tolerance=float(prop["tolerance"]),
        reproduction_min_mean_top100_jaccard=float(
            prop["reproduction_min_mean_top100_jaccard"]),
        reproduction_min_spearman=float(prop["reproduction_min_spearman"]),
        folds=int(train["folds"]), repeats=int(train["repeats"]),
        seeds=tuple(int(x) for x in train["seeds"]), epochs=int(train["epochs"]),
        patience=int(train["patience"]), learning_rate=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]), hidden_dim=int(train["hidden_dim"]),
        pair_margin=float(train["pair_margin"]),
        max_pairs_per_query=int(train["max_pairs_per_query"]),
        lambda_regression=float(train["lambda_regression"]),
    )
    cfg.validate()
    return cfg
