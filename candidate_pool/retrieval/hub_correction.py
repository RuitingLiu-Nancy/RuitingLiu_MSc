"""Read-only hub penalties for Official HippoRAG2 transition profiles.

The official graph is undirected, so a target-only degree penalty cannot be
represented by one igraph edge-weight vector.  We therefore use the symmetric
geometric degree penalty below.  It changes transition weights only; topology,
restart mass and persisted graph state remain untouched.
"""
from __future__ import annotations

import numpy as np


def symmetric_degree_penalty(
    degrees: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Return ``((deg(u)+1)(deg(v)+1))^(-gamma/2)`` per edge."""
    if gamma < 0:
        raise ValueError("hub gamma must be non-negative")
    degrees = np.asarray(degrees, dtype=np.float64)
    src = np.asarray(edge_sources, dtype=np.int64)
    dst = np.asarray(edge_targets, dtype=np.int64)
    if src.shape != dst.shape:
        raise ValueError("edge endpoint arrays must be aligned")
    if gamma == 0:
        return np.ones(src.shape, dtype=np.float64)
    product = (degrees[src] + 1.0) * (degrees[dst] + 1.0)
    return np.power(product, -0.5 * gamma)


def high_degree_mask(degrees: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Flag the high-degree tail used only for diagnostics."""
    if not 0 < quantile < 1:
        raise ValueError("degree quantile must be in (0, 1)")
    values = np.asarray(degrees, dtype=np.float64)
    threshold = float(np.quantile(values, quantile)) if len(values) else np.inf
    return values >= threshold
