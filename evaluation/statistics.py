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
