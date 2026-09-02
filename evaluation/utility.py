"""Utility-label contract shared by annotation and evaluation."""
from __future__ import annotations

import json
import re

import configuration as config

DIMS_V2 = (
    "relevance",
    "usefulness",
    "novelty",
    "actionability",
    "resonance",
    "safety",
)
WEIGHTS_V2 = {
    "relevance": 0.25,
    "usefulness": 0.30,
    "actionability": 0.15,
    "novelty": 0.10,
    "resonance": 0.10,
    "safety": 0.10,
}


def parse_judgment(text: str) -> dict:
    """Parse one JSON judgment, with numeric-field recovery for malformed text."""
    value = text.strip()
    try:
        return json.loads(value)
    except Exception:
        match = re.search(r"\{.*\}", value, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    result = {key: int(score) for key, score in re.findall(
        r'"(\w+)"\s*:\s*(-?\d+)', value
    )}
    rationale = re.search(r'"rationale"\s*:\s*"(.*?)"\s*[,}]', value, re.S)
    if rationale:
        result["rationale"] = rationale.group(1)[:1000]
    if not result:
        raise ValueError("could not parse utility judgment")
    return result


def clamp_score(value) -> int:
    """Round and clamp one rubric score to the 1--7 scale."""
    try:
        score = int(round(float(value)))
    except Exception:
        score = 1
    return max(1, min(7, score))


def annotation_prompt(
    query: str,
    snippet: str,
    facets: dict,
    rubric_version: str = "v2",
) -> str:
    """Render one utility-annotation request from runtime templates."""
    rubric = config.prompt(f"evidence_card_judge_{rubric_version}")
    wrapper = config.prompt("evidence_card_annotation_wrapper")
    return wrapper.format(
        rubric=rubric,
        query=query,
        snippet=snippet,
        facets=json.dumps(facets, ensure_ascii=False),
    )


def utility_v2(ratings: dict) -> float:
    """Compute weighted utility with the safety cap used in the dissertation."""
    value = sum(float(ratings[name]) * weight for name, weight in WEIGHTS_V2.items())
    return min(value, 2.0) if float(ratings["safety"]) <= 2 else value
