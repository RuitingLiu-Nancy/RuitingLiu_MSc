"""Safety filter: exclude medication-related queries from the study.

Per the research/ethics decision, ADHD MEDICATION questions (dose, side effects,
stopping/switching meds, drug interactions, etc.) must NOT be shown to
participants — a RAG system answering them carries real clinical risk. This
filter runs at BUILD time, so flagged queries never enter the item pool and
participants never see them.

Detection = a maintained keyword/phrase list with WORD-BOUNDARY matching (so
'immediately'/'medley' are NOT false-flagged). Keep it conservative: when in
doubt, exclude — a missed support query is cheap; a shown med query is not.

Tune the list in study_config.yaml -> safety.drug_patterns (regex, case-insensitive)
or rely on the built-in DEFAULT_DRUG_PATTERNS below.
"""
from __future__ import annotations

import re

# Word-boundary regex patterns. Case-insensitive. Conservative but precise.
DEFAULT_DRUG_PATTERNS = [
    # brand / generic ADHD drugs
    r"\badderall\b", r"\bvyvanse\b", r"\belvanse\b", r"\britalin\b",
    r"\bconcerta\b", r"\bstrattera\b", r"\bwellbutrin\b", r"\bintuniv\b",
    r"\bdexedrine\b", r"\bfocalin\b", r"\bmydayis\b", r"\bquillivant\b",
    r"\bmethylphenidate\b", r"\bamphetamine\b", r"\bdexamphetamine\b",
    r"\bdextroamphetamine\b", r"\blisdexamfetamine\b", r"\batomoxetine\b",
    r"\bguanfacine\b", r"\bbupropion\b", r"\bclonidine\b", r"\bmodafinil\b",
    r"\bmodofinil\b",
    # drug classes
    r"\bstimulants?\b", r"\bnon[-\s]?stimulants?\b", r"\bssris?\b",
    # dosing / pharmacology language
    r"\bmeds?\b", r"\bmedication", r"\bmedicine", r"\bmedicating\b",
    r"\bdosages?\b", r"\bdose[ds]?\b", r"\bdosing\b", r"\bmicro[-\s]?dos",
    r"\b\d+\s?mg\b", r"\bmilligrams?\b", r"\btitrat", r"\btaper(?:ing|ed)?\b",
    r"\bside[-\s.]?effects?\b", r"\bprescri", r"\bpsychiatrist",
    r"\bextended[-\s]?release\b", r"\binstant[-\s]?release\b",
    r"\bcomedown\b", r"\bcrash(?:ing|es)?\b.*\b(med|dose|adderall|vyvanse)\b",
    # explicit med actions / risks
    r"\b(start|stop|switch|come\s?off|quit|skip)\s+(my\s+)?(meds?|medication|adderall|vyvanse|ritalin|stimulants?)\b",
    r"\btake\s+(my\s+)?(meds?|medication|adderall|vyvanse|ritalin)\b",
]


class DrugFilter:
    def __init__(self, patterns: list[str] | None = None):
        pats = patterns or DEFAULT_DRUG_PATTERNS
        self._rx = re.compile("|".join(pats), re.IGNORECASE)

    def is_drug_related(self, text: str) -> bool:
        return bool(self._rx.search(text or ""))

    def match(self, text: str) -> str | None:
        m = self._rx.search(text or "")
        return m.group(0) if m else None

    def filter_queries(self, queries: list[dict], text_key: str = "text"
                       ) -> tuple[list[dict], list[dict]]:
        """Return (kept, excluded). Each excluded query gets a 'drug_match' field
        recording WHY it was excluded, so the exclusion is auditable for ethics."""
        kept, excluded = [], []
        for q in queries:
            m = self.match(q.get(text_key, ""))
            if m:
                q = dict(q, drug_match=m)
                excluded.append(q)
            else:
                kept.append(q)
        return kept, excluded


def from_config(cfg: dict | None = None) -> tuple[DrugFilter | None, bool]:
    """Build a filter from study_config.yaml. Returns (filter, enabled)."""
    safety = (cfg or {}).get("safety", {}) if cfg else {}
    enabled = safety.get("exclude_drug_queries", True)   # default ON (safe by default)
    patterns = safety.get("drug_patterns") or None
    return (DrugFilter(patterns) if enabled else None), enabled
