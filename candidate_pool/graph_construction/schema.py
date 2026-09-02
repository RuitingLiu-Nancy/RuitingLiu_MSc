#!/usr/bin/env python3
"""Annotation and graph schema constants used by the construction pipeline.

Every fixed annotation axis below has an anchoring citation in the dissertation.
The only label without a single source is the open-vocabulary fallback (*_OTHER),
which by design means "none of the fixed classes fit" and is governed by a
mandatory rationale + a periodic consolidation/up-grading flow (doc sec 2.5).
"""
from __future__ import annotations

# Relation signatures used to validate open-entity edges during graph assembly.
# An empty set means that any entity type is accepted on that endpoint.
RELATION_TYPES = {
    "USED_FOR": (
        {"TOOL", "STRATEGY", "MED_TREATMENT"},
        {"LIFE_TASK", "EXEC_DIFFICULTY"},
    ),
    "ADDRESSES_BARRIER": (
        {"STRATEGY", "TOOL", "MED_TREATMENT"},
        {"EXEC_DIFFICULTY"},
    ),
    "HELPS_WITH": (
        {"STRATEGY", "MED_TREATMENT", "TOOL"},
        {"AFFECT_STATE", "EXEC_DIFFICULTY"},
    ),
    "OCCURS_IN_CONTEXT": (set(), {"BODY_CONTEXT"}),
    "ASSOCIATED_WITH": (
        {
            "LIFE_TASK",
            "EXEC_DIFFICULTY",
            "MED_TREATMENT",
            "AFFECT_STATE",
            "BODY_CONTEXT",
            "COMMUNITY_TERM",
            "STRATEGY",
        },
        {"AFFECT_STATE", "LIFE_TASK", "EXEC_DIFFICULTY"},
    ),
    "CO_OCCURS_WITH": (set(), set()),
}

# ---------------------------------------------------------------------------
# COMMENT side
# ---------------------------------------------------------------------------

# L1 — Support Function (6, all comments, multi-label).
# Anchor: Cutrona & Russell (1990) SSBC; Isser & Gazit (2025).  [doc sec 2.1]
SUPPORT_FUNCTIONS = [
    "informational_support",
    "emotional_support",
    "esteem_support",
    "experiential_sharing",
    "resource_referral",
    "safety_response",
]

# L1b — EPITOME empathy mechanism, only on emotional_support / esteem_support.
# Each scored 0/1/2 intensity.  Anchor: Sharma et al. (EMNLP 2020).  [doc sec 2.2]
EPITOME_MECHANISMS = [
    "emotional_reactions",
    "interpretations",
    "explorations",
]
EPITOME_FUNCTIONS = {"emotional_support", "esteem_support"}  # which L1 trigger EPITOME

# L2a — Strategy Domain, only inside informational_support (multi-label, optional).
# Anchor: Canela et al. (2017, PLOS One) empirically-derived 5 categories.  [doc sec 2.3]
STRATEGY_DOMAINS = [
    "domain_organizational",
    "domain_attentional",
    "domain_motoric",
    "domain_social",
    "domain_pharmacological",
]
STRATEGY_DOMAIN_OTHER = "domain_other"   # open-vocabulary fallback (doc sec 2.5)

# L2b — EF Mechanism, only inside informational_support (multi-label, optional).
# Anchor: Barkley BDEFS 5 dimensions.  [doc sec 2.3]
EF_MECHANISMS = [
    "ef_time_management",
    "ef_organization_problem_solving",
    "ef_self_restraint",
    "ef_self_motivation",
    "ef_emotion_regulation",
]
EF_MECHANISM_OTHER = "ef_other"          # open-vocabulary fallback (doc sec 2.5)

# Both strategy axes only annotated when L1 contains this function.  [doc sec 2.3]
STRATEGY_TRIGGER_FUNCTION = "informational_support"

# X — Horizontal descriptors (NOT a classification layer).  [doc sec 2.6]
SUPPORT_STYLES = [
    "practical_steps",
    "validating",
    "personal_story",
    "playful_reframe",
    "caution",
]

# ---------------------------------------------------------------------------
# POST side
# ---------------------------------------------------------------------------

# L1 Situation coarse layer borrowed from PATCHES (K-CAP 2025) skeleton.  [doc sec 3.1]
# Maps existing fine scenarios up to a 6-class coarse Situation tier.
SITUATION_OF_SCENARIO = {
    "academic_study": "AcademicSituation",
    "housework_chores": "OrganizationSituation",
    "finance_admin": "OrganizationSituation",
    "leaving_house_transitions": "OrganizationSituation",
    "workplace_job": "TimeManagementSituation",
    "digital_distraction": "TimeManagementSituation",
    "sleep_routine": "TimeManagementSituation",
    "relationships_social": "SocialSituation",
    "parenting_caregiving": "SocialSituation",      # also cross-hangs to FamilyDynamicSituation
    "emotional_regulation": "EmotionalRegulationSituation",
    "diagnosis_identity": "EmotionalRegulationSituation",
    "hygiene_selfcare": "EmotionalRegulationSituation",
    "medication_healthcare": "EmotionalRegulationSituation",
    "general_unspecified": "general_unspecified",
    "emergent_other": "emergent_other",
}
SITUATIONS = sorted({v for v in SITUATION_OF_SCENARIO.values()})

# Constraint -> MeSH anchor (doc sec 3.2). Reddit-specific barriers w/ no MeSH kept.
CONSTRAINT_MESH = {
    "executive_dysfunction": "D056344",
    "anxiety": "D001007",
    "shame": "D012752",
    "sleep_deprivation": "D012892",
    "social_anxiety": "D001007",
    "lack_of_access_to_care": "D006297",
}
# Reddit-specific barriers, deliberately retained without MeSH (community contribution).
CONSTRAINT_REDDIT_ONLY = [
    "lack_of_money", "unclear_task", "too_many_steps",
    "digital_distraction", "housework_chores",
]

# Fine scenarios (14, doc 3.1) — the keys of SITUATION_OF_SCENARIO minus the
# two fallbacks, which are handled separately. general_unspecified is a real
# pickable label; emergent_other is the open-vocab fallback.
SCENARIOS = [s for s in SITUATION_OF_SCENARIO if s != "emergent_other"]
SCENARIO_OTHER = "emergent_other"

# Full constraint vocabulary (doc 3.2): MeSH-anchored ∪ Reddit-specific ∪ the
# broader everyday-barrier set the project already annotated. Single source of
# truth for post-side L4 validation.
CONSTRAINTS = sorted(set(CONSTRAINT_MESH) | set(CONSTRAINT_REDDIT_ONLY) | {
    "low_energy", "time_pressure", "sensory_sensitivity", "physical_pain",
})
CONSTRAINT_OTHER = "emergent_other"

# Need -> expected Support Function (doc sec 3.3, demand-supply axis).
NEED_TO_FUNCTION = {
    "action_need": ["informational_support"],
    "affective_need": ["emotional_support", "esteem_support"],
    "experience_request": ["experiential_sharing"],
    "resource_need": ["resource_referral"],
    "safety_need": ["safety_response"],
    "diagnosis_information_need": ["informational_support"],
    "clarification_need": ["informational_support"],
}

# ---------------------------------------------------------------------------
# OLD (19 response_family) -> NEW mapping  (doc sec 6)
# Each old label maps to (L1 function, L2a domain|None, L2b ef_mechanism|None,
# epitome dimension|None).  "informational only" axes are None for affective.
# ---------------------------------------------------------------------------
OLD_TO_NEW = {
    # action_AS* -> informational_support + dual-axis subtype
    "action_AS01_planning_scheduling":
        ("informational_support", "domain_organizational", "ef_time_management", None),
    "action_AS02_task_initiation":
        ("informational_support", "domain_organizational", "ef_self_restraint", None),
    "action_AS03_external_structure":
        ("informational_support", "domain_organizational", "ef_organization_problem_solving", None),
    "action_AS04_prioritisation_decision_support":
        ("informational_support", "domain_organizational", "ef_organization_problem_solving", None),
    "action_AS05_rewards_accountability":
        ("informational_support", "domain_social", "ef_self_motivation", None),
    "action_AS06_reframing_emotional_regulation":
        ("informational_support", "domain_organizational", "ef_emotion_regulation", None),
    "action_AS07_social_scaffolding":
        ("informational_support", "domain_social", "ef_self_motivation", None),
    "action_AS08_self_monitoring":
        ("informational_support", "domain_organizational", "ef_organization_problem_solving", None),
    "action_AS09_attention_distraction":
        ("informational_support", "domain_attentional", "ef_self_restraint", None),
    # AS10 -> informational, strategy axes left for content judgement; use-case=domain scenario
    "action_AS10_academic_workplace":
        ("informational_support", None, None, None),
    # CS01 -> experiential_sharing (+informational if it gave a method)
    "action_CS01_lived_experience_knowledge":
        ("experiential_sharing", None, None, None),
    # affective_* -> emotional/esteem + EPITOME dimension
    "affective_shared_lived_experience":
        ("experiential_sharing", None, None, None),
    "affective_validates_struggle":
        ("emotional_support", None, None, "emotional_reactions"),
    "affective_validates_frustration":
        ("emotional_support", None, None, "interpretations"),
    "affective_gentle_encouragement":
        ("emotional_support", None, None, "emotional_reactions"),
    "affective_celebrates_progress":
        ("esteem_support", None, None, None),
    "affective_other":
        ("emotional_support", None, None, None),  # re-judge by content; default emotional
    # operational categories
    "resource": ("resource_referral", None, None, None),
    "safety": ("safety_response", None, None, None),
}

# ---------------------------------------------------------------------------
# Node / edge type vocabulary for the three-tier graph (doc sec 5)
# ---------------------------------------------------------------------------
NODE_TYPES = {
    # Tier 1 (coarse / global)
    "support_function", "situation", "need",
    # Tier 2 (mid / bridge)
    "strategy_domain", "ef_mechanism", "epitome_mechanism",
    "scenario", "constraint",
    # Tier 3 (fine / source)
    "post", "comment", "evidence_chunk", "canonical_entity",
    # abstraction
    "problem_profile",
}

EDGE_TYPES = {
    # HiRAG SUMMARY_OF 統辖 edges (doc sec 5.2) -- the core of the redesign
    "SUMMARY_OF",
    # structural
    "answered_by", "has_situation", "has_scenario", "has_need", "has_constraint",
    "has_support_function", "has_empathy", "mentions_entity",
    # recommendation / retrieval abstraction
    "recommends_function", "recommends_subtype", "addressed_by",
    # entity relations + weak co-occurrence
    "USED_FOR", "HELPS_WITH", "ADDRESSES_BARRIER", "CO_OCCURS_WITH",
}
