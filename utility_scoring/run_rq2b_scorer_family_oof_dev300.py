#!/usr/bin/env python3
"""Out-of-fold scores for the Stage-2 scorer family, under the frozen contract.

Stage 2 currently carries three scorers and the residual prior helps exactly
one.  Two explanations fit that and they are different claims: the prior helps
a scorer because it RANKS, or because its training loss leaves the score
UNCALIBRATED.  Pairwise and listwise surrogates put scores only in the form
s_i - s_j, so adding a constant inside a query leaves the loss unchanged and
nothing anchors the score to the label scale; pointwise regression anchors it
directly.  The two explanations coincide on the three frozen arms and separate
on a scorer that ranks with a calibrated score.

This runner fits the arms that separate them, inside the same frozen 5x5
query-grouped folds, on the same eight-feature contract, over the same pools,
and emits out-of-fold predictions in the schema the Stage-2 pipeline already
consumes.  It fits models and nothing else: no candidate set is selected here,
no selection hyperparameter is touched, and the utility label enters only as a
training target inside training folds.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEY = "rq2b_scorer_family_dev300"

sys.path.insert(0, str(ROOT))
from evidence_selection import run_selection_action_space_repair as repair  # noqa: E402

try:
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from utility_scoring.learned_diffusion import reranker_validation as canonical
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from utility_scoring.learned_diffusion import reranker_validation as canonical

import torch  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

CALIBRATED, RANKING = "calibrated", "ranking"
UTILITY_LOW, UTILITY_HIGH = 1.0, 7.0

REQUIRED = ("_load_contract", "_build_pools_and_features", "_candidate_arrays",
            "_query_mean_mae", "_query_mean_ndcg_at8", "_reject_test",
            "STATIC_PREDICTOR_FEATURES", "EPS")
_missing = [name for name in REQUIRED if not hasattr(repair, name)]
if _missing:
    raise ImportError(
        "run_selection_action_space_repair no longer exposes: " + ", ".join(_missing)
        + ".  This runner mirrors its contract and must be re-checked against it."
    )


class _Scaled:
    def __init__(self, scaler: StandardScaler, model: Any):
        self.scaler, self.model = scaler, model

    def _x(self, matrix: np.ndarray) -> np.ndarray:
        return self.scaler.transform(matrix).astype(np.float32)


class GBDTRegression(_Scaled):
    """Squared-error trees: a tree ensemble whose score carries utility scale."""

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(self._x(matrix)), dtype=np.float64)
        return np.clip(raw, UTILITY_LOW, UTILITY_HIGH)


class XGBRankerArm(_Scaled):
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._x(matrix)), dtype=np.float64)


class LGBMRankerArm(_Scaled):
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._x(matrix)), dtype=np.float64)


class MLPRankNet(_Scaled):
    """The project's SmallMLP with the RankNet pairwise loss.

    Architecture, optimiser and seed handling are the pointwise MLP arm's; only
    the loss changes, so a move in the selected beta is attributable to the loss
    alone.  The output stays a bare logit rather than being mapped onto 1-7,
    because a pairwise loss constrains differences and not levels.
    """

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(self._x(matrix), dtype=torch.float32))
        return logits.squeeze(-1).numpy().astype(np.float64)


class RankSVMArm(_Scaled):
    """Ranking SVM: a linear score fitted on within-query feature differences.

    Joachims (KDD 2002) turns ranking into binary classification on pairwise
    feature differences.  The loss sees only ``w . (x_i - x_j)``, so the score
    is translation invariant inside a query and carries no utility scale: this
    is the linear cell of the uncalibrated column.
    """

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model.decision_function(self._x(matrix)), dtype=np.float64
        )


class CatBoostArm(_Scaled):
    """A CatBoost ranker.  The score is returned raw, on whatever scale the loss
    leaves it.

    QueryRMSE in particular centres the residual within each group, so its
    output has no identifiable utility level.  Clipping such a score onto the
    rubric's 1-7 range is not a harmless guard: it is a floor, and any query
    whose whole score vector lies below it collapses to a single value, which
    destroys the ordering the arm exists to provide.  The objective does not
    promise unit-scale centred predictions on unseen queries; selection uses
    only within-query order after query-local normalisation anyway.
    """

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._x(matrix)), dtype=np.float64)


class MLPListNet(MLPRankNet):
    """The project's SmallMLP under the ListNet top-one loss.

    Cao et al. (ICML 2007) compare the softmax of the scores with the softmax of
    the labels inside each query.  A softmax is unchanged by adding a constant
    to every score in the query, so this loss is translation invariant exactly
    as the pairwise ones are, and it reaches that property through a listwise
    route rather than a pairwise one.  Prediction is inherited: a bare logit.
    """


class LogisticUseful(_Scaled):
    """Logistic regression on the rubric's useful threshold, read as a level.

    The rubric already defines a candidate as useful at ``u >= 4``.  A logistic
    fit to that label produces a probability that IS anchored - it is a
    calibrated estimate of a labelled event - and mapping it affinely onto the
    utility range makes the arm's score comparable with the other calibrated
    arms and with the query-mean MAE criterion they are selected on.  The map
    is monotone, so the induced within-query order is the probability's order.
    """

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        probability = self.model.predict_proba(self._x(matrix))[:, 1]
        return UTILITY_LOW + (UTILITY_HIGH - UTILITY_LOW) * np.asarray(
            probability, dtype=np.float64
        )


class CalibratedRanker:
    """A ranker whose within-query normalised score is mapped back onto utility.

    The base ranker is fitted first; its scores are min-max normalised inside
    each training query, which removes the per-query affine freedom a
    translation-invariant loss leaves undetermined; an isotonic regression then
    maps the normalised score onto the utility scale.  The map is monotone, so
    the induced ranking inside a query is identical to the base arm's and only
    the calibration differs.
    """

    def __init__(self, base: Any, isotonic: IsotonicRegression):
        self.base, self.isotonic = base, isotonic

    def predict_grouped(self, matrix: np.ndarray, groups: list[str]) -> np.ndarray:
        raw = np.asarray(self.base.predict(matrix), dtype=np.float64)
        normalised = _normalise_within_group(raw, groups)
        return np.clip(self.isotonic.predict(normalised), UTILITY_LOW, UTILITY_HIGH)


def _normalise_within_group(values: np.ndarray, groups: list[str]) -> np.ndarray:
    out = np.empty_like(values)
    index: dict[str, list[int]] = defaultdict(list)
    for position, group in enumerate(groups):
        index[group].append(position)
    for positions in index.values():
        block = values[positions]
        low, high = float(block.min()), float(block.max())
        span = high - low
        out[positions] = 0.5 if span <= repair.EPS else (block - low) / span
    return out


def _graded_labels(target: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Quantise continuous utility into the integer grades a grouped ranker takes.

    ``scale = 1`` reproduces the project's frozen seven-level grade exactly, the
    one ``canonical.historical_utility_grade`` applies, and is the default so
    every existing arm is untouched.  A larger scale keeps more of the label's
    resolution: at ``scale = 4`` a utility difference of a quarter point still
    separates two candidates instead of collapsing them into one grade.  This is
    the single knob the quantisation ablation turns.
    """
    clipped = np.clip(target, UTILITY_LOW, UTILITY_HIGH)
    if float(scale) == 1.0:
        return (np.rint(clipped) - UTILITY_LOW).astype(np.int32)
    return np.rint((clipped - UTILITY_LOW) * float(scale)).astype(np.int32)


def _group_codes(ordered: list[str]) -> np.ndarray:
    seen: dict[str, int] = {}
    codes = []
    for qid in ordered:
        if qid not in seen:
            seen[qid] = len(seen)
        codes.append(seen[qid])
    if codes != sorted(codes):
        raise ValueError("grouped objectives require contiguous query blocks")
    return np.asarray(codes, dtype=np.int32)


def _group_sizes(ordered: list[str]) -> list[int]:
    sizes: list[int] = []
    for position, qid in enumerate(ordered):
        if position == 0 or qid != ordered[position - 1]:
            sizes.append(0)
        sizes[-1] += 1
    return sizes


def _group_blocks(ordered: list[str]) -> list[tuple[int, int]]:
    blocks, start = [], 0
    for position in range(1, len(ordered) + 1):
        if position == len(ordered) or ordered[position] != ordered[start]:
            blocks.append((start, position))
            start = position
    return blocks


def _ranknet_pairs(blocks, utilities, setting, seed):
    """Within-query preference pairs, built as canonical.fit_mlp builds them.

    Eligible pairs are the ordered within-query pairs whose utilities differ by
    at least the project's pair margin, shuffled under the fold seed and
    truncated to the project's per-query cap.  Pair weights are query balanced
    in the same form the pointwise arm's candidate weights take, so the two
    losses weight queries identically and differ only in what they compare.
    """
    margin = float(setting["pair_margin"])
    cap = int(setting["max_pairs_per_query"])
    rng = random.Random(seed)
    rows = []
    for start, stop in blocks:
        judged = [
            index
            for index in range(start, stop)
            if bool(torch.isfinite(utilities[index]))
        ]
        eligible = [
            (i, j)
            for i in judged
            for j in judged
            if float(utilities[i] - utilities[j]) >= margin
        ]
        rng.shuffle(eligible)
        eligible = eligible[:cap]
        if eligible:
            rows.append(eligible)
    if not rows:
        raise ValueError("no within-query preference pair survived the pair margin")
    left = torch.tensor(
        [i for pairs in rows for i, _ in pairs], dtype=torch.long
    )
    right = torch.tensor(
        [j for pairs in rows for _, j in pairs], dtype=torch.long
    )
    weights = torch.tensor(
        [1.0 / (len(rows) * len(pairs)) for pairs in rows for _ in pairs],
        dtype=torch.float32,
    )
    return left, right, weights


def _unit_mean_weights(weights: np.ndarray) -> np.ndarray:
    """Put query-balanced weights on the scale a learner's penalties assume.

    ``repair._candidate_arrays`` returns weights that sum to one, which is what
    the frozen linear arm wants: a linear fit is invariant to the scale of its
    sample weights except through its penalty term.  A gradient-boosted tree is
    not.  ``min_child_weight`` and ``reg_lambda`` are compared against sums of
    hessians, and under a squared-error objective a row's hessian IS its sample
    weight, so weights that sum to one make the entire training set weigh
    exactly one and ``min_child_weight = 1.0`` rejects every candidate split:
    every tree becomes a single leaf and the arm predicts one constant per
    query.  Rescaling to mean one leaves every relative weight untouched, so
    the query balancing is unchanged, and restores the conventional per-row
    scale that the library defaults and the ranker arms already assume.  The
    same argument applies to any learner whose penalty is not rescaled with the
    weights: a support vector machine's C multiplies a SUM of hinge losses, so
    weights summing to one shrink the data term by the number of rows and leave
    the margin term to dominate.
    """
    values = np.asarray(weights, dtype=np.float64)
    average = float(values.mean())
    if not average > 0.0:
        raise AssertionError("sample weights do not have a positive mean")
    # Divide by the mean rather than multiply by the row count, so this holds
    # for a weight vector that sums to anything: the candidate arrays sum to
    # one, but a pairwise arm that mirrors each pair sums to two.  For a vector
    # summing to one the two are the same operation.
    scaled = values / average
    if not math.isclose(float(scaled.mean()), 1.0, rel_tol=1e-9):
        raise AssertionError("rescaled sample weights do not average one")
    return scaled


def _fit_gbdt_regression(qids, candidate_ids, static, registry, setting, seed, feature_names=None):
    import xgboost

    _, matrix, target, weights = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    params = {k: v for k, v in setting.items() if k != "objective"}
    model = xgboost.XGBRegressor(
        objective=str(setting["objective"]), random_state=int(seed), **params
    )
    model.fit(
        scaler.transform(matrix).astype(np.float32),
        target.astype(np.float32),
        sample_weight=_unit_mean_weights(weights).astype(np.float32),
    )
    return GBDTRegression(scaler, model)


def _fit_xgb_ranker(qids, candidate_ids, static, registry, setting, seed,
                    feature_names=None):
    import xgboost

    pairs, matrix, target, _ = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    scale = float(setting.get("label_scale", 1.0))
    params = {
        k: v for k, v in setting.items()
        if k not in ("objective", "label_scale")
    }
    model = xgboost.XGBRanker(
        objective=str(setting["objective"]), random_state=int(seed), **params
    )
    model.fit(
        scaler.transform(matrix).astype(np.float32),
        _graded_labels(target, scale),
        qid=_group_codes([qid for qid, _ in pairs]),
    )
    return XGBRankerArm(scaler, model)


def _fit_lgbm_ranker(qids, candidate_ids, static, registry, setting, seed, feature_names=None):
    import lightgbm

    pairs, matrix, target, _ = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    scale = float(setting.get("label_scale", 1.0))
    params = {
        k: v for k, v in setting.items()
        if k not in ("objective", "label_scale")
    }
    model = lightgbm.LGBMRanker(
        objective=str(setting["objective"]), random_state=int(seed),
        verbose=-1, **params
    )
    model.fit(
        scaler.transform(matrix).astype(np.float32),
        _graded_labels(target, scale),
        group=_group_sizes([qid for qid, _ in pairs]),
    )
    return LGBMRankerArm(scaler, model)


def _fit_mlp_ranknet(qids, candidate_ids, static, registry, setting, seed,
                     feature_names=None):
    """The frozen pointwise MLP arm with its loss replaced by RankNet.

    Everything the pointwise arm fixes is fixed here, in the same order and
    with the same values: the three seeds, the query-balanced candidate
    arrays, the standardiser, the canonical SmallMLP settings dict, the Adam
    optimiser and the epoch count.  The loss becomes the RankNet cross entropy
    of Burges (2010) eq. (1) over within-query preference pairs, which is the
    project's own canonical ranking term with its pointwise regression anchor
    removed.  A move in the selected beta is therefore attributable to the
    loss alone.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pairs, matrix, target, _ = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    values = torch.tensor(
        scaler.transform(matrix).astype(np.float32), dtype=torch.float32
    )
    utilities = torch.tensor(target, dtype=torch.float32)
    canonical_setting = {
        "hidden_dim": int(setting["hidden_dim"]),
        "layers": int(setting["layers"]),
        "dropout": float(setting["dropout"]),
    }
    model = canonical.SmallMLP(values.shape[1], canonical_setting)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(setting["learning_rate"]),
        weight_decay=float(setting["weight_decay"]),
    )
    left, right, pair_weights = _ranknet_pairs(
        _group_blocks([qid for qid, _ in pairs]), utilities, setting, seed
    )
    for _ in range(int(setting["epochs"])):
        model.train()
        optimizer.zero_grad()
        scores = model(values)
        margin = scores[left] - scores[right]
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            margin, torch.ones_like(margin), reduction="none"
        )
        loss = (losses * pair_weights).sum()
        loss.backward()
        optimizer.step()
    return MLPRankNet(scaler, model)


def _fit_calibrated_ranker(qids, candidate_ids, static, registry, setting, seed, feature_names=None):
    base = _fit_xgb_ranker(
        qids, candidate_ids, static, registry, dict(setting["base"]), seed,
        feature_names
    )
    pairs, matrix, target, _ = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    raw = np.asarray(base.predict(matrix), dtype=np.float64)
    normalised = _normalise_within_group(raw, [qid for qid, _ in pairs])
    isotonic = IsotonicRegression(
        y_min=UTILITY_LOW, y_max=UTILITY_HIGH, increasing=True, out_of_bounds="clip"
    ).fit(normalised, target)
    return CalibratedRanker(base, isotonic)


def _fit_ranksvm(qids, candidate_ids, static, registry, setting, seed, feature_names=None):
    from sklearn.svm import LinearSVC

    pairs, matrix, target, _ = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    values = scaler.transform(matrix)
    left, right, pair_weights = _ranknet_pairs(
        _group_blocks([qid for qid, _ in pairs]),
        torch.tensor(target, dtype=torch.float32), setting, seed,
    )
    difference = values[left.numpy()] - values[right.numpy()]
    # Both orientations of every pair, so the decision boundary is forced
    # through the origin by the data as well as by fit_intercept=False.
    features = np.vstack([difference, -difference])
    labels = np.concatenate([
        np.ones(len(difference)), -np.ones(len(difference))
    ])
    weights = _unit_mean_weights(
        np.concatenate([pair_weights.numpy(), pair_weights.numpy()])
    )
    model = LinearSVC(
        C=float(setting["C"]), loss="hinge", fit_intercept=False,
        max_iter=int(setting["max_iter"]), tol=float(setting["tol"]),
        random_state=int(seed),
    )
    model.fit(features, labels, sample_weight=weights)
    return RankSVMArm(scaler, model)


def _fit_catboost_ranker(qids, candidate_ids, static, registry, setting, seed, feature_names=None):
    """CatBoostRanker under whichever grouped loss the grid names.

    YetiRank is pairwise, so its score is free up to a within-query shift.
    QueryRMSE centres the residual within each query, so its score has no
    identifiable level.  Unlike an ordinary pointwise RMSE it therefore does
    not support a query-mean MAE tuning criterion, and unlike a calibrated
    regressor it makes no unit-scale promise on unseen queries.  Both CatBoost
    arms go through one fitter because only the registered grouped loss changes.
    """
    import catboost

    pairs, matrix, target, _ = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    params = {
        key: value for key, value in setting.items() if key != "loss_function"
    }
    model = catboost.CatBoostRanker(
        loss_function=str(setting["loss_function"]),
        random_seed=int(seed), verbose=False, **params
    )
    # CatBoost requires the rows of a group to be contiguous and the groups to
    # be ordered.  The candidate arrays are already grouped by query, but the
    # query order comes from the caller, so sort rather than assume.  Row order
    # is not otherwise a modelling choice.
    groups = [qid for qid, _ in pairs]
    order = sorted(range(len(groups)), key=lambda index: (groups[index], index))
    model.fit(
        scaler.transform(matrix).astype(np.float32)[order],
        target.astype(np.float64)[order],
        group_id=[groups[index] for index in order],
    )
    return CatBoostArm(scaler, model)


def _fit_mlp_listnet(qids, candidate_ids, static, registry, setting, seed, feature_names=None):
    """The frozen pointwise MLP arm under the ListNet top-one loss.

    Constructed exactly as ``_fit_mlp_ranknet`` is, for the same reason: the
    architecture, optimiser, epoch count and seeding are the pointwise arm's
    and only the loss changes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pairs, matrix, target, _ = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    values = torch.tensor(
        scaler.transform(matrix).astype(np.float32), dtype=torch.float32
    )
    utilities = torch.tensor(target, dtype=torch.float32)
    model = canonical.SmallMLP(values.shape[1], {
        "hidden_dim": int(setting["hidden_dim"]),
        "layers": int(setting["layers"]),
        "dropout": float(setting["dropout"]),
    })
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(setting["learning_rate"]),
        weight_decay=float(setting["weight_decay"]),
    )
    temperature = float(setting["label_temperature"])
    blocks = [
        (start, stop)
        for start, stop in _group_blocks([qid for qid, _ in pairs])
        if stop - start > 1
    ]
    if not blocks:
        raise ValueError("no query carries more than one candidate")
    # One index per row naming its query, so every per-query softmax is a
    # single segmented operation rather than a Python loop over queries.  A
    # loop would put one autograd node per query per epoch into the graph.
    rows = torch.cat([torch.arange(start, stop) for start, stop in blocks])
    membership = torch.cat([
        torch.full((stop - start,), index, dtype=torch.long)
        for index, (start, stop) in enumerate(blocks)
    ])
    count = len(blocks)

    def _log_softmax_by_query(raw: torch.Tensor) -> torch.Tensor:
        highest = torch.full((count,), float("-inf")).scatter_reduce(
            0, membership, raw, reduce="amax", include_self=False
        )
        shifted = raw - highest[membership]
        total = torch.zeros(count).index_add(0, membership, shifted.exp())
        return shifted - total.log()[membership]

    target_log = _log_softmax_by_query(utilities[rows] / temperature)
    target_probability = target_log.exp().detach()
    for _ in range(int(setting["epochs"])):
        model.train()
        optimizer.zero_grad()
        scores = model(values)
        # Query balanced, matching the weighting the candidate arrays give:
        # each query's cross entropy contributes once, divided by the count.
        loss = -(
            target_probability * _log_softmax_by_query(scores[rows])
        ).sum() / count
        loss.backward()
        optimizer.step()
    return MLPListNet(scaler, model)


def _fit_logistic_useful(qids, candidate_ids, static, registry, setting, seed, feature_names=None):
    from sklearn.linear_model import LogisticRegression

    _, matrix, target, weights = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    labels = (target >= float(setting["useful_threshold"])).astype(np.int64)
    if len(set(labels.tolist())) < 2:
        raise ValueError("the useful threshold does not split this training fold")
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        C=float(setting["C"]), max_iter=int(setting["max_iter"]),
        solver=str(setting["solver"]), random_state=int(seed),
    )
    model.fit(
        scaler.transform(matrix), labels,
        sample_weight=_unit_mean_weights(weights),
    )
    return LogisticUseful(scaler, model)


def _fit_principal_huber(qids, candidate_ids, static, registry, setting, seed,
                         feature_names=None):
    """Canonical frozen Huber (repair implementation); the fit is deterministic
    so the seed is accepted for interface parity and unused."""
    return repair._fit_huber(qids, candidate_ids, static, registry, setting,
                             feature_names)


def _fit_principal_mlp(qids, candidate_ids, static, registry, setting, seed, feature_names=None):
    return repair._fit_mlp(qids, candidate_ids, static, registry, setting, seed,
                           feature_names)


# --- additional lightweight families -----------------------------------
# Added for the expanded lightweight scorer search.  Each mirrors an existing
# fitter's contract exactly: same signature, same StandardScaler-then-model
# pipeline, same query-balanced weights, and a wrapper with the same predict
# interface, so the canonical tuner and OOF loop need no special cases.

def _fit_ridge(qids, candidate_ids, static, registry, setting, seed,
               feature_names=None):
    from sklearn.linear_model import Ridge

    _, matrix, target, weights = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    model = Ridge(alpha=float(setting["alpha"]), fit_intercept=True,
                  random_state=int(seed))
    model.fit(scaler.transform(matrix), target, sample_weight=weights)
    return repair.CandidateHuber(scaler, model)


def _fit_elasticnet(qids, candidate_ids, static, registry, setting, seed,
                    feature_names=None):
    from sklearn.linear_model import ElasticNet

    _, matrix, target, weights = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    model = ElasticNet(
        alpha=float(setting["alpha"]), l1_ratio=float(setting["l1_ratio"]),
        max_iter=int(setting.get("max_iter", 10000)),
        tol=float(setting.get("tol", 1e-4)),
        fit_intercept=True, random_state=int(seed),
    )
    model.fit(scaler.transform(matrix), target, sample_weight=weights)
    return repair.CandidateHuber(scaler, model)


def _fit_hist_gbr(qids, candidate_ids, static, registry, setting, seed,
                  feature_names=None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    _, matrix, target, weights = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    params = {k: v for k, v in setting.items()}
    model = HistGradientBoostingRegressor(random_state=int(seed), **params)
    model.fit(scaler.transform(matrix), target, sample_weight=weights)
    return repair.CandidateHuber(scaler, model)


def _fit_catboost_regression(qids, candidate_ids, static, registry, setting,
                             seed, feature_names=None):
    """Pointwise CatBoost regression on the utility scale.

    Distinct from ``_fit_catboost_ranker``: that one fits a grouped ranking
    loss whose score has no identifiable level, so it cannot be tuned on MAE.
    This one is an ordinary calibrated regressor and can.
    """
    import catboost

    _, matrix, target, weights = repair._candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    params = {k: v for k, v in setting.items() if k != "loss_function"}
    model = catboost.CatBoostRegressor(
        loss_function=str(setting.get("loss_function", "RMSE")),
        random_seed=int(seed), verbose=False, **params
    )
    model.fit(scaler.transform(matrix).astype(np.float32),
              target.astype(np.float64), sample_weight=weights)
    return repair.CandidateHuber(scaler, model)


FITTERS: dict[str, Callable] = {
    "huber": _fit_principal_huber,
    "ridge": _fit_ridge,
    "elasticnet": _fit_elasticnet,
    "hist_gbr": _fit_hist_gbr,
    "catboost_regression": _fit_catboost_regression,
    "small_mlp": _fit_principal_mlp,
    "gbdt_regression": _fit_gbdt_regression,
    "xgb_ranker": _fit_xgb_ranker,
    "lgbm_ranker": _fit_lgbm_ranker,
    "mlp_ranknet": _fit_mlp_ranknet,
    "calibrated_ranker": _fit_calibrated_ranker,
    "ranksvm": _fit_ranksvm,
    "catboost_ranker": _fit_catboost_ranker,
    "mlp_listnet": _fit_mlp_listnet,
    "logistic_useful": _fit_logistic_useful,
}

# Third-party packages each fitter needs, imported before any fitting starts so
# a missing package fails in a second rather than twenty minutes in.
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "huber": (),
    "ridge": (),
    "elasticnet": (),
    "hist_gbr": (),
    "catboost_regression": ("catboost",),
    "small_mlp": (),
    "gbdt_regression": ("xgboost",),
    "xgb_ranker": ("xgboost",),
    "lgbm_ranker": ("lightgbm",),
    "mlp_ranknet": (),
    "calibrated_ranker": ("xgboost",),
    "ranksvm": (),
    "catboost_ranker": ("catboost",),
    "mlp_listnet": (),
    "logistic_useful": (),
}


def _preflight_dependencies(arms) -> None:
    import importlib

    missing: dict[str, list[str]] = {}
    for arm in arms:
        for module in DEPENDENCIES.get(str(arm["fitter"]), ()):
            try:
                importlib.import_module(module)
            except ImportError:
                missing.setdefault(module, []).append(str(arm["scorer"]))
    if missing:
        raise ImportError(
            "; ".join(
                f"{module} is not installed but {', '.join(scorers)} needs it"
                for module, scorers in sorted(missing.items())
            )
        )


def _predict_pairs(model: Any, qids, candidate_ids, static,
                   feature_names=None) -> dict:
    names = tuple(repair.STATIC_PREDICTOR_FEATURES if feature_names is None
                  else feature_names)
    pairs = [(qid, cid) for qid in qids for cid in candidate_ids[qid]]
    matrix = np.asarray(
        [
            [float(static[pair][name]) for name in names]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    if hasattr(model, "predict_grouped"):
        values = model.predict_grouped(matrix, [qid for qid, _ in pairs])
    else:
        values = model.predict(matrix)
    return dict(zip(pairs, map(float, values), strict=True))


def _inner_score(family, predicted, qids, candidate_ids, registry) -> float:
    if family == RANKING:
        return repair._query_mean_ndcg_at8(predicted, qids, candidate_ids, registry)
    return repair._query_mean_mae(predicted, qids, candidate_ids, registry)


def _tune_and_fit(arm, train_qids, inner_splits, candidate_ids, static, registry,
                  seed, feature_names=None):
    """Grid search inside the outer training fold, on the family's own criterion.

    A ranking arm is selected on inner query-mean NDCG@8 and a calibrated arm on
    inner query-mean MAE, which is the convention the three frozen arms already
    follow.
    """
    fitter = FITTERS[str(arm["fitter"])]
    family = str(arm["family"])
    traces = []
    for config_index, setting in enumerate(arm["grid"]):
        scores = []
        for inner_index, (inner_train, inner_valid) in enumerate(inner_splits):
            fit_seed = seed + 10_000 * config_index + inner_index
            model = fitter(
                inner_train, candidate_ids, static, registry, setting, fit_seed,
                feature_names,
            )
            predicted = _predict_pairs(model, inner_valid, candidate_ids, static,
                                       feature_names)
            scores.append(
                _inner_score(family, predicted, inner_valid, candidate_ids, registry)
            )
        traces.append({
            "config_index": config_index,
            "inner_mean": statistics.fmean(scores),
            "inner_folds": scores,
            "setting": setting,
        })
    best = (
        max(traces, key=lambda row: (row["inner_mean"], -row["config_index"]))
        if family == RANKING
        else min(traces, key=lambda row: (row["inner_mean"], row["config_index"]))
    )
    model = fitter(
        train_qids, candidate_ids, static, registry, best["setting"],
        seed + 500_000, feature_names,
    )
    return model, {
        "selected_config_index": best["config_index"],
        "selected_inner_mean": best["inner_mean"],
        "criterion": "inner_query_mean_ndcg_at8" if family == RANKING
        else "inner_query_mean_mae",
    }


def _run_arm_oof(arm, contract, pools, backend, feature_names=None):
    candidate_ids = pools["max_pool_ids"][backend]
    static = pools["features"][backend]
    raw_predictions: dict[tuple[str, str], list[dict]] = defaultdict(list)
    audit: list[dict] = []
    for split_index, split in enumerate(contract["splits"]):
        repeat, fold = int(split["repeat"]), int(split["fold"])
        fold_seed = int(split["seed"])
        train = list(map(str, split["train_query_ids"]))
        valid = list(map(str, split["validation_query_ids"]))
        if set(train) & set(valid):
            raise AssertionError("a held-out query entered training")
        inner = canonical.inner_folds(
            train, int(contract["paths"]["inner_folds"]), fold_seed + 7000
        )
        model, trace = _tune_and_fit(
            arm, train, inner, candidate_ids, static, contract["registry"],
            fold_seed + 700_000 + 1_000 * split_index, feature_names,
        )
        predicted = _predict_pairs(model, valid, candidate_ids, static,
                                   feature_names)
        expected = {(qid, cid) for qid in valid for cid in candidate_ids[qid]}
        if set(predicted) != expected:
            raise ValueError(f"{arm['scorer']}: held-out coverage changed")
        for pair, value in predicted.items():
            raw_predictions[pair].append(
                {"repeat": repeat, "fold": fold, "prediction": float(value)}
            )
        audit.append({
            "scorer": str(arm["scorer"]), "repeat": repeat, "fold": fold,
            "train_queries": len(train), "validation_queries": len(valid), **trace,
        })
    rows = [
        {
            "backend": backend,
            "scorer": str(arm["scorer"]),
            "query_id": qid,
            "candidate_id": cid,
            "oof_prediction_mean": statistics.fmean(
                entry["prediction"] for entry in entries
            ),
            "repeat_folds": [int(entry["fold"]) for entry in entries],
            "oof_repeats": len(entries),
        }
        for (qid, cid), entries in sorted(raw_predictions.items())
    ]
    return rows, audit


def _load_config(config_key: str) -> dict[str, Any]:
    raw = dict(project_config.load()[config_key])
    if raw.get("allow_external_calls") or raw.get("allow_frozen_test"):
        raise ValueError("scorer-family fitting is development-only and offline")
    if list(repair.STATIC_PREDICTOR_FEATURES) != list(raw["expected_feature_names"]):
        raise ValueError("the eight-feature contract changed")
    seen: set[str] = set()
    for arm in raw["arms"]:
        name = str(arm["scorer"])
        if name in seen:
            raise ValueError(f"duplicate arm: {name}")
        seen.add(name)
        if str(arm["family"]) not in (CALIBRATED, RANKING):
            raise ValueError(f"{name}: family must be calibrated or ranking")
        if str(arm["fitter"]) not in FITTERS:
            raise ValueError(f"{name}: unknown fitter {arm['fitter']}")
        if not arm.get("grid"):
            raise ValueError(f"{name}: empty hyperparameter grid")
    return raw


def _write_csv(path: Path, rows: list[dict]) -> None:
    header = list(rows[0])
    path.write_text(
        "\n".join(
            [",".join(header)]
            + [",".join(str(row[key]) for key in header) for row in rows]
        ) + "\n",
        encoding="utf-8",
    )


def run(config_key: str = CONFIG_KEY, output_dir: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = _load_config(config_key)
    backend = str(cfg["backend"])
    source_raw = project_config.load()[str(cfg["source_config_key"])]
    contract = repair._load_contract(source_raw)
    max_pool_override = None
    if cfg.get("max_pool_override"):
        override_path = Path(str(cfg["max_pool_override"]))
        if not override_path.is_absolute():
            override_path = ROOT / override_path
        repair._reject_test(override_path)
        max_pool_override = {}
        with override_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                max_pool_override[str(row["query_id"])] = [
                    str(cid) for cid in row["ordered_ids"]
                ]
        missing_label = [
            (qid, cid)
            for qid, ids in max_pool_override.items()
            for cid in ids
            if (qid, cid) not in contract["registry"]
        ]
        if missing_label:
            raise ValueError(
                f"pool override has {len(missing_label)} unlabelled pairs; "
                "the utility registry must cover the training pool"
            )
    pools = repair._build_pools_and_features(
        contract, max_pool_override=max_pool_override
    )

    destination = Path(cfg["output_dir"] if output_dir is None else output_dir)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination = destination.resolve()
    repair._reject_test(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    _preflight_dependencies(cfg["arms"])
    expected_pairs = int(cfg["expected_scorer_pool_pairs"])
    journal_dir = None
    if cfg.get("journal_dir"):
        journal_dir = Path(str(cfg["journal_dir"]))
        if not journal_dir.is_absolute():
            journal_dir = ROOT / journal_dir
        repair._reject_test(journal_dir)
        journal_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    audits: list[dict] = []
    per_arm: list[dict] = []
    for arm in cfg["arms"]:
        arm_started = time.perf_counter()
        journal_rows = (
            journal_dir / f"{arm['scorer']}.parquet" if journal_dir else None
        )
        journal_audit = (
            journal_dir / f"{arm['scorer']}_audit.json" if journal_dir else None
        )
        if journal_rows and journal_rows.exists() and journal_audit.exists():
            arm_rows = pq.read_table(journal_rows).to_pylist()
            arm_audit = json.loads(journal_audit.read_text(encoding="utf-8"))
            if len(arm_rows) != expected_pairs:
                raise ValueError(
                    f"{arm['scorer']}: journal has {len(arm_rows)} rows, "
                    f"expected {expected_pairs}; delete the journal to refit"
                )
            print(json.dumps({"scorer": arm["scorer"],
                              "journal": "reused"}, sort_keys=True), flush=True)
        else:
            arm_rows, arm_audit = _run_arm_oof(arm, contract, pools, backend)
            if journal_rows:
                pq.write_table(pa.Table.from_pylist(arm_rows), journal_rows,
                               compression="zstd")
                journal_audit.write_text(
                    json.dumps(arm_audit, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
        if len(arm_rows) != expected_pairs:
            raise ValueError(
                f"{arm['scorer']}: {len(arm_rows)} scorer-pool pairs, "
                f"expected {expected_pairs}"
            )
        if any(row["oof_repeats"] != 5 for row in arm_rows):
            raise ValueError(f"{arm['scorer']}: a pair lacks five repeats")
        values = [row["oof_prediction_mean"] for row in arm_rows]
        # A Stage-2 scorer earns its place by ordering candidates WITHIN a
        # query.  A globally varying score can still be constant inside every
        # query - a degenerate tree arm predicts one value per query - and such
        # an arm carries no ordering information at all while looking healthy
        # in a score range.  Measure the property that matters and refuse to
        # write an arm that lacks it.
        by_query: dict[str, list[float]] = defaultdict(list)
        for row in arm_rows:
            by_query[str(row["query_id"])].append(float(row["oof_prediction_mean"]))
        spreads = [
            max(scores) - min(scores)
            for scores in by_query.values()
            if len(scores) > 1
        ]
        tied = sum(1 for spread in spreads if spread <= repair.EPS)
        if tied > max(1, len(spreads) // 100):
            raise ValueError(
                f"{arm['scorer']}: no within-query ordering on {tied} of "
                f"{len(spreads)} queries; the arm is degenerate"
            )
        rows.extend(arm_rows)
        audits.extend(arm_audit)
        per_arm.append({
            "scorer": str(arm["scorer"]), "family": str(arm["family"]),
            "fitter": str(arm["fitter"]), "grid_size": len(arm["grid"]),
            "rows": len(arm_rows), "score_min": min(values), "score_max": max(values),
            "mean_within_query_spread": round(
                sum(spreads) / len(spreads), 6
            ),
            "within_query_tied_queries": tied,
            "provenance": str(arm.get("provenance", "unrecorded")),
            "reference_implementation": str(arm.get("reference_implementation", "")),
            "elapsed_seconds": round(time.perf_counter() - arm_started, 2),
        })
        print(json.dumps(per_arm[-1], ensure_ascii=False, sort_keys=True), flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        out = Path(tmp)
        pq.write_table(
            pa.Table.from_pylist(rows),
            out / "oof_scorer_pool_predictions.parquet", compression="zstd",
        )
        _write_csv(out / "tuning_audit.csv", audits)
        software = {
            "python": platform.python_version(), "numpy": np.__version__,
            "pyarrow": pa.__version__, "torch": torch.__version__,
        }
        for module in ("xgboost", "lightgbm"):
            try:
                software[module] = __import__(module).__version__
            except ModuleNotFoundError:
                software[module] = "not installed"
        manifest = {
            "schema": "rq2b-scorer-family-oof",
            "version": str(cfg["version"]),
            "status": "COMPLETE",
            "created_utc": now(),
            "backend": backend,
            "source_config_key": str(cfg["source_config_key"]),
            **({"protocol_status": str(cfg["protocol_status"])}
               if "protocol_status" in cfg else {}),
            **({"max_pool_override": {
                    "path": str(cfg["max_pool_override"]),
                    "sha256": sha256_file(
                        override_path if max_pool_override is not None else None
                    ),
                }} if cfg.get("max_pool_override") else {}),
            "feature_names": list(repair.STATIC_PREDICTOR_FEATURES),
            "outer_folds": len(contract["splits"]),
            "inner_folds": int(contract["paths"]["inner_folds"]),
            "expected_scorer_pool_pairs": expected_pairs,
            "arms": per_arm,
            "boundaries": {
                "external_requests_made": 0,
                "frozen_test_read": False,
                "candidate_sets_selected_here": False,
                "selection_hyperparameter_touched": False,
                "utility_used_outside_training_folds": False,
            },
            "software": software,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["outputs"] = {
            name: sha256_file(out / name)
            for name in ("oof_scorer_pool_predictions.parquet", "tuning_audit.csv")
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(out).rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-key", default=CONFIG_KEY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = run(args.config_key, args.output_dir)
    print(json.dumps(
        {"status": manifest["status"], "arms": manifest["arms"],
         "elapsed_seconds": manifest["elapsed_seconds"]},
        ensure_ascii=False, indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
