#!/usr/bin/env python3
"""Fit the frozen final LambdaMART transfer model from a numeric NPZ bundle.

This runtime bridge isolates fitting and prediction in the pinned Python 3.12
reranker environment. It changes no model semantics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility_scoring.learned_diffusion.reranker_validation import (  # noqa: E402
    fit_xgb_lambdamart,
)


def _save_scaler(scaler: StandardScaler, path: Path) -> None:
    np.savez_compressed(
        path,
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        var=np.asarray(scaler.var_, dtype=np.float64),
        n_features=np.asarray([int(scaler.n_features_in_)], dtype=np.int64),
    )


def run(
    bundle: Path,
    setting_path: Path,
    output: Path,
    seed: int,
    *,
    model_output: Path | None = None,
    scaler_output: Path | None = None,
) -> dict:
    values = np.load(bundle, allow_pickle=False)
    train_x = np.asarray(values["train_x"], dtype=np.float64)
    train_y = np.asarray(values["train_y"], dtype=np.float64)
    train_qids = list(map(str, values["train_qids"].tolist()))
    predict_x = np.asarray(values["predict_x"], dtype=np.float64)
    setting = json.loads(setting_path.read_text(encoding="utf-8"))
    scaler = StandardScaler().fit(train_x)
    model = fit_xgb_lambdamart(
        scaler.transform(train_x).astype(np.float32),
        train_y.astype(np.float32),
        train_qids,
        setting,
        int(seed),
    )
    predictions = np.asarray(
        model.predict(scaler.transform(predict_x).astype(np.float32)),
        dtype=np.float64,
    )
    if model_output is not None:
        model.save_model(model_output)
    if scaler_output is not None:
        _save_scaler(scaler, scaler_output)
    np.save(output, predictions, allow_pickle=False)
    return {
        "training_rows": int(train_x.shape[0]),
        "prediction_rows": int(predict_x.shape[0]),
        "features": int(train_x.shape[1]),
        "seed": int(seed),
        "model_saved": model_output is not None,
        "scaler_saved": scaler_output is not None,
    }


def predict(
    bundle: Path,
    output: Path,
    *,
    model_path: Path,
    scaler_path: Path,
) -> dict:
    from xgboost import XGBRanker

    values = np.load(bundle, allow_pickle=False)
    predict_x = np.asarray(values["predict_x"], dtype=np.float64)
    scaler_values = np.load(scaler_path, allow_pickle=False)
    mean = np.asarray(scaler_values["mean"], dtype=np.float64)
    scale = np.asarray(scaler_values["scale"], dtype=np.float64)
    if predict_x.ndim != 2 or predict_x.shape[1] != len(mean):
        raise ValueError("prediction feature width differs from frozen scaler")
    if len(scale) != len(mean) or np.any(scale <= 0):
        raise ValueError("frozen scaler is invalid")
    model = XGBRanker()
    model.load_model(model_path)
    predictions = np.asarray(
        model.predict(((predict_x - mean) / scale).astype(np.float32)),
        dtype=np.float64,
    )
    np.save(output, predictions, allow_pickle=False)
    return {
        "prediction_rows": int(predict_x.shape[0]),
        "features": int(predict_x.shape[1]),
        "model_loaded": True,
        "scaler_loaded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("setting", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("fit", "predict"), default="fit")
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--scaler-output", type=Path)
    parser.add_argument("--model-input", type=Path)
    parser.add_argument("--scaler-input", type=Path)
    args = parser.parse_args()
    if args.mode == "fit":
        result = run(
            args.bundle,
            args.setting,
            args.output,
            args.seed,
            model_output=args.model_output,
            scaler_output=args.scaler_output,
        )
    else:
        if args.model_input is None or args.scaler_input is None:
            parser.error("predict mode requires --model-input and --scaler-input")
        result = predict(
            args.bundle,
            args.output,
            model_path=args.model_input,
            scaler_path=args.scaler_input,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
