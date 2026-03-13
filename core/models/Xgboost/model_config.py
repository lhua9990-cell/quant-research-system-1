from __future__ import annotations

from pathlib import Path

import pandas as pd

HORIZON = 20

FEATURES = [
    "liq__turnover",
    "liq__close_vwap_dev",
    "ret__roc__20",
    "ret__roc__60",
    "rng__atr__14",
]

INPUT_DIR = Path("dataset/trainable_universe/A_stock/2016-2025")
OUTPUT_DATASET = Path("dataset/ml_dataset/ml_dataset.parquet")
MODEL_PATH = Path("models/Xgboost/model_params/xgboost_alpha.model")
PREDICT_OUTPUT = Path("models/Xgboost/alpha_predictions/alpha.parquet")
EVAL_OUTPUT = Path("models/Xgboost/alpha_evaluation/alpha_eval.json")
EVAL_TIMESERIES = Path("models/Xgboost/alpha_evaluation/alpha_eval_timeseries.parquet")

TRAIN_END = pd.Timestamp("2022-01-01")
VALID_END = pd.Timestamp("2024-01-01")
EVAL_METRIC = "rmse"
EARLY_STOPPING_ROUNDS = 50
TOP_QUANTILE = 0.2
