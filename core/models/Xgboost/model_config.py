from __future__ import annotations

from pathlib import Path

import pandas as pd

HORIZON = 20

FEATURES = [
]

INPUT_DIR = Path("dataset/trainable_universe/A_stock/2016-2025")
OUTPUT_DATASET = Path("dataset/ml_dataset/ml_dataset.parquet")
MODEL_PATH = Path("core/models/Xgboost/model_params/xgboost_alpha.model")
PREDICT_OUTPUT = Path("core/models/Xgboost/alpha_predictions/alpha.parquet")
EVAL_OUTPUT = Path("core/models/Xgboost/alpha_evaluation/alpha_eval.json")
EVAL_TIMESERIES = Path("core/models/Xgboost/alpha_evaluation/alpha_eval_timeseries.parquet")
SANITY_CHECK_OUTPUT = Path("core/models/Xgboost/alpha_evaluation/sanity_check_report.json")
SANITY_CHECK_MD = Path("core/models/Xgboost/alpha_evaluation/sanity_check_report.md")

TRAIN_END = pd.Timestamp("2022-01-01")
VALID_END = pd.Timestamp("2024-01-01")
EVAL_METRIC = "rmse"
EARLY_STOPPING_ROUNDS = 50
TOP_QUANTILE = 0.2