from __future__ import annotations

import pandas as pd
import xgboost as xgb

from core.models.Xgboost.dataset_utils import load_dataset
from core.models.Xgboost.model_config import (
    EARLY_STOPPING_ROUNDS,
    EVAL_METRIC,
    FEATURES,
    MODEL_PATH,
    OUTPUT_DATASET,
    TRAIN_END,
    VALID_END,
)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "future_date" not in df.columns:
        raise ValueError("dataset must contain future_date for leakage-safe split")
    train = df[(df["date"] < TRAIN_END) & (df["future_date"] < TRAIN_END)]
    valid = df[
        (df["date"] >= TRAIN_END)
        & (df["date"] < VALID_END)
        & (df["future_date"] < VALID_END)
    ]
    test = df[df["date"] >= VALID_END]
    return train, valid, test


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in FEATURES + ["future_return"] if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")
    subset = df[FEATURES + ["future_return"]].dropna()
    return subset[FEATURES], subset["future_return"]


def train() -> dict[str, int]:
    df = load_dataset(OUTPUT_DATASET)
    train_df, valid_df, test_df = time_split(df)
    X_train, y_train = build_xy(train_df)
    X_valid, y_valid = build_xy(valid_df)
    X_test, y_test = build_xy(test_df)
    if X_train.empty or X_valid.empty:
        raise ValueError("train/valid split has no usable rows")
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "eval_metric": EVAL_METRIC,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=True,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print("model saved:", MODEL_PATH)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        print("best_iteration:", best_iteration)
    return {
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
    }


if __name__ == "__main__":
    raise SystemExit(train())
