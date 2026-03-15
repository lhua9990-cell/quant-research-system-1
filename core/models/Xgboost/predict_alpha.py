from __future__ import annotations

import pandas as pd
import xgboost as xgb

from core.models.Xgboost.dataset_utils import load_dataset
from core.models.Xgboost.model_config import FEATURES, MODEL_PATH, OUTPUT_DATASET, PREDICT_OUTPUT


def predict() -> pd.DataFrame:
    df = load_dataset(OUTPUT_DATASET)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing features: {missing}")
    X = df[FEATURES]
    df = df.copy()
    dmatrix = xgb.DMatrix(X)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        df["alpha"] = model.predict(dmatrix, iteration_range=(0, best_iteration + 1))
    else:
        df["alpha"] = model.predict(dmatrix)
    df = df.sort_values(["date", "symbol"])
    PREDICT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "symbol", "alpha"]].to_parquet(PREDICT_OUTPUT, index=False)
    print("alpha saved:", PREDICT_OUTPUT)
    return df[["date", "symbol", "alpha"]]


if __name__ == "__main__":
    raise SystemExit(predict())
