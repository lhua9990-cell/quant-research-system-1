from __future__ import annotations

import joblib
import pandas as pd

from core.models.Xgboost.dataset_utils import load_dataset
from core.models.Xgboost.model_config import FEATURES, MODEL_PATH, OUTPUT_DATASET, PREDICT_OUTPUT


def predict() -> pd.DataFrame:
    df = load_dataset(OUTPUT_DATASET)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing features: {missing}")
    X = df[FEATURES]
    df = df.copy()
    best_ntree_limit = None
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        best_ntree_limit = model.best_iteration + 1
    if best_ntree_limit is not None:
        df["alpha"] = model.predict(X, ntree_limit=best_ntree_limit)
    else:
        df["alpha"] = model.predict(X)
    df = df.sort_values(["date", "symbol"])
    PREDICT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "symbol", "alpha"]].to_parquet(PREDICT_OUTPUT, index=False)
    print("alpha saved:", PREDICT_OUTPUT)
    return df[["date", "symbol", "alpha"]]


if __name__ == "__main__":
    raise SystemExit(predict())
