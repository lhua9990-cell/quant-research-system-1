from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.models.Xgboost.dataset_utils import load_symbol_data
from core.models.Xgboost.model_config import FEATURES, HORIZON, INPUT_DIR, OUTPUT_DATASET


def build_dataset() -> Path:
    files = sorted(Path(INPUT_DIR).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found in {INPUT_DIR}")
    frames: list[pd.DataFrame] = []
    for file_path in files:
        df = load_symbol_data(file_path, FEATURES, HORIZON)
        frames.append(df)
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.dropna()
    dataset = dataset.sort_values(["date", "symbol"])
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_DATASET, index=False)
    print("dataset saved:", OUTPUT_DATASET)
    return OUTPUT_DATASET


if __name__ == "__main__":
    raise SystemExit(build_dataset())
