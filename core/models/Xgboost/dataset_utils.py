from __future__ import annotations

from pathlib import Path

import pandas as pd


def compute_future_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("input must contain date and close")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    close = out["close"].astype(float)
    out["future_date"] = out["date"].shift(-horizon)
    out["future_return"] = close.shift(-horizon) / close - 1.0
    if horizon > 0:
        out = out.iloc[:-horizon]
    return out


def load_symbol_data(file_path: Path, features: list[str], horizon: int) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"missing date/close in {file_path}")
    if "symbol" not in df.columns:
        df = df.copy()
        df["symbol"] = file_path.stem
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns in {file_path}: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = compute_future_return(df, horizon=horizon)
    cols = ["date", "future_date", "symbol"] + features + ["future_return"]
    return df[cols]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    if "date" not in df.columns:
        raise ValueError("dataset must contain date column")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "future_date" in df.columns:
        df["future_date"] = pd.to_datetime(df["future_date"])
    return df
