from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.models.Xgboost.dataset_utils import load_dataset
from core.models.Xgboost.model_config import (
    EVAL_OUTPUT,
    EVAL_TIMESERIES,
    OUTPUT_DATASET,
    PREDICT_OUTPUT,
    TOP_QUANTILE,
)


def load_alpha_predictions() -> pd.DataFrame:
    if not PREDICT_OUTPUT.exists():
        raise FileNotFoundError(f"alpha predictions not found: {PREDICT_OUTPUT}")
    df = pd.read_parquet(PREDICT_OUTPUT)
    if "date" not in df.columns or "symbol" not in df.columns or "alpha" not in df.columns:
        raise ValueError("alpha predictions must contain date, symbol, alpha")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "symbol", "alpha"]]


def build_eval_panel() -> pd.DataFrame:
    alpha_df = load_alpha_predictions()
    dataset = load_dataset(OUTPUT_DATASET)
    missing = [c for c in ["date", "symbol", "future_return"] if c not in dataset.columns]
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")
    dataset = dataset[["date", "symbol", "future_return"]].copy()
    dataset["date"] = pd.to_datetime(dataset["date"])
    panel = alpha_df.merge(dataset, on=["date", "symbol"], how="inner")
    return panel


def compute_daily_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.dropna(subset=["alpha", "future_return"])
    rows = []
    for date, group in panel.groupby("date", sort=False):
        if len(group) < 2:
            continue
        rank_ic = group["alpha"].corr(group["future_return"], method="spearman")
        group = group.sort_values("alpha")
        n = len(group)
        k = max(1, int(n * TOP_QUANTILE))
        top = group.tail(k)
        bottom = group.head(k)
        long_short = top["future_return"].mean() - bottom["future_return"].mean()
        top_return = top["future_return"].mean()
        rows.append(
            {
                "date": date,
                "rank_ic": rank_ic,
                "long_short_return": long_short,
                "top_quantile_return": top_return,
                "group_size": n,
                "quantile_size": k,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "rank_ic",
                "long_short_return",
                "top_quantile_return",
                "group_size",
                "quantile_size",
            ]
        )
    return pd.DataFrame(rows).sort_values("date")


def summarize_series(series: pd.Series) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"mean": 0.0, "std": 0.0, "ir": 0.0, "IC_IR": 0.0, "win_rate": 0.0}
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    ir = float(mean / std) if std > 0 else 0.0
    win_rate = float((clean > 0).mean())
    return {"mean": mean, "std": std, "ir": ir, "IC_IR": ir, "win_rate": win_rate}


def evaluate() -> dict[str, object]:
    panel = build_eval_panel()
    eval_start = pd.Timestamp("2024-01-01")
    panel = panel[panel["date"] >= eval_start]
    daily = compute_daily_metrics(panel)
    report = {
        "run_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "row_count": int(len(panel)),
            "date_count": int(daily["date"].nunique()) if not daily.empty else 0,
            "top_quantile": float(TOP_QUANTILE),
            "eval_start": str(eval_start.date()),
        },
        "rank_ic": summarize_series(daily["rank_ic"]) if not daily.empty else summarize_series(pd.Series(dtype=float)),
        "long_short": summarize_series(daily["long_short_return"]) if not daily.empty else summarize_series(pd.Series(dtype=float)),
        "top_quantile": summarize_series(daily["top_quantile_return"]) if not daily.empty else summarize_series(pd.Series(dtype=float)),
    }
    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not daily.empty:
        EVAL_TIMESERIES.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(EVAL_TIMESERIES, index=False)
    print("alpha evaluation saved:", EVAL_OUTPUT)
    if not daily.empty:
        print("alpha evaluation timeseries saved:", EVAL_TIMESERIES)
    return report


if __name__ == "__main__":
    evaluate()
    raise SystemExit(0)
