from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset_utils import load_dataset
from ..model_config import OUTPUT_DATASET, PREDICT_OUTPUT, VALID_END

OUTPUT_JSON = Path("core/models/Xgboost/alpha_evaluation/ic_vs_liquidity_h20.json")
OUTPUT_DAILY = Path("core/models/Xgboost/alpha_evaluation/ic_vs_liquidity_h20_daily.parquet")


def summarize(series: pd.Series) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"mean": 0.0, "std": 0.0, "IC_IR": 0.0, "win_rate": 0.0, "obs": 0}
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    return {
        "mean": mean,
        "std": std,
        "IC_IR": float(mean / std) if std > 0 else 0.0,
        "win_rate": float((clean > 0).mean()),
        "obs": int(clean.count()),
    }


def assign_buckets(values: pd.Series, bucket_num: int = 5) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = values.dropna()
    if len(valid) < bucket_num:
        return out
    rank = valid.rank(method="first")
    bucket = pd.qcut(rank, bucket_num, labels=False, duplicates="drop")
    out.loc[bucket.index] = bucket.astype(float)
    return out


def build_panel() -> pd.DataFrame:
    if not PREDICT_OUTPUT.exists():
        raise FileNotFoundError(f"alpha predictions not found: {PREDICT_OUTPUT}")
    pred = pd.read_parquet(PREDICT_OUTPUT)
    need_pred = ["date", "symbol", "alpha"]
    miss_pred = [c for c in need_pred if c not in pred.columns]
    if miss_pred:
        raise ValueError(f"alpha predictions missing columns: {miss_pred}")

    ds = load_dataset(OUTPUT_DATASET)
    need_ds = ["date", "symbol", "future_return", "amount"]
    miss_ds = [c for c in need_ds if c not in ds.columns]
    if miss_ds:
        raise ValueError(f"dataset missing columns: {miss_ds}")

    pred = pred[need_pred].copy()
    pred["date"] = pd.to_datetime(pred["date"])
    ds = ds[need_ds].copy()
    ds["date"] = pd.to_datetime(ds["date"])
    ds = ds.sort_values(["symbol", "date"])
    ds["amount_20"] = ds.groupby("symbol", sort=False)["amount"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    panel = pred.merge(ds[["date", "symbol", "future_return", "amount_20"]], on=["date", "symbol"], how="inner")
    panel = panel[panel["date"] >= VALID_END].copy()
    panel["liq_bucket"] = panel.groupby("date", sort=False)["amount_20"].transform(assign_buckets)
    panel = panel.dropna(subset=["alpha", "future_return", "liq_bucket"]).copy()
    panel["liq_bucket"] = panel["liq_bucket"].astype(int)
    return panel


def evaluate_ic_vs_liquidity() -> dict[str, object]:
    panel = build_panel()
    rows = []
    for (date, bucket), group in panel.groupby(["date", "liq_bucket"], sort=False):
        if len(group) < 2:
            continue
        rank_ic = group["alpha"].corr(group["future_return"], method="spearman")
        rows.append(
            {
                "date": date,
                "liq_bucket": int(bucket),
                "rank_ic": rank_ic,
                "row_count": int(len(group)),
            }
        )
    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily = daily.sort_values(["liq_bucket", "date"])

    bucket_stats: dict[str, dict[str, float]] = {}
    for bucket in sorted(panel["liq_bucket"].unique()):
        series = daily.loc[daily["liq_bucket"] == bucket, "rank_ic"] if not daily.empty else pd.Series(dtype=float)
        stats = summarize(series)
        stats["avg_rows_per_day"] = float(
            daily.loc[daily["liq_bucket"] == bucket, "row_count"].mean()
        ) if not daily.empty else 0.0
        bucket_stats[str(int(bucket))] = stats

    report = {
        "run_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horizon": 20,
            "eval_start": str(pd.Timestamp(VALID_END).date()),
            "bucket_num": 5,
            "liquidity_proxy": "amount_20",
            "rolling_window": 20,
            "bucket_definition": "cross_section_quantile_by_date",
        },
        "sample_summary": {
            "row_count": int(len(panel)),
            "date_count": int(panel["date"].nunique()),
            "bucket_counts": {str(int(k)): int(v) for k, v in panel["liq_bucket"].value_counts().sort_index().items()},
        },
        "bucket_rank_ic": bucket_stats,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not daily.empty:
        daily.to_parquet(OUTPUT_DAILY, index=False)
    print("ic vs liquidity saved:", OUTPUT_JSON)
    if not daily.empty:
        print("daily bucket ic saved:", OUTPUT_DAILY)
    return report


if __name__ == "__main__":
    evaluate_ic_vs_liquidity()
    raise SystemExit(0)
