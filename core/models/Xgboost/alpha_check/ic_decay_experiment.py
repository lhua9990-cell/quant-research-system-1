from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb

from ..dataset_utils import load_symbol_data
from ..model_config import (
    EARLY_STOPPING_ROUNDS,
    EVAL_METRIC,
    FEATURES,
    INPUT_DIR,
    TRAIN_END,
    VALID_END,
)

HORIZONS = [5, 10, 20, 40, 60, 80, 120]
OUTPUT_JSON = Path("core/models/Xgboost/alpha_evaluation/ic_decay_results.json")


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["date"] < TRAIN_END) & (df["future_date"] < TRAIN_END)]
    valid = df[
        (df["date"] >= TRAIN_END)
        & (df["date"] < VALID_END)
        & (df["future_date"] < VALID_END)
    ]
    test = df[df["date"] >= VALID_END]
    return train, valid, test


def summarize(series: pd.Series) -> dict[str, float]:
    clean = series.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if clean.empty:
        return {"mean": 0.0, "std": 0.0, "IC_IR": 0.0, "win_rate": 0.0}
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    ic_ir = float(mean / std) if std > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "IC_IR": ic_ir,
        "win_rate": float((clean > 0).mean()),
    }


def build_dataset(horizon: int) -> pd.DataFrame:
    files = sorted(Path(INPUT_DIR).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found in {INPUT_DIR}")
    frames: list[pd.DataFrame] = []
    for file_path in files:
        frames.append(load_symbol_data(file_path, FEATURES, horizon))
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.dropna()
    dataset = dataset.sort_values(["date", "symbol"])
    return dataset


def train_and_eval_for_horizon(horizon: int) -> dict[str, object]:
    dataset = build_dataset(horizon)
    train_df, valid_df, test_df = time_split(dataset)

    train_subset = train_df[FEATURES + ["future_return"]].dropna()
    valid_subset = valid_df[FEATURES + ["future_return"]].dropna()
    test_subset = test_df[["date", "symbol", "future_return"] + FEATURES].dropna()

    if train_subset.empty or valid_subset.empty or test_subset.empty:
        raise ValueError(f"empty split for horizon={horizon}")

    dtrain = xgb.DMatrix(train_subset[FEATURES], label=train_subset["future_return"])
    dvalid = xgb.DMatrix(valid_subset[FEATURES], label=valid_subset["future_return"])
    dtest = xgb.DMatrix(test_subset[FEATURES])

    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "eval_metric": EVAL_METRIC,
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        alpha = model.predict(dtest, iteration_range=(0, best_iteration + 1))
    else:
        alpha = model.predict(dtest)

    panel = test_subset[["date", "symbol", "future_return"]].copy()
    panel["alpha"] = alpha
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= VALID_END]

    daily_rank_ic = (
        panel.groupby("date", sort=False)[["alpha", "future_return"]]
        .apply(lambda x: x["alpha"].corr(x["future_return"], method="spearman"))
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
    )
    rank_ic_summary = summarize(daily_rank_ic)
    return {
        "horizon": horizon,
        "rank_ic": rank_ic_summary,
        "row_count": int(len(panel)),
        "date_count": int(panel["date"].nunique()),
        "train_rows": int(len(train_subset)),
        "valid_rows": int(len(valid_subset)),
        "test_rows": int(len(test_subset)),
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
    }


def run_ic_decay_experiment() -> dict[str, object]:
    results = []
    for horizon in HORIZONS:
        item = train_and_eval_for_horizon(horizon)
        results.append(item)
        print(
            f"h={horizon} rank_ic_mean={item['rank_ic']['mean']:.6f} "
            f"IC_IR={item['rank_ic']['IC_IR']:.6f} "
            f"win_rate={item['rank_ic']['win_rate']:.2%}"
        )
    report = {
        "run_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eval_start": str(pd.Timestamp(VALID_END).date()),
            "horizons": HORIZONS,
        },
        "results": results,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("ic decay report saved:", OUTPUT_JSON)
    return report


if __name__ == "__main__":
    run_ic_decay_experiment()
    raise SystemExit(0)
