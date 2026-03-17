from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset_utils import load_dataset
from ..model_config import HORIZON, OUTPUT_DATASET, PREDICT_OUTPUT, VALID_END

OUTPUT_JSON = Path("core/models/Xgboost/alpha_evaluation/capacity_simulation_h20.json")
OUTPUT_DAILY = Path("core/models/Xgboost/alpha_evaluation/capacity_simulation_h20_daily.parquet")


def load_panel() -> pd.DataFrame:
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
    panel = pred.merge(
        ds[["date", "symbol", "future_return", "amount_20"]],
        on=["date", "symbol"],
        how="inner",
    )
    panel = panel[panel["date"] >= VALID_END].copy()
    panel = panel.dropna(subset=["alpha", "future_return", "amount_20"])
    return panel.sort_values(["date", "symbol"])


def calc_turnover(prev_weights: dict[str, float], curr_weights: dict[str, float]) -> float:
    symbols = set(prev_weights.keys()) | set(curr_weights.keys())
    diff_sum = 0.0
    for symbol in symbols:
        diff_sum += abs(curr_weights.get(symbol, 0.0) - prev_weights.get(symbol, 0.0))
    return diff_sum / 2.0


def build_daily_capacity(
    panel: pd.DataFrame,
    participation_rate: float = 0.05,
    quantile: float = 0.2,
    min_universe: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    prev_weights: dict[str, float] = {}
    for date, group in panel.groupby("date", sort=True):
        group = group.sort_values("alpha")
        n = len(group)
        if n < min_universe:
            continue
        k = max(1, int(n * quantile))
        long_group = group.tail(k)
        short_group = group.head(k)
        portfolio = pd.concat([long_group, short_group], ignore_index=True)

        long_weight = 0.5 / k
        short_weight = -0.5 / k
        curr_weights = {s: short_weight for s in short_group["symbol"].tolist()}
        for symbol in long_group["symbol"].tolist():
            curr_weights[symbol] = long_weight

        turnover = calc_turnover(prev_weights, curr_weights) if prev_weights else np.nan
        prev_weights = curr_weights

        daily_capacity_notional = float((portfolio["amount_20"] * participation_rate).sum())
        turnover_adjusted_capacity = (
            float(daily_capacity_notional / turnover) if turnover and turnover > 0 else np.nan
        )
        rows.append(
            {
                "date": date,
                "universe_size": int(n),
                "leg_size": int(k),
                "position_count": int(len(portfolio)),
                "turnover": float(turnover) if not np.isnan(turnover) else np.nan,
                "daily_capacity_notional": daily_capacity_notional,
                "turnover_adjusted_aum_capacity": turnover_adjusted_capacity,
            }
        )
    return pd.DataFrame(rows)


def summarize_daily_capacity(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "avg_daily_capacity_notional": 0.0,
            "median_daily_capacity_notional": 0.0,
            "avg_turnover": 0.0,
            "annualized_turnover": 0.0,
            "avg_turnover_adjusted_aum_capacity": 0.0,
            "p10_turnover_adjusted_aum_capacity": 0.0,
            "p50_turnover_adjusted_aum_capacity": 0.0,
            "p90_turnover_adjusted_aum_capacity": 0.0,
            "obs_days": 0,
        }

    turnover_clean = daily["turnover"].dropna()
    aum_clean = daily["turnover_adjusted_aum_capacity"].dropna()
    return {
        "avg_daily_capacity_notional": float(daily["daily_capacity_notional"].mean()),
        "median_daily_capacity_notional": float(daily["daily_capacity_notional"].median()),
        "avg_turnover": float(turnover_clean.mean()) if not turnover_clean.empty else 0.0,
        "annualized_turnover": float(turnover_clean.mean() * 252.0) if not turnover_clean.empty else 0.0,
        "avg_turnover_adjusted_aum_capacity": float(aum_clean.mean()) if not aum_clean.empty else 0.0,
        "p10_turnover_adjusted_aum_capacity": float(aum_clean.quantile(0.1)) if not aum_clean.empty else 0.0,
        "p50_turnover_adjusted_aum_capacity": float(aum_clean.quantile(0.5)) if not aum_clean.empty else 0.0,
        "p90_turnover_adjusted_aum_capacity": float(aum_clean.quantile(0.9)) if not aum_clean.empty else 0.0,
        "obs_days": int(len(daily)),
    }


def capacity_table(
    avg_daily_capacity_notional: float,
    estimated_aum_capacity: float,
    capital_list: list[float],
) -> list[dict[str, float]]:
    rows = []
    for capital in capital_list:
        utilization_daily_notional = (
            float(capital / avg_daily_capacity_notional) if avg_daily_capacity_notional > 0 else np.nan
        )
        utilization_turnover_adjusted = (
            float(capital / estimated_aum_capacity) if estimated_aum_capacity > 0 else np.nan
        )
        rows.append(
            {
                "capital": float(capital),
                "capacity_utilization_vs_daily_notional": utilization_daily_notional,
                "capacity_utilization_vs_turnover_adjusted_aum": utilization_turnover_adjusted,
            }
        )
    return rows


def run_capacity_simulation(
    participation_rate: float = 0.05,
    quantile: float = 0.2,
    capital_list: list[float] | None = None,
) -> dict[str, object]:
    if capital_list is None:
        capital_list = [50e6, 100e6, 200e6, 500e6, 1e9, 2e9, 5e9]

    panel = load_panel()
    daily = build_daily_capacity(panel, participation_rate=participation_rate, quantile=quantile)
    summary = summarize_daily_capacity(daily)
    est_aum_capacity = summary["avg_turnover_adjusted_aum_capacity"]

    report = {
        "run_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horizon": int(HORIZON),
            "eval_start": str(pd.Timestamp(VALID_END).date()),
            "participation_rate": float(participation_rate),
            "quantile": float(quantile),
            "liquidity_proxy": "amount_20",
            "turnover_formula": "abs(weight_t - weight_t-1).sum()/2",
        },
        "sample_summary": {
            "row_count": int(len(panel)),
            "date_count": int(panel["date"].nunique()),
        },
        "capacity_summary": summary,
        "capital_utilization_table": capacity_table(
            summary["avg_daily_capacity_notional"],
            est_aum_capacity,
            capital_list,
        ),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not daily.empty:
        daily.to_parquet(OUTPUT_DAILY, index=False)

    print("capacity simulation saved:", OUTPUT_JSON)
    if not daily.empty:
        print("capacity simulation daily saved:", OUTPUT_DAILY)
    return report


if __name__ == "__main__":
    run_capacity_simulation()
    raise SystemExit(0)
