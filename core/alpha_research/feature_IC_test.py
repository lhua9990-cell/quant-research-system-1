from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_COLUMNS = {
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "pct_change",
    "change",
    "ret_1d",
    "logret_1d",
    "hl_range",
    "oc_range",
    "amplitude",
    "dow",
    "month",
    "is_month_end",
    "days_since_listed",
    "is_trading_day",
    "turnover",
}


def compute_future_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("feature data must contain date and close")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    close = out["close"].astype(float)
    out["future_return"] = close.shift(-horizon) / close - 1.0
    return out[["date", "future_return"]]


def load_feature_data(
    features_dir: Path,
    horizon: int,
    max_symbols: int = 0,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    feature_files = sorted(features_dir.glob("*.parquet"))
    if not feature_files:
        raise FileNotFoundError(f"no feature parquet files found in {features_dir}")
    if max_symbols > 0:
        feature_files = feature_files[:max_symbols]
    frames: List[pd.DataFrame] = []
    feature_order: List[str] = []
    feature_seen = set()
    invalid_symbol_files = 0
    for feature_path in feature_files:
        symbol = feature_path.stem
        feature_df = pd.read_parquet(feature_path)
        if "date" not in feature_df.columns or "close" not in feature_df.columns:
            invalid_symbol_files += 1
            continue
        symbol_features = [c for c in feature_df.columns if c not in BASE_COLUMNS]
        if not symbol_features:
            invalid_symbol_files += 1
            continue
        for col in symbol_features:
            if col not in feature_seen:
                feature_seen.add(col)
                feature_order.append(col)
        feature_df = feature_df.copy()
        feature_df["date"] = pd.to_datetime(feature_df["date"])
        future_return_df = compute_future_return(
            feature_df[["date", "close"]],
            horizon=horizon,
        )
        merged = feature_df.merge(future_return_df, on="date", how="inner")
        if merged.empty:
            invalid_symbol_files += 1
            continue
        if "symbol" not in merged.columns:
            merged["symbol"] = symbol
        keep_cols = ["date", "symbol", "future_return"] + symbol_features
        frames.append(merged[keep_cols])
    if not frames:
        raise ValueError("no valid symbol data after loading feature inputs")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    stats = {
        "input_symbols": int(len(feature_files)),
        "loaded_symbols": int(len(frames)),
        "invalid_symbol_files": int(invalid_symbol_files),
    }
    return panel, feature_order, stats


def compute_daily_ic(panel: pd.DataFrame, feature_col: str) -> pd.Series:
    subset = panel[["date", feature_col, "future_return"]].dropna()
    if subset.empty:
        return pd.Series(dtype=float)
    grouped = subset.groupby("date")[[feature_col, "future_return"]]
    return grouped.apply(
        lambda x: x[feature_col].corr(x["future_return"], method="pearson")
    )


def compute_rank_ic(panel: pd.DataFrame, feature_col: str) -> pd.Series:
    subset = panel[["date", feature_col, "future_return"]].dropna()
    if subset.empty:
        return pd.Series(dtype=float)
    grouped = subset.groupby("date")[[feature_col, "future_return"]]
    return grouped.apply(
        lambda x: x[feature_col].corr(x["future_return"], method="spearman")
    )


def aggregate_ic_statistics(ic_series: pd.Series, rank_ic_series: pd.Series) -> Dict[str, float]:
    ic_clean = ic_series.replace([np.inf, -np.inf], np.nan).dropna()
    rank_clean = rank_ic_series.replace([np.inf, -np.inf], np.nan).dropna()
    if ic_clean.empty:
        ic_mean = 0.0
        ic_std = 0.0
        ic_ir = 0.0
        positive_ratio = 0.0
    else:
        ic_mean = float(ic_clean.mean())
        ic_std = float(ic_clean.std(ddof=0))
        ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
        positive_ratio = float((ic_clean > 0).mean())
    if rank_clean.empty:
        rank_mean = 0.0
        rank_std = 0.0
    else:
        rank_mean = float(rank_clean.mean())
        rank_std = float(rank_clean.std(ddof=0))
    valid_feature = abs(ic_mean) > 0.01 and abs(ic_ir) > 0.5
    return {
        "IC_mean": round(ic_mean, 6),
        "IC_std": round(ic_std, 6),
        "IC_IR": round(ic_ir, 6),
        "Rank_IC_mean": round(rank_mean, 6),
        "Rank_IC_std": round(rank_std, 6),
        "positive_ratio": round(positive_ratio, 6),
        "valid_feature": bool(valid_feature),
    }


def compute_feature_correlation(panel: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    if not feature_cols:
        return pd.DataFrame()
    existing = [c for c in feature_cols if c in panel.columns]
    if not existing:
        return pd.DataFrame()
    return panel[existing].corr(method="pearson").round(6)


def generate_report(
    panel: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    load_stats: Dict[str, int],
) -> Tuple[Dict[str, object], pd.DataFrame]:
    feature_metrics: Dict[str, Dict[str, float]] = {}
    series_frames: List[pd.DataFrame] = []
    for feature in feature_cols:
        if feature not in panel.columns:
            continue
        ic_series = compute_daily_ic(panel, feature)
        rank_ic_series = compute_rank_ic(panel, feature)
        feature_metrics[feature] = aggregate_ic_statistics(ic_series, rank_ic_series)
        date_index = sorted(set(ic_series.index.tolist()) | set(rank_ic_series.index.tolist()))
        if date_index:
            ts = pd.DataFrame({"date": date_index})
            ts["feature"] = feature
            ts["IC"] = ts["date"].map(ic_series.to_dict())
            ts["Rank_IC"] = ts["date"].map(rank_ic_series.to_dict())
            series_frames.append(ts)
    if series_frames:
        ic_timeseries = pd.concat(series_frames, ignore_index=True)
        ic_timeseries = ic_timeseries.sort_values(["date", "feature"]).reset_index(drop=True)
    else:
        ic_timeseries = pd.DataFrame(columns=["date", "feature", "IC", "Rank_IC"])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    symbols = panel["symbol"].astype(str).nunique() if "symbol" in panel.columns else 0
    report: Dict[str, object] = {
        "run_meta": {
            "generated_at": now,
            "horizon": int(horizon),
            "symbol_count": int(symbols),
            "feature_count": int(len(feature_metrics)),
            "input_stats": load_stats,
        },
        "horizon": int(horizon),
        "features": feature_metrics,
    }
    return report, ic_timeseries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir",
        default="dataset/trainable_universe/A_stock/2016-2025",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--report-dir",
        default="core/alpha_research/feature_ic_reports",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    features_dir = Path(args.features_dir)
    report_dir = Path(args.report_dir)
    panel, feature_cols, load_stats = load_feature_data(
        features_dir=features_dir,
        horizon=args.horizon,
        max_symbols=args.max_symbols,
    )
    report, ic_timeseries = generate_report(
        panel=panel,
        feature_cols=feature_cols,
        horizon=args.horizon,
        load_stats=load_stats,
    )
    corr = compute_feature_correlation(panel, feature_cols)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "feature_ic_report.json"
    timeseries_path = report_dir / "feature_ic_timeseries.parquet"
    corr_path = report_dir / "feature_corr_matrix.csv"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    ic_timeseries.to_parquet(timeseries_path, index=False)
    corr.to_csv(corr_path, encoding="utf-8")
    print(f"feature_ic_report written to {report_path}")
    print(f"feature_ic_timeseries written to {timeseries_path}")
    print(f"feature_corr_matrix written to {corr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
