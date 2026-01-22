import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_canonical_columns(canonical_dir: Path) -> List[str]:
    files = sorted(canonical_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no canonical parquet files in {canonical_dir}")
    sample = pd.read_parquet(files[0])
    return list(sample.columns)


def compute_time_integrity(df: pd.DataFrame) -> Dict[str, int]:
    total_days = int(len(df))
    if total_days <= 1 or "date" not in df.columns:
        return {
            "total_days": total_days,
            "missing_trading_days": 0,
            "max_gap_days": 0,
        }
    d = pd.to_datetime(df["date"]).sort_values()
    diff = d.diff().dt.days
    if diff.empty:
        missing_trading_days = 0
        max_gap_days = 0
    else:
        gaps = diff[diff > 3]
        if gaps.empty:
            missing_trading_days = 0
            max_gap_days = 0
        else:
            gap_sizes = gaps - 1
            missing_trading_days = int(gap_sizes.sum())
            max_gap_days = int(gap_sizes.max())
    return {
        "total_days": total_days,
        "missing_trading_days": missing_trading_days,
        "max_gap_days": max_gap_days,
    }


def compute_returns_shape(df: pd.DataFrame) -> Dict[str, float]:
    if "ret_1d" not in df.columns:
        return {
            "zero_return_ratio": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
        }
    s = df["ret_1d"].dropna()
    if s.empty:
        return {
            "zero_return_ratio": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
        }
    zero_ratio = float((s == 0).sum() / len(s))
    skew = float(s.skew())
    kurt = float(s.kurtosis())
    return {
        "zero_return_ratio": zero_ratio,
        "skew": skew,
        "kurtosis": kurt,
    }


def compute_feature_health(
    df: pd.DataFrame, canonical_cols: List[str]
) -> Dict[str, object]:
    canonical_set = set(canonical_cols)
    feature_cols = [c for c in df.columns if c not in canonical_set]
    if not feature_cols:
        return {
            "nan_ratio": 0.0,
            "inf_count": 0,
            "unstable_features": [],
        }
    feature_df = df[feature_cols]
    total_values = feature_df.shape[0] * feature_df.shape[1]
    if total_values == 0:
        nan_ratio = 0.0
    else:
        nan_ratio = float(feature_df.isna().values.sum() / total_values)
    inf_count = int(np.isinf(feature_df.to_numpy()).sum())
    unstable_features = detect_unstable_features(feature_df, df)
    return {
        "nan_ratio": nan_ratio,
        "inf_count": inf_count,
        "unstable_features": unstable_features,
    }


def detect_unstable_features(
    feature_df: pd.DataFrame, full_df: pd.DataFrame
) -> List[str]:
    if "date" not in full_df.columns:
        return []
    time_feature_patterns = [
        "time__days_since_listed",
        "time__dow",
        "time__day_of_week",
        "time__month",
        "time__year",
    ]
    dates = pd.to_datetime(full_df["date"])
    if dates.empty:
        return []
    bounds: List[Tuple[str, str, str]] = [
        ("2016-01-01", "2018-12-31", "2016-2018"),
        ("2019-01-01", "2021-12-31", "2019-2021"),
        ("2022-01-01", "2024-12-31", "2022-2024"),
    ]
    unstable: List[str] = []
    for col in feature_df.columns:
        name = str(col)
        if any(pat in name for pat in time_feature_patterns):
            continue
        s_all = feature_df[col].astype(float)
        s_all = s_all.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s_all) < 100:
            continue
        global_std = float(s_all.std(ddof=0))
        if not np.isfinite(global_std) or global_std == 0.0:
            continue
        means: List[float] = []
        for start, end, _label in bounds:
            mask = (dates >= start) & (dates <= end)
            seg = feature_df.loc[mask, col].astype(float)
            seg = seg.replace([np.inf, -np.inf], np.nan).dropna()
            if len(seg) < 20:
                continue
            means.append(float(seg.mean()))
        if len(means) < 2:
            continue
        max_diff = max(abs(a - b) for a in means for b in means)
        if max_diff > 1.5 * global_std:
            unstable.append(col)
    return sorted(set(unstable))


def compute_cross_section_outliers(
    panel: pd.DataFrame, z_thresh: float = 8.0
) -> Tuple[Dict[str, int], Dict[str, int]]:
    pct_counts: Dict[str, int] = {}
    vol_counts: Dict[str, int] = {}
    if panel.empty:
        return pct_counts, vol_counts
    for date, g in panel.groupby("date"):
        for col, acc in (("pct_change", pct_counts), ("volume", vol_counts)):
            if col not in g.columns:
                continue
            s = g[col].astype(float)
            s = s.replace([np.inf, -np.inf], np.nan)
            if s.notna().sum() < 10:
                continue
            mean = float(s.mean())
            std = float(s.std(ddof=0))
            if std == 0.0 or not np.isfinite(std):
                continue
            z = (s - mean) / std
            mask = z.abs() > z_thresh
            if not mask.any():
                continue
            syms = g.loc[mask, "symbol"]
            for sym in syms:
                acc[sym] = acc.get(sym, 0) + 1
    return pct_counts, vol_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir",
        default="dataset/features/A_stock/2016-2025",
    )
    parser.add_argument(
        "--canonical-dir",
        default="dataset/canonical/A_stock/2016-2025",
    )
    parser.add_argument(
        "--report-dir",
        default="data_quality/dq_feature_reports",
    )
    parser.add_argument(
        "--pipeline-stage",
        default="feature_full",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="limit number of symbols for debugging; 0 means all",
    )
    args = parser.parse_args()
    features_dir = Path(args.features_dir)
    canonical_dir = Path(args.canonical_dir)
    report_root = Path(args.report_dir)
    feature_files = sorted(features_dir.glob("*.parquet"))
    if not feature_files:
        raise FileNotFoundError(f"no feature parquet files found in {features_dir}")
    if args.max_symbols > 0:
        feature_files = feature_files[: args.max_symbols]
    canonical_cols = load_canonical_columns(canonical_dir)
    symbol_metrics: Dict[str, Dict[str, object]] = {}
    panel_records: List[Dict[str, object]] = []
    date_min: datetime | None = None
    date_max: datetime | None = None
    for path in feature_files:
        sym = path.stem
        df = pd.read_parquet(path)
        if "date" not in df.columns or "symbol" not in df.columns:
            continue
        df = df.sort_values("date")
        d = pd.to_datetime(df["date"])
        if not d.empty:
            d_min = d.iloc[0]
            d_max = d.iloc[-1]
            if date_min is None or d_min < date_min:
                date_min = d_min
            if date_max is None or d_max > date_max:
                date_max = d_max
        time_integrity = compute_time_integrity(df)
        returns_shape = compute_returns_shape(df)
        feature_health = compute_feature_health(df, canonical_cols)
        symbol_metrics[sym] = {
            "time_integrity": time_integrity,
            "returns_shape": returns_shape,
            "extreme_values": {
                "pct_change_outliers": 0,
                "volume_outliers": 0,
            },
            "feature_health": feature_health,
            "quality_tags": [],
        }
        cols = []
        for col in ("symbol", "date", "pct_change", "volume", "ret_1d"):
            if col in df.columns:
                cols.append(col)
        if set(("symbol", "date")) <= set(cols):
            sub = df[cols].copy()
            panel_records.append(sub)
    if panel_records:
        panel = pd.concat(panel_records, ignore_index=True)
    else:
        panel = pd.DataFrame(columns=["symbol", "date", "pct_change", "volume", "ret_1d"])
    pct_counts, vol_counts = compute_cross_section_outliers(panel)
    for sym, count in pct_counts.items():
        if sym in symbol_metrics:
            symbol_metrics[sym]["extreme_values"]["pct_change_outliers"] = int(count)
    for sym, count in vol_counts.items():
        if sym in symbol_metrics:
            symbol_metrics[sym]["extreme_values"]["volume_outliers"] = int(count)
    flags_index: Dict[str, List[str]] = {}
    unstable_feature_names: set[str] = set()
    for sym, metrics in symbol_metrics.items():
        ti = metrics["time_integrity"]
        rs = metrics["returns_shape"]
        ev = metrics["extreme_values"]
        fh = metrics["feature_health"]
        tags: List[str] = []
        max_gap = ti.get("max_gap_days", 0)
        if max_gap > 10:
            tags.append("TIME_GAP_LARGE")
        elif max_gap > 0:
            tags.append("TIME_GAP_MEDIUM")
        kurt = rs.get("kurtosis", 0.0) or 0.0
        if kurt > 10.0:
            tags.append("RET_KURTOSIS_HIGH")
        zero_ratio = rs.get("zero_return_ratio", 0.0) or 0.0
        if zero_ratio > 0.3:
            tags.append("ZERO_RETURN_HIGH")
        unstable = fh.get("unstable_features", []) or []
        if unstable:
            tags.append("FEATURE_UNSTABLE")
            unstable_feature_names.update(unstable)
        outlier_count = int(ev.get("pct_change_outliers", 0)) + int(
            ev.get("volume_outliers", 0)
        )
        if outlier_count > 10:
            tags.append("EXTREME_EVENT_DENSE")
        metrics["quality_tags"] = tags
        for tag in tags:
            flags_index.setdefault(tag, []).append(sym)
    for tag in list(flags_index.keys()):
        flags_index[tag] = sorted(flags_index[tag])
    if panel.empty or "ret_1d" not in panel.columns:
        ret_summary = {
            "ret_1d": {
                "mean": 0.0,
                "std": 0.0,
                "skew": 0.0,
                "kurtosis": 0.0,
            }
        }
    else:
        s_ret = panel["ret_1d"].dropna()
        if s_ret.empty:
            ret_summary = {
                "ret_1d": {
                    "mean": 0.0,
                    "std": 0.0,
                    "skew": 0.0,
                    "kurtosis": 0.0,
                }
            }
        else:
            ret_summary = {
                "ret_1d": {
                    "mean": float(s_ret.mean()),
                    "std": float(s_ret.std(ddof=0)),
                    "skew": float(s_ret.skew()),
                    "kurtosis": float(s_ret.kurtosis()),
                }
            }
    max_gap_values = [m["time_integrity"]["max_gap_days"] for m in symbol_metrics.values()]
    if max_gap_values:
        max_gap_days = int(max(max_gap_values))
        median_gap_days = float(np.median(max_gap_values))
    else:
        max_gap_days = 0
        median_gap_days = 0.0
    symbols_with_large_gaps = sum(1 for v in max_gap_values if v > 10)
    global_summary = {
        "time_integrity": {
            "symbols_with_large_gaps": int(symbols_with_large_gaps),
            "max_gap_days": max_gap_days,
            "median_gap_days": median_gap_days,
        },
        "returns_distribution": ret_summary,
        "extreme_events": {
            "cross_section_outliers": int(
                sum(pct_counts.values()) + sum(vol_counts.values())
            )
        },
        "feature_stability": {
            "checked_windows": ["2016-2018", "2019-2021", "2022-2024"],
            "unstable_features": sorted(unstable_feature_names),
        },
    }
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y-%m-%dT%H:%M:%SZ_feature_full_v1")
    if date_min is None or date_max is None:
        data_window = {
            "start": None,
            "end": None,
        }
    else:
        data_window = {
            "start": date_min.strftime("%Y-%m-%d"),
            "end": date_max.strftime("%Y-%m-%d"),
        }
    run_meta = {
        "run_id": run_id,
        "pipeline_stage": args.pipeline_stage,
        "data_window": data_window,
        "input_universe_size": int(len(feature_files)),
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    report = {
        "run_meta": run_meta,
        "global_summary": global_summary,
        "symbol_metrics": symbol_metrics,
        "flags_index": flags_index,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    run_dir = report_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    out_path = run_dir / "dq_feature_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"dq_feature_report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
