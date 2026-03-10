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


def compute_time_integrity(
    df: pd.DataFrame, trading_calendar: pd.DatetimeIndex | None = None
) -> Dict[str, int]:
    total_days = int(len(df))
    if total_days <= 1 or "date" not in df.columns:
        return {
            "total_days": total_days,
            "missing_trading_days": 0,
            "max_gap_days": 0,
            "large_gap_count": 0,
            "days_since_listed_violation_count": 0,
            "non_trading_day_rows": 0,
        }
    sorted_df = df.sort_values("date")
    d = pd.to_datetime(sorted_df["date"]).dropna()
    if d.empty or len(d) <= 1:
        missing_trading_days = 0
        max_gap_days = 0
        large_gap_count = 0
    else:
        if trading_calendar is not None and len(trading_calendar) > 0:
            start = d.iloc[0]
            end = d.iloc[-1]
            calendar = trading_calendar[(trading_calendar >= start) & (trading_calendar <= end)]
            if len(calendar) <= 1:
                missing_trading_days = 0
                max_gap_days = 0
                large_gap_count = 0
            else:
                calendar_index = pd.Index(calendar)
                sym_dates = pd.Index(d.unique()).sort_values()
                positions = calendar_index.get_indexer(sym_dates)
                valid_positions = positions[positions >= 0]
                if len(valid_positions) <= 1:
                    missing_trading_days = 0
                    max_gap_days = 0
                    large_gap_count = 0
                else:
                    gaps = np.diff(np.sort(valid_positions)) - 1
                    gaps = gaps[gaps > 0]
                    if len(gaps) == 0:
                        missing_trading_days = 0
                        max_gap_days = 0
                        large_gap_count = 0
                    else:
                        missing_trading_days = int(gaps.sum())
                        max_gap_days = int(gaps.max())
                        large_gap_count = int((gaps > 30).sum())
        else:
            diff = d.diff().dt.days
            if diff.empty or len(d) <= 1:
                missing_trading_days = 0
                max_gap_days = 0
                large_gap_count = 0
            else:
                estimated_missing = [max(0, int(v) - 1) for v in diff.iloc[1:].fillna(0)]
                if not estimated_missing:
                    missing_trading_days = 0
                    max_gap_days = 0
                    large_gap_count = 0
                else:
                    missing_trading_days = int(sum(estimated_missing))
                    max_gap_days = int(max(estimated_missing))
                    large_gap_count = int(sum(1 for v in estimated_missing if v > 30))
    if "days_since_listed" in sorted_df.columns:
        dsl = sorted_df["days_since_listed"].astype(float)
        dsl_diff = dsl.diff()
        date_diff = d.diff().dt.days
        valid = dsl_diff.notna() & date_diff.notna()
        dsl_viol = dsl_diff[valid].ne(date_diff[valid]).sum()
        days_since_listed_violation_count = int(dsl_viol)
    else:
        days_since_listed_violation_count = 0
    if "is_trading_day" in sorted_df.columns:
        non_trading_day_rows = int((sorted_df["is_trading_day"].astype(float) != 1.0).sum())
    elif trading_calendar is not None and len(trading_calendar) > 0 and not d.empty:
        calendar_index = pd.Index(trading_calendar)
        positions = calendar_index.get_indexer(pd.Index(d.unique()))
        non_trading_day_rows = int((positions < 0).sum())
    else:
        non_trading_day_rows = 0
    return {
        "total_days": total_days,
        "missing_trading_days": missing_trading_days,
        "max_gap_days": max_gap_days,
        "large_gap_count": large_gap_count,
        "days_since_listed_violation_count": days_since_listed_violation_count,
        "non_trading_day_rows": non_trading_day_rows,
    }


def compute_returns_shape(df: pd.DataFrame) -> Dict[str, float]:
    if "ret_1d" not in df.columns:
        return {
            "zero_return_ratio": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "logret_zero_return_ratio": 0.0,
            "logret_skew": 0.0,
            "logret_kurtosis": 0.0,
        }
    s = df["ret_1d"].dropna()
    if s.empty:
        return {
            "zero_return_ratio": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "logret_zero_return_ratio": 0.0,
            "logret_skew": 0.0,
            "logret_kurtosis": 0.0,
        }
    if "logret_1d" in df.columns:
        ls = df["logret_1d"].dropna()
    else:
        ls = pd.Series(dtype=float)
    zero_ratio = float((s == 0).sum() / len(s))
    skew = float(s.skew())
    kurt = float(s.kurtosis())
    if ls.empty:
        logret_zero_ratio = 0.0
        logret_skew = 0.0
        logret_kurt = 0.0
    else:
        logret_zero_ratio = float((ls == 0).sum() / len(ls))
        logret_skew = float(ls.skew())
        logret_kurt = float(ls.kurtosis())
    return {
        "zero_return_ratio": zero_ratio,
        "skew": skew,
        "kurtosis": kurt,
        "logret_zero_return_ratio": logret_zero_ratio,
        "logret_skew": logret_skew,
        "logret_kurtosis": logret_kurt,
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
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    pct_counts: Dict[str, int] = {}
    vol_counts: Dict[str, int] = {}
    amt_counts: Dict[str, int] = {}
    if panel.empty:
        return pct_counts, vol_counts, amt_counts
    for _, g in panel.groupby("date"):
        for col, acc in (
            ("pct_change", pct_counts),
            ("volume", vol_counts),
            ("amount", amt_counts),
        ):
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
    return pct_counts, vol_counts, amt_counts


def compute_prefix_invariance(
    feature_files: List[Path],
    canonical_dir: Path,
    canonical_cols: List[str],
    samples: int = 5,
) -> Dict[str, object]:
    try:
        from core.features.feature_factory_talib import compute_features
    except Exception:
        return {
            "checked_symbols": 0,
            "passed_symbols": 0,
            "failed_symbols": [],
        }
    if not feature_files:
        return {
            "checked_symbols": 0,
            "passed_symbols": 0,
            "failed_symbols": [],
        }
    candidate_files = sorted(feature_files)[: max(1, min(samples, len(feature_files)))]
    failed_symbols: List[str] = []
    checked = 0
    passed = 0
    canonical_set = set(canonical_cols)
    for f in candidate_files:
        sym = f.stem
        cpath = canonical_dir / f"{sym}.parquet"
        if not cpath.exists():
            continue
        raw_df = pd.read_parquet(cpath).sort_values(["symbol", "date"])
        n = len(raw_df)
        if n < 120:
            continue
        t0 = int(n * 0.7)
        if t0 < 60:
            continue
        checked += 1
        full_feat = compute_features(raw_df)
        trunc_feat = compute_features(raw_df.iloc[: t0 + 1].copy())
        feature_cols = [c for c in full_feat.columns if c not in canonical_set]
        full_row = full_feat.iloc[t0][feature_cols]
        trunc_row = trunc_feat.iloc[-1][feature_cols]
        same = full_row.equals(trunc_row)
        if same:
            passed += 1
        else:
            failed_symbols.append(sym)
    return {
        "checked_symbols": int(checked),
        "passed_symbols": int(passed),
        "failed_symbols": sorted(failed_symbols),
    }


def update_weak_ic_stats(
    df: pd.DataFrame,
    canonical_cols: List[str],
    ic_values: Dict[str, List[float]],
) -> None:
    if "ret_1d" not in df.columns:
        return
    target = df["ret_1d"].shift(-1)
    canonical_set = set(canonical_cols)
    feature_cols = [c for c in df.columns if c not in canonical_set and c != "ret_1d"]
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(target, errors="coerce")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 60:
            continue
        x_rank = x[valid].rank(method="average")
        y_rank = y[valid].rank(method="average")
        ic = x_rank.corr(y_rank, method="pearson")
        if pd.notna(ic) and np.isfinite(ic):
            ic_values.setdefault(col, []).append(float(ic))


def summarize_weak_ic(ic_values: Dict[str, List[float]]) -> Dict[str, object]:
    if not ic_values:
        return {
            "checked_features": 0,
            "all_near_zero": True,
            "suspicious_features": [],
            "median_rank_ic": {},
        }
    median_ic: Dict[str, float] = {}
    for col, vals in ic_values.items():
        if not vals:
            continue
        median_ic[col] = float(np.median(vals))
    if not median_ic:
        return {
            "checked_features": 0,
            "all_near_zero": True,
            "suspicious_features": [],
            "median_rank_ic": {},
        }
    suspicious = sorted([k for k, v in median_ic.items() if abs(v) >= 0.5])
    all_near_zero = all(abs(v) < 0.005 for v in median_ic.values())
    return {
        "checked_features": int(len(median_ic)),
        "all_near_zero": bool(all_near_zero),
        "suspicious_features": suspicious,
        "median_rank_ic": {k: round(v, 6) for k, v in sorted(median_ic.items())},
    }


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
        default="dq/dq_feature_reports",
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
    ic_values: Dict[str, List[float]] = {}
    date_min: datetime | None = None
    date_max: datetime | None = None
    for path in feature_files:
        df_dates = pd.read_parquet(path, columns=["date"])
        if "date" not in df_dates.columns:
            continue
        d = pd.to_datetime(df_dates["date"]).dropna()
        if d.empty:
            continue
        d_min = d.min()
        d_max = d.max()
        if date_min is None or d_min < date_min:
            date_min = d_min
        if date_max is None or d_max > date_max:
            date_max = d_max
    if date_min is None or date_max is None:
        trading_calendar = pd.DatetimeIndex([])
    else:
        from dq.calendar import load_trading_calendar

        trading_calendar = load_trading_calendar(
            date_min.strftime("%Y-%m-%d"),
            date_max.strftime("%Y-%m-%d"),
            exchange="XSHG",
        )
    for path in feature_files:
        sym = path.stem
        df = pd.read_parquet(path)
        if "date" not in df.columns or "symbol" not in df.columns:
            continue
        df = df.sort_values("date")
        time_integrity = compute_time_integrity(df, trading_calendar)
        returns_shape = compute_returns_shape(df)
        feature_health = compute_feature_health(df, canonical_cols)
        symbol_metrics[sym] = {
            "time_integrity": time_integrity,
            "returns_shape": returns_shape,
            "extreme_values": {
                "pct_change_outliers": 0,
                "volume_outliers": 0,
                "amount_outliers": 0,
            },
            "feature_health": feature_health,
            "quality_tags": [],
        }
        cols = []
        update_weak_ic_stats(df, canonical_cols, ic_values)
        for col in ("symbol", "date", "pct_change", "volume", "amount", "ret_1d", "logret_1d"):
            if col in df.columns:
                cols.append(col)
        if set(("symbol", "date")) <= set(cols):
            sub = df[cols].copy()
            panel_records.append(sub)
    if panel_records:
        panel = pd.concat(panel_records, ignore_index=True)
    else:
        panel = pd.DataFrame(
            columns=["symbol", "date", "pct_change", "volume", "amount", "ret_1d", "logret_1d"]
        )
    pct_counts, vol_counts, amt_counts = compute_cross_section_outliers(panel)
    for sym, count in pct_counts.items():
        if sym in symbol_metrics:
            symbol_metrics[sym]["extreme_values"]["pct_change_outliers"] = int(count)
    for sym, count in vol_counts.items():
        if sym in symbol_metrics:
            symbol_metrics[sym]["extreme_values"]["volume_outliers"] = int(count)
    for sym, count in amt_counts.items():
        if sym in symbol_metrics:
            symbol_metrics[sym]["extreme_values"]["amount_outliers"] = int(count)
    flags_index: Dict[str, List[str]] = {}
    unstable_feature_names: set[str] = set()
    for sym, metrics in symbol_metrics.items():
        ti = metrics["time_integrity"]
        rs = metrics["returns_shape"]
        ev = metrics["extreme_values"]
        fh = metrics["feature_health"]
        tags: List[str] = []
        max_gap = ti.get("max_gap_days", 0)
        large_gap_count = ti.get("large_gap_count", 0)
        dsl_viol = ti.get("days_since_listed_violation_count", 0)
        non_trading_rows = ti.get("non_trading_day_rows", 0)
        if max_gap > 60 or large_gap_count > 0:
            tags.append("TIME_GAP_LARGE")
        elif max_gap > 20 or dsl_viol > 0 or non_trading_rows > 0:
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
        ) + int(ev.get("amount_outliers", 0))
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
            },
            "logret_1d": {
                "mean": 0.0,
                "std": 0.0,
                "skew": 0.0,
                "kurtosis": 0.0,
            }
        }
    else:
        s_ret = panel["ret_1d"].dropna()
        if "logret_1d" in panel.columns:
            s_logret = panel["logret_1d"].dropna()
        else:
            s_logret = pd.Series(dtype=float)
        if s_ret.empty:
            ret_summary = {
                "ret_1d": {
                    "mean": 0.0,
                    "std": 0.0,
                    "skew": 0.0,
                    "kurtosis": 0.0,
                },
                "logret_1d": {
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
                },
                "logret_1d": {
                    "mean": float(s_logret.mean()) if not s_logret.empty else 0.0,
                    "std": float(s_logret.std(ddof=0)) if not s_logret.empty else 0.0,
                    "skew": float(s_logret.skew()) if not s_logret.empty else 0.0,
                    "kurtosis": float(s_logret.kurtosis()) if not s_logret.empty else 0.0,
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
    prefix_invariance = compute_prefix_invariance(
        feature_files=feature_files,
        canonical_dir=canonical_dir,
        canonical_cols=canonical_cols,
        samples=5,
    )
    weak_ic_sanity = summarize_weak_ic(ic_values)
    global_summary = {
        "time_integrity": {
            "symbols_with_large_gaps": int(symbols_with_large_gaps),
            "max_gap_days": max_gap_days,
            "median_gap_days": median_gap_days,
        },
        "returns_distribution": ret_summary,
        "extreme_events": {
            "cross_section_outliers": int(
                sum(pct_counts.values()) + sum(vol_counts.values()) + sum(amt_counts.values())
            )
        },
        "feature_stability": {
            "checked_windows": ["2016-2018", "2019-2021", "2022-2024"],
            "unstable_features": sorted(unstable_feature_names),
        },
        "prefix_invariance": prefix_invariance,
        "weak_ic_sanity": weak_ic_sanity,
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
