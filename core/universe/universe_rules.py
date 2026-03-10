from __future__ import annotations

from typing import Dict


def extract_symbol_stats(metrics: Dict[str, object]) -> Dict[str, float]:
    time_integrity = metrics.get("time_integrity", {}) if isinstance(metrics, dict) else {}
    feature_health = metrics.get("feature_health", {}) if isinstance(metrics, dict) else {}
    returns_shape = metrics.get("returns_shape", {}) if isinstance(metrics, dict) else {}
    extreme_values = metrics.get("extreme_values", {}) if isinstance(metrics, dict) else {}

    max_gap_days = int(time_integrity.get("max_gap_days", 0) or 0)
    missing_trading_days = int(time_integrity.get("missing_trading_days", 0) or 0)
    nan_ratio = float(feature_health.get("nan_ratio", 0) or 0)
    inf_count = int(feature_health.get("inf_count", 0) or 0)
    zero_return_ratio = float(returns_shape.get("zero_return_ratio", 0) or 0)
    extreme_events = int(
        int(extreme_values.get("pct_change_outliers", 0) or 0)
        + int(extreme_values.get("volume_outliers", 0) or 0)
        + int(extreme_values.get("amount_outliers", 0) or 0)
    )
    return {
        "max_gap_days": max_gap_days,
        "missing_trading_days": missing_trading_days,
        "nan_ratio": nan_ratio,
        "inf_count": inf_count,
        "zero_return_ratio": zero_return_ratio,
        "extreme_events": extreme_events,
    }


def filter_reasons(stats: Dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if stats.get("max_gap_days", 0) > 30:
        reasons.append("max_gap_days")
    if stats.get("missing_trading_days", 0) > 60:
        reasons.append("missing_trading_days")
    if stats.get("nan_ratio", 0) > 0.05:
        reasons.append("nan_ratio")
    if stats.get("inf_count", 0) > 0:
        reasons.append("inf_count")
    if stats.get("extreme_events", 0) > 200:
        reasons.append("extreme_events")
    if stats.get("zero_return_ratio", 0) > 0.5:
        reasons.append("zero_return_ratio")
    return reasons


def compute_quality_score(stats: Dict[str, float]) -> float:
    max_gap_days = float(stats.get("max_gap_days", 0))
    missing_trading_days = float(stats.get("missing_trading_days", 0))
    nan_ratio = float(stats.get("nan_ratio", 0))
    extreme_events = float(stats.get("extreme_events", 0))

    gap_penalty = min(max_gap_days / 30, 1)
    missing_penalty = min(missing_trading_days / 60, 1)
    nan_penalty = min(nan_ratio / 0.05, 1)
    extreme_penalty = min(extreme_events / 200, 1)

    total_penalty = (
        40 * gap_penalty
        + 20 * missing_penalty
        + 30 * nan_penalty
        + 10 * extreme_penalty
    )
    return max(0.0, 100 - total_penalty)


def default_filters(score_threshold: int, max_symbols: int) -> Dict[str, float]:
    return {
        "max_gap_days": 30,
        "missing_trading_days": 60,
        "nan_ratio": 0.05,
        "inf_count": 0,
        "extreme_events": 200,
        "zero_return_ratio": 0.5,
        "score_threshold": score_threshold,
        "max_symbols": max_symbols,
    }
