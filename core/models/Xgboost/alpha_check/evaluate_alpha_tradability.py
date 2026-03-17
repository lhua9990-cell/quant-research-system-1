from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..dataset_utils import load_dataset
from ..model_config import OUTPUT_DATASET, PREDICT_OUTPUT, TOP_QUANTILE, VALID_END


def load_eval_panel() -> pd.DataFrame:
    if not PREDICT_OUTPUT.exists():
        raise FileNotFoundError(f"alpha predictions not found: {PREDICT_OUTPUT}")
    alpha = pd.read_parquet(PREDICT_OUTPUT)
    needed_alpha = ["date", "symbol", "alpha"]
    missing_alpha = [c for c in needed_alpha if c not in alpha.columns]
    if missing_alpha:
        raise ValueError(f"alpha predictions missing columns: {missing_alpha}")
    dataset = load_dataset(OUTPUT_DATASET)
    needed_dataset = ["date", "symbol", "future_return", "ret_1d", "turnover", "amount", "volume"]
    missing_dataset = [c for c in needed_dataset if c not in dataset.columns]
    if missing_dataset:
        raise ValueError(f"dataset missing columns: {missing_dataset}")
    alpha = alpha[needed_alpha].copy()
    alpha["date"] = pd.to_datetime(alpha["date"])
    dataset = dataset[needed_dataset].copy()
    dataset["date"] = pd.to_datetime(dataset["date"])
    panel = alpha.merge(dataset, on=["date", "symbol"], how="inner")
    panel = panel[panel["date"] >= VALID_END]
    return panel


def build_tradability_mask(
    panel: pd.DataFrame,
    price_limit: float = 0.095,
    min_turnover: float = 0.5,
    min_amount: float = 50_000_000.0,
) -> pd.Series:
    mask = pd.Series(True, index=panel.index)
    mask &= panel["ret_1d"].abs() < price_limit
    mask &= panel["volume"] > 0
    mask &= panel["turnover"] > min_turnover
    mask &= panel["amount"] > min_amount
    return mask


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
        rows.append(
            {
                "date": date,
                "rank_ic": rank_ic,
                "long_short_return": top["future_return"].mean() - bottom["future_return"].mean(),
                "top_quantile_return": top["future_return"].mean(),
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


def summarize(series: pd.Series) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"mean": 0.0, "std": 0.0, "IC_IR": 0.0, "win_rate": 0.0}
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    ir = float(mean / std) if std > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "IC_IR": ir,
        "win_rate": float((clean > 0).mean()),
    }


def collect_summary(daily: pd.DataFrame) -> dict[str, object]:
    if daily.empty:
        empty = summarize(pd.Series(dtype=float))
        return {"date_count": 0, "rank_ic": empty, "long_short": empty, "top_quantile": empty}
    return {
        "date_count": int(daily["date"].nunique()),
        "rank_ic": summarize(daily["rank_ic"]),
        "long_short": summarize(daily["long_short_return"]),
        "top_quantile": summarize(daily["top_quantile_return"]),
    }


def evaluate_tradability() -> dict[str, object]:
    panel = load_eval_panel()
    baseline_daily = compute_daily_metrics(panel)

    tradable_mask = build_tradability_mask(panel)
    tradable_panel = panel[tradable_mask]
    tradable_daily = compute_daily_metrics(tradable_panel)

    output_dir = PREDICT_OUTPUT.parent.parent / "alpha_evaluation"
    json_path = output_dir / "alpha_eval_tradability_validation.json"
    md_path = output_dir / "alpha_eval_tradability_validation.md"
    tradable_daily_path = output_dir / "alpha_eval_tradability_daily.parquet"

    report = {
        "run_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eval_start": str(pd.Timestamp(VALID_END).date()),
            "top_quantile": float(TOP_QUANTILE),
            "tradability_filter": {
                "abs_ret_1d_lt": 0.095,
                "volume_gt": 0,
                "turnover_gt": 0.5,
                "amount_gt": 50_000_000.0,
            },
        },
        "baseline": {
            "row_count": int(len(panel)),
            **collect_summary(baseline_daily),
        },
        "tradable": {
            "row_count": int(len(tradable_panel)),
            "row_keep_ratio": float(len(tradable_panel) / len(panel)) if len(panel) > 0 else 0.0,
            **collect_summary(tradable_daily),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not tradable_daily.empty:
        tradable_daily.to_parquet(tradable_daily_path, index=False)

    lines = [
        "# Alpha Tradability Validation",
        "",
        "## 过滤条件",
        "- |ret_1d| < 9.5%",
        "- volume > 0",
        "- turnover > 0.5",
        "- amount > 50,000,000",
        "",
        "## Baseline",
        f"- row_count: {report['baseline']['row_count']}",
        f"- date_count: {report['baseline']['date_count']}",
        "| Metric | Mean | Std | IC_IR | Win Rate |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Rank IC | {report['baseline']['rank_ic']['mean']:.6f} | {report['baseline']['rank_ic']['std']:.6f} | {report['baseline']['rank_ic']['IC_IR']:.6f} | {report['baseline']['rank_ic']['win_rate']:.2%} |",
        f"| Long-Short Return | {report['baseline']['long_short']['mean']:.6f} | {report['baseline']['long_short']['std']:.6f} | {report['baseline']['long_short']['IC_IR']:.6f} | {report['baseline']['long_short']['win_rate']:.2%} |",
        f"| Top Quantile Return | {report['baseline']['top_quantile']['mean']:.6f} | {report['baseline']['top_quantile']['std']:.6f} | {report['baseline']['top_quantile']['IC_IR']:.6f} | {report['baseline']['top_quantile']['win_rate']:.2%} |",
        "",
        "## Tradable",
        f"- row_count: {report['tradable']['row_count']}",
        f"- row_keep_ratio: {report['tradable']['row_keep_ratio']:.2%}",
        f"- date_count: {report['tradable']['date_count']}",
        "| Metric | Mean | Std | IC_IR | Win Rate |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Rank IC | {report['tradable']['rank_ic']['mean']:.6f} | {report['tradable']['rank_ic']['std']:.6f} | {report['tradable']['rank_ic']['IC_IR']:.6f} | {report['tradable']['rank_ic']['win_rate']:.2%} |",
        f"| Long-Short Return | {report['tradable']['long_short']['mean']:.6f} | {report['tradable']['long_short']['std']:.6f} | {report['tradable']['long_short']['IC_IR']:.6f} | {report['tradable']['long_short']['win_rate']:.2%} |",
        f"| Top Quantile Return | {report['tradable']['top_quantile']['mean']:.6f} | {report['tradable']['top_quantile']['std']:.6f} | {report['tradable']['top_quantile']['IC_IR']:.6f} | {report['tradable']['top_quantile']['win_rate']:.2%} |",
        "",
        "## 输出文件",
        f"- {json_path}",
        f"- {md_path}",
        f"- {tradable_daily_path}",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("tradability validation saved:", json_path)
    print("tradability validation markdown saved:", md_path)
    if not tradable_daily.empty:
        print("tradable daily timeseries saved:", tradable_daily_path)
    return report


if __name__ == "__main__":
    evaluate_tradability()
    raise SystemExit(0)
