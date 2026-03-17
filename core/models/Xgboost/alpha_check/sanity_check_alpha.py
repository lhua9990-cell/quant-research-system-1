from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..dataset_utils import load_dataset
from ..model_config import (
    OUTPUT_DATASET,
    PREDICT_OUTPUT,
    SANITY_CHECK_MD,
    SANITY_CHECK_OUTPUT,
)


def load_panel() -> pd.DataFrame:
    if not PREDICT_OUTPUT.exists():
        raise FileNotFoundError(f"alpha predictions not found: {PREDICT_OUTPUT}")
    pred = pd.read_parquet(PREDICT_OUTPUT)
    need_pred = ["date", "symbol", "alpha"]
    missing_pred = [c for c in need_pred if c not in pred.columns]
    if missing_pred:
        raise ValueError(f"alpha predictions missing columns: {missing_pred}")
    ds = load_dataset(OUTPUT_DATASET)
    need_ds = ["date", "symbol", "future_return", "ret__roc__20"]
    missing_ds = [c for c in need_ds if c not in ds.columns]
    if missing_ds:
        raise ValueError(f"dataset missing columns: {missing_ds}")
    pred = pred[need_pred].copy()
    pred["date"] = pd.to_datetime(pred["date"])
    ds = ds[need_ds].copy()
    ds["date"] = pd.to_datetime(ds["date"])
    panel = pred.merge(ds, on=["date", "symbol"], how="inner")
    panel = panel[panel["date"] >= pd.Timestamp("2024-01-01")]
    return panel.dropna(subset=["alpha", "ret__roc__20", "future_return"])


def daily_corr(panel: pd.DataFrame, x_col: str, y_col: str, method: str) -> pd.Series:
    grouped = panel.groupby("date", sort=False)[[x_col, y_col]]
    series = grouped.apply(lambda x: x[x_col].corr(x[y_col], method=method))
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def summarize(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"mean": 0.0, "std": 0.0, "ir": 0.0, "win_rate": 0.0}
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    ir = float(mean / std) if std > 0 else 0.0
    win_rate = float((series > 0).mean())
    return {"mean": mean, "std": std, "ir": ir, "win_rate": win_rate}


def run_sanity_checks() -> dict[str, object]:
    panel = load_panel()
    rng = np.random.default_rng(42)
    random_alpha = rng.standard_normal(len(panel))
    shuffled_future_return = (
        panel["future_return"].sample(frac=1.0, random_state=42).reset_index(drop=True)
    )
    shuffled_panel = panel.reset_index(drop=True).copy()
    shuffled_panel["future_return_shuffled"] = shuffled_future_return

    feature_ic = daily_corr(panel, "ret__roc__20", "future_return", method="pearson")
    feature_rank_ic = daily_corr(panel, "ret__roc__20", "future_return", method="spearman")

    random_panel = panel.copy()
    random_panel["random_alpha"] = random_alpha
    random_ic = daily_corr(random_panel, "random_alpha", "future_return", method="spearman")

    shuffled_target_ic = daily_corr(
        shuffled_panel, "alpha", "future_return_shuffled", method="spearman"
    )

    report = {
        "run_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eval_start": "2024-01-01",
            "rows": int(len(panel)),
            "days": int(panel["date"].nunique()),
        },
        "single_feature_ic_ret__roc__20": {
            "IC": summarize(feature_ic),
            "Rank_IC": summarize(feature_rank_ic),
        },
        "random_alpha_rank_ic": summarize(random_ic),
        "model_alpha_vs_shuffled_future_return_rank_ic": summarize(shuffled_target_ic),
    }
    SANITY_CHECK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SANITY_CHECK_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Sanity Check Report",
        "",
        "## Rule",
        '- Test-only: `df[df.date >= "2024-01-01"]`',
        "",
        "## 1) Single Feature IC: ret__roc__20",
        f'- IC mean: {report["single_feature_ic_ret__roc__20"]["IC"]["mean"]:.6f}, IR: {report["single_feature_ic_ret__roc__20"]["IC"]["ir"]:.6f}, win_rate: {report["single_feature_ic_ret__roc__20"]["IC"]["win_rate"]:.2%}',
        f'- Rank IC mean: {report["single_feature_ic_ret__roc__20"]["Rank_IC"]["mean"]:.6f}, IR: {report["single_feature_ic_ret__roc__20"]["Rank_IC"]["ir"]:.6f}, win_rate: {report["single_feature_ic_ret__roc__20"]["Rank_IC"]["win_rate"]:.2%}',
        "",
        "## 2) Random Alpha",
        f'- Rank IC mean: {report["random_alpha_rank_ic"]["mean"]:.6f}, IR: {report["random_alpha_rank_ic"]["ir"]:.6f}, win_rate: {report["random_alpha_rank_ic"]["win_rate"]:.2%}',
        "",
        "## 3) Shuffled Target",
        f'- Rank IC mean: {report["model_alpha_vs_shuffled_future_return_rank_ic"]["mean"]:.6f}, IR: {report["model_alpha_vs_shuffled_future_return_rank_ic"]["ir"]:.6f}, win_rate: {report["model_alpha_vs_shuffled_future_return_rank_ic"]["win_rate"]:.2%}',
        "",
        "## Output",
        f"- {SANITY_CHECK_OUTPUT}",
    ]
    SANITY_CHECK_MD.parent.mkdir(parents=True, exist_ok=True)
    SANITY_CHECK_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print("sanity check saved:", SANITY_CHECK_OUTPUT)
    print("sanity check markdown saved:", SANITY_CHECK_MD)
    return report


if __name__ == "__main__":
    run_sanity_checks()
    raise SystemExit(0)
