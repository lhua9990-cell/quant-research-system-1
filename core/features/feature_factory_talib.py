from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import talib


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    std = std.replace(0, np.nan)
    return (series - mean) / std


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if "symbol" not in df.columns:
        raise ValueError("input must contain 'symbol' column")
    if "date" not in df.columns:
        raise ValueError("input must contain 'date' column")
    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vwap",
        "turnover",
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
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = df.copy()
    out = out.sort_values(["symbol", "date"])
    grouped = out.groupby("symbol", group_keys=False)
    out["ret__roc__20"] = grouped["close"].pct_change(20)
    out["ret__roc__60"] = grouped["close"].pct_change(60)
    rsi_parts: List[pd.Series] = []
    adx_parts: List[pd.Series] = []
    atr_parts: List[pd.Series] = []
    bb_width_parts: List[pd.Series] = []
    close_off_parts: List[pd.Series] = []
    turnover_z_parts: List[pd.Series] = []
    atr_z_parts: List[pd.Series] = []
    for _, sub in grouped:
        close_series = sub["close"].astype(float)
        high_series = sub["high"].astype(float)
        low_series = sub["low"].astype(float)
        turnover = sub["turnover"]
        close_np = close_series.to_numpy()
        high_np = high_series.to_numpy()
        low_np = low_series.to_numpy()
        rsi_np = talib.RSI(close_np, timeperiod=14)
        adx_np = talib.ADX(high_np, low_np, close_np, timeperiod=14)
        atr_np = talib.ATR(high_np, low_np, close_np, timeperiod=14)
        rsi_series = pd.Series(rsi_np, index=close_series.index)
        adx_series = pd.Series(adx_np, index=close_series.index)
        atr_series = pd.Series(atr_np, index=close_series.index)
        rsi_parts.append(rsi_series)
        adx_parts.append(adx_series)
        atr_parts.append(atr_series)
        atr_z = _rolling_zscore(atr_series, 60)
        atr_z_parts.append(atr_z)
        turnover_z = _rolling_zscore(turnover, 60)
        turnover_z_parts.append(turnover_z)
        upper_np, middle_np, lower_np = talib.BBANDS(
            close_np, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
        )
        upper = pd.Series(upper_np, index=close_series.index)
        middle = pd.Series(middle_np, index=close_series.index)
        lower = pd.Series(lower_np, index=close_series.index)
        middle = middle.replace(0, np.nan)
        band_span = upper - lower
        band_span = band_span.replace(0, np.nan)
        bb_width = band_span / middle
        std = band_span / 4.0
        std = std.replace(0, np.nan)
        close_off = (close_series - middle) / std
        bb_width_parts.append(bb_width)
        close_off_parts.append(close_off)
    out["ret__rsi__14"] = pd.concat(rsi_parts).sort_index()
    out["ret__adx__14"] = pd.concat(adx_parts).sort_index()
    out["rng__atr__14"] = pd.concat(atr_parts).sort_index()
    out["rng__atr_z__60"] = pd.concat(atr_z_parts).sort_index()
    out["rng__bb_width__20"] = pd.concat(bb_width_parts).sort_index()
    out["rng__close_off__20"] = pd.concat(close_off_parts).sort_index()
    out["trn__rsi_delta__1"] = out.groupby("symbol")["ret__rsi__14"].diff(1)
    out["trn__adx_delta__1"] = out.groupby("symbol")["ret__adx__14"].diff(1)
    out["trn__bb_width_delta__1"] = out.groupby("symbol")["rng__bb_width__20"].diff(1)
    turnover_z_all = pd.concat(turnover_z_parts).sort_index()
    out["liq__turnover_z__60"] = turnover_z_all
    vwap = out["vwap"]
    close_all = out["close"]
    out["liq__close_vwap_dev"] = np.where(
        vwap > 0,
        (close_all - vwap) / vwap,
        np.nan,
    )
    out["liq__turnover"] = out["turnover"] / 100.0
    out["trn__turnover_z_delta__1"] = out.groupby("symbol")["liq__turnover_z__60"].diff(1)
    round_cols = [
        "ret__roc__20",
        "ret__roc__60",
        "ret__rsi__14",
        "ret__adx__14",
        "rng__atr__14",
        "rng__atr_z__60",
        "rng__bb_width__20",
        "rng__close_off__20",
        "trn__rsi_delta__1",
        "trn__adx_delta__1",
        "trn__bb_width_delta__1",
        "liq__turnover_z__60",
        "liq__close_vwap_dev",
        "liq__turnover",
        "trn__turnover_z_delta__1",
    ]
    existing_round_cols = [c for c in round_cols if c in out.columns]
    out[existing_round_cols] = out[existing_round_cols].round(4)
    out = out.sort_values(["symbol", "date"])
    return out


def list_canonical_files(
    canonical_dir: Path, symbols: Optional[List[str]]
) -> List[Path]:
    if symbols:
        files: List[Path] = []
        for s in symbols:
            name = f"{s.strip().zfill(6)}.parquet"
            p = canonical_dir / name
            if p.exists():
                files.append(p)
        return files
    return sorted(p for p in canonical_dir.glob("*.parquet") if p.is_file())


def transform_file(canonical_path: Path, out_path: Path) -> None:
    df = pd.read_parquet(canonical_path)
    df_feat = compute_features(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-dir",
        default="dataset/canonical/A_stock/_exclude_negprice_qfq_2016-2025",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset/features/A_stock/2016-2025",
    )
    parser.add_argument("--symbols", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    canonical_dir = Path(args.canonical_dir)
    out_dir = Path(args.out_dir)
    error_log_path = Path("logs") / "feature_factory_talib_errors.log"
    logger = logging.getLogger("feature_factory_talib")
    logger.setLevel(logging.INFO)
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(error_log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s\t%(levelname)s\t%(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if args.symbols.strip():
        symbols = [s for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = None
    canonical_files = list_canonical_files(canonical_dir, symbols)
    total = len(canonical_files)
    if total == 0:
        print("FeatureFactoryTalib: no canonical files found")
        return 0
    print(f"FeatureFactoryTalib: start transform {total} symbols")
    failed = False
    for idx, canonical_path in enumerate(canonical_files, start=1):
        sym = canonical_path.stem
        out_path = out_dir / f"{sym}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"FeatureFactoryTalib: {idx}/{total} {sym} skipped (exists)")
            continue
        print(f"FeatureFactoryTalib: {idx}/{total} {sym} transforming")
        try:
            transform_file(canonical_path, out_path)
        except Exception:
            failed = True
            logger.exception(
                f"failed to transform {canonical_path} -> {out_path}"
            )
            print(f"FeatureFactoryTalib: {idx}/{total} {sym} FAILED")
        else:
            print(f"FeatureFactoryTalib: {idx}/{total} {sym} done")
    if failed:
        print(f"FeatureFactoryTalib: completed with errors, see {error_log_path}")
        return 1
    print("FeatureFactoryTalib: all symbols completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
