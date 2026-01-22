from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    std = std.replace(0, np.nan)
    return (series - mean) / std


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def _rsi_native(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rsi = pd.Series(np.nan, index=series.index, dtype=float)
    if len(series) < window + 1:
        return rsi
    ag = avg_gain.iloc[window]
    al = avg_loss.iloc[window]
    for i in range(window, len(series)):
        if i > window:
            g = gain.iloc[i]
            l = loss.iloc[i]
            ag = (ag * (window - 1) + g) / window
            al = (al * (window - 1) + l) / window
        if al == 0:
            if ag > 0:
                rsi.iloc[i] = 100.0
            else:
                rsi.iloc[i] = 0.0
        else:
            rs = ag / al
            rsi.iloc[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr_native(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr = _true_range(high, low, close)
    atr = pd.Series(np.nan, index=high.index, dtype=float)
    if len(tr) < window + 1:
        return atr
    atr_initial = tr.rolling(window=window, min_periods=window).mean()
    prev_atr = atr_initial.iloc[window]
    atr.iloc[window] = prev_atr
    for i in range(window + 1, len(tr)):
        prev_atr = (prev_atr * (window - 1) + tr.iloc[i]) / window
        atr.iloc[i] = prev_atr
    return atr


def _adx_native(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    high_shift = high.shift(1)
    low_shift = low.shift(1)
    plus_dm = high - high_shift
    minus_dm = low_shift - low
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)
    tr = _true_range(high, low, close)
    tr_smoothed = pd.Series(np.nan, index=high.index, dtype=float)
    plus_dm_smoothed = pd.Series(np.nan, index=high.index, dtype=float)
    minus_dm_smoothed = pd.Series(np.nan, index=high.index, dtype=float)
    if len(tr) < window + 1:
        return pd.Series(np.nan, index=high.index, dtype=float)
    tr_initial = tr.rolling(window=window, min_periods=window).sum()
    plus_dm_initial = plus_dm.rolling(window=window, min_periods=window).sum()
    minus_dm_initial = minus_dm.rolling(window=window, min_periods=window).sum()
    tr_prev = tr_initial.iloc[window]
    plus_dm_prev = plus_dm_initial.iloc[window]
    minus_dm_prev = minus_dm_initial.iloc[window]
    tr_smoothed.iloc[window] = tr_prev
    plus_dm_smoothed.iloc[window] = plus_dm_prev
    minus_dm_smoothed.iloc[window] = minus_dm_prev
    for i in range(window + 1, len(tr)):
        tr_prev = tr_prev - (tr_prev / window) + tr.iloc[i]
        plus_dm_prev = plus_dm_prev - (plus_dm_prev / window) + plus_dm.iloc[i]
        minus_dm_prev = minus_dm_prev - (minus_dm_prev / window) + minus_dm.iloc[i]
        tr_smoothed.iloc[i] = tr_prev
        plus_dm_smoothed.iloc[i] = plus_dm_prev
        minus_dm_smoothed.iloc[i] = minus_dm_prev
    plus_di = 100.0 * (plus_dm_smoothed / tr_smoothed)
    minus_di = 100.0 * (minus_dm_smoothed / tr_smoothed)
    denom = (plus_di + minus_di).abs()
    denom = denom.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    adx = pd.Series(np.nan, index=high.index, dtype=float)
    dx_initial = dx.rolling(window=window, min_periods=window).mean()
    prev_adx = dx_initial.iloc[window]
    adx.iloc[window] = prev_adx
    for i in range(window + 1, len(dx)):
        prev_adx = ((prev_adx * (window - 1)) + dx.iloc[i]) / window
        adx.iloc[i] = prev_adx
    return adx


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
        "vwap",
        "turnover",
        "dow",
        "month",
        "is_month_end",
        "days_since_listed",
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
        rsi_series = _rsi_native(close_series, 14)
        adx_series = _adx_native(high_series, low_series, close_series, 14)
        atr_series = _atr_native(high_series, low_series, close_series, 14)
        rsi_parts.append(rsi_series)
        adx_parts.append(adx_series)
        atr_parts.append(atr_series)
        atr_z = _rolling_zscore(atr_series, 60)
        atr_z_parts.append(atr_z)
        turnover_z = _rolling_zscore(turnover, 60)
        turnover_z_parts.append(turnover_z)
        middle = close_series.rolling(window=20, min_periods=20).mean()
        middle = middle.replace(0, np.nan)
        band_std = close_series.rolling(window=20, min_periods=20).std(ddof=0)
        band_std = band_std.replace(0, np.nan)
        bb_width = (4.0 * band_std) / middle
        close_off = (close_series - middle) / band_std
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
    out["liq__turnover"] = out["turnover"]
    turnover_z_all = pd.concat(turnover_z_parts).sort_index()
    out["liq__turnover_z__60"] = turnover_z_all
    vwap = out["vwap"]
    close_all = out["close"]
    out["liq__close_vwap_dev"] = np.where(
        vwap > 0,
        (close_all - vwap) / vwap,
        np.nan,
    )
    out["trn__turnover_z_delta__1"] = out.groupby("symbol")["liq__turnover_z__60"].diff(1)
    out["time__dow"] = out["dow"]
    out["time__month"] = out["month"]
    out["time__is_month_end"] = out["is_month_end"].astype("int64")
    out["time__days_since_listed"] = out["days_since_listed"]
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
        default="dataset/canonical/A_stock/2016-2025",
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
    error_log_path = Path("logs") / "feature_factory_errors.log"
    logger = logging.getLogger("feature_factory")
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
        print("FeatureFactory: no canonical files found")
        return 0
    print(f"FeatureFactory: start transform {total} symbols")
    failed = False
    for idx, canonical_path in enumerate(canonical_files, start=1):
        sym = canonical_path.stem
        out_path = out_dir / f"{sym}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"FeatureFactory: {idx}/{total} {sym} skipped (exists)")
            continue
        print(f"FeatureFactory: {idx}/{total} {sym} transforming")
        try:
            transform_file(canonical_path, out_path)
        except Exception:
            failed = True
            logger.exception(
                f"failed to transform {canonical_path} -> {out_path}"
            )
            print(f"FeatureFactory: {idx}/{total} {sym} FAILED")
        else:
            print(f"FeatureFactory: {idx}/{total} {sym} done")
    if failed:
        print(f"FeatureFactory: completed with errors, see {error_log_path}")
        return 1
    print("FeatureFactory: all symbols completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
