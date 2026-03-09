from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, TextIO

import numpy as np
import pandas as pd


REQUIRED_RAW_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTIONAL_RAW_COLUMNS = [
    "amount",
    "amplitude",
    "pct_change",
    "change",
    "turnover",
]


CANONICAL_COLUMNS = [
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
]


def canonicalize_ashare_df(df_raw: pd.DataFrame):
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"missing required raw columns: {missing}")

    df = df_raw.copy()

    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["symbol", "date"])

    mask_notna = df[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
    mask_price_nonneg = (df["open"] >= 0) & (df["high"] >= 0) & (df["low"] >= 0) & (df["close"] >= 0)
    hi_max = np.maximum(df["open"].to_numpy(), df["close"].to_numpy())
    lo_min = np.minimum(df["open"].to_numpy(), df["close"].to_numpy())
    mask_price_logic = (df["high"].to_numpy() >= hi_max) & (df["low"].to_numpy() <= lo_min)
    mask_volume_nonneg = df["volume"] >= 0
    if "amount" in df.columns:
        mask_amount_nonneg = df["amount"].isna() | (df["amount"] >= 0)
    else:
        mask_amount_nonneg = pd.Series(True, index=df.index)

    mask_valid = mask_notna & mask_price_nonneg & mask_price_logic & mask_volume_nonneg & mask_amount_nonneg
    invalid_mask = ~mask_valid

    summary = {
        "total_rows": int(len(df)),
        "valid_rows": int(mask_valid.sum()),
        "invalid_rows": int(invalid_mask.sum()),
        "reasons": {
            "missing_ohlcv": int((~mask_notna).sum()),
            "price_nonneg": int((~mask_price_nonneg).sum()),
            "price_logic": int((~mask_price_logic).sum()),
            "volume_nonneg": int((~mask_volume_nonneg).sum()),
            "amount_nonneg": int((~mask_amount_nonneg).sum()),
        },
    }

    df = df.loc[mask_valid].copy()

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    dup_mask = df.duplicated(subset=["symbol", "date"], keep=False)
    if dup_mask.any():
        dup_keys = (
            df.loc[dup_mask, ["symbol", "date"]]
            .value_counts()
            .reset_index(name="count")
        )
        raise ValueError(
            f"duplicate (symbol, date) keys found: {dup_keys.to_dict(orient='records')}"
        )

    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    mask_prev = df["prev_close"] > 0
    ret_1d = df["close"] / df["prev_close"] - 1.0
    logret_1d = np.log(df["close"] / df["prev_close"])
    df["ret_1d"] = np.where(mask_prev, np.round(ret_1d, 4), np.nan)
    df["logret_1d"] = np.where(mask_prev, np.round(logret_1d, 4), np.nan)

    vol = df["volume"]
    if "amount" in df.columns:
        amt = df["amount"]
    else:
        amt = pd.Series(np.nan, index=df.index)
    df["vwap"] = np.where((vol > 0) & amt.notna(), amt / (vol * 100.0), np.nan)
    df["vwap"] = np.round(df["vwap"], 4)

    hl_range = df["high"] / df["low"] - 1.0
    oc_range = df["close"] / df["open"] - 1.0
    df["hl_range"] = np.where(df["low"] > 0, np.round(hl_range, 4), np.nan)
    df["oc_range"] = np.where(df["open"] > 0, np.round(oc_range, 4), np.nan)

    dt = df["date"]
    df["dow"] = dt.dt.weekday + 1
    df["month"] = dt.dt.month
    df["is_month_end"] = dt.dt.is_month_end

    first_date = df.groupby("symbol")["date"].transform("min")
    df["days_since_listed"] = (df["date"] - first_date).dt.days

    df["is_trading_day"] = (df["volume"] > 0) & (df["close"] > 0)

    for col in OPTIONAL_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    out = df[CANONICAL_COLUMNS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out, summary


def canonicalize_file(
    raw_path: Path, out_path: Path, fmt: str, log_f: TextIO | None = None
) -> None:
    try:
        df_raw = pd.read_csv(raw_path)
    except pd.errors.EmptyDataError:
        msg = f"[canonicalize] {raw_path.name} error: EmptyDataError - skipping"
        print(msg)
        if log_f is not None:
            log_f.write(msg + "\n")
        return
    except Exception as exc:
        msg = f"[canonicalize] {raw_path.name} error: {exc} - skipping"
        print(msg)
        if log_f is not None:
            log_f.write(msg + "\n")
        return
    df_can, summary = canonicalize_ashare_df(df_raw)
    if summary["invalid_rows"] > 0:
        msg = f"[canonicalize] {raw_path.name} discard_summary={summary}"
        print(msg)
        if log_f is not None:
            log_f.write(msg + "\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df_can.to_parquet(out_path, index=False)
    else:
        df_can.to_csv(out_path, index=False)


def list_raw_files(raw_dir: Path, symbols: Optional[List[str]]) -> List[Path]:
    if symbols:
        files = []
        for s in symbols:
            name = f"{s.strip().zfill(6)}.csv"
            p = raw_dir / name
            if p.exists():
                files.append(p)
        return files
    return sorted(raw_dir.glob("*.csv"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="dataset/raw_data/A_stock/2016-2025")
    parser.add_argument("--out-dir", default="dataset/canonical/A_stock/2016-2025")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.symbols.strip():
        symbols = [s for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = None

    raw_files = list_raw_files(raw_dir, symbols)

    suffix = ".parquet" if args.format == "parquet" else ".csv"

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"canonicalize_ashare_{out_dir.name}.log"

    with log_path.open("w", encoding="utf-8") as log_f:
        for raw_path in raw_files:
            sym = raw_path.stem
            out_path = out_dir / f"{sym}{suffix}"
            if out_path.exists() and not args.overwrite:
                continue
            canonicalize_file(raw_path, out_path, args.format, log_f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
