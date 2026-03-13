from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_build.canonicalize_ashare import canonicalize_ashare_df


RAW_DIR = ROOT / "dataset/raw_data/A_stock/2016-2025"


def gate1_deterministic() -> None:
    raw_path = RAW_DIR / "000001.csv"
    if not raw_path.exists():
        raise AssertionError(f"raw file not found: {raw_path}")

    df_raw = pd.read_csv(raw_path)

    can1, sum1 = canonicalize_ashare_df(df_raw)
    can2, sum2 = canonicalize_ashare_df(df_raw)

    if can1.shape[0] != can2.shape[0]:
        raise AssertionError(f"row_count_mismatch: {can1.shape[0]} vs {can2.shape[0]}")

    if list(can1.columns) != list(can2.columns):
        raise AssertionError("columns_order_mismatch")

    if not can1.isna().sum().equals(can2.isna().sum()):
        raise AssertionError("nan_count_mismatch")

    if not can1.isna().equals(can2.isna()):
        raise AssertionError("nan_mask_mismatch")

    keys1 = set(zip(can1["date"], can1["symbol"]))
    keys2 = set(zip(can2["date"], can2["symbol"]))
    if keys1 != keys2:
        raise AssertionError("primary_key_set_mismatch")

    num_cols = [
        c
        for c in can1.columns
        if pd.api.types.is_numeric_dtype(can1[c]) and not pd.api.types.is_bool_dtype(can1[c])
    ]
    for c in num_cols:
        s1 = can1[c]
        s2 = can2[c]
        mask = s1.notna() & s2.notna()
        diff = (s1[mask] - s2[mask]).to_numpy()
        if diff.size > 0:
            max_abs = float(np.max(np.abs(diff)))
        else:
            max_abs = 0.0
        if not np.allclose(s1[mask].to_numpy(), s2[mask].to_numpy(), rtol=1e-10, atol=1e-12):
            raise AssertionError(f"numeric_mismatch_{c}_max_abs={max_abs}")

    if sum1 != sum2:
        raise AssertionError(f"summary_mismatch: {sum1} vs {sum2}")


def gate2_invalid_input() -> None:
    df_dup = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": ["2020-01-02", "2020-01-02"],
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.5, 9.5],
            "close": [10.5, 10.5],
            "volume": [1000, 2000],
        }
    )
    duplicate_ok = False
    try:
        canonicalize_ashare_df(df_dup)
    except ValueError:
        duplicate_ok = True
    if not duplicate_ok:
        raise AssertionError("duplicate_key_not_rejected")

    df_ohlc = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": ["2020-01-03", "2020-01-04"],
            "open": [10.0, 10.0],
            "high": [11.0, 9.0],
            "low": [9.5, 9.0],
            "close": [10.5, 9.5],
            "volume": [1000, 1000],
        }
    )
    can_ohlc, sum_ohlc = canonicalize_ashare_df(df_ohlc)
    if can_ohlc.shape[0] != 1:
        raise AssertionError(f"ohlc_invalid_row_not_dropped: rows={can_ohlc.shape[0]}")
    if sum_ohlc["invalid_rows"] < 1:
        raise AssertionError("ohlc_invalid_not_counted")

    df_act = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": ["2020-01-05", "2020-01-06"],
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.5, 9.5],
            "close": [10.5, 10.5],
            "volume": [1000, -100],
            "amount": [10500.0, 1000.0],
        }
    )
    can_act, sum_act = canonicalize_ashare_df(df_act)
    if can_act.shape[0] != 1:
        raise AssertionError(f"activity_invalid_row_not_dropped: rows={can_act.shape[0]}")
    if sum_act["invalid_rows"] < 1:
        raise AssertionError("activity_invalid_not_counted")


def gate3_small_batch() -> None:
    if not RAW_DIR.exists():
        raise AssertionError(f"raw dir not found: {RAW_DIR}")

    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise AssertionError("no raw files found for Gate3")

    sample_files = files[: min(50, len(files))]

    frames = []
    for p in sample_files:
        try:
            df_raw = pd.read_csv(p)
        except Exception:
            continue
        if df_raw.empty:
            continue
        can_df, _ = canonicalize_ashare_df(df_raw)
        frames.append(can_df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        raise AssertionError("combined canonical output is empty")

    if combined.duplicated(subset=["date", "symbol"]).any():
        raise AssertionError("duplicate_keys_in_combined_panel")

    required_no_nan = ["symbol", "date", "open", "high", "low", "close", "volume"]
    for c in required_no_nan:
        if combined[c].isna().any():
            raise AssertionError(f"unexpected_nan_in_required_field: {c}")

    for c in ["dow", "month", "is_month_end", "days_since_listed", "is_trading_day"]:
        if combined[c].isna().any():
            raise AssertionError(f"unexpected_nan_in_lifecycle_field: {c}")


def main() -> int:
    gates = [
        ("Gate1_Deterministic", gate1_deterministic),
        ("Gate2_InvalidInput", gate2_invalid_input),
        ("Gate3_SmallBatch", gate3_small_batch),
    ]

    failures: list[tuple[str, str]] = []

    for name, fn in gates:
        try:
            fn()
            print(f"{name}: PASS")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"{name}: FAIL: {exc}")

    if failures:
        for name, msg in failures:
            print(f"[Result] {name} -> FAIL: {msg}")
        return 1

    print("[Result] All canonicalize gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
