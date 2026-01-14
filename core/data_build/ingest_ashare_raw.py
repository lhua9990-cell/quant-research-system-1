#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Raw data ingestion ONLY.
# - Source: AkShare
# - Output: dataset/raw_data/A_stock/2016-2025/{symbol}.csv
# - No cleaning, no feature engineering, no canonical transform.

# Step 1: A-share raw daily data ingestion (AkShare)
# - Fetch A-share daily K-line data (qfq by default)
# - Save as symbol.csv under dataset/raw_data/A_stock/2016-2025/
#
# Usage examples:
#   python ingest_ashare_raw.py
#   python ingest_ashare_raw.py --start 20160101 --end 20251231 --adjust qfq
#   python ingest_ashare_raw.py --symbols 000001,600000,300750
#   python ingest_ashare_raw.py --limit 200
#
# AkShare 日线原始字段含义（与本文件中的英文列名对应）：
# - 振幅 (amplitude): (当日最高价 - 当日最低价) / 昨日收盘价 * 100
# - 涨跌幅 (pct_change): (当日收盘价 - 昨日收盘价) / 昨日收盘价 * 100
# - 涨跌额 (change): 当日收盘价 - 昨日收盘价
# - 换手率 (turnover): 当日成交量 / 流通股本 * 100

from __future__ import annotations

import argparse
import os
import sys
import time
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter("ignore", SettingWithCopyWarning)

try:
    import akshare as ak
except Exception as e:
    print("ERROR: akshare is not installed. Try: pip install akshare", file=sys.stderr)
    raise


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    # Atomic-ish write
    tmp_path = out_path.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    tmp_path.replace(out_path)


def get_all_ashare_symbols() -> List[str]:
    """
    Pull A-share spot list from Eastmoney via AkShare.
    Returns 6-digit codes as strings: '000001', '600000', '300750', ...
    """
    spot = ak.stock_zh_a_spot_em()
    # common columns: 代码, 名称, 最新价, ...
    code_col = None
    for c in spot.columns:
        if c in ("代码", "code", "股票代码"):
            code_col = c
            break
    if code_col is None:
        # fallback: first column often is 代码
        code_col = spot.columns[0]
    symbols = spot[code_col].astype(str).str.zfill(6).unique().tolist()
    symbols.sort()
    return symbols


def fetch_symbol_hist(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    retries: int = 3,
    sleep_base: float = 0.3,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily historical data for a single A-share symbol using AkShare.
    Returns DataFrame or None on permanent failure.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,  # "", "qfq", "hfq"
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            col_map = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "换手率": "turnover",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "股票代码" in df.columns:
                df = df.drop(columns=["股票代码"])
            return df
        except Exception as e:
            last_err = e
            # exponential-ish backoff + jitter
            sleep_s = sleep_base * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
            print(f"[{now_str()}] WARN fetch failed symbol={symbol} attempt={attempt}/{retries} err={e}. sleep={sleep_s:.2f}s")
            time.sleep(sleep_s)

    print(f"[{now_str()}] ERROR fetch permanently failed symbol={symbol} err={last_err}", file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest A-share raw daily data into dataset/raw_data/...")
    parser.add_argument("--start", default="20160101", help="Start date YYYYMMDD (default: 20160101)")
    parser.add_argument("--end", default="20251231", help="End date YYYYMMDD (default: 20251231)")
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="Price adjustment (default: qfq)")
    parser.add_argument("--out", default="dataset/raw_data/A_stock/2016-2025", help="Output directory")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol list, e.g. 000001,600000. If empty, fetch all A-shares.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of symbols (for testing). 0 = no limit.")
    parser.add_argument("--batch-size", type=int, default=0, help="0 = no batching; >0 number of symbols in this run.")
    parser.add_argument("--batch-index", type=int, default=0, help="0-based batch index when using --batch-size.")
    parser.add_argument("--sleep", type=float, default=0.35, help="Base sleep seconds between symbols (default: 0.35)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty csv files")
    args = parser.parse_args()

    # Basic date validation
    for d in (args.start, args.end):
        if len(d) != 8 or not d.isdigit():
            print(f"ERROR: invalid date {d}. Expect YYYYMMDD.", file=sys.stderr)
            return 2

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # Determine symbols
    if args.symbols.strip():
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]
        symbols = sorted(set(symbols))
        print(f"[{now_str()}] Using user-provided symbols: {len(symbols)}")
    else:
        print(f"[{now_str()}] Fetching A-share symbol universe via ak.stock_zh_a_spot_em() ...")
        symbols = get_all_ashare_symbols()
        print(f"[{now_str()}] Universe size: {len(symbols)}")

    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]
        print(f"[{now_str()}] Limit enabled. Using first {len(symbols)} symbols.")

    if args.batch_size and args.batch_size > 0:
        start_idx = args.batch_index * args.batch_size
        if start_idx >= len(symbols):
            print(f"[{now_str()}] Batch index {args.batch_index} out of range for {len(symbols)} symbols.", file=sys.stderr)
            return 0
        end_idx = start_idx + args.batch_size
        symbols = symbols[start_idx:end_idx]
        print(
            f"[{now_str()}] Batching enabled. batch_index={args.batch_index} batch_size={args.batch_size} -> {len(symbols)} symbols."
        )

    ok, skipped, empty, failed = 0, 0, 0, 0
    t0 = time.time()

    for i, sym in enumerate(symbols, start=1):
        out_path = out_dir / f"{sym}.csv"

        if out_path.exists() and not args.overwrite:
            try:
                if out_path.stat().st_size > 10:  # non-empty
                    skipped += 1
                    if i % 200 == 0 or i == 1:
                        print(f"[{now_str()}] Progress {i}/{len(symbols)} ok={ok} skipped={skipped} empty={empty} failed={failed}")
                    continue
            except Exception:
                # If stat fails, try overwrite
                pass

        df = fetch_symbol_hist(
            symbol=sym,
            start_date=args.start,
            end_date=args.end,
            adjust=args.adjust,
            retries=3,
            sleep_base=0.4,
        )

        if df is None:
            failed += 1
        elif df.empty:
            # Still write an empty file? Better to write nothing and count empty.
            empty += 1
            # Optionally: write an empty marker file
            safe_write_csv(pd.DataFrame(), out_path)
        else:
            # Add minimal provenance columns (still "raw"): symbol + fetch_time
            df = df.copy()
            df.insert(0, "symbol", sym)
            df.insert(1, "fetched_at", now_str())
            safe_write_csv(df, out_path)
            ok += 1

        # Rate limiting: sleep + jitter
        time.sleep(args.sleep + random.uniform(0, 0.15))

        if i % 200 == 0 or i == 1:
            print(f"[{now_str()}] Progress {i}/{len(symbols)} ok={ok} skipped={skipped} empty={empty} failed={failed}")

    dt = time.time() - t0
    print(f"[{now_str()}] DONE. ok={ok} skipped={skipped} empty={empty} failed={failed} elapsed={dt/60:.1f} min out={out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f"\n[{now_str()}] Interrupted by user.", file=sys.stderr)
        raise
    except Exception:
        print(f"[{now_str()}] FATAL:\n{traceback.format_exc()}", file=sys.stderr)
        raise
