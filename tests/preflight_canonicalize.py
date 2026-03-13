from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from core.data_build.canonicalize_ashare import canonicalize_ashare_df


def sample_raw_files(raw_dir: Path, max_symbols: int) -> List[Path]:
    files = sorted(raw_dir.glob("*.csv"))
    if max_symbols > 0:
        files = files[:max_symbols]
    return files


def compute_percentiles(values: List[int]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def run_preflight(raw_dir: Path, max_symbols: int) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    files = sample_raw_files(raw_dir, max_symbols)
    if not files:
        raise RuntimeError(f"no raw files found in {raw_dir}")

    per_symbol_rows: List[int] = []
    per_symbol_invalid_ratio: List[float] = []
    duplicate_key_symbols: List[str] = []
    failures: List[Tuple[str, str]] = []

    for p in files:
        sym = p.stem
        try:
            df_raw = pd.read_csv(p)
        except Exception as exc:
            failures.append((sym, f"read_error: {exc}"))
            continue

        if df_raw.empty:
            failures.append((sym, "empty_raw_file"))
            continue

        try:
            df_can, summary = canonicalize_ashare_df(df_raw)
        except Exception as exc:
            failures.append((sym, f"canonicalize_error: {exc}"))
            continue

        raw_rows = int(summary["total_rows"])
        can_rows = int(summary["valid_rows"])
        invalid_rows = int(summary["invalid_rows"])

        per_symbol_rows.append(can_rows)
        if raw_rows > 0:
            per_symbol_invalid_ratio.append(invalid_rows / raw_rows)

        if df_can.duplicated(subset=["date", "symbol"]).any():
            duplicate_key_symbols.append(sym)

    if not per_symbol_rows:
        raise RuntimeError("no successful symbols processed in preflight")

    rows_stats = compute_percentiles(per_symbol_rows)
    invalid_stats = compute_percentiles(
        [int(r * 1e4) for r in per_symbol_invalid_ratio]
    ) if per_symbol_invalid_ratio else {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}

    print(f"[Preflight] symbols_checked={len(per_symbol_rows)} failures={len(failures)}")
    print(f"[Preflight] rows_per_symbol: {rows_stats}")
    print(f"[Preflight] invalid_ratio_per_symbol_x1e4: {invalid_stats}")
    if duplicate_key_symbols:
        print(f"[Preflight] duplicate_key_symbols: {duplicate_key_symbols}")
    if failures:
        for sym, msg in failures[:20]:
            print(f"[Preflight] failure sym={sym} reason={msg}")
        if len(failures) > 20:
            print(f"[Preflight] ... {len(failures) - 20} more failures not shown")

    return len(duplicate_key_symbols), rows_stats, invalid_stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="dataset/raw_data/A_stock/2016-2025")
    parser.add_argument("--max-symbols", type=int, default=500)
    args = parser.parse_args()

    raw_dir = ROOT / args.raw_dir if not Path(args.raw_dir).is_absolute() else Path(args.raw_dir)

    dup_count, _, _ = run_preflight(raw_dir, args.max_symbols)

    if dup_count > 0:
        print(f"[Preflight] FAIL: duplicate keys found in {dup_count} symbols")
        return 1

    print("[Preflight] PASS: no duplicate keys detected in sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

