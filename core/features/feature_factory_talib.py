from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if "symbol" not in df.columns:
        raise ValueError("input must contain 'symbol' column")
    if "date" not in df.columns:
        raise ValueError("input must contain 'date' column")
    # Public demo only: keep generic, non-proprietary feature transformations.
    out = df.copy()
    out = out.sort_values(["symbol", "date"])
    grouped = out.groupby("symbol", group_keys=False)
    out["feat_001"] = grouped["close"].pct_change(5)
    out["feat_002"] = grouped["turnover"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    out["feat_003"] = np.where(
        out["vwap"] > 0,
        (out["close"] - out["vwap"]) / out["vwap"],
        np.nan,
    )
    round_cols = [
        "feat_001",
        "feat_002",
        "feat_003",
    ]
    out[round_cols] = out[round_cols].round(4)
    out = out.sort_values(["symbol", "date"])
    return out


def list_canonical_files(canonical_dir: Path, symbols: Optional[List[str]]) -> List[Path]:
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
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    symbols = [s for s in args.symbols.split(",") if s.strip()] if args.symbols.strip() else None
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
            logger.exception(f"failed to transform {canonical_path} -> {out_path}")
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
