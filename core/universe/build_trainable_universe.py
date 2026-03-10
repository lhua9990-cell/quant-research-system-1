# 基于 universe_core 指定的 symbols，从 features 目录筛选并写入 trainable_universe

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def load_universe_symbols(universe_path: Path) -> List[str]:
    if not universe_path.exists():
        raise FileNotFoundError(f"universe file not found: {universe_path}")
    payload = json.loads(universe_path.read_text())
    symbols = payload.get("symbols")
    if isinstance(symbols, list) and symbols:
        return [str(sym) for sym in symbols]
    symbol_scores = payload.get("symbol_scores")
    if isinstance(symbol_scores, dict) and symbol_scores:
        return sorted(str(sym) for sym in symbol_scores.keys())
    raise ValueError("universe file does not contain symbols or symbol_scores")


def resolve_feature_file(features_dir: Path, symbol: str) -> Optional[Path]:
    parquet_path = features_dir / f"{symbol}.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = features_dir / f"{symbol}.csv"
    if csv_path.exists():
        return csv_path
    return None


def build_trainable_universe(
    universe_path: Path,
    features_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> Dict[str, object]:
    symbols = load_universe_symbols(universe_path)
    if not features_dir.exists():
        raise FileNotFoundError(f"features dir not found: {features_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    missing: List[str] = []

    for symbol in symbols:
        src = resolve_feature_file(features_dir, symbol)
        if src is None:
            missing.append(symbol)
            continue
        dst = out_dir / src.name
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    payload: Dict[str, object] = {
        "universe_file": str(universe_path),
        "features_dir": str(features_dir),
        "out_dir": str(out_dir),
        "symbol_count": len(symbols),
        "copied_files": copied,
        "skipped_files": skipped,
        "missing_files": len(missing),
        "missing_symbols": missing[:10],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build trainable universe from feature files."
    )
    parser.add_argument(
        "--universe-file",
        default="core/universe/data/universe_core_v1.json",
        help="path to universe json file",
    )
    parser.add_argument(
        "--features-dir",
        default="dataset/features/A_stock/2016-2025",
        help="feature dataset directory",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset/trainable_universe/A_stock/2016-2025",
        help="trainable universe output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing files",
    )
    args = parser.parse_args()

    payload = build_trainable_universe(
        universe_path=Path(args.universe_file),
        features_dir=Path(args.features_dir),
        out_dir=Path(args.out_dir),
        overwrite=args.overwrite,
    )

    print("build_trainable_universe")
    print(f"universe_file: {payload['universe_file']}")
    print(f"features_dir: {payload['features_dir']}")
    print(f"out_dir: {payload['out_dir']}")
    print(f"symbol_count: {payload['symbol_count']}")
    print(f"copied_files: {payload['copied_files']}")
    print(f"skipped_files: {payload['skipped_files']}")
    print(f"missing_files: {payload['missing_files']}")
    if payload["missing_symbols"]:
        print("missing_symbols")
        for symbol in payload["missing_symbols"]:
            print(symbol)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
