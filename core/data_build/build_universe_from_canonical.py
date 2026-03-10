#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Universe build rule:
# - Use canonicalize_ashare log discard_summary to exclude any symbol with invalid_rows > 0.
# - invalid_rows: 前复权数据拉取存在负 price 的行，当前暂不处理。
# - Copy remaining canonical files into dataset/universe/A_stock/2016-2025.

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple


def parse_invalid_symbols(log_path: Path) -> Dict[str, Tuple[int, int, float]]:
    result: Dict[str, Tuple[int, int, float]] = {}
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "discard_summary=" not in line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[1]
            if not name.endswith(".csv"):
                continue
            symbol = name.replace(".csv", "")
            idx = line.find("discard_summary=")
            if idx == -1:
                continue
            dict_str = line[idx + len("discard_summary=") :]
            try:
                summary = ast.literal_eval(dict_str)
            except Exception:
                continue
            total = int(summary.get("total_rows", 0))
            invalid = int(summary.get("invalid_rows", 0))
            if total <= 0:
                continue
            if invalid > 0:
                ratio = invalid / total
                result[symbol] = (total, invalid, ratio)
    return result


def list_canonical_files(canonical_dir: Path) -> list[Path]:
    if not canonical_dir.exists():
        raise FileNotFoundError(f"canonical dir not found: {canonical_dir}")
    files: list[Path] = []
    for p in canonical_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix not in (".csv", ".parquet"):
            continue
        files.append(p)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build universe from canonical data, excluding symbols with invalid rows."
    )
    parser.add_argument(
        "--log-path",
        default="logs/canonicalize_ashare_2016-2025.log",
        help="path to canonicalize log file",
    )
    parser.add_argument(
        "--canonical-dir",
        default="dataset/canonical/A_stock/2016-2025",
        help="canonical data directory",
    )
    parser.add_argument(
        "--out",
        default="dataset/universe/A_stock/2016-2025",
        help="universe output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing universe files",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    canonical_dir = Path(args.canonical_dir)
    out_dir = Path(args.out)

    invalid_map = parse_invalid_symbols(log_path)
    invalid_symbols: Set[str] = set(invalid_map.keys())

    files = list_canonical_files(canonical_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    excluded = 0

    for src in files:
        symbol = src.stem
        if symbol in invalid_symbols:
            excluded += 1
            continue
        dst = out_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print("build_universe_from_canonical")
    print(f"log_path: {log_path}")
    print(f"canonical_dir: {canonical_dir}")
    print(f"out_dir: {out_dir}")
    print(f"invalid_symbols: {len(invalid_symbols)}")
    print(f"copied_files: {copied}")
    print(f"skipped_files: {skipped}")
    print(f"excluded_files: {excluded}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
