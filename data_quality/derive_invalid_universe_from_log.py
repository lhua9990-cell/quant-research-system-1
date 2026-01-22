import argparse
import ast
from pathlib import Path
from typing import Dict, Set, Tuple


def parse_log(path: Path, threshold: float) -> Dict[str, Tuple[int, int, float]]:
    result: Dict[str, Tuple[int, int, float]] = {}
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
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
            ratio = invalid / total
            if ratio >= threshold:
                result[symbol] = (total, invalid, ratio)
    return result


def parse_gate_csv(path: Path) -> Dict[str, Tuple[int, int, float]]:
    result: Dict[str, Tuple[int, int, float]] = {}
    if not path.exists():
        raise FileNotFoundError(f"gate csv not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("symbol,"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            symbol = parts[0]
            try:
                raw_rows = int(parts[1])
                invalid_rows = int(parts[2])
                invalid_ratio = float(parts[3])
            except ValueError:
                continue
            result[symbol] = (raw_rows, invalid_rows, invalid_ratio)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-path",
        default="logs/canonicalize_ashare_2016-2025.log",
        help="path to canonicalize log file",
    )
    parser.add_argument(
        "--gate-csv",
        default="data_quality/suspicious_invalid_ratio_v0_1.csv",
        help="path to reference gate csv",
    )
    # 默认门槛与 suspicious_invalid_ratio_v0_1.csv 中的 invalid_ratio 定义保持一致：invalid_ratio >= 0.01
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="invalid_ratio threshold for exclusion",
    )
    parser.add_argument(
        "--canonical-dir",
        default="dataset/canonical/A_stock/2016-2025",
        help="canonical output directory (for moving excluded files)",
    )
    parser.add_argument(
        "--move-files",
        action="store_true",
        help="move excluded symbols' canonical files into _exclude_negprice_qfq_2016-2025",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    gate_path = Path(args.gate_csv)

    from_log = parse_log(log_path, args.threshold)
    from_csv = parse_gate_csv(gate_path)

    symbols_log: Set[str] = set(from_log.keys())
    symbols_csv: Set[str] = set(from_csv.keys())

    only_in_log = sorted(symbols_log - symbols_csv)
    only_in_csv = sorted(symbols_csv - symbols_log)
    in_both = sorted(symbols_log & symbols_csv)

    print(f"gate threshold: invalid_ratio >= {args.threshold}")
    print(f"symbols from log: {len(symbols_log)}")
    print(f"symbols from csv: {len(symbols_csv)}")
    print(f"intersection: {len(in_both)}")
    print(f"only_in_log: {only_in_log}")
    print(f"only_in_csv: {only_in_csv}")

    if args.move_files:
        canonical_dir = Path(args.canonical_dir)
        excluded_dir = canonical_dir.parent / "_exclude_negprice_qfq_2016-2025"
        excluded_dir.mkdir(parents=True, exist_ok=True)

        excluded_symbols = sorted(symbols_log)
        moved_count = 0
        missing: Dict[str, int] = {}

        for sym in excluded_symbols:
            for ext in (".csv", ".parquet"):
                src = canonical_dir / f"{sym}{ext}"
                if src.exists():
                    dst = excluded_dir / src.name
                    src.replace(dst)
                    moved_count += 1
                else:
                    missing[sym] = missing.get(sym, 0) + 1

        print(f"moved canonical files: {moved_count} to {excluded_dir}")
        if missing:
            missing_syms = sorted(missing.keys())
            print(f"symbols with no canonical file found (csv/parquet): {missing_syms}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
