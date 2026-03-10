from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from core.universe.universe_rules import (
    compute_quality_score,
    default_filters,
    extract_symbol_stats,
    filter_reasons,
)


def load_report(report_path: Path) -> Dict[str, object]:
    if not report_path.exists():
        raise FileNotFoundError(f"report file not found: {report_path}")
    return json.loads(report_path.read_text())


def build_universe_core(
    dq_report_path: Path,
    output_path: Path,
    score_threshold: int = 80,
    max_symbols: int = 3000,
) -> Dict[str, object]:
    report = load_report(dq_report_path)
    symbol_metrics: Dict[str, object] = report.get("symbol_metrics", {})
    candidates: List[Tuple[str, float]] = []
    reason_counts: Dict[str, int] = {}

    for sym, metrics in symbol_metrics.items():
        stats = extract_symbol_stats(metrics if isinstance(metrics, dict) else {})
        reasons = filter_reasons(stats)
        if reasons:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue
        score = compute_quality_score(stats)
        candidates.append((sym, score))

    candidates.sort(key=lambda x: (-x[1], x[0]))
    selected_pairs = [(sym, score) for sym, score in candidates if score >= score_threshold]
    if max_symbols > 0:
        selected_pairs = selected_pairs[:max_symbols]
    selected = [sym for sym, _ in selected_pairs]
    symbol_scores = {sym: round(score, 4) for sym, score in selected_pairs}
    mean_score = (
        sum(symbol_scores.values()) / len(symbol_scores) if symbol_scores else 0.0
    )

    payload: Dict[str, object] = {
        "universe_version": "core_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_report": str(dq_report_path),
        "filters": default_filters(score_threshold, max_symbols),
        "summary": {
            "original_symbols": len(symbol_metrics),
            "after_filter": len(candidates),
            "final_universe": len(selected),
            "mean_score": round(mean_score, 4),
        },
        "symbols": selected,
        "symbol_scores": symbol_scores,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"original_symbols = {len(symbol_metrics)}")
    print(f"filtered_symbols = {len(candidates)}")
    print(f"final_universe = {len(selected)}")
    print(f"mean_score = {round(mean_score, 4)}")
    if reason_counts:
        top_reasons = sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))[:10]
        print("top_removed_reasons")
        for reason, count in top_reasons:
            print(f"{reason}: {count}")

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training universe from DQ report.")
    parser.add_argument(
        "--dq-report",
        default="reports/dq_feature_report.json",
    )
    parser.add_argument(
        "--output",
        default="core/universe/data/universe_core_v1.json",
    )
    parser.add_argument(
        "--score-threshold",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=3000,
    )
    args = parser.parse_args()

    build_universe_core(
        dq_report_path=Path(args.dq_report),
        output_path=Path(args.output),
        score_threshold=args.score_threshold,
        max_symbols=args.max_symbols,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
