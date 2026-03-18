from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.backtest.run import run_backtest

OUTPUT_DIR = Path("core/backtest/output")


def parse_capitals(text: str) -> list[int]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("capitals is empty")
    return [int(v) for v in values]


def parse_quantiles(text: str) -> list[float]:
    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError("quantiles range must be start:step:end")
        start = float(parts[0])
        step = float(parts[1])
        end = float(parts[2])
        if step <= 0:
            raise ValueError("quantile step must be positive")
        result: list[float] = []
        q = start
        while q <= end + 1e-12:
            result.append(round(q, 10))
            q += step
        return result
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("quantiles is empty")
    return [float(v) for v in values]


def evaluate_grid(capitals: list[int], quantiles: list[float], trailing_stop_pct: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for capital in capitals:
        for quantile in quantiles:
            result = run_backtest(
                initial_capital=float(capital),
                trailing_stop_pct=float(trailing_stop_pct),
                signal_quantile=float(quantile),
            )
            summary = result["summary"]
            rows.append(
                {
                    "capital": float(capital),
                    "quantile": float(quantile),
                    "fill_rate": float(summary["execution"]["fill_rate"]),
                    "buy_fail_rate": float(summary["execution"]["buy_fail_rate"]),
                    "sell_fail_rate": float(summary["execution"]["sell_fail_rate"]),
                    "capital_utilization": float(summary["execution"]["capital_utilization"]),
                    "annual_return": float(summary["return"]["annual_return"]),
                    "sharpe": float(summary["return"]["sharpe"]),
                    "max_drawdown": float(summary["risk"]["max_drawdown"]),
                    "volatility": float(summary["risk"]["volatility"]),
                    "win_rate": float(summary["trade"]["win_rate"]),
                }
            )
    return rows


def select_rows(rows: list[dict[str, float]], condition) -> list[dict[str, float]]:
    return [row for row in rows if condition(row)]


def sort_rows(rows: list[dict[str, float]], keys: list[str]) -> list[dict[str, float]]:
    return sorted(rows, key=lambda x: tuple(x[k] for k in keys), reverse=True)


def markdown_table(rows: list[dict[str, float]], columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if col == "capital":
                cells.append(f"{int(value):,}")
            elif col == "quantile":
                cells.append(f"{value:.2f}")
            else:
                cells.append(f"{value:.6f}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def build_markdown(
    rows: list[dict[str, float]],
    capitals: list[int],
    quantiles: list[float],
    trailing_stop_pct: float,
) -> str:
    columns = [
        "capital",
        "quantile",
        "fill_rate",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "capital_utilization",
        "win_rate",
    ]
    all_rows = sorted(rows, key=lambda x: (x["capital"], x["quantile"]))
    top_sharpe = sort_rows(rows, ["sharpe", "annual_return"])[:10]
    top_return = sort_rows(rows, ["annual_return", "sharpe"])[:10]
    filter_a = sort_rows(
        select_rows(rows, lambda x: x["fill_rate"] >= 0.90 and x["max_drawdown"] >= -0.15),
        ["sharpe", "annual_return"],
    )
    filter_b = sort_rows(
        select_rows(rows, lambda x: x["fill_rate"] >= 0.70 and x["max_drawdown"] >= -0.20),
        ["sharpe", "annual_return"],
    )
    best_sharpe = max(rows, key=lambda x: x["sharpe"])
    best_return = max(rows, key=lambda x: x["annual_return"])
    best_mdd = max(rows, key=lambda x: x["max_drawdown"])

    lines: list[str] = []
    lines.append("# Backtest Grid Tuning Report")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append(f"- trailing_stop_pct: {trailing_stop_pct:.2%}")
    lines.append(f"- capitals: {', '.join([f'{c:,}' for c in capitals])}")
    lines.append(f"- quantiles: {', '.join([f'{q:.2f}' for q in quantiles])}")
    lines.append(f"- total_combinations: {len(rows)}")
    lines.append("")
    lines.append("## Full Grid")
    lines.extend(markdown_table(all_rows, columns))
    lines.append("")
    lines.append("## Top 10 by Sharpe")
    lines.extend(markdown_table(top_sharpe, columns))
    lines.append("")
    lines.append("## Top 10 by Annual Return")
    lines.extend(markdown_table(top_return, columns))
    lines.append("")
    lines.append("## Filter A (fill_rate >= 0.90 and max_drawdown >= -0.15)")
    lines.extend(markdown_table(filter_a, columns) if filter_a else ["none"])
    lines.append("")
    lines.append("## Filter B (fill_rate >= 0.70 and max_drawdown >= -0.20)")
    lines.extend(markdown_table(filter_b, columns) if filter_b else ["none"])
    lines.append("")
    lines.append("## Best")
    lines.append("- best_sharpe")
    lines.extend(markdown_table([best_sharpe], columns))
    lines.append("")
    lines.append("- best_annual_return")
    lines.extend(markdown_table([best_return], columns))
    lines.append("")
    lines.append("- best_drawdown_least_negative")
    lines.extend(markdown_table([best_mdd], columns))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capitals", type=str, default="1000000,1500000,2000000,2500000,3000000")
    parser.add_argument("--quantiles", type=str, default="0.01:0.01:0.10")
    parser.add_argument("--trailing-stop-pct", type=float, default=0.05)
    parser.add_argument("--output-prefix", type=str, default="grid_tuning_stop5_cap100_300_q001_010")
    args = parser.parse_args()

    capitals = parse_capitals(args.capitals)
    quantiles = parse_quantiles(args.quantiles)
    rows = evaluate_grid(capitals=capitals, quantiles=quantiles, trailing_stop_pct=args.trailing_stop_pct)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{args.output_prefix}.json"
    csv_path = OUTPUT_DIR / f"{args.output_prefix}.csv"
    md_path = OUTPUT_DIR / f"{args.output_prefix}.md"

    payload = {
        "run_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trailing_stop_pct": float(args.trailing_stop_pct),
            "capitals": capitals,
            "quantiles": quantiles,
            "total_combinations": len(rows),
        },
        "results": rows,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    md_path.write_text(
        build_markdown(
            rows=rows,
            capitals=capitals,
            quantiles=quantiles,
            trailing_stop_pct=float(args.trailing_stop_pct),
        ),
        encoding="utf-8",
    )

    print("grid json saved:", json_path)
    print("grid csv saved:", csv_path)
    print("grid markdown saved:", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
