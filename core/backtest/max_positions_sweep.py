from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.backtest.run import run_backtest

OUTPUT_DIR = Path("core/backtest/output")
OUTPUT_JSON = OUTPUT_DIR / "max_positions_sweep_report.json"
OUTPUT_CSV = OUTPUT_DIR / "max_positions_sweep_curve.csv"
OUTPUT_MD = OUTPUT_DIR / "max_positions_sweep_report.md"


def run_max_positions_sweep(
    initial_capital: float = 2_000_000.0,
    signal_quantile: float = 0.05,
    trailing_stop_pct: float = 0.05,
    atr_stop_k: float = 1.0,
) -> dict[str, object]:
    n_values = list(range(10, 51, 5))
    rows: list[dict[str, float]] = []
    for n in n_values:
        report = run_backtest(
            initial_capital=initial_capital,
            signal_quantile=signal_quantile,
            trailing_stop_pct=trailing_stop_pct,
            atr_stop_k=atr_stop_k,
            stop_mode="fixed",
            max_positions=n,
        )
        summary = report["summary"]
        sharpe = float(summary["return"]["sharpe"])
        max_drawdown = float(summary["risk"]["max_drawdown"])
        rows.append(
            {
                "max_positions": int(n),
                "annual_return": float(summary["return"]["annual_return"]),
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "volatility": float(summary["risk"]["volatility"]),
                "fill_rate": float(summary["execution"]["fill_rate"]),
                "capital_utilization": float(summary["execution"]["capital_utilization"]),
                "win_rate": float(summary["trade"]["win_rate"]),
                "risk_adjusted_score": float(sharpe / abs(max_drawdown)) if max_drawdown < 0 else 0.0,
            }
        )

    baseline = run_backtest(
        initial_capital=initial_capital,
        signal_quantile=signal_quantile,
        trailing_stop_pct=trailing_stop_pct,
        atr_stop_k=atr_stop_k,
        stop_mode="fixed",
        max_positions=None,
    )
    baseline_summary = baseline["summary"]
    baseline_row = {
        "max_positions": None,
        "annual_return": float(baseline_summary["return"]["annual_return"]),
        "sharpe": float(baseline_summary["return"]["sharpe"]),
        "max_drawdown": float(baseline_summary["risk"]["max_drawdown"]),
        "volatility": float(baseline_summary["risk"]["volatility"]),
        "fill_rate": float(baseline_summary["execution"]["fill_rate"]),
        "capital_utilization": float(baseline_summary["execution"]["capital_utilization"]),
        "win_rate": float(baseline_summary["trade"]["win_rate"]),
    }
    best_by_sharpe = max(rows, key=lambda x: x["sharpe"])
    best_by_drawdown = max(rows, key=lambda x: x["max_drawdown"])
    best_by_score = max(rows, key=lambda x: x["risk_adjusted_score"])
    payload = {
        "run_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "initial_capital": float(initial_capital),
            "signal_quantile": float(signal_quantile),
            "trailing_stop_rule": "min(5%, k*atr/price)",
            "atr_stop_k": float(atr_stop_k),
            "max_positions_range": "10:5:50",
        },
        "baseline_no_limit": baseline_row,
        "curve": rows,
        "best": {
            "best_by_sharpe": best_by_sharpe,
            "best_by_drawdown": best_by_drawdown,
            "best_by_risk_adjusted_score": best_by_score,
        },
    }
    return payload


def write_outputs(report: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    curve_df = pd.DataFrame(report["curve"]).sort_values("max_positions")
    curve_df.to_csv(OUTPUT_CSV, index=False)

    baseline = report["baseline_no_limit"]
    lines = [
        "# Max Positions Sweep Report",
        "",
        f"- generated_at: {report['run_meta']['generated_at']}",
        f"- initial_capital: {report['run_meta']['initial_capital']}",
        f"- signal_quantile: {report['run_meta']['signal_quantile']}",
        f"- trailing_stop_rule: {report['run_meta']['trailing_stop_rule']}",
        f"- atr_stop_k: {report['run_meta']['atr_stop_k']}",
        f"- max_positions_range: {report['run_meta']['max_positions_range']}",
        "",
        "## Baseline (No max_positions limit)",
        f"- annual_return: {baseline['annual_return']:.6f}",
        f"- sharpe: {baseline['sharpe']:.6f}",
        f"- max_drawdown: {baseline['max_drawdown']:.6f}",
        f"- volatility: {baseline['volatility']:.6f}",
        f"- fill_rate: {baseline['fill_rate']:.6f}",
        f"- capital_utilization: {baseline['capital_utilization']:.6f}",
        f"- win_rate: {baseline['win_rate']:.6f}",
        "",
        "## Curve",
        "| max_positions | annual_return | sharpe | max_drawdown | volatility | fill_rate | capital_utilization | win_rate | risk_adjusted_score |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["curve"]:
        lines.append(
            f"| {row['max_positions']} | {row['annual_return']:.6f} | {row['sharpe']:.6f} | {row['max_drawdown']:.6f} | {row['volatility']:.6f} | {row['fill_rate']:.6f} | {row['capital_utilization']:.6f} | {row['win_rate']:.6f} | {row['risk_adjusted_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Best",
            f"- best_by_sharpe: max_positions={report['best']['best_by_sharpe']['max_positions']}, sharpe={report['best']['best_by_sharpe']['sharpe']:.6f}, max_drawdown={report['best']['best_by_sharpe']['max_drawdown']:.6f}",
            f"- best_by_drawdown: max_positions={report['best']['best_by_drawdown']['max_positions']}, sharpe={report['best']['best_by_drawdown']['sharpe']:.6f}, max_drawdown={report['best']['best_by_drawdown']['max_drawdown']:.6f}",
            f"- best_by_risk_adjusted_score: max_positions={report['best']['best_by_risk_adjusted_score']['max_positions']}, score={report['best']['best_by_risk_adjusted_score']['risk_adjusted_score']:.6f}",
            "",
            "## Output Files",
            f"- {OUTPUT_JSON}",
            f"- {OUTPUT_CSV}",
            f"- {OUTPUT_MD}",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = run_max_positions_sweep()
    write_outputs(report)
    print("max_positions sweep report saved:", OUTPUT_JSON)
    print("max_positions sweep curve saved:", OUTPUT_CSV)
    print("max_positions sweep markdown saved:", OUTPUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
