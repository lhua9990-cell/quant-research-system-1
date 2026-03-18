from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.backtest.run import run_backtest

OUTPUT_DIR = Path("core/backtest/output")
OUTPUT_JSON = OUTPUT_DIR / "atr_risk_sweep_report.json"
OUTPUT_CSV = OUTPUT_DIR / "atr_risk_sweep_curve.csv"
OUTPUT_MD = OUTPUT_DIR / "atr_risk_sweep_report.md"


def run_atr_sweep(
    initial_capital: float = 2_000_000.0,
    signal_quantile: float = 0.05,
    baseline_trailing_stop_pct: float = 0.05,
) -> dict[str, object]:
    k_values = [round(1.0 + 0.2 * i, 1) for i in range(11)]
    rows: list[dict[str, float]] = []
    for k in k_values:
        report = run_backtest(
            initial_capital=initial_capital,
            signal_quantile=signal_quantile,
            stop_mode="atr_ratio",
            atr_stop_k=float(k),
            trailing_stop_pct=baseline_trailing_stop_pct,
        )
        summary = report["summary"]
        sharpe = float(summary["return"]["sharpe"])
        max_drawdown = float(summary["risk"]["max_drawdown"])
        rows.append(
            {
                "k": float(k),
                "annual_return": float(summary["return"]["annual_return"]),
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "volatility": float(summary["risk"]["volatility"]),
                "fill_rate": float(summary["execution"]["fill_rate"]),
                "win_rate": float(summary["trade"]["win_rate"]),
                "risk_adjusted_score": float(sharpe / abs(max_drawdown)) if max_drawdown < 0 else 0.0,
            }
        )

    baseline = run_backtest(
        initial_capital=initial_capital,
        signal_quantile=signal_quantile,
        trailing_stop_pct=baseline_trailing_stop_pct,
        stop_mode="fixed",
    )
    baseline_summary = baseline["summary"]
    baseline_row = {
        "stop_mode": "fixed",
        "trailing_stop_pct": float(baseline_trailing_stop_pct),
        "annual_return": float(baseline_summary["return"]["annual_return"]),
        "sharpe": float(baseline_summary["return"]["sharpe"]),
        "max_drawdown": float(baseline_summary["risk"]["max_drawdown"]),
        "volatility": float(baseline_summary["risk"]["volatility"]),
        "fill_rate": float(baseline_summary["execution"]["fill_rate"]),
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
            "k_range": "1:0.2:3",
            "baseline_trailing_stop_pct": float(baseline_trailing_stop_pct),
        },
        "baseline_fixed_stop": baseline_row,
        "atr_curve": rows,
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
    curve_df = pd.DataFrame(report["atr_curve"]).sort_values("k")
    curve_df.to_csv(OUTPUT_CSV, index=False)

    baseline = report["baseline_fixed_stop"]
    lines = [
        "# ATR Risk Control Sweep Report",
        "",
        f"- generated_at: {report['run_meta']['generated_at']}",
        f"- initial_capital: {report['run_meta']['initial_capital']}",
        f"- signal_quantile: {report['run_meta']['signal_quantile']}",
        f"- k_range: {report['run_meta']['k_range']}",
        f"- baseline_fixed_stop: {report['run_meta']['baseline_trailing_stop_pct']:.2%}",
        "",
        "## Baseline (Fixed Stop)",
        f"- annual_return: {baseline['annual_return']:.6f}",
        f"- sharpe: {baseline['sharpe']:.6f}",
        f"- max_drawdown: {baseline['max_drawdown']:.6f}",
        f"- volatility: {baseline['volatility']:.6f}",
        f"- fill_rate: {baseline['fill_rate']:.6f}",
        f"- win_rate: {baseline['win_rate']:.6f}",
        "",
        "## Sharpe vs MaxDrawdown Curve (ATR stop_pct = k * atr / price)",
        "| k | annual_return | sharpe | max_drawdown | volatility | fill_rate | win_rate | risk_adjusted_score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["atr_curve"]:
        lines.append(
            f"| {row['k']:.1f} | {row['annual_return']:.6f} | {row['sharpe']:.6f} | {row['max_drawdown']:.6f} | {row['volatility']:.6f} | {row['fill_rate']:.6f} | {row['win_rate']:.6f} | {row['risk_adjusted_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Best",
            f"- best_by_sharpe: k={report['best']['best_by_sharpe']['k']:.1f}, sharpe={report['best']['best_by_sharpe']['sharpe']:.6f}, max_drawdown={report['best']['best_by_sharpe']['max_drawdown']:.6f}",
            f"- best_by_drawdown: k={report['best']['best_by_drawdown']['k']:.1f}, sharpe={report['best']['best_by_drawdown']['sharpe']:.6f}, max_drawdown={report['best']['best_by_drawdown']['max_drawdown']:.6f}",
            f"- best_by_risk_adjusted_score: k={report['best']['best_by_risk_adjusted_score']['k']:.1f}, score={report['best']['best_by_risk_adjusted_score']['risk_adjusted_score']:.6f}",
            "",
            "## Output Files",
            f"- {OUTPUT_JSON}",
            f"- {OUTPUT_CSV}",
            f"- {OUTPUT_MD}",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = run_atr_sweep()
    write_outputs(report)
    print("atr sweep report saved:", OUTPUT_JSON)
    print("atr sweep curve saved:", OUTPUT_CSV)
    print("atr sweep markdown saved:", OUTPUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
