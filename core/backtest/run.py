from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.backtest.stratege_report import build_stratege_report
from core.models.Xgboost.model_config import INPUT_DIR, PREDICT_OUTPUT, VALID_END

DAILY_OUTPUT = Path("core/backtest/output/daily_equity.parquet")
ORDERS_OUTPUT = Path("core/backtest/output/orders_log.parquet")
POSITIONS_OUTPUT = Path("core/backtest/output/positions_log.parquet")
REPORT_OUTPUT = Path("core/backtest/output/backtest_report.json")


def load_alpha() -> pd.DataFrame:
    if not PREDICT_OUTPUT.exists():
        raise FileNotFoundError(f"alpha predictions not found: {PREDICT_OUTPUT}")
    alpha = pd.read_parquet(PREDICT_OUTPUT)
    required = ["date", "symbol", "alpha"]
    missing = [c for c in required if c not in alpha.columns]
    if missing:
        raise ValueError(f"alpha predictions missing columns: {missing}")
    alpha = alpha[required].copy()
    alpha["date"] = pd.to_datetime(alpha["date"])
    return alpha[alpha["date"] >= VALID_END].sort_values(["date", "symbol"])


def generate_signals(alpha: pd.DataFrame, quantile: float = 0.10) -> pd.DataFrame:
    panel = alpha.copy()
    panel["rank"] = panel.groupby("date", sort=False)["alpha"].rank(method="first", pct=True)
    panel["signal"] = (panel["rank"] >= 1.0 - quantile) & (panel["alpha"] > 0)
    return panel[panel["signal"]].sort_values(["date", "alpha"], ascending=[True, False])


def load_market(symbols: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    symbol_set = set(symbols)
    for file_path in sorted(Path(INPUT_DIR).glob("*.parquet")):
        symbol = file_path.stem
        if symbol not in symbol_set:
            continue
        df = pd.read_parquet(file_path)
        if "symbol" not in df.columns:
            df = df.copy()
            df["symbol"] = symbol
        required = ["date", "symbol", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"market data missing columns in {file_path}: {missing}")
        df = df[required].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if not df.empty:
            rows.append(df)
    if not rows:
        raise ValueError("no market data loaded")
    market = pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])
    market["prev_close"] = market.groupby("symbol", sort=False)["close"].shift(1)
    return market


def run_backtest(
    initial_capital: float = 1000000.0,
    position_unit: float = 5000.0,
    lot_size: int = 100,
    max_invest_ratio: float = 0.9,
    trailing_stop_pct: float = 0.08,
    limit_up_pct: float = 0.10,
    limit_down_pct: float = 0.10,
    signal_quantile: float = 0.10,
) -> dict[str, object]:
    alpha = load_alpha()
    signals = generate_signals(alpha, quantile=signal_quantile)
    start_date = pd.Timestamp(alpha["date"].min()) - pd.Timedelta(days=5)
    end_date = pd.Timestamp(alpha["date"].max()) + pd.Timedelta(days=5)
    market = load_market(alpha["symbol"].astype(str).unique().tolist(), start_date=start_date, end_date=end_date)

    market_lookup = {}
    for _, row in market.iterrows():
        market_lookup[(pd.Timestamp(row["date"]), str(row["symbol"]))] = {
            "open": float(row["open"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "prev_close": float(row["prev_close"]) if pd.notna(row["prev_close"]) else np.nan,
        }

    trading_dates = sorted(pd.to_datetime(market["date"].unique()))
    next_date = {trading_dates[i]: trading_dates[i + 1] for i in range(len(trading_dates) - 1)}
    signals_map = {
        pd.Timestamp(date): group.sort_values("alpha", ascending=False)[["symbol", "alpha"]].copy()
        for date, group in signals.groupby("date", sort=False)
    }

    pending_orders: dict[pd.Timestamp, list[dict[str, object]]] = {}
    positions: dict[str, dict[str, object]] = {}
    closed_positions: list[dict[str, object]] = []
    orders_log: list[dict[str, object]] = []
    daily_log: list[dict[str, object]] = []
    cash = float(initial_capital)
    prev_equity = float(initial_capital)

    for date in trading_dates:
        date = pd.Timestamp(date)
        orders = pending_orders.pop(date, [])
        for order in orders:
            symbol = str(order["symbol"])
            row = market_lookup.get((date, symbol))
            if row is None:
                orders_log.append(
                    {
                        "signal_date": order["signal_date"],
                        "execute_date": date,
                        "symbol": symbol,
                        "side": order["side"],
                        "status": "failed_no_data",
                        "price": np.nan,
                        "shares": 0,
                        "trade_value": 0.0,
                    }
                )
                continue
            open_price = row["open"]
            prev_close = row["prev_close"]
            volume = row["volume"]
            if open_price <= 0 or volume <= 0:
                orders_log.append(
                    {
                        "signal_date": order["signal_date"],
                        "execute_date": date,
                        "symbol": symbol,
                        "side": order["side"],
                        "status": "failed_volume",
                        "price": np.nan,
                        "shares": 0,
                        "trade_value": 0.0,
                    }
                )
                continue

            if order["side"] == "buy":
                if np.isfinite(prev_close) and open_price >= prev_close * (1.0 + limit_up_pct):
                    status = "failed_limit_up"
                    shares = 0
                    trade_value = 0.0
                else:
                    shares = int(math.ceil(position_unit / open_price / lot_size) * lot_size)
                    trade_value = float(shares * open_price)
                    invested_before = initial_capital - cash
                    if shares <= 0 or trade_value > cash or invested_before + trade_value > initial_capital * max_invest_ratio:
                        status = "failed_cash"
                        shares = 0
                        trade_value = 0.0
                    elif symbol in positions:
                        status = "failed_existing_position"
                        shares = 0
                        trade_value = 0.0
                    else:
                        cash -= trade_value
                        positions[symbol] = {
                            "entry_date": date,
                            "entry_price": open_price,
                            "shares": shares,
                            "peak_price": row["close"],
                        }
                        status = "executed"
                orders_log.append(
                    {
                        "signal_date": order["signal_date"],
                        "execute_date": date,
                        "symbol": symbol,
                        "side": "buy",
                        "status": status,
                        "price": float(open_price) if shares > 0 else np.nan,
                        "shares": int(shares),
                        "trade_value": float(trade_value),
                    }
                )
            else:
                position = positions.get(symbol)
                if position is None:
                    status = "failed_no_position"
                    shares = 0
                    trade_value = 0.0
                elif np.isfinite(prev_close) and open_price <= prev_close * (1.0 - limit_down_pct):
                    status = "failed_limit_down"
                    shares = 0
                    trade_value = 0.0
                else:
                    shares = int(position["shares"])
                    trade_value = float(shares * open_price)
                    cash += trade_value
                    closed_positions.append(
                        {
                            "symbol": symbol,
                            "entry_date": position["entry_date"],
                            "exit_date": date,
                            "entry_price": float(position["entry_price"]),
                            "exit_price": float(open_price),
                            "shares": int(shares),
                            "holding_days": int((date - pd.Timestamp(position["entry_date"])).days),
                        }
                    )
                    del positions[symbol]
                    status = "executed"
                orders_log.append(
                    {
                        "signal_date": order["signal_date"],
                        "execute_date": date,
                        "symbol": symbol,
                        "side": "sell",
                        "status": status,
                        "price": float(open_price) if shares > 0 else np.nan,
                        "shares": int(shares),
                        "trade_value": float(trade_value),
                    }
                )

        for symbol, pos in list(positions.items()):
            row = market_lookup.get((date, symbol))
            if row is None:
                continue
            close_price = float(row["close"])
            pos["peak_price"] = max(float(pos["peak_price"]), close_price)
            drawdown = (close_price - float(pos["peak_price"])) / float(pos["peak_price"]) if pos["peak_price"] > 0 else 0.0
            if drawdown <= -trailing_stop_pct:
                d1 = next_date.get(date)
                if d1 is not None:
                    q = pending_orders.setdefault(d1, [])
                    if not any((x["side"] == "sell" and x["symbol"] == symbol) for x in q):
                        q.append({"signal_date": date, "symbol": symbol, "side": "sell"})

        sig = signals_map.get(date)
        if sig is not None:
            d1 = next_date.get(date)
            if d1 is not None:
                q = pending_orders.setdefault(d1, [])
                queued_buy = {x["symbol"] for x in q if x["side"] == "buy"}
                for _, r in sig.iterrows():
                    symbol = str(r["symbol"])
                    if symbol in positions or symbol in queued_buy:
                        continue
                    q.append({"signal_date": date, "symbol": symbol, "side": "buy"})
                    queued_buy.add(symbol)

        position_value = 0.0
        for symbol, pos in positions.items():
            row = market_lookup.get((date, symbol))
            if row is None:
                continue
            position_value += float(pos["shares"] * row["close"])
        equity = float(cash + position_value)
        daily_pnl = float(equity - prev_equity)
        daily_return = float(daily_pnl / prev_equity) if prev_equity > 0 else 0.0
        daily_log.append(
            {
                "date": date,
                "equity": equity,
                "cash": float(cash),
                "position_value": position_value,
                "daily_pnl": daily_pnl,
                "daily_return": daily_return,
                "invest_ratio": float(position_value / initial_capital) if initial_capital > 0 else 0.0,
            }
        )
        prev_equity = equity

    open_positions = [
        {
            "symbol": symbol,
            "entry_date": pos["entry_date"],
            "exit_date": pd.NaT,
            "entry_price": float(pos["entry_price"]),
            "exit_price": np.nan,
            "shares": int(pos["shares"]),
            "holding_days": int((trading_dates[-1] - pd.Timestamp(pos["entry_date"])).days),
        }
        for symbol, pos in positions.items()
    ]

    daily_df = pd.DataFrame(daily_log).sort_values("date")
    orders_df = pd.DataFrame(orders_log).sort_values(["execute_date", "symbol", "side"]) if orders_log else pd.DataFrame(
        columns=["signal_date", "execute_date", "symbol", "side", "status", "price", "shares", "trade_value"]
    )
    positions_df = pd.DataFrame(closed_positions + open_positions)

    report = {
        "run_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eval_start": str(pd.Timestamp(alpha["date"].min()).date()),
            "eval_end": str(pd.Timestamp(alpha["date"].max()).date()),
            "params": {
                "initial_capital": initial_capital,
                "position_unit": position_unit,
                "lot_size": lot_size,
                "max_invest_ratio": max_invest_ratio,
                "trailing_stop_pct": trailing_stop_pct,
                "limit_up_pct": limit_up_pct,
                "limit_down_pct": limit_down_pct,
                "signal_quantile": signal_quantile,
            },
        },
        "summary": build_stratege_report(daily_df, orders_df, positions_df),
    }

    DAILY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(DAILY_OUTPUT, index=False)
    orders_df.to_parquet(ORDERS_OUTPUT, index=False)
    positions_df.to_parquet(POSITIONS_OUTPUT, index=False)
    with REPORT_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("daily output saved:", DAILY_OUTPUT)
    print("orders output saved:", ORDERS_OUTPUT)
    print("positions output saved:", POSITIONS_OUTPUT)
    print("report output saved:", REPORT_OUTPUT)
    return report


if __name__ == "__main__":
    run_backtest()
    raise SystemExit(0)
