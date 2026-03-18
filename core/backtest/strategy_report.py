from __future__ import annotations

import math

import numpy as np
import pandas as pd


def build_stratege_report(
    daily_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    positions_df: pd.DataFrame,
) -> dict[str, object]:
    equity = daily_df["equity"] if not daily_df.empty else pd.Series(dtype=float)
    daily_return = daily_df["daily_return"] if not daily_df.empty else pd.Series(dtype=float)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    annual_return = float((equity.iloc[-1] / equity.iloc[0]) ** (252.0 / len(equity)) - 1.0) if len(equity) > 1 else 0.0
    mean_return = float(daily_return.mean()) if not daily_return.empty else 0.0
    std_return = float(daily_return.std(ddof=0)) if not daily_return.empty else 0.0
    sharpe = float(mean_return / std_return * math.sqrt(252.0)) if std_return > 0 else 0.0

    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0.0, np.nan)
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    volatility = float(std_return * math.sqrt(252.0))

    win_rate = float((daily_return > 0).mean()) if not daily_return.empty else 0.0

    total_orders = int(len(orders_df))
    executed_orders = int((orders_df["status"] == "executed").sum()) if not orders_df.empty else 0
    buy_orders = orders_df[orders_df["side"] == "buy"] if not orders_df.empty else pd.DataFrame()
    sell_orders = orders_df[orders_df["side"] == "sell"] if not orders_df.empty else pd.DataFrame()
    buy_fail = int(((buy_orders["status"] == "failed_limit_up") | (buy_orders["status"] == "failed_volume")).sum()) if not buy_orders.empty else 0
    sell_fail = int(((sell_orders["status"] == "failed_limit_down") | (sell_orders["status"] == "failed_volume")).sum()) if not sell_orders.empty else 0

    fill_rate = float(executed_orders / total_orders) if total_orders > 0 else 0.0
    buy_fail_rate = float(buy_fail / len(buy_orders)) if len(buy_orders) > 0 else 0.0
    sell_fail_rate = float(sell_fail / len(sell_orders)) if len(sell_orders) > 0 else 0.0
    capital_utilization = float(daily_df["invest_ratio"].mean()) if not daily_df.empty else 0.0

    traded_value = float(orders_df.loc[orders_df["status"] == "executed", "trade_value"].sum()) if not orders_df.empty else 0.0
    total_equity = float(daily_df["equity"].sum()) if not daily_df.empty else 0.0
    turnover = float(traded_value / total_equity) if total_equity > 0 else 0.0
    avg_holding_period = float(positions_df.loc[positions_df["exit_date"].notna(), "holding_days"].mean()) if not positions_df.empty else 0.0

    return {
        "return": {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": sharpe,
        },
        "risk": {
            "max_drawdown": max_drawdown,
            "volatility": volatility,
        },
        "trade": {
            "win_rate": win_rate,
        },
        "execution": {
            "fill_rate": fill_rate,
            "buy_fail_rate": buy_fail_rate,
            "sell_fail_rate": sell_fail_rate,
            "capital_utilization": capital_utilization,
        },
        "diagnostics": {
            "turnover": turnover,
            "avg_holding_period": avg_holding_period,
        },
    }
