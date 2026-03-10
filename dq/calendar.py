import pandas as pd
import exchange_calendars as xc


def load_trading_calendar(
    start_date: str, end_date: str, exchange: str = "XSHG"
) -> pd.DatetimeIndex:
    cal = xc.get_calendar(exchange)
    sessions = cal.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
    return pd.DatetimeIndex(sessions).tz_localize(None)
