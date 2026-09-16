"""Fetches OHLCV history for a stock code, auto-detecting the market (US / HK / China A-share).

Live fetches require outbound network access to Yahoo Finance (yfinance) or
akshare's data sources. Results are cached to CSV under ``data_cache/`` so a
repeated request for the same code/period does not re-hit the network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

_CN_CODE_RE = re.compile(r"^\d{6}$")
_HK_CODE_RE = re.compile(r"^\d{4,5}\.HK$", re.IGNORECASE)


@dataclass
class FetchResult:
    code: str
    market: str
    df: pd.DataFrame
    source: str


def detect_market(code: str) -> str:
    code = code.strip()
    if _HK_CODE_RE.match(code):
        return "hk"
    if _CN_CODE_RE.match(code):
        return "cn"
    return "us"


def _cn_yfinance_symbol(code: str) -> str:
    # Shanghai main board / STAR market starts with 6, Shenzhen/ChiNext with 0/3, Beijing with 4/8.
    suffix = ".SS" if code.startswith("6") else ".SZ"
    return code + suffix


def _fetch_yfinance(symbol: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _period_to_days(period: str) -> int:
    unit = period[-1]
    value = int(period[:-1]) if period[:-1].isdigit() else 2
    return {"d": 1, "mo": 30, "y": 365}.get(unit, 365) * value


def _fetch_akshare(code: str, period: str) -> pd.DataFrame:
    import akshare as ak

    days = _period_to_days(period)
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y%m%d")
    end = pd.Timestamp.today().strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        raise RuntimeError(f"akshare returned no data for {code}")
    df = df.rename(columns={
        "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def make_synthetic_ohlcv(code: str, days: int = 500, seed: int | None = None) -> pd.DataFrame:
    """Generate a plausible-looking random-walk OHLCV series for offline demos and tests."""
    rng = np.random.default_rng(seed if seed is not None else abs(hash(code)) % (2**32))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)

    drift = 0.0002
    vol = 0.018
    shocks = rng.normal(drift, vol, size=days)
    # A few momentum regimes so indicators have real crossovers to react to.
    for start in range(0, days, 60):
        regime = rng.choice([-1, 0, 1], p=[0.3, 0.3, 0.4])
        shocks[start:start + 60] += regime * 0.0015

    close = 50 * np.exp(np.cumsum(shocks))
    daily_range = close * rng.uniform(0.005, 0.03, size=days)
    open_ = close * (1 + rng.normal(0, 0.005, size=days))
    high = np.maximum(open_, close) + daily_range * rng.uniform(0.1, 0.6, size=days)
    low = np.minimum(open_, close) - daily_range * rng.uniform(0.1, 0.6, size=days)
    volume = rng.integers(1_000_000, 20_000_000, size=days).astype(float)
    volume *= 1 + np.abs(shocks) * 10  # higher volume on bigger moves

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    return df


def fetch_history(code: str, period: str = "2y", market: str | None = None,
                   use_cache: bool = True, demo: bool = False) -> FetchResult:
    """Fetch OHLCV history for ``code``. Falls back to cache, then raises if all sources fail."""
    code = code.strip().upper() if detect_market(code) != "cn" else code.strip()
    market = market or detect_market(code)
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{code.replace('.', '_')}_{period}.csv"

    if demo:
        return FetchResult(code, market, make_synthetic_ohlcv(code), source="synthetic-demo")

    errors = []
    try:
        if market == "cn":
            df = _fetch_akshare(code, period)
            source = "akshare"
        elif market == "hk":
            df = _fetch_yfinance(code, period)
            source = "yfinance"
        else:
            df = _fetch_yfinance(code, period)
            source = "yfinance"
        if use_cache:
            df.to_csv(cache_path)
        return FetchResult(code, market, df, source=source)
    except Exception as exc:  # noqa: BLE001 - we want to try every fallback below
        errors.append(str(exc))

    if market == "cn":
        try:
            symbol = _cn_yfinance_symbol(code)
            df = _fetch_yfinance(symbol, period)
            if use_cache:
                df.to_csv(cache_path)
            return FetchResult(code, market, df, source="yfinance-fallback")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return FetchResult(code, market, df, source="cache")

    raise RuntimeError(
        f"Could not fetch data for '{code}' (market={market}). Tried: {errors}. "
        "Pass demo=True / --demo to test offline with synthetic data, or check network access."
    )
