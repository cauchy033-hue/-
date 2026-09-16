"""Simple long-only backtest of the rule-based signals, compared to buy-and-hold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import BUY_THRESHOLD, SELL_THRESHOLD


@dataclass
class BacktestResult:
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    n_trades: int
    sharpe: float
    buy_hold_return_pct: float
    equity_curve: pd.Series


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min() * 100)


def run_backtest(scored_df: pd.DataFrame, initial_capital: float = 100_000.0) -> BacktestResult:
    """``scored_df`` is the output of ``signals.generate_signals(...).scored_df``."""
    df = scored_df.dropna(subset=["signal_score"]).copy()
    position = 0  # 0 = flat, 1 = long
    cash = initial_capital
    shares = 0.0
    equity = []
    trade_returns = []
    entry_price = None

    for _, row in df.iterrows():
        price = row["Close"]
        score = row["signal_score"]

        if position == 0 and score >= BUY_THRESHOLD:
            shares = cash / price
            cash = 0.0
            position = 1
            entry_price = price
        elif position == 1 and score <= SELL_THRESHOLD:
            cash = shares * price
            trade_returns.append(price / entry_price - 1)
            shares = 0.0
            position = 0
            entry_price = None

        equity.append(cash + shares * price)

    if position == 1:
        trade_returns.append(df["Close"].iloc[-1] / entry_price - 1)

    equity_curve = pd.Series(equity, index=df.index)
    total_return = equity_curve.iloc[-1] / initial_capital - 1
    n_days = len(equity_curve)
    years = max(n_days / 252, 1e-6)
    annualized_return = (1 + total_return) ** (1 / years) - 1

    daily_returns = equity_curve.pct_change().dropna()
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

    wins = [r for r in trade_returns if r > 0]
    win_rate = (len(wins) / len(trade_returns) * 100) if trade_returns else 0.0

    buy_hold_return = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1

    return BacktestResult(
        total_return_pct=total_return * 100,
        annualized_return_pct=annualized_return * 100,
        max_drawdown_pct=_max_drawdown(equity_curve),
        win_rate_pct=win_rate,
        n_trades=len(trade_returns),
        sharpe=sharpe,
        buy_hold_return_pct=buy_hold_return * 100,
        equity_curve=equity_curve,
    )
