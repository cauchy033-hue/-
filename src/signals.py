"""Rule-based buy/sell signal generation from technical indicators.

Each indicator contributes a small integer score (positive = bullish,
negative = bearish) to a composite per-day score. Local extrema of that
composite score above/below a threshold become discrete buy/sell points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BUY_THRESHOLD = 3
SELL_THRESHOLD = -3


@dataclass
class SignalPoint:
    date: pd.Timestamp
    price: float
    kind: str  # "BUY" or "SELL"
    score: int
    reasons: list[str]


@dataclass
class SignalReport:
    points: list[SignalPoint] = field(default_factory=list)
    scored_df: pd.DataFrame | None = None
    current_score: int = 0
    current_recommendation: str = "HOLD"
    current_reasons: list[str] = field(default_factory=list)
    support: float | None = None
    resistance: float | None = None


def _score_row(row: pd.Series, prev: pd.Series) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if not np.isnan(row.get("ma5", np.nan)) and not np.isnan(row.get("ma20", np.nan)):
        if prev["ma5"] <= prev["ma20"] and row["ma5"] > row["ma20"]:
            score += 2
            reasons.append("5日均线上穿20日均线（金叉）")
        elif prev["ma5"] >= prev["ma20"] and row["ma5"] < row["ma20"]:
            score -= 2
            reasons.append("5日均线下穿20日均线（死叉）")
        elif row["ma5"] > row["ma20"]:
            score += 1
        else:
            score -= 1

    if not np.isnan(row.get("macd_dif", np.nan)):
        if prev["macd_dif"] <= prev["macd_dea"] and row["macd_dif"] > row["macd_dea"]:
            score += 2
            reasons.append("MACD金叉（DIF上穿DEA）")
        elif prev["macd_dif"] >= prev["macd_dea"] and row["macd_dif"] < row["macd_dea"]:
            score -= 2
            reasons.append("MACD死叉（DIF下穿DEA）")

    if not np.isnan(row.get("rsi14", np.nan)):
        if row["rsi14"] < 30:
            score += 1
            reasons.append(f"RSI超卖（{row['rsi14']:.1f}）")
        elif row["rsi14"] > 70:
            score -= 1
            reasons.append(f"RSI超买（{row['rsi14']:.1f}）")

    if not np.isnan(row.get("kdj_k", np.nan)):
        if prev["kdj_k"] <= prev["kdj_d"] and row["kdj_k"] > row["kdj_d"] and row["kdj_k"] < 30:
            score += 1
            reasons.append("KDJ低位金叉")
        elif prev["kdj_k"] >= prev["kdj_d"] and row["kdj_k"] < row["kdj_d"] and row["kdj_k"] > 70:
            score -= 1
            reasons.append("KDJ高位死叉")

    if not np.isnan(row.get("boll_lower", np.nan)):
        if row["Close"] <= row["boll_lower"]:
            score += 1
            reasons.append("股价触及布林带下轨")
        elif row["Close"] >= row["boll_upper"]:
            score -= 1
            reasons.append("股价触及布林带上轨")

    if not np.isnan(row.get("volume_ratio", np.nan)) and row["volume_ratio"] > 1.5:
        if score > 0:
            score += 1
            reasons.append(f"放量上涨（量比{row['volume_ratio']:.1f}）")
        elif score < 0:
            score -= 1
            reasons.append(f"放量下跌（量比{row['volume_ratio']:.1f}）")

    return score, reasons


def generate_signals(df: pd.DataFrame) -> SignalReport:
    """``df`` must already contain the indicator columns from ``indicators.add_all_indicators``."""
    scores = []
    reasons_col = []
    for i in range(len(df)):
        if i == 0:
            scores.append(0)
            reasons_col.append([])
            continue
        s, r = _score_row(df.iloc[i], df.iloc[i - 1])
        scores.append(s)
        reasons_col.append(r)

    scored = df.copy()
    scored["signal_score"] = scores
    scored["signal_reasons"] = reasons_col

    points: list[SignalPoint] = []
    for date, row in scored.iterrows():
        if row["signal_score"] >= BUY_THRESHOLD:
            points.append(SignalPoint(date, float(row["Close"]), "BUY", int(row["signal_score"]), row["signal_reasons"]))
        elif row["signal_score"] <= SELL_THRESHOLD:
            points.append(SignalPoint(date, float(row["Close"]), "SELL", int(row["signal_score"]), row["signal_reasons"]))

    lookback = scored.tail(60)
    support = float(lookback["Low"].min()) if not lookback.empty else None
    resistance = float(lookback["High"].max()) if not lookback.empty else None

    last = scored.iloc[-1]
    current_score = int(last["signal_score"])
    if current_score >= BUY_THRESHOLD:
        rec = "BUY"
    elif current_score >= 1:
        rec = "WEAK_BUY"
    elif current_score <= SELL_THRESHOLD:
        rec = "SELL"
    elif current_score <= -1:
        rec = "WEAK_SELL"
    else:
        rec = "HOLD"

    return SignalReport(
        points=points,
        scored_df=scored,
        current_score=current_score,
        current_recommendation=rec,
        current_reasons=last["signal_reasons"],
        support=support,
        resistance=resistance,
    )
