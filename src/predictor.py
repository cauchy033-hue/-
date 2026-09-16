"""Machine-learning direction/return prediction on top of the technical indicator features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score

FEATURE_COLUMNS = [
    "rsi14", "macd_dif", "macd_dea", "macd_hist",
    "kdj_k", "kdj_d", "kdj_j",
    "volatility20", "volume_ratio", "atr14",
]


@dataclass
class PredictionResult:
    horizon_days: int
    up_probability: float
    direction: str
    expected_return_pct: float
    predicted_price: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    feature_importance: dict[str, float]
    n_train: int
    n_test: int


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    feats["price_vs_ma5"] = feats["Close"] / feats["ma5"] - 1
    feats["price_vs_ma20"] = feats["Close"] / feats["ma20"] - 1
    feats["price_vs_ma60"] = feats["Close"] / feats["ma60"] - 1
    feats["ma5_vs_ma20"] = feats["ma5"] / feats["ma20"] - 1
    for lag in (1, 2, 3, 5):
        feats[f"return_lag{lag}"] = feats["daily_return"].shift(lag - 1)
    return feats


ALL_FEATURES = FEATURE_COLUMNS + [
    "price_vs_ma5", "price_vs_ma20", "price_vs_ma60", "ma5_vs_ma20",
    "return_lag1", "return_lag2", "return_lag3", "return_lag5",
]


def predict(df: pd.DataFrame, horizon_days: int = 5, test_fraction: float = 0.2) -> PredictionResult:
    """Train on all-but-the-last-``horizon_days`` rows, predict the current setup's outlook.

    Uses a chronological (non-shuffled) split, appropriate for time series: the model
    never sees future data during "training" evaluation.
    """
    feats = _build_features(df)
    feats["future_return"] = feats["Close"].shift(-horizon_days) / feats["Close"] - 1
    feats["future_up"] = (feats["future_return"] > 0).astype(int)

    usable = feats.dropna(subset=ALL_FEATURES + ["future_return"])
    if len(usable) < 60:
        raise ValueError("Not enough history to train a reliable model (need 60+ usable rows).")

    X = usable[ALL_FEATURES]
    y_class = usable["future_up"]
    y_reg = usable["future_return"]

    split = int(len(usable) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_class.iloc[:split], y_class.iloc[split:]
    y_reg_train = y_reg.iloc[:split]

    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    reg = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    reg.fit(X_train, y_reg_train)

    latest_features = feats[ALL_FEATURES].iloc[[-1]]
    if latest_features.isna().any(axis=None):
        latest_features = latest_features.fillna(X_train.median())

    up_prob = float(clf.predict_proba(latest_features)[0][1])
    expected_return = float(reg.predict(latest_features)[0])
    current_price = float(df["Close"].iloc[-1])
    predicted_price = current_price * (1 + expected_return)

    direction = "UP" if up_prob >= 0.55 else ("DOWN" if up_prob <= 0.45 else "NEUTRAL")

    importance = dict(zip(ALL_FEATURES, clf.feature_importances_.round(4).tolist()))
    importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    return PredictionResult(
        horizon_days=horizon_days,
        up_probability=up_prob,
        direction=direction,
        expected_return_pct=expected_return * 100,
        predicted_price=predicted_price,
        test_accuracy=accuracy,
        test_precision=precision,
        test_recall=recall,
        feature_importance=importance,
        n_train=len(X_train),
        n_test=len(X_test),
    )
