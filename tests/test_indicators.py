import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_fetcher import make_synthetic_ohlcv
from src.indicators import add_all_indicators


def test_add_all_indicators_columns_present():
    df = make_synthetic_ohlcv("TEST", days=200, seed=1)
    out = add_all_indicators(df)
    for col in ["ma5", "ma20", "ma60", "macd_dif", "macd_dea", "rsi14",
                "kdj_k", "kdj_d", "boll_upper", "boll_lower", "atr14", "obv"]:
        assert col in out.columns
    assert len(out) == len(df)


def test_rsi_bounds():
    df = make_synthetic_ohlcv("TEST2", days=200, seed=2)
    out = add_all_indicators(df)
    valid = out["rsi14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_bollinger_ordering():
    df = make_synthetic_ohlcv("TEST3", days=200, seed=3)
    out = add_all_indicators(df).dropna(subset=["boll_upper", "boll_lower"])
    assert (out["boll_upper"] >= out["boll_lower"]).all()
