import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_fetcher import make_synthetic_ohlcv
from src.indicators import add_all_indicators
from src.signals import generate_signals


def test_generate_signals_basic_shape():
    df = make_synthetic_ohlcv("TEST", days=300, seed=42)
    enriched = add_all_indicators(df)
    report = generate_signals(enriched)

    assert report.scored_df is not None
    assert "signal_score" in report.scored_df.columns
    assert report.current_recommendation in {"BUY", "WEAK_BUY", "HOLD", "WEAK_SELL", "SELL"}
    assert report.support is not None and report.resistance is not None
    assert report.support <= report.resistance


def test_signal_points_have_valid_kind():
    df = make_synthetic_ohlcv("TEST", days=300, seed=7)
    enriched = add_all_indicators(df)
    report = generate_signals(enriched)
    for p in report.points:
        assert p.kind in {"BUY", "SELL"}
        assert p.price > 0
