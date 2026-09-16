import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtester import run_backtest
from src.data_fetcher import make_synthetic_ohlcv
from src.indicators import add_all_indicators
from src.signals import generate_signals


def test_backtest_runs_and_returns_reasonable_fields():
    df = make_synthetic_ohlcv("TEST", days=300, seed=99)
    enriched = add_all_indicators(df)
    report = generate_signals(enriched)
    result = run_backtest(report.scored_df)

    assert isinstance(result.total_return_pct, float)
    assert result.max_drawdown_pct <= 0
    assert 0 <= result.win_rate_pct <= 100
    assert len(result.equity_curve) > 0
