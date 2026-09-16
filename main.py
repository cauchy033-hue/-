#!/usr/bin/env python3
"""CLI entry point: given a stock code, print a full quant analysis report and save a chart.

Examples:
    python main.py AAPL
    python main.py 600519 --period 3y
    python main.py 0700.HK --predict-days 10
    python main.py 600519 --demo   # offline test with synthetic data, no network needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.backtester import run_backtest
from src.data_fetcher import FetchResult, detect_market, fetch_history, load_csv
from src.indicators import add_all_indicators
from src.predictor import predict
from src.report import build_text_report, plot_chart
from src.signals import generate_signals

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def analyze(code: str, period: str = "2y", predict_days: int = 5, demo: bool = False,
            save_chart: bool = True, csv_path: str | None = None) -> str:
    if csv_path:
        result = FetchResult(code, detect_market(code), load_csv(csv_path), source=f"csv:{csv_path}")
    else:
        result = fetch_history(code, period=period, demo=demo)
    df = result.df
    if len(df) < 80:
        raise ValueError(f"Only {len(df)} rows of data fetched; need at least ~80 for reliable indicators.")

    enriched = add_all_indicators(df)
    signal_report = generate_signals(enriched)
    prediction = predict(enriched, horizon_days=predict_days)
    backtest = run_backtest(signal_report.scored_df)

    text = build_text_report(result.code, result.source, signal_report, prediction, backtest)

    if save_chart:
        OUTPUT_DIR.mkdir(exist_ok=True)
        chart_path = OUTPUT_DIR / f"{result.code.replace('.', '_')}_chart.png"
        plot_chart(result.code, signal_report, chart_path)
        text += f"\n\n图表已保存至: {chart_path}"

    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="股票量化分析：输入代码，输出买卖点与预测分析")
    parser.add_argument("code", help="股票代码，如 AAPL / 600519 / 0700.HK")
    parser.add_argument("--period", default="2y", help="历史数据时长，如 1y/2y/5y/500d (默认 2y)")
    parser.add_argument("--predict-days", type=int, default=5, help="预测未来N个交易日走势 (默认 5)")
    parser.add_argument("--no-chart", action="store_true", help="跳过图表生成")
    parser.add_argument("--demo", action="store_true", help="离线演示模式：使用随机生成的模拟数据，无需联网")
    parser.add_argument("--csv", help="使用本地CSV历史行情文件而非联网获取（需包含日期/开盘/最高/最低/收盘/成交量列）")
    args = parser.parse_args()

    try:
        report = analyze(
            args.code, period=args.period, predict_days=args.predict_days,
            demo=args.demo, save_chart=not args.no_chart, csv_path=args.csv,
        )
        print(report)
    except Exception as exc:  # noqa: BLE001
        print(f"分析失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
