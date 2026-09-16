"""Renders the combined analysis (signals + prediction + backtest) as text and a chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

for _font in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei"):
    if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.sans-serif"] = [_font]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

from .backtester import BacktestResult
from .predictor import PredictionResult
from .signals import SignalReport

RECOMMENDATION_LABELS = {
    "BUY": "买入",
    "WEAK_BUY": "偏多 / 轻仓买入",
    "HOLD": "观望",
    "WEAK_SELL": "偏空 / 减仓",
    "SELL": "卖出",
}


def build_text_report(code: str, source: str, signal_report: SignalReport,
                       prediction: PredictionResult, backtest: BacktestResult) -> str:
    df = signal_report.scored_df
    last = df.iloc[-1]
    lines = []
    lines.append(f"===== 股票量化分析报告：{code} (数据源: {source}) =====")
    lines.append(f"分析日期: {df.index[-1].date()}    最新收盘价: {last['Close']:.2f}")
    lines.append("")

    lines.append("【当前信号】")
    rec = RECOMMENDATION_LABELS.get(signal_report.current_recommendation, signal_report.current_recommendation)
    lines.append(f"  综合评分: {signal_report.current_score:+d}   建议: {rec}")
    if signal_report.current_reasons:
        lines.append("  依据: " + "；".join(signal_report.current_reasons))
    else:
        lines.append("  依据: 当前无明显技术信号触发，指标处于中性区间")
    lines.append(f"  近期支撑位: {signal_report.support:.2f}   近期压力位: {signal_report.resistance:.2f}")
    lines.append("")

    lines.append("【关键技术指标】")
    lines.append(f"  MA5 / MA20 / MA60: {last['ma5']:.2f} / {last['ma20']:.2f} / {last['ma60']:.2f}")
    lines.append(f"  MACD (DIF/DEA/HIST): {last['macd_dif']:.3f} / {last['macd_dea']:.3f} / {last['macd_hist']:.3f}")
    lines.append(f"  RSI(14): {last['rsi14']:.1f}")
    lines.append(f"  KDJ (K/D/J): {last['kdj_k']:.1f} / {last['kdj_d']:.1f} / {last['kdj_j']:.1f}")
    lines.append(f"  布林带 (上/中/下): {last['boll_upper']:.2f} / {last['boll_mid']:.2f} / {last['boll_lower']:.2f}")
    lines.append(f"  20日波动率: {last['volatility20'] * 100:.2f}%   量比: {last['volume_ratio']:.2f}")
    lines.append("")

    lines.append(f"【近期历史买卖点】(最近{min(10, len(signal_report.points))}个)")
    for p in signal_report.points[-10:]:
        tag = "买入" if p.kind == "BUY" else "卖出"
        reason = "；".join(p.reasons) if p.reasons else "-"
        lines.append(f"  {p.date.date()}  {tag}  价格 {p.price:.2f}  评分 {p.score:+d}  ({reason})")
    if not signal_report.points:
        lines.append("  历史数据中未触发达到阈值的强烈买卖点。")
    lines.append("")

    lines.append(f"【机器学习预测】(未来 {prediction.horizon_days} 个交易日)")
    lines.append(f"  上涨概率: {prediction.up_probability * 100:.1f}%   预测方向: {prediction.direction}")
    lines.append(f"  预期收益率: {prediction.expected_return_pct:+.2f}%   预测目标价: {prediction.predicted_price:.2f}")
    lines.append(f"  模型样本外表现: 准确率 {prediction.test_accuracy * 100:.1f}% / "
                 f"精确率 {prediction.test_precision * 100:.1f}% / 召回率 {prediction.test_recall * 100:.1f}% "
                 f"(训练集{prediction.n_train}条 / 测试集{prediction.n_test}条)")
    top_features = list(prediction.feature_importance.items())[:5]
    lines.append("  最重要特征: " + "、".join(f"{k}({v:.3f})" for k, v in top_features))
    lines.append("")

    lines.append("【历史信号回测】(与买入持有对比)")
    lines.append(f"  策略累计收益: {backtest.total_return_pct:+.2f}%   年化收益: {backtest.annualized_return_pct:+.2f}%")
    lines.append(f"  最大回撤: {backtest.max_drawdown_pct:.2f}%   夏普比率: {backtest.sharpe:.2f}")
    lines.append(f"  交易次数: {backtest.n_trades}   胜率: {backtest.win_rate_pct:.1f}%")
    lines.append(f"  同期买入持有收益: {backtest.buy_hold_return_pct:+.2f}%")
    lines.append("")

    lines.append("【风险提示】")
    lines.append("  本报告由技术指标规则与统计学习模型自动生成，仅供参考，不构成投资建议。")
    lines.append("  历史表现不代表未来收益，股市有风险，投资需谨慎，请结合基本面与仓位管理自主决策。")

    return "\n".join(lines)


def plot_chart(code: str, signal_report: SignalReport, out_path: str | Path) -> Path:
    df = signal_report.scored_df.tail(250)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    ax1.plot(df.index, df["Close"], label="收盘价", color="black", linewidth=1)
    ax1.plot(df.index, df["ma5"], label="MA5", linewidth=0.8)
    ax1.plot(df.index, df["ma20"], label="MA20", linewidth=0.8)
    ax1.plot(df.index, df["ma60"], label="MA60", linewidth=0.8)
    ax1.fill_between(df.index, df["boll_lower"], df["boll_upper"], color="gray", alpha=0.1, label="布林带")

    buys = [p for p in signal_report.points if p.kind == "BUY" and p.date in df.index]
    sells = [p for p in signal_report.points if p.kind == "SELL" and p.date in df.index]
    if buys:
        ax1.scatter([p.date for p in buys], [p.price for p in buys], marker="^", color="red", s=90,
                    label="买点", zorder=5)
    if sells:
        ax1.scatter([p.date for p in sells], [p.price for p in sells], marker="v", color="green", s=90,
                    label="卖点", zorder=5)

    ax1.set_title(f"{code} 股价走势与买卖点")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(df.index, df["macd_dif"], label="DIF", linewidth=0.8)
    ax2.plot(df.index, df["macd_dea"], label="DEA", linewidth=0.8)
    ax2.bar(df.index, df["macd_hist"], label="MACD柱", width=1.0, alpha=0.4)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylabel("MACD")
    ax2.grid(alpha=0.3)

    ax3.plot(df.index, df["rsi14"], label="RSI14", color="purple", linewidth=0.8)
    ax3.axhline(70, color="red", linestyle="--", linewidth=0.6)
    ax3.axhline(30, color="green", linestyle="--", linewidth=0.6)
    ax3.set_ylabel("RSI")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
