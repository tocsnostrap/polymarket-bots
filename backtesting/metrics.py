"""
Performance metrics for backtesting results.

All functions accept a list/Series of per-trade PnL values (positive = win, negative = loss).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expected_value_per_dollar: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    kelly_fraction: float = 0.0
    kelly_half: float = 0.0
    trade_frequency_per_day: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    per_bucket: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    pnl: Sequence[float],
    trade_days: float = 90.0,
    risk_free_rate: float = 0.0,
    bankroll: float = 1000.0,
) -> PerformanceReport:
    pnl = np.array(pnl, dtype=float)
    r = PerformanceReport()

    if len(pnl) == 0:
        return r

    r.total_trades = len(pnl)
    r.wins = int(np.sum(pnl > 0))
    r.losses = int(np.sum(pnl <= 0))
    r.win_rate = r.wins / r.total_trades

    winners = pnl[pnl > 0]
    losers = pnl[pnl <= 0]

    r.gross_profit = float(np.sum(winners)) if len(winners) else 0.0
    r.gross_loss = float(np.sum(losers)) if len(losers) else 0.0
    r.net_profit = float(np.sum(pnl))

    r.profit_factor = abs(r.gross_profit / r.gross_loss) if r.gross_loss != 0 else float("inf")

    r.avg_win = float(np.mean(winners)) if len(winners) else 0.0
    r.avg_loss = float(np.mean(losers)) if len(losers) else 0.0
    r.expected_value_per_dollar = float(np.mean(pnl))

    r.best_trade = float(np.max(pnl))
    r.worst_trade = float(np.min(pnl))

    # Sharpe ratio (annualized, assuming ~96 trades/day for 15-min intervals)
    trades_per_day = r.total_trades / trade_days if trade_days > 0 else 1
    r.trade_frequency_per_day = trades_per_day

    if np.std(pnl) > 0:
        daily_returns = pnl.reshape(-1, max(1, int(trades_per_day))).sum(axis=1) if trades_per_day > 1 else pnl
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        r.sharpe_ratio = float((mean_daily - risk_free_rate / 365) / std_daily * math.sqrt(365)) if std_daily > 0 else 0.0

        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 1e-9
        r.sortino_ratio = float((mean_daily - risk_free_rate / 365) / downside_std * math.sqrt(365))

    # Max drawdown
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    r.max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    r.max_drawdown_pct = float(r.max_drawdown / bankroll * 100) if bankroll > 0 else 0.0

    # Kelly criterion: f* = (bp - q) / b
    # where b = avg_win/avg_loss_abs, p = win_rate, q = 1-p
    if r.avg_loss != 0 and r.win_rate > 0:
        b = abs(r.avg_win / r.avg_loss)
        p = r.win_rate
        q = 1.0 - p
        kelly = (b * p - q) / b
        r.kelly_fraction = max(0.0, float(kelly))
        r.kelly_half = r.kelly_fraction / 2.0

    return r


def bucket_metrics(
    trades: pd.DataFrame,
    group_col: str,
    pnl_col: str = "pnl",
    trade_days: float = 90.0,
) -> dict[str, dict]:
    """Compute metrics for each bucket/group in a trades DataFrame."""
    result = {}
    for name, group in trades.groupby(group_col):
        m = compute_metrics(group[pnl_col].values, trade_days=trade_days)
        result[str(name)] = m.to_dict()
    return result


def format_report(report: PerformanceReport, title: str = "Performance Report") -> str:
    lines = [
        f"\n{'=' * 60}",
        f"  {title}",
        f"{'=' * 60}",
        f"  Total Trades:           {report.total_trades:>10,}",
        f"  Wins / Losses:          {report.wins:>6,} / {report.losses:<6,}",
        f"  Win Rate:               {report.win_rate:>10.2%}",
        f"  Net Profit:            ${report.net_profit:>10,.2f}",
        f"  Gross Profit:          ${report.gross_profit:>10,.2f}",
        f"  Gross Loss:            ${report.gross_loss:>10,.2f}",
        f"  Profit Factor:          {report.profit_factor:>10.2f}",
        f"  Avg Win:               ${report.avg_win:>10.4f}",
        f"  Avg Loss:              ${report.avg_loss:>10.4f}",
        f"  EV per $1 Trade:       ${report.expected_value_per_dollar:>10.4f}",
        f"  Best Trade:            ${report.best_trade:>10.4f}",
        f"  Worst Trade:           ${report.worst_trade:>10.4f}",
        f"  Sharpe Ratio:           {report.sharpe_ratio:>10.2f}",
        f"  Sortino Ratio:          {report.sortino_ratio:>10.2f}",
        f"  Max Drawdown:          ${report.max_drawdown:>10.2f} ({report.max_drawdown_pct:.1f}%)",
        f"  Kelly Fraction:         {report.kelly_fraction:>10.4f}",
        f"  Half-Kelly:             {report.kelly_half:>10.4f}",
        f"  Trades/Day:             {report.trade_frequency_per_day:>10.1f}",
        f"{'=' * 60}",
    ]
    return "\n".join(lines)
