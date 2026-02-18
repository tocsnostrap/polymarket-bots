#!/usr/bin/env python3
"""
Phase 1 – 15-Minute BTC Momentum Strategy Validation

Reproduces the Polymarket 15-minute BTC Up/Down strategy from polymarket-bot:
    1. Fetches 90 days of BTC 1-minute candles (Coinbase).
    2. Constructs synthetic 15-minute intervals aligned to the quarter-hour.
    3. For every interval, at each candidate entry second (60-840 s elapsed),
       measures the BTC % move from the interval open, looks up the
       calibrated fair-value probability, and decides whether to trade.
    4. Resolution: did BTC close the 15-min interval higher or lower than open?
    5. PnL = payout based on implied Polymarket pricing vs fair value.
    6. Outputs detailed metrics, per-bucket validation, and charts.

Usage:
    python 15min_backtest.py              # normal run (uses cached data)
    python 15min_backtest.py --refresh    # force re-download
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from data_fetcher import fetch_btc_candles
from metrics import (
    PerformanceReport,
    bucket_metrics,
    compute_metrics,
    format_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Calibrated fair-value table from polymarket-bot/STRATEGY.md ──────────────
# (move_low%, move_high%, elapsed_low_s, elapsed_high_s, win_probability)
FAIR_VALUE_TABLE: list[tuple[float, float, int, int, float]] = [
    # move 0.03-0.05 %
    (0.03, 0.05, 60, 180, 0.5976),
    (0.03, 0.05, 180, 420, 0.6522),
    (0.03, 0.05, 420, 600, 0.6968),
    (0.03, 0.05, 600, 840, 0.8003),
    # move 0.05-0.10 %
    (0.05, 0.10, 60, 180, 0.6546),
    (0.05, 0.10, 180, 420, 0.7144),
    (0.05, 0.10, 420, 600, 0.7733),
    (0.05, 0.10, 600, 840, 0.8714),
    # move 0.10-0.20 %
    (0.10, 0.20, 60, 180, 0.6864),
    (0.10, 0.20, 180, 420, 0.7772),
    (0.10, 0.20, 420, 600, 0.8557),
    (0.10, 0.20, 600, 840, 0.9433),
    # move 0.20 %+
    (0.20, 100.0, 60, 180, 0.7454),
    (0.20, 100.0, 180, 420, 0.8672),
    (0.20, 100.0, 420, 600, 0.9360),
    (0.20, 100.0, 600, 840, 0.9823),
]

# Minimum edge over the market price to enter (basis points of probability)
MIN_EDGE = 0.03
# Market spread (simulated): fair_value → market_price = fair_value - spread/2
MARKET_SPREAD = 0.04
TRADE_SIZE = 1.0  # $1 per trade for EV calculations


class Interval(NamedTuple):
    start: pd.Timestamp
    open_price: float
    close_price: float
    went_up: bool


class TradeRecord(NamedTuple):
    interval_start: pd.Timestamp
    direction: str       # "UP" or "DOWN"
    move_pct: float
    elapsed_s: int
    fair_value: float
    market_price: float
    edge: float
    won: bool
    pnl: float
    move_bucket: str
    time_bucket: str


def lookup_fair_value(abs_move_pct: float, elapsed_s: int) -> float | None:
    for mlo, mhi, tlo, thi, fv in FAIR_VALUE_TABLE:
        if mlo <= abs_move_pct < mhi and tlo <= elapsed_s < thi:
            return fv
    return None


def _bucket_label(move_pct: float, elapsed_s: int) -> tuple[str, str]:
    if move_pct < 0.03:
        mb = "<0.03%"
    elif move_pct < 0.05:
        mb = "0.03-0.05%"
    elif move_pct < 0.10:
        mb = "0.05-0.10%"
    elif move_pct < 0.20:
        mb = "0.10-0.20%"
    else:
        mb = "0.20%+"

    if elapsed_s < 180:
        tb = "60-180s"
    elif elapsed_s < 420:
        tb = "180-420s"
    elif elapsed_s < 600:
        tb = "420-600s"
    else:
        tb = "600-840s"
    return mb, tb


def build_intervals(candles: pd.DataFrame) -> list[Interval]:
    """Group 1-minute candles into aligned 15-minute intervals."""
    df = candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Align to 15-minute boundaries
    df["interval"] = df.index.floor("15min")

    intervals: list[Interval] = []
    for ts, group in df.groupby("interval"):
        if len(group) < 10:
            continue
        open_price = group.iloc[0]["open"]
        close_price = group.iloc[-1]["close"]
        went_up = close_price > open_price
        intervals.append(Interval(start=ts, open_price=open_price, close_price=close_price, went_up=went_up))

    log.info("Built %d valid 15-minute intervals", len(intervals))
    return intervals


def simulate_entry_points(
    candles: pd.DataFrame,
    intervals: list[Interval],
) -> list[TradeRecord]:
    """
    For each interval, check every minute from 60 s to 840 s elapsed.
    If the BTC move maps to a fair-value bucket and offers sufficient edge
    over the simulated market price, record a trade.
    Only the *first* qualifying entry per interval is taken (single entry).
    """
    df = candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    trades: list[TradeRecord] = []

    for iv in intervals:
        window = df.loc[iv.start : iv.start + pd.Timedelta(minutes=14)]
        if len(window) < 2:
            continue
        open_price = iv.open_price

        entered = False
        for i, (ts, row) in enumerate(window.iterrows()):
            elapsed_s = int((ts - iv.start).total_seconds())
            if elapsed_s < 60 or elapsed_s > 840:
                continue

            current_price = row["close"]
            move_pct = abs(current_price - open_price) / open_price * 100

            fv = lookup_fair_value(move_pct, elapsed_s)
            if fv is None:
                continue

            direction = "UP" if current_price > open_price else "DOWN"

            # Simulated market price: slightly worse than fair value
            market_price = fv - MARKET_SPREAD / 2
            edge = fv - market_price

            if edge < MIN_EDGE:
                continue

            # Did the trade win?
            if direction == "UP":
                won = iv.went_up
            else:
                won = not iv.went_up

            # Binary payout: pay market_price, receive $1 if win, $0 if lose
            pnl = (TRADE_SIZE - market_price * TRADE_SIZE) if won else (-market_price * TRADE_SIZE)

            mb, tb = _bucket_label(move_pct, elapsed_s)

            trades.append(TradeRecord(
                interval_start=iv.start,
                direction=direction,
                move_pct=move_pct,
                elapsed_s=elapsed_s,
                fair_value=fv,
                market_price=market_price,
                edge=edge,
                won=won,
                pnl=pnl,
                move_bucket=mb,
                time_bucket=tb,
            ))
            entered = True
            break  # one entry per interval

    log.info("Simulated %d trades across %d intervals", len(trades), len(intervals))
    return trades


def validate_win_rates(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compare simulated win rates against the claimed calibration table."""
    claimed = {
        ("0.03-0.05%", "60-180s"): 0.5976,
        ("0.03-0.05%", "180-420s"): 0.6522,
        ("0.03-0.05%", "420-600s"): 0.6968,
        ("0.03-0.05%", "600-840s"): 0.8003,
        ("0.05-0.10%", "60-180s"): 0.6546,
        ("0.05-0.10%", "180-420s"): 0.7144,
        ("0.05-0.10%", "420-600s"): 0.7733,
        ("0.05-0.10%", "600-840s"): 0.8714,
        ("0.10-0.20%", "60-180s"): 0.6864,
        ("0.10-0.20%", "180-420s"): 0.7772,
        ("0.10-0.20%", "420-600s"): 0.8557,
        ("0.10-0.20%", "600-840s"): 0.9433,
        ("0.20%+", "60-180s"): 0.7454,
        ("0.20%+", "180-420s"): 0.8672,
        ("0.20%+", "420-600s"): 0.9360,
        ("0.20%+", "600-840s"): 0.9823,
    }

    rows = []
    for (mb, tb), claimed_wr in claimed.items():
        subset = trades_df[(trades_df["move_bucket"] == mb) & (trades_df["time_bucket"] == tb)]
        if len(subset) == 0:
            rows.append({
                "move_bucket": mb,
                "time_bucket": tb,
                "claimed_wr": claimed_wr,
                "actual_wr": None,
                "n_trades": 0,
                "delta": None,
            })
            continue
        actual_wr = subset["won"].mean()
        rows.append({
            "move_bucket": mb,
            "time_bucket": tb,
            "claimed_wr": claimed_wr,
            "actual_wr": actual_wr,
            "n_trades": len(subset),
            "delta": actual_wr - claimed_wr,
        })

    return pd.DataFrame(rows)


# ── Visualizations ───────────────────────────────────────────────────────────

def plot_equity_curve(trades_df: pd.DataFrame, path: Path):
    cumulative = trades_df["pnl"].cumsum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative.values, linewidth=1.2, color="#2196F3")
    ax.fill_between(range(len(cumulative)), cumulative.values, alpha=0.15, color="#2196F3")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("15-Minute Strategy – Equity Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved equity curve → %s", path)


def plot_win_rate_heatmap(validation_df: pd.DataFrame, path: Path, column: str = "actual_wr", title_suffix: str = "Actual"):
    pivot = validation_df.pivot(index="move_bucket", columns="time_bucket", values=column)
    move_order = ["0.03-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20%+"]
    time_order = ["60-180s", "180-420s", "420-600s", "600-840s"]
    pivot = pivot.reindex(index=move_order, columns=time_order)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        linewidths=1,
        ax=ax,
    )
    ax.set_title(f"15-Min Strategy Win Rates ({title_suffix})")
    ax.set_ylabel("BTC Move Size")
    ax.set_xlabel("Time Elapsed in Interval")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved heatmap → %s", path)


def plot_claimed_vs_actual(validation_df: pd.DataFrame, path: Path):
    vdf = validation_df.dropna(subset=["actual_wr"]).copy()
    if vdf.empty:
        return
    vdf["label"] = vdf["move_bucket"] + "\n" + vdf["time_bucket"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(vdf))
    width = 0.35
    ax.bar(x - width / 2, vdf["claimed_wr"], width, label="Claimed", color="#FF9800", alpha=0.85)
    ax.bar(x + width / 2, vdf["actual_wr"], width, label="Backtest", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(vdf["label"], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Win Rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Claimed vs Backtest Win Rates by Bucket")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved comparison chart → %s", path)


def plot_pnl_distribution(trades_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(trades_df["pnl"], bins=50, edgecolor="white", color="#673AB7", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(trades_df["pnl"].mean(), color="green", linestyle="--", linewidth=1.5, label=f"Mean = ${trades_df['pnl'].mean():.4f}")
    ax.set_xlabel("Trade PnL ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("PnL Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved PnL distribution → %s", path)


def plot_monthly_returns(trades_df: pd.DataFrame, path: Path):
    df = trades_df.copy()
    df["month"] = df["interval_start"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["pnl"].agg(["sum", "count", "mean"]).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in monthly["sum"]]
    ax1.bar(monthly["month"], monthly["sum"], color=colors, alpha=0.85)
    ax1.set_ylabel("Net PnL ($)")
    ax1.set_title("Monthly PnL")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(monthly["month"], monthly["count"], color="#2196F3", alpha=0.75)
    ax2.set_ylabel("Trade Count")
    ax2.set_xlabel("Month")
    ax2.set_title("Monthly Trade Volume")
    ax2.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved monthly returns → %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

async def run_backtest(days: int = 90, force_refresh: bool = False):
    log.info("═══ Phase 1: 15-Minute Strategy Validation ═══")

    # 1. Fetch data
    candles = await fetch_btc_candles(days=days, granularity=60, force_refresh=force_refresh)
    log.info("Data: %d candles from %s to %s",
             len(candles), candles["timestamp"].min(), candles["timestamp"].max())

    # 2. Build intervals
    intervals = build_intervals(candles)
    up_count = sum(1 for iv in intervals if iv.went_up)
    log.info("Intervals: %d total, %d up (%.1f%%), %d down (%.1f%%)",
             len(intervals), up_count, up_count / len(intervals) * 100,
             len(intervals) - up_count, (len(intervals) - up_count) / len(intervals) * 100)

    # 3. Simulate trades
    trades = simulate_entry_points(candles, intervals)
    if not trades:
        log.error("No trades generated — check data range or strategy params")
        return

    trades_df = pd.DataFrame(trades)

    # 4. Compute overall metrics
    overall = compute_metrics(trades_df["pnl"].values, trade_days=days)
    print(format_report(overall, "15-Minute Strategy – Overall"))

    # 5. Validate claimed vs actual win rates
    validation = validate_win_rates(trades_df)
    print("\n── Win Rate Validation ─────────────────────────────────────")
    print(validation.to_string(index=False, float_format="%.4f"))

    # 6. Per-bucket metrics
    trades_df["bucket"] = trades_df["move_bucket"] + " / " + trades_df["time_bucket"]
    per_bucket = bucket_metrics(trades_df, "bucket", trade_days=days)

    # 7. Save results JSON
    result = {
        "strategy": "15min_btc_momentum",
        "data_days": days,
        "total_intervals": len(intervals),
        "overall": overall.to_dict(),
        "validation": validation.to_dict(orient="records"),
        "per_bucket": per_bucket,
    }
    results_path = RESULTS_DIR / "15min_performance.json"
    results_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("Results saved → %s", results_path)

    # 8. Generate charts
    plot_equity_curve(trades_df, RESULTS_DIR / "15min_equity_curve.png")
    plot_win_rate_heatmap(validation, RESULTS_DIR / "15min_winrate_actual.png", "actual_wr", "Backtest")
    plot_win_rate_heatmap(validation, RESULTS_DIR / "15min_winrate_claimed.png", "claimed_wr", "Claimed")
    plot_claimed_vs_actual(validation, RESULTS_DIR / "15min_claimed_vs_actual.png")
    plot_pnl_distribution(trades_df, RESULTS_DIR / "15min_pnl_distribution.png")
    plot_monthly_returns(trades_df, RESULTS_DIR / "15min_monthly_returns.png")

    # 9. Generate markdown summary
    _write_strategy_comparison(overall, validation, per_bucket, days, len(intervals))

    log.info("═══ Phase 1 Complete ═══")
    return result


def _write_strategy_comparison(
    overall: PerformanceReport,
    validation: pd.DataFrame,
    per_bucket: dict,
    days: int,
    total_intervals: int,
):
    lines = [
        "# 15-Minute BTC Momentum Strategy – Backtest Report",
        "",
        f"**Data Period**: {days} days of BTC 1-minute candles",
        f"**Total 15-min Intervals**: {total_intervals:,}",
        f"**Total Trades Taken**: {overall.total_trades:,}",
        "",
        "## Overall Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Win Rate | {overall.win_rate:.2%} |",
        f"| Net Profit (per $1 trades) | ${overall.net_profit:,.2f} |",
        f"| EV per $1 Trade | ${overall.expected_value_per_dollar:.4f} |",
        f"| Profit Factor | {overall.profit_factor:.2f} |",
        f"| Sharpe Ratio | {overall.sharpe_ratio:.2f} |",
        f"| Max Drawdown | ${overall.max_drawdown:.2f} ({overall.max_drawdown_pct:.1f}%) |",
        f"| Kelly Fraction | {overall.kelly_fraction:.4f} |",
        f"| Half-Kelly | {overall.kelly_half:.4f} |",
        f"| Trades per Day | {overall.trade_frequency_per_day:.1f} |",
        "",
        "## Win Rate Validation: Claimed vs Backtest",
        "",
        "| Move Bucket | Time Bucket | Claimed | Backtest | Delta | N Trades |",
        "|-------------|-------------|---------|----------|-------|----------|",
    ]

    for _, row in validation.iterrows():
        actual = f"{row['actual_wr']:.2%}" if pd.notna(row.get("actual_wr")) else "N/A"
        delta = f"{row['delta']:+.2%}" if pd.notna(row.get("delta")) else "N/A"
        lines.append(
            f"| {row['move_bucket']} | {row['time_bucket']} | "
            f"{row['claimed_wr']:.2%} | {actual} | {delta} | {row['n_trades']} |"
        )

    lines += [
        "",
        "## Visualizations",
        "",
        "- `15min_equity_curve.png` – Cumulative PnL over time",
        "- `15min_winrate_actual.png` – Heatmap of backtest win rates",
        "- `15min_winrate_claimed.png` – Heatmap of claimed win rates",
        "- `15min_claimed_vs_actual.png` – Side-by-side comparison",
        "- `15min_pnl_distribution.png` – Distribution of trade outcomes",
        "- `15min_monthly_returns.png` – Monthly PnL and volume",
        "",
        "## Conclusions",
        "",
        "See `15min_performance.json` for full machine-readable results.",
        "",
    ]

    path = RESULTS_DIR / "strategy_comparison.md"
    path.write_text("\n".join(lines))
    log.info("Markdown report → %s", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="15-min BTC strategy backtest")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--refresh", action="store_true", help="Force re-download data")
    args = parser.parse_args()
    asyncio.run(run_backtest(days=args.days, force_refresh=args.refresh))
