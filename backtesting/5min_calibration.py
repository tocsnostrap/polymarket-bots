#!/usr/bin/env python3
"""
Phase 2 – 5-Minute BTC Strategy Calibration

Generates an empirical fair-value probability table for 5-minute BTC intervals,
analogous to the captured 15-minute calibration.

Approach:
    1. Fetch 60 days of BTC 1-minute candles.
    2. Build 5-minute intervals aligned on the 5-min mark.
    3. For each interval, at each second-offset (20-240 s), measure the
       absolute BTC move from interval open.
    4. Bin by (move_pct, elapsed_s) and compute the empirical probability
       that the direction at entry matches the direction at close.
    5. Output the calibrated table + backtest metrics + charts.

Usage:
    python 5min_calibration.py
    python 5min_calibration.py --days 60 --refresh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from data_fetcher import fetch_btc_candles
from metrics import compute_metrics, format_report, bucket_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 5-minute bins from the hypothesis in 5min_strategy_plan.md
MOVE_BINS = [
    (0.01, 0.03, "0.01-0.03%"),
    (0.03, 0.06, "0.03-0.06%"),
    (0.06, 0.10, "0.06-0.10%"),
    (0.10, 100.0, "0.10%+"),
]

TIME_BINS = [
    (20, 60, "20-60s"),
    (60, 120, "60-120s"),
    (120, 180, "120-180s"),
    (180, 241, "180-240s"),
]

MARKET_SPREAD = 0.04
MIN_EDGE = 0.02
TRADE_SIZE = 1.0


def _classify(move_pct: float, elapsed_s: int) -> tuple[str | None, str | None]:
    mb = None
    for lo, hi, label in MOVE_BINS:
        if lo <= move_pct < hi:
            mb = label
            break
    tb = None
    for lo, hi, label in TIME_BINS:
        if lo <= elapsed_s < hi:
            tb = label
            break
    return mb, tb


def build_5min_intervals(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df["interval"] = df.index.floor("5min")

    rows = []
    for ts, grp in df.groupby("interval"):
        if len(grp) < 3:
            continue
        rows.append({
            "interval_start": ts,
            "open_price": grp.iloc[0]["open"],
            "close_price": grp.iloc[-1]["close"],
            "went_up": grp.iloc[-1]["close"] > grp.iloc[0]["open"],
        })
    return pd.DataFrame(rows)


def calibrate_table(candles: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """
    For every (move_bucket, time_bucket) cell, compute the empirical
    win probability – i.e. the fraction of times the direction at the
    entry point matches the direction at interval close.
    """
    df = candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    observations: list[dict] = []

    for _, iv in intervals.iterrows():
        start = iv["interval_start"]
        open_price = iv["open_price"]
        went_up = iv["went_up"]

        window = df.loc[start : start + pd.Timedelta(minutes=4)]
        for ts, row in window.iterrows():
            elapsed_s = int((ts - start).total_seconds())
            if elapsed_s < 20 or elapsed_s > 240:
                continue

            current = row["close"]
            move_pct = abs(current - open_price) / open_price * 100
            mb, tb = _classify(move_pct, elapsed_s)
            if mb is None or tb is None:
                continue

            direction_up = current > open_price
            won = (direction_up and went_up) or (not direction_up and not went_up)

            observations.append({
                "move_bucket": mb,
                "time_bucket": tb,
                "move_pct": move_pct,
                "elapsed_s": elapsed_s,
                "won": won,
            })

    obs_df = pd.DataFrame(observations)
    if obs_df.empty:
        log.warning("No calibration observations produced")
        return pd.DataFrame()

    table = (
        obs_df.groupby(["move_bucket", "time_bucket"])
        .agg(win_rate=("won", "mean"), n_obs=("won", "count"))
        .reset_index()
    )
    return table


def simulate_5min_trades(
    candles: pd.DataFrame,
    intervals: pd.DataFrame,
    fair_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backtest: for each interval, find the first qualifying entry using the
    freshly calibrated table, then measure PnL.
    """
    fv_lookup: dict[tuple[str, str], float] = {}
    for _, row in fair_table.iterrows():
        fv_lookup[(row["move_bucket"], row["time_bucket"])] = row["win_rate"]

    df = candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    trades: list[dict] = []

    for _, iv in intervals.iterrows():
        start = iv["interval_start"]
        open_price = iv["open_price"]
        went_up = iv["went_up"]

        window = df.loc[start : start + pd.Timedelta(minutes=4)]
        for ts, row in window.iterrows():
            elapsed_s = int((ts - start).total_seconds())
            if elapsed_s < 20 or elapsed_s > 240:
                continue

            current = row["close"]
            move_pct = abs(current - open_price) / open_price * 100
            mb, tb = _classify(move_pct, elapsed_s)
            if mb is None or tb is None:
                continue

            fv = fv_lookup.get((mb, tb))
            if fv is None or fv < 0.52:
                continue

            market_price = fv - MARKET_SPREAD / 2
            edge = fv - market_price
            if edge < MIN_EDGE:
                continue

            direction = "UP" if current > open_price else "DOWN"
            won = (direction == "UP" and went_up) or (direction == "DOWN" and not went_up)
            pnl = (TRADE_SIZE - market_price * TRADE_SIZE) if won else (-market_price * TRADE_SIZE)

            trades.append({
                "interval_start": start,
                "direction": direction,
                "move_pct": move_pct,
                "elapsed_s": elapsed_s,
                "fair_value": fv,
                "market_price": market_price,
                "edge": edge,
                "won": won,
                "pnl": pnl,
                "move_bucket": mb,
                "time_bucket": tb,
            })
            break

    return pd.DataFrame(trades)


# ── Visualizations ───────────────────────────────────────────────────────────

def plot_calibration_heatmap(fair_table: pd.DataFrame, path: Path):
    pivot = fair_table.pivot(index="move_bucket", columns="time_bucket", values="win_rate")
    move_order = [m[2] for m in MOVE_BINS]
    time_order = [t[2] for t in TIME_BINS]
    pivot = pivot.reindex(index=move_order, columns=time_order)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot.astype(float), annot=True, fmt=".1%",
                cmap="RdYlGn", vmin=0.5, vmax=1.0, linewidths=1, ax=ax)
    ax.set_title("5-Min Strategy – Calibrated Win Rates")
    ax.set_ylabel("BTC Move Size")
    ax.set_xlabel("Time Elapsed in Interval")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved heatmap → %s", path)


def plot_5min_equity(trades_df: pd.DataFrame, path: Path):
    if trades_df.empty:
        return
    cumulative = trades_df["pnl"].cumsum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative.values, linewidth=1.2, color="#E91E63")
    ax.fill_between(range(len(cumulative)), cumulative.values, alpha=0.15, color="#E91E63")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("5-Minute Strategy – Equity Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved equity curve → %s", path)


def plot_observation_counts(fair_table: pd.DataFrame, path: Path):
    pivot = fair_table.pivot(index="move_bucket", columns="time_bucket", values="n_obs")
    move_order = [m[2] for m in MOVE_BINS]
    time_order = [t[2] for t in TIME_BINS]
    pivot = pivot.reindex(index=move_order, columns=time_order)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot.astype(float), annot=True, fmt=",.0f",
                cmap="Blues", linewidths=1, ax=ax)
    ax.set_title("5-Min Calibration – Observation Counts")
    ax.set_ylabel("BTC Move Size")
    ax.set_xlabel("Time Elapsed")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved observation counts → %s", path)


def plot_hypothesis_comparison(fair_table: pd.DataFrame, path: Path):
    """Compare calibrated values against initial hypothesis."""
    hypothesis = {
        ("0.01-0.03%", "20-60s"): 0.575,
        ("0.03-0.06%", "20-60s"): 0.625,
        ("0.06-0.10%", "20-60s"): 0.675,
        ("0.10%+", "20-60s"): 0.725,
        ("0.01-0.03%", "60-120s"): 0.625,
        ("0.03-0.06%", "60-120s"): 0.675,
        ("0.06-0.10%", "60-120s"): 0.725,
        ("0.10%+", "60-120s"): 0.775,
        ("0.01-0.03%", "120-180s"): 0.675,
        ("0.03-0.06%", "120-180s"): 0.725,
        ("0.06-0.10%", "120-180s"): 0.775,
        ("0.10%+", "120-180s"): 0.825,
        ("0.01-0.03%", "180-240s"): 0.725,
        ("0.03-0.06%", "180-240s"): 0.775,
        ("0.06-0.10%", "180-240s"): 0.825,
        ("0.10%+", "180-240s"): 0.875,
    }

    rows = []
    for _, row in fair_table.iterrows():
        key = (row["move_bucket"], row["time_bucket"])
        hyp = hypothesis.get(key, None)
        rows.append({
            "label": f"{row['move_bucket']}\n{row['time_bucket']}",
            "calibrated": row["win_rate"],
            "hypothesis": hyp,
        })

    cdf = pd.DataFrame(rows).dropna(subset=["hypothesis"])
    if cdf.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(cdf))
    w = 0.35
    ax.bar(x - w / 2, cdf["hypothesis"], w, label="Hypothesis", color="#FF9800", alpha=0.85)
    ax.bar(x + w / 2, cdf["calibrated"], w, label="Calibrated", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cdf["label"], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Win Rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("5-Min Hypothesis vs Calibrated Win Rates")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved hypothesis comparison → %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

async def run_calibration(days: int = 60, force_refresh: bool = False):
    log.info("═══ Phase 2: 5-Minute Strategy Calibration ═══")

    candles = await fetch_btc_candles(days=days, granularity=60, force_refresh=force_refresh)
    log.info("Data: %d candles, %s to %s",
             len(candles), candles["timestamp"].min(), candles["timestamp"].max())

    intervals = build_5min_intervals(candles)
    log.info("Built %d valid 5-minute intervals", len(intervals))

    # Calibrate
    fair_table = calibrate_table(candles, intervals)
    if fair_table.empty:
        log.error("Calibration produced no data")
        return

    print("\n── Calibrated 5-Minute Fair Value Table ────────────────────")
    print(fair_table.to_string(index=False, float_format="%.4f"))

    # Backtest with calibrated table
    trades_df = simulate_5min_trades(candles, intervals, fair_table)
    overall_dict: dict = {}
    per_bucket: dict = {}
    if trades_df.empty:
        log.warning("No trades generated from calibrated table")
    else:
        overall = compute_metrics(trades_df["pnl"].values, trade_days=days)
        print(format_report(overall, "5-Minute Strategy – Overall"))

        trades_df["bucket"] = trades_df["move_bucket"] + " / " + trades_df["time_bucket"]
        per_bucket = bucket_metrics(trades_df, "bucket", trade_days=days)
        overall_dict = overall.to_dict()

    # Save results
    result = {
        "strategy": "5min_btc_calibration",
        "data_days": days,
        "total_intervals": len(intervals),
        "fair_value_table": fair_table.to_dict(orient="records"),
        "overall": overall_dict,
        "per_bucket": per_bucket,
    }
    out_path = RESULTS_DIR / "5min_performance.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("Results → %s", out_path)

    # Charts
    plot_calibration_heatmap(fair_table, RESULTS_DIR / "5min_calibration_heatmap.png")
    plot_observation_counts(fair_table, RESULTS_DIR / "5min_observation_counts.png")
    plot_hypothesis_comparison(fair_table, RESULTS_DIR / "5min_hypothesis_vs_calibrated.png")
    if not trades_df.empty:
        plot_5min_equity(trades_df, RESULTS_DIR / "5min_equity_curve.png")

    log.info("═══ Phase 2 Complete ═══")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-min BTC strategy calibration")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_calibration(days=args.days, force_refresh=args.refresh))
