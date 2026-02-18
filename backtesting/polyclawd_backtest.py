#!/usr/bin/env python3
"""
Phase 3 – Polyclawd Strategy Backtesting

Tests each of the 7 arbitrage/edge strategies from polyclawd using
Monte-Carlo simulation with parameters inferred from the captured code.

Since historical Polymarket order-book data, third-party odds feeds, and
on-chain whale data are not freely downloadable, each strategy is modeled
as a parameterised random process calibrated from the documented edge
profiles.  This gives realistic performance distributions for sizing,
risk budgeting, and strategy comparison – not a tick-perfect replay.

Strategies:
    1. Sharp vs Soft Line Arbitrage
    2. Manifold → Polymarket Flow
    3. Cross-Platform Arbitrage
    4. Whale Fade
    5. News Speed Edge
    6. Correlation Violation
    7. Injury Impact Edge

Usage:
    python polyclawd_backtest.py
    python polyclawd_backtest.py --sims 50000
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from metrics import compute_metrics, format_report, PerformanceReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)


@dataclass
class StrategyProfile:
    """Parameters that define a strategy's statistical edge."""
    name: str
    win_rate: float
    avg_win_pct: float          # average payout on a win (fraction of stake)
    avg_loss_pct: float         # average loss on a loss (fraction of stake, positive)
    trades_per_day: float
    edge_decay_daily: float     # probability that edge disappears each day
    capacity_usd: float         # max daily volume before market impact
    description: str


STRATEGIES: list[StrategyProfile] = [
    StrategyProfile(
        name="Sharp vs Soft Line Arbitrage",
        win_rate=0.58,
        avg_win_pct=0.92,
        avg_loss_pct=1.0,
        trades_per_day=8,
        edge_decay_daily=0.001,
        capacity_usd=5000,
        description=(
            "Exploit the gap between sharp Vegas/Betfair odds (vig-removed via "
            "Shin method) and retail Polymarket prices.  Requires ≥5% edge to enter."
        ),
    ),
    StrategyProfile(
        name="Manifold → Polymarket Flow",
        win_rate=0.62,
        avg_win_pct=0.85,
        avg_loss_pct=1.0,
        trades_per_day=3,
        edge_decay_daily=0.002,
        capacity_usd=3000,
        description=(
            "Play-money signals on Manifold precede real-money Polymarket moves "
            "by 1-4 hours.  Trade Polymarket before it catches up.  ≥10% move filter."
        ),
    ),
    StrategyProfile(
        name="Cross-Platform Arbitrage",
        win_rate=0.64,
        avg_win_pct=0.80,
        avg_loss_pct=1.0,
        trades_per_day=2,
        edge_decay_daily=0.003,
        capacity_usd=2000,
        description=(
            "PredictIt vs Polymarket price gaps on politics markets.  ≥12% gap "
            "threshold.  Limited by PredictIt position caps ($850/contract)."
        ),
    ),
    StrategyProfile(
        name="Whale Fade",
        win_rate=0.57,
        avg_win_pct=0.95,
        avg_loss_pct=1.0,
        trades_per_day=4,
        edge_decay_daily=0.002,
        capacity_usd=4000,
        description=(
            "Track losing whale wallets on-chain and bet opposite their next "
            "trades.  55-60% historical hit rate with near-even payouts."
        ),
    ),
    StrategyProfile(
        name="News Speed Edge",
        win_rate=0.65,
        avg_win_pct=0.75,
        avg_loss_pct=1.0,
        trades_per_day=1.5,
        edge_decay_daily=0.004,
        capacity_usd=2500,
        description=(
            "Trade stale Polymarket prices within seconds of breaking news.  "
            "Requires low-latency news feeds (Twitter firehose, AP, Reuters)."
        ),
    ),
    StrategyProfile(
        name="Correlation Violation",
        win_rate=0.72,
        avg_win_pct=0.65,
        avg_loss_pct=1.0,
        trades_per_day=1,
        edge_decay_daily=0.001,
        capacity_usd=3000,
        description=(
            "Detect mathematical inconsistencies between parent/child markets "
            "(e.g., P(A) must ≥ P(A and B)).  High conviction but rare signals."
        ),
    ),
    StrategyProfile(
        name="Injury Impact Edge",
        win_rate=0.60,
        avg_win_pct=0.88,
        avg_loss_pct=1.0,
        trades_per_day=0.5,
        edge_decay_daily=0.005,
        capacity_usd=2000,
        description=(
            "Key player injuries move lines 3-4 points.  Trade before "
            "Polymarket adjusts.  Seasonal (sports calendar dependent)."
        ),
    ),
]


def simulate_strategy(
    profile: StrategyProfile,
    days: int = 90,
    stake: float = 50.0,
    n_sims: int = 10000,
) -> dict:
    """
    Run a Monte-Carlo simulation of the strategy over `days` trading days.

    Returns per-simulation final PnL and a single "representative" trade list
    for detailed metrics.
    """
    all_final_pnl: list[float] = []
    representative_trades: list[float] = []

    for sim_idx in range(n_sims):
        cumulative = 0.0
        current_wr = profile.win_rate
        trades_this_sim: list[float] = []

        for day in range(days):
            # Decay: small chance edge shrinks
            if RNG.random() < profile.edge_decay_daily:
                current_wr = max(0.50, current_wr - RNG.uniform(0.005, 0.02))

            n_trades = RNG.poisson(profile.trades_per_day)
            for _ in range(n_trades):
                won = RNG.random() < current_wr
                if won:
                    pnl = stake * profile.avg_win_pct * RNG.uniform(0.8, 1.2)
                else:
                    pnl = -stake * profile.avg_loss_pct * RNG.uniform(0.8, 1.2)
                cumulative += pnl
                trades_this_sim.append(pnl)

        all_final_pnl.append(cumulative)
        if sim_idx == 0:
            representative_trades = trades_this_sim

    return {
        "profile": profile,
        "final_pnl_dist": np.array(all_final_pnl),
        "representative_trades": representative_trades,
    }


def analyze_strategy(sim_result: dict, days: int) -> dict:
    profile: StrategyProfile = sim_result["profile"]
    pnl_dist = sim_result["final_pnl_dist"]
    rep_trades = sim_result["representative_trades"]

    report = compute_metrics(rep_trades, trade_days=days)

    analysis = {
        "strategy": profile.name,
        "description": profile.description,
        "parameters": {
            "win_rate": profile.win_rate,
            "avg_win_pct": profile.avg_win_pct,
            "avg_loss_pct": profile.avg_loss_pct,
            "trades_per_day": profile.trades_per_day,
            "edge_decay_daily": profile.edge_decay_daily,
            "capacity_usd": profile.capacity_usd,
        },
        "monte_carlo": {
            "n_sims": len(pnl_dist),
            "mean_final_pnl": float(np.mean(pnl_dist)),
            "median_final_pnl": float(np.median(pnl_dist)),
            "std_final_pnl": float(np.std(pnl_dist)),
            "pct_profitable": float(np.mean(pnl_dist > 0) * 100),
            "p5": float(np.percentile(pnl_dist, 5)),
            "p25": float(np.percentile(pnl_dist, 25)),
            "p75": float(np.percentile(pnl_dist, 75)),
            "p95": float(np.percentile(pnl_dist, 95)),
            "worst_case": float(np.min(pnl_dist)),
            "best_case": float(np.max(pnl_dist)),
        },
        "representative_run": report.to_dict(),
    }
    return analysis


# ── Visualizations ───────────────────────────────────────────────────────────

def plot_strategy_comparison(analyses: list[dict], path: Path):
    names = [a["strategy"] for a in analyses]
    means = [a["monte_carlo"]["mean_final_pnl"] for a in analyses]
    p5 = [a["monte_carlo"]["p5"] for a in analyses]
    p95 = [a["monte_carlo"]["p95"] for a in analyses]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    colors = ["#4CAF50" if m > 0 else "#F44336" for m in means]
    bars = ax.bar(x, means, color=colors, alpha=0.85)
    ax.errorbar(x, means,
                yerr=[np.array(means) - np.array(p5), np.array(p95) - np.array(means)],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Expected Final PnL ($)")
    ax.set_title("Polyclawd Strategies – Expected PnL (90 days, $50 stake)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved strategy comparison → %s", path)


def plot_pnl_distributions(sim_results: list[dict], path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, sim in enumerate(sim_results):
        if i >= len(axes):
            break
        ax = axes[i]
        pnl = sim["final_pnl_dist"]
        color = "#4CAF50" if np.mean(pnl) > 0 else "#F44336"
        ax.hist(pnl, bins=50, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(np.mean(pnl), color="blue", linestyle="--", linewidth=1.2)
        ax.set_title(sim["profile"].name, fontsize=9)
        ax.set_xlabel("Final PnL ($)", fontsize=8)
        ax.tick_params(labelsize=7)

    if len(sim_results) < len(axes):
        for j in range(len(sim_results), len(axes)):
            axes[j].set_visible(False)

    fig.suptitle("Monte-Carlo PnL Distributions (10k sims each)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved PnL distributions → %s", path)


def plot_profitability_radar(analyses: list[dict], path: Path):
    metrics_keys = [
        ("monte_carlo", "pct_profitable", "% Profitable"),
        ("representative_run", "win_rate", "Win Rate (×100)"),
        ("representative_run", "profit_factor", "Profit Factor (÷5)"),
        ("representative_run", "sharpe_ratio", "Sharpe (÷10)"),
        ("representative_run", "kelly_fraction", "Kelly (×10)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    strategy_names = []
    metric_names = [m[2] for m in metrics_keys]

    for a in analyses:
        name = a["strategy"].split()[0]  # short name
        strategy_names.append(name)
        vals = []
        for section, key, label in metrics_keys:
            raw = a[section].get(key, 0) or 0
            if "×100" in label:
                raw *= 100
            elif "÷5" in label:
                raw = min(raw / 5, 1.0)
            elif "÷10" in label:
                raw = min(raw / 10, 1.0)
            elif "×10" in label:
                raw = min(raw * 10, 1.0)
            vals.append(max(0, raw))

    # Simple grouped bar for readability
    x = np.arange(len(strategy_names))
    width = 0.15
    for i, (_, key, label) in enumerate(metrics_keys):
        vals = []
        for a in analyses:
            section = metrics_keys[i][0]
            raw = a[section].get(key, 0) or 0
            vals.append(raw)
        ax.bar(x + i * width, vals, width, label=label, alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(strategy_names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("Strategy Quality Metrics Comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved quality radar → %s", path)


def plot_risk_return(analyses: list[dict], path: Path):
    fig, ax = plt.subplots(figsize=(10, 7))

    for a in analyses:
        mc = a["monte_carlo"]
        mean_pnl = mc["mean_final_pnl"]
        std_pnl = mc["std_final_pnl"]
        pct_prof = mc["pct_profitable"]
        size = max(pct_prof * 3, 30)
        color = "#4CAF50" if mean_pnl > 0 else "#F44336"
        ax.scatter(std_pnl, mean_pnl, s=size, color=color, alpha=0.8, edgecolors="black", linewidths=0.8)
        ax.annotate(a["strategy"].split()[0], (std_pnl, mean_pnl),
                    fontsize=7, textcoords="offset points", xytext=(8, 4))

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Risk (Std Dev of Final PnL)")
    ax.set_ylabel("Return (Mean Final PnL)")
    ax.set_title("Risk-Return Scatter – Polyclawd Strategies")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved risk-return scatter → %s", path)


# ── Report ───────────────────────────────────────────────────────────────────

def write_polyclawd_report(analyses: list[dict], path: Path):
    lines = [
        "# Polyclawd Strategy Backtest Report",
        "",
        "Monte-Carlo simulation of 7 strategies over 90 days, $50 stake, 10k simulations each.",
        "",
        "## Strategy Ranking (by Mean Final PnL)",
        "",
        "| Rank | Strategy | Mean PnL | Median PnL | % Profitable | Sharpe | Kelly | Trades/Day |",
        "|------|----------|----------|------------|--------------|--------|-------|------------|",
    ]

    sorted_analyses = sorted(analyses, key=lambda a: a["monte_carlo"]["mean_final_pnl"], reverse=True)
    for rank, a in enumerate(sorted_analyses, 1):
        mc = a["monte_carlo"]
        rep = a["representative_run"]
        lines.append(
            f"| {rank} | {a['strategy']} | "
            f"${mc['mean_final_pnl']:,.0f} | "
            f"${mc['median_final_pnl']:,.0f} | "
            f"{mc['pct_profitable']:.1f}% | "
            f"{rep.get('sharpe_ratio', 0):.2f} | "
            f"{rep.get('kelly_fraction', 0):.4f} | "
            f"{a['parameters']['trades_per_day']:.1f} |"
        )

    lines += ["", "## Strategy Details", ""]
    for a in sorted_analyses:
        mc = a["monte_carlo"]
        lines += [
            f"### {a['strategy']}",
            "",
            f"**Description:** {a['description']}",
            "",
            f"- Win Rate: {a['parameters']['win_rate']:.0%}",
            f"- Avg Win: {a['parameters']['avg_win_pct']:.0%} of stake",
            f"- Trades/day: {a['parameters']['trades_per_day']}",
            f"- Capacity: ${a['parameters']['capacity_usd']:,.0f}/day",
            f"- Mean 90-day PnL: ${mc['mean_final_pnl']:,.0f}",
            f"- 5th-95th percentile: ${mc['p5']:,.0f} to ${mc['p95']:,.0f}",
            f"- % Profitable sims: {mc['pct_profitable']:.1f}%",
            "",
        ]

    lines += [
        "## Recommendations",
        "",
        "1. **Top Tier** – Strategies with >70% profitable simulations and positive Sharpe:",
    ]

    for a in sorted_analyses:
        mc = a["monte_carlo"]
        rep = a["representative_run"]
        if mc["pct_profitable"] > 70 and rep.get("sharpe_ratio", 0) > 0:
            lines.append(f"   - {a['strategy']} ({mc['pct_profitable']:.0f}% profitable)")

    lines += [
        "",
        "2. **Proceed with caution** – Positive EV but higher variance:",
    ]
    for a in sorted_analyses:
        mc = a["monte_carlo"]
        rep = a["representative_run"]
        if 50 < mc["pct_profitable"] <= 70:
            lines.append(f"   - {a['strategy']} ({mc['pct_profitable']:.0f}% profitable)")

    lines += [
        "",
        "3. **Skip** – Negative or marginal EV after costs:",
    ]
    for a in sorted_analyses:
        mc = a["monte_carlo"]
        if mc["pct_profitable"] <= 50:
            lines.append(f"   - {a['strategy']} ({mc['pct_profitable']:.0f}% profitable)")

    if not any(a["monte_carlo"]["pct_profitable"] <= 50 for a in sorted_analyses):
        lines.append("   - (none)")

    lines += [
        "",
        "## Visualizations",
        "",
        "- `polyclawd_strategy_comparison.png` – Bar chart of expected PnL with confidence intervals",
        "- `polyclawd_pnl_distributions.png` – Monte-Carlo histograms for each strategy",
        "- `polyclawd_quality_metrics.png` – Grouped metric comparison",
        "- `polyclawd_risk_return.png` – Risk-return scatter plot",
        "",
    ]

    path.write_text("\n".join(lines))
    log.info("Report → %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_polyclawd_backtest(days: int = 90, n_sims: int = 10000):
    log.info("═══ Phase 3: Polyclawd Strategy Backtesting ═══")

    sim_results: list[dict] = []
    analyses: list[dict] = []

    for profile in STRATEGIES:
        log.info("Simulating: %s", profile.name)
        sim = simulate_strategy(profile, days=days, n_sims=n_sims)
        sim_results.append(sim)

        analysis = analyze_strategy(sim, days)
        analyses.append(analysis)

        rep = compute_metrics(sim["representative_trades"], trade_days=days)
        print(format_report(rep, profile.name))

    # Save JSON results
    out_path = RESULTS_DIR / "polyclawd_performance.json"
    out_path.write_text(json.dumps(analyses, indent=2, default=str))
    log.info("Results → %s", out_path)

    # Charts
    plot_strategy_comparison(analyses, RESULTS_DIR / "polyclawd_strategy_comparison.png")
    plot_pnl_distributions(sim_results, RESULTS_DIR / "polyclawd_pnl_distributions.png")
    plot_profitability_radar(analyses, RESULTS_DIR / "polyclawd_quality_metrics.png")
    plot_risk_return(analyses, RESULTS_DIR / "polyclawd_risk_return.png")

    # Report
    write_polyclawd_report(analyses, RESULTS_DIR / "polyclawd_report.md")

    log.info("═══ Phase 3 Complete ═══")
    return analyses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polyclawd strategy backtest")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--sims", type=int, default=10000)
    args = parser.parse_args()
    run_polyclawd_backtest(days=args.days, n_sims=args.sims)
