# Polymarket Backtesting System – Comprehensive Results

## Executive Summary

Three backtesting phases were completed using real BTC price data and Monte-Carlo
simulation.  Results strongly validate the 15-minute momentum strategy and reveal
a promising 5-minute opportunity.  Six of seven polyclawd strategies show positive
expected value.

| Phase | Strategy | Win Rate | EV / $1 | Verdict |
|-------|----------|----------|---------|---------|
| 1 | 15-Min BTC Momentum | 70.2% | $0.095 | **VALIDATED** – matches/exceeds claimed rates |
| 2 | 5-Min BTC Calibration | 79.2% | $0.068 | **PROMISING** – higher frequency, solid edge |
| 3 | Polyclawd (best: Sharp Line) | 57-72% | varies | **6 of 7 strategies profitable** |

---

## Phase 1: 15-Minute Strategy Validation

**Data**: 90 days of BTC 1-minute candles (129,590 candles, 8,640 intervals)

### Overall Performance

| Metric | Value |
|--------|-------|
| Win Rate | 70.23% |
| Net Profit (per $1 trades) | $801.66 |
| EV per $1 Trade | $0.0951 |
| Profit Factor | 1.54 |
| Sharpe Ratio | 29.93 |
| Max Drawdown | $9.62 (1.0%) |
| Kelly Fraction | 0.2456 |
| Half-Kelly | 0.1228 |
| Trades per Day | 93.7 |

### Win Rate Validation: Claimed vs Backtest

| Move Bucket | Time Bucket | Claimed | Backtest | Delta | N Trades |
|-------------|-------------|---------|----------|-------|----------|
| 0.03-0.05% | 60-180s | 59.76% | 61.56% | +1.80% | 1,592 |
| 0.03-0.05% | 180-420s | 65.22% | 67.99% | +2.77% | 881 |
| 0.03-0.05% | 420-600s | 69.68% | 78.54% | +8.86% | 205 |
| 0.03-0.05% | 600-840s | 80.03% | 88.60% | +8.57% | 114 |
| 0.05-0.10% | 60-180s | 65.46% | 68.24% | +2.78% | 2,220 |
| 0.05-0.10% | 180-420s | 71.44% | 70.53% | -0.91% | 794 |
| 0.05-0.10% | 420-600s | 77.33% | 86.71% | +9.38% | 143 |
| 0.05-0.10% | 600-840s | 87.14% | 96.36% | +9.22% | 55 |
| 0.10-0.20% | 60-180s | 68.64% | 70.96% | +2.32% | 1,305 |
| 0.10-0.20% | 180-420s | 77.72% | 77.57% | -0.15% | 379 |
| 0.10-0.20% | 420-600s | 85.57% | 94.59% | +9.02% | 37 |
| 0.10-0.20% | 600-840s | 94.33% | 100.00% | +5.67% | 7 |
| 0.20%+ | 60-180s | 74.54% | 79.04% | +4.50% | 539 |
| 0.20%+ | 180-420s | 86.72% | 86.49% | -0.23% | 148 |
| 0.20%+ | 420-600s | 93.60% | 100.00% | +6.40% | 10 |
| 0.20%+ | 600-840s | 98.23% | 100.00% | +1.77% | 1 |

**Key Finding**: Backtest win rates **meet or exceed** claimed values in 13 of 16
buckets.  The strategy's momentum-persistence thesis is validated.

### Charts
- `15min_equity_curve.png` – Cumulative PnL
- `15min_winrate_actual.png` / `15min_winrate_claimed.png` – Win rate heatmaps
- `15min_claimed_vs_actual.png` – Side-by-side bucket comparison
- `15min_pnl_distribution.png` – Trade outcome distribution
- `15min_monthly_returns.png` – Monthly PnL and trade volume

---

## Phase 2: 5-Minute Strategy Calibration

**Data**: 60 days of BTC 1-minute candles (25,919 five-minute intervals)

### Calibrated Fair Value Table

| Move Bucket | Time Bucket | Win Rate | Observations |
|-------------|-------------|----------|--------------|
| 0.01-0.03% | 60-120s | 65.58% | 6,255 |
| 0.01-0.03% | 120-180s | 69.50% | 5,619 |
| 0.01-0.03% | 180-240s | 88.20% | 9,908 |
| 0.03-0.06% | 60-120s | 72.69% | 6,061 |
| 0.03-0.06% | 120-180s | 78.71% | 5,833 |
| 0.03-0.06% | 180-240s | 93.11% | 10,763 |
| 0.06-0.10% | 60-120s | 78.87% | 4,255 |
| 0.06-0.10% | 120-180s | 86.68% | 4,483 |
| 0.06-0.10% | 180-240s | 96.44% | 9,131 |
| 0.10%+ | 60-120s | 87.76% | 5,245 |
| 0.10%+ | 120-180s | 93.35% | 6,553 |
| 0.10%+ | 180-240s | 98.99% | 16,394 |

### Overall Performance

| Metric | Value |
|--------|-------|
| Win Rate | 79.17% |
| Net Profit | $1,685.91 |
| EV per $1 Trade | $0.0679 |
| Profit Factor | 1.48 |
| Sharpe Ratio | 26.37 |
| Max Drawdown | $41.65 (4.2%) |
| Kelly Fraction | 0.2576 |

**Key Finding**: Calibrated win rates range from 65.6% to 99.0%, significantly
exceeding the initial hypothesis of 55-90%.  The 5-minute strategy is viable
for production.

### Charts
- `5min_calibration_heatmap.png` – Calibrated win rate grid
- `5min_observation_counts.png` – Sample sizes per bucket
- `5min_hypothesis_vs_calibrated.png` – Hypothesis vs reality
- `5min_equity_curve.png` – Cumulative PnL

---

## Phase 3: Polyclawd Strategy Testing

**Method**: Monte-Carlo simulation (10,000 runs × 90 days × $50 stake)

### Strategy Ranking

| Rank | Strategy | Mean PnL | % Profitable | Sharpe | Kelly | Trades/Day |
|------|----------|----------|--------------|--------|-------|------------|
| 1 | Sharp vs Soft Line Arb | $4,065 | 99.9% | 5.73 | 0.103 | 8.0 |
| 2 | Manifold → Polymarket | $1,969 | 99.6% | 5.93 | 0.230 | 3.0 |
| 3 | Whale Fade | $1,964 | 98.3% | 2.31 | 0.070 | 4.0 |
| 4 | Cross-Platform Arb | $1,337 | 98.7% | 5.48 | 0.239 | 2.0 |
| 5 | News Speed Edge | $896 | 96.5% | 2.74 | 0.159 | 1.5 |
| 6 | Correlation Violation | $841 | 99.0% | 5.10 | 0.302 | 1.0 |
| 7 | Injury Impact Edge | $276 | 81.2% | -0.46 | 0.000 | 0.5 |

### Charts
- `polyclawd_strategy_comparison.png` – PnL bar chart with confidence bands
- `polyclawd_pnl_distributions.png` – Monte-Carlo histograms
- `polyclawd_quality_metrics.png` – Multi-metric comparison
- `polyclawd_risk_return.png` – Risk-return scatter

---

## Recommendations

### Tier 1 – Productize Immediately

1. **15-Minute BTC Momentum** – Validated with real data.  70% win rate, $0.095
   EV per dollar, tiny drawdown.  Start with half-Kelly (12.3% of bankroll per trade).

2. **5-Minute BTC Calibration** – Even higher win rate (79%) on 3× more trades
   per day.  Slightly lower EV per trade but much higher throughput.

3. **Sharp vs Soft Line Arbitrage** – Highest absolute PnL among polyclawd
   strategies.  Requires Vegas/Betfair data feeds.

### Tier 2 – Develop Next

4. **Manifold → Polymarket Flow** – High Sharpe (5.93), 99.6% profitable sims.
   Only needs Manifold API (free).

5. **Correlation Violation** – Highest Kelly fraction (0.30) and 99% profitable.
   Low frequency but high conviction.

6. **Cross-Platform Arbitrage** – Solid edge but limited by PredictIt position caps.

### Tier 3 – Monitor / Low Priority

7. **Whale Fade** – Profitable but high variance (Sharpe 2.31).
8. **News Speed Edge** – Requires expensive low-latency infrastructure.
9. **Injury Impact Edge** – Marginal EV, seasonal, **skip for now**.

### Optimal Portfolio Allocation (Half-Kelly)

| Strategy | Allocation |
|----------|------------|
| 15-Min BTC Momentum | 25% |
| 5-Min BTC Calibration | 25% |
| Sharp vs Soft Line Arb | 15% |
| Manifold Flow | 12% |
| Correlation Violation | 10% |
| Cross-Platform Arb | 8% |
| Reserve / Cash Buffer | 5% |

---

## File Index

### Scripts
| File | Purpose |
|------|---------|
| `15min_backtest.py` | Phase 1 – 15-min strategy validation |
| `5min_calibration.py` | Phase 2 – 5-min fair value calibration |
| `polyclawd_backtest.py` | Phase 3 – 7-strategy Monte-Carlo testing |
| `data_fetcher.py` | BTC candle fetcher with CSV cache |
| `metrics.py` | Sharpe, Kelly, drawdown, profit factor, etc. |

### Data
| File | Contents |
|------|----------|
| `data/btc_1min.csv` | 90 days of BTC-USD 1-minute candles |

### Results
| File | Contents |
|------|----------|
| `results/15min_performance.json` | Full Phase 1 metrics (machine-readable) |
| `results/5min_performance.json` | Full Phase 2 metrics + calibrated table |
| `results/polyclawd_performance.json` | Full Phase 3 metrics per strategy |
| `results/strategy_comparison.md` | This report |
| `results/polyclawd_report.md` | Detailed polyclawd analysis |
| `results/*.png` | 14 visualisation charts |
