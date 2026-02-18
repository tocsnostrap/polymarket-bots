# Polyclawd Strategy Backtest Report

Monte-Carlo simulation of 7 strategies over 90 days, $50 stake, 10k simulations each.

## Strategy Ranking (by Mean Final PnL)

| Rank | Strategy | Mean PnL | Median PnL | % Profitable | Sharpe | Kelly | Trades/Day |
|------|----------|----------|------------|--------------|--------|-------|------------|
| 1 | Sharp vs Soft Line Arbitrage | $4,065 | $4,056 | 99.9% | 5.73 | 0.1030 | 8.0 |
| 2 | Manifold → Polymarket Flow | $1,969 | $1,970 | 99.6% | 5.93 | 0.2297 | 3.0 |
| 3 | Whale Fade | $1,964 | $1,960 | 98.3% | 2.31 | 0.0701 | 4.0 |
| 4 | Cross-Platform Arbitrage | $1,337 | $1,334 | 98.7% | 5.48 | 0.2386 | 2.0 |
| 5 | News Speed Edge | $896 | $899 | 96.5% | 2.74 | 0.1593 | 1.5 |
| 6 | Correlation Violation | $841 | $841 | 99.0% | 5.10 | 0.3017 | 1.0 |
| 7 | Injury Impact Edge | $276 | $278 | 81.2% | -0.46 | 0.0000 | 0.5 |

## Strategy Details

### Sharp vs Soft Line Arbitrage

**Description:** Exploit the gap between sharp Vegas/Betfair odds (vig-removed via Shin method) and retail Polymarket prices.  Requires ≥5% edge to enter.

- Win Rate: 58%
- Avg Win: 92% of stake
- Trades/day: 8
- Capacity: $5,000/day
- Mean 90-day PnL: $4,065
- 5th-95th percentile: $1,974 to $6,163
- % Profitable sims: 99.9%

### Manifold → Polymarket Flow

**Description:** Play-money signals on Manifold precede real-money Polymarket moves by 1-4 hours.  Trade Polymarket before it catches up.  ≥10% move filter.

- Win Rate: 62%
- Avg Win: 85% of stake
- Trades/day: 3
- Capacity: $3,000/day
- Mean 90-day PnL: $1,969
- 5th-95th percentile: $729 to $3,214
- % Profitable sims: 99.6%

### Whale Fade

**Description:** Track losing whale wallets on-chain and bet opposite their next trades.  55-60% historical hit rate with near-even payouts.

- Win Rate: 57%
- Avg Win: 95% of stake
- Trades/day: 4
- Capacity: $4,000/day
- Mean 90-day PnL: $1,964
- 5th-95th percentile: $456 to $3,510
- % Profitable sims: 98.3%

### Cross-Platform Arbitrage

**Description:** PredictIt vs Polymarket price gaps on politics markets.  ≥12% gap threshold.  Limited by PredictIt position caps ($850/contract).

- Win Rate: 64%
- Avg Win: 80% of stake
- Trades/day: 2
- Capacity: $2,000/day
- Mean 90-day PnL: $1,337
- 5th-95th percentile: $350 to $2,311
- % Profitable sims: 98.7%

### News Speed Edge

**Description:** Trade stale Polymarket prices within seconds of breaking news.  Requires low-latency news feeds (Twitter firehose, AP, Reuters).

- Win Rate: 65%
- Avg Win: 75% of stake
- Trades/day: 1.5
- Capacity: $2,500/day
- Mean 90-day PnL: $896
- 5th-95th percentile: $77 to $1,698
- % Profitable sims: 96.5%

### Correlation Violation

**Description:** Detect mathematical inconsistencies between parent/child markets (e.g., P(A) must ≥ P(A and B)).  High conviction but rare signals.

- Win Rate: 72%
- Avg Win: 65% of stake
- Trades/day: 1
- Capacity: $3,000/day
- Mean 90-day PnL: $841
- 5th-95th percentile: $248 to $1,431
- % Profitable sims: 99.0%

### Injury Impact Edge

**Description:** Key player injuries move lines 3-4 points.  Trade before Polymarket adjusts.  Seasonal (sports calendar dependent).

- Win Rate: 60%
- Avg Win: 88% of stake
- Trades/day: 0.5
- Capacity: $2,000/day
- Mean 90-day PnL: $276
- 5th-95th percentile: $-246 to $792
- % Profitable sims: 81.2%

## Recommendations

1. **Top Tier** – Strategies with >70% profitable simulations and positive Sharpe:
   - Sharp vs Soft Line Arbitrage (100% profitable)
   - Manifold → Polymarket Flow (100% profitable)
   - Whale Fade (98% profitable)
   - Cross-Platform Arbitrage (99% profitable)
   - News Speed Edge (97% profitable)
   - Correlation Violation (99% profitable)

2. **Proceed with caution** – Positive EV but higher variance:

3. **Skip** – Negative or marginal EV after costs:
   - (none)

## Visualizations

- `polyclawd_strategy_comparison.png` – Bar chart of expected PnL with confidence intervals
- `polyclawd_pnl_distributions.png` – Monte-Carlo histograms for each strategy
- `polyclawd_quality_metrics.png` – Grouped metric comparison
- `polyclawd_risk_return.png` – Risk-return scatter plot
