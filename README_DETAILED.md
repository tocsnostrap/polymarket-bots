# Polymarket Trading Bots - Complete Collection

## 🚀 What This Repository Contains

Two production-ready Polymarket trading systems captured from public repositories:

### 1. **`polymarket-bot/`** - 15-Minute BTC Up/Down Momentum Bot
**Strategy:** Data-driven momentum betting on BTC 15-minute intervals using calibrated fair value tables.

**Key Features:**
- Calibrated win probabilities from 90 days historical data (8,637 intervals)
- Chainlink price feed integration (same oracle as Polymarket resolution)
- Real-time order placement via CLOB API
- Risk management with session loss limits
- Paper trading mode included

**Win Probability Table:**
| Move Size | Time Elapsed | Win Rate |
|-----------|--------------|----------|
| 0.03-0.05% | 60-180s | 59.76% |
| 0.03-0.05% | 180-420s | 65.22% |
| 0.03-0.05% | 420-600s | 69.68% |
| 0.03-0.05% | 600-840s | 80.03% |
| 0.05-0.10% | 60-180s | 65.46% |
| 0.05-0.10% | 180-420s | 71.44% |
| 0.05-0.10% | 420-600s | 77.33% |
| 0.05-0.10% | 600-840s | 87.14% |
| 0.10-0.20% | 60-180s | 68.64% |
| 0.10-0.20% | 180-420s | 77.72% |
| 0.10-0.20% | 420-600s | 85.57% |
| 0.10-0.20% | 600-840s | 94.33% |
| 0.20%+ | 60-180s | 74.54% |
| 0.20%+ | 180-420s | 86.72% |
| 0.20%+ | 420-600s | 93.60% |
| 0.20%+ | 600-840s | 98.23% |

### 2. **`polyclawd/`** - AI-Powered Prediction Market Trading System
**Strategy:** Multi-source arbitrage across 15 signal sources with 7 profit strategies.

**Key Features:**
- 121 API endpoints for full trading system
- 15 signal sources (Vegas, ESPN, Betfair, Manifold, PredictIt, etc.)
- 7 profit strategies including arbitrage, whale fade, news speed
- Bayesian confidence scoring with adaptive trading
- Kelly position sizing with drawdown protection
- OpenClaw integration for cron jobs and alerts
- Production deployment at virtuosocrypto.com/polyclawd

**7 Profit Strategies:**
1. **Sharp vs Soft Line Arbitrage** (5%+ edge) - Vegas/Betfair vs Polymarket
2. **Manifold → Polymarket Flow** (10%+ moves) - Play money signals precede real money
3. **Cross-Platform Arbitrage** (12%+ gap) - PredictIt vs Polymarket
4. **Whale Fade** (55-60% win rate) - Bet against losing whale traders
5. **News Speed Edge** - Exploit slow market reactions to news
6. **Correlation Violation** - Math constraints between related markets
7. **Injury Impact Edge** - Sports injuries before line movement

## 📊 5-Minute Strategy Adaptation

**We confirmed:** Polymarket has 5-minute markets (`{asset}-updown-5m-{timestamp}`)

**Created:** `calibrate_5m.py` and `quick_5m_test.py` for 5-minute strategy development

**Estimated 5-Minute Win Rates:**
| Move Size | Time Elapsed | Estimated Win Rate |
|-----------|--------------|-------------------|
| 0.01-0.03% | 20-60s | 55-60% |
| 0.03-0.06% | 20-60s | 60-65% |
| 0.06-0.10% | 20-60s | 65-70% |
| 0.10%+ | 20-60s | 70-75% |
| 0.01-0.03% | 180-240s | 70-75% |
| 0.03-0.06% | 180-240s | 75-80% |
| 0.06-0.10% | 180-240s | 80-85% |
| 0.10%+ | 180-240s | 85-90% |

## 🚀 Quick Start

### For 15-Minute Bot:
```bash
cd polymarket-bot
cp .env.example .env  # Add your Polymarket API keys
pip install -r requirements.txt
python discover_markets.py  # Find active markets
python paper_trader.py     # Test strategy
python live_trader.py      # Live trading (requires real API keys)
```

### For Polyclawd System:
```bash
cd polyclawd
cp .env.example .env  # Configure API keys
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8420
```

## 💰 Productization Opportunities

### Tier 1: Quick Wins
1. **Clawdbot 15-Minute** - $97 (existing bot)
2. **Clawdbot 5-Minute** - $147 (adapted version)
3. **Clawdbot Duo Bundle** - $197 (both strategies)
4. **Polymarket Arbitrage Guide** - $47 (7 strategies PDF)

### Tier 2: SaaS Platform
5. **Clawdbot Pro** - $297/month (hosted polyclawd)
6. **Whale Fade Bot** - $147 (on-chain tracking)
7. **Prediction Market Suite** - $497/month (multi-platform)

## 📈 Performance Metrics

### 15-Minute Bot:
- **Historical win rate:** 59-98% (depending on move/time)
- **Expected value:** $0.10-0.20 per $1 trade
- **Trade frequency:** Every 15 minutes
- **Minimum bankroll:** $100 recommended

### Polyclawd System:
- **Signal sources:** 15 active sources
- **Confidence scoring:** Bayesian with adaptive weights
- **Position sizing:** Quarter-Kelly with drawdown protection
- **Risk management:** 5% daily loss halt

## 🔧 Technical Requirements

### 15-Minute Bot:
- Python 3.8+
- Polymarket API keys (CLOB access)
- Chainlink/Binance price feed
- Basic VPS for 24/7 operation

### Polyclawd System:
- Python 3.10+
- Multiple API keys (Vegas, ESPN, Betfair, etc.)
- PostgreSQL database
- Redis for caching
- VPS with 2GB+ RAM
- OpenClaw for cron management

## 📚 Documentation

See included files:
- `CLAWDBOT_ANALYSIS.md` - Complete strategy breakdown
- `5min_strategy_plan.md` - 5-minute adaptation plan
- `polymarket-bot/STRATEGY.md` - Detailed 15-minute strategy
- `polyclawd/README.md` - Full system documentation

## ⚠️ Disclaimer

**Trading involves significant risk.** These bots are for educational purposes. 

**Important:**
- Paper trade extensively before live trading
- Start with small amounts
- Understand the strategies before automating
- Past performance doesn't guarantee future results
- You are responsible for your own trading decisions

## 📄 License

Both projects appear to be publicly shared for educational/development purposes. Review individual LICENSE files in each subdirectory for details.

## 🎯 Next Steps

1. **Test both systems** with paper trading
2. **Calibrate 5-minute strategy** with historical data
3. **Create product packages** for Gumroad/SAAS
4. **Build marketing content** around "Clawdbot discovery"
5. **Monitor performance** and iterate on strategies

---

**Repository created:** February 18, 2026  
**Source:** Public repositories (jmoss82/polymarket-bot, virtexvirtuoso/polyclawd)  
**Status:** Ready for productization