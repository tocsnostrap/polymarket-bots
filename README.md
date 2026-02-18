# Polymarket Trading Bots Collection

## Two Production-Ready Polymarket Trading Systems

### 1. **polymarket-bot** - 15-Minute BTC Up/Down Momentum Bot
**Strategy:** Data-driven momentum betting on BTC 15-minute intervals
**Key Features:**
- Calibrated fair value table from 90 days of historical data
- Chainlink price feed integration (same oracle as Polymarket resolution)
- Real-time order placement via CLOB
- Risk management with session loss limits
- Paper trading mode available

**Performance:** Based on 8,637 historical intervals with calibrated win probabilities

### 2. **polyclawd** - AI-Powered Prediction Market Trading System  
**Strategy:** Multi-source arbitrage across 15 signal sources
**Key Features:**
- 121 API endpoints for full trading system
- 7 profit strategies (arbitrage, whale fade, news speed, etc.)
- Bayesian confidence scoring with adaptive trading
- Kelly position sizing
- OpenClaw integration for cron jobs and alerts
- Production deployment at virtuosocrypto.com/polyclawd

**Signal Sources:** Vegas odds, ESPN, Betfair, Manifold, PredictIt, Kalshi, Metaculus, on-chain whale tracking, and more

## Quick Start

### For polymarket-bot:
```bash
cd polymarket-bot
cp .env.example .env  # Add your Polymarket API keys
pip install -r requirements.txt
python discover_markets.py  # Find active markets
python paper_trader.py     # Test strategy
python live_trader.py      # Live trading
```

### For polyclawd:
```bash
cd polyclawd
cp .env.example .env  # Configure API keys
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8420
```

## Trading Strategies

### 15-Minute Momentum (polymarket-bot)
- Trades BTC 15-minute Up/Down markets
- Uses calibrated probability table based on BTC move size and time elapsed
- Only enters when fair value exceeds market price by minimum edge
- Exits with profit targets or before resolution

### Multi-Source Arbitrage (polyclawd)
1. **Sharp vs Soft Line Arbitrage** - Vegas/Betfair odds vs Polymarket
2. **Manifold → Polymarket Flow** - Play money signals preceding real money
3. **Cross-Platform Arbitrage** - PredictIt vs Polymarket price gaps
4. **Whale Fade** - Bet against losing whale traders
5. **News Speed Edge** - Exploit slow market reactions
6. **Correlation Violation** - Math constraints between related markets
7. **Injury Impact Edge** - Sports injuries before line movement

## Architecture

### polymarket-bot:
```
Binance WS/Chainlink WS → Price Feed
Gamma API → Market Discovery  
CLOB API → Order Execution
Calibrated Table → Signal Generation
Risk Manager → Position Control
```

### polyclawd:
```
15 Data Sources → Edge Scanner → Signal Aggregator
Bayesian Scoring → Trading Engine → Paper/Live Trading
OpenClaw → Cron Alerts → Telegram
```

## Risk Management

**polymarket-bot:**
- Session loss limits ($15 default)
- Maximum spread limits (6% default)
- Entry/exit timeouts
- Position monitoring

**polyclawd:**
- Drawdown breaker (5% daily loss halt)
- Adaptive confidence scoring
- Quarter-Kelly position sizing
- Phase management with position limits

## License

Both projects appear to be publicly shared for educational/development purposes. Review individual LICENSE files for details.

## Disclaimer

Trading involves significant risk. These bots are for educational purposes. Paper trade extensively before considering live trading. Past performance does not guarantee future results.