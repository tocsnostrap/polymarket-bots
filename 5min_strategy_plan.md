# 5-MINUTE POLYMARKET STRATEGY PLAN

## OPPORTUNITY
The tweet mentions "5m & 15m Polymarket Clawdbot setup." We have the 15-minute bot. Need to create the 5-minute version.

## STRATEGY ADAPTATION

### From 15-minute to 5-minute:
1. **Interval:** 300 seconds instead of 900 seconds
2. **Move bins:** Need recalibration (smaller moves matter more)
3. **Time bins:** Different time windows (e.g., 20-60, 60-120, 120-180, 180-240 seconds)
4. **Market availability:** Check if Polymarket has 5-minute BTC markets

### Hypothetical 5-Minute Fair Value Table:
```
Move Size   | Time Elapsed | Expected Win Rate
------------|--------------|------------------
0.01-0.03%  | 20-60s       | 55-60%
0.03-0.06%  | 20-60s       | 60-65%
0.06-0.10%  | 20-60s       | 65-70%
0.10%+      | 20-60s       | 70-75%

0.01-0.03%  | 60-120s      | 60-65%
0.03-0.06%  | 60-120s      | 65-70%
0.06-0.10%  | 60-120s      | 70-75%
0.10%+      | 60-120s      | 75-80%

0.01-0.03%  | 120-180s     | 65-70%
0.03-0.06%  | 120-180s     | 70-75%
0.06-0.10%  | 120-180s     | 75-80%
0.10%+      | 120-180s     | 80-85%

0.01-0.03%  | 180-240s     | 70-75%
0.03-0.06%  | 180-240s     | 75-80%
0.06-0.10%  | 180-240s     | 80-85%
0.10%+      | 180-240s     | 85-90%
```

## IMPLEMENTATION STEPS

### 1. Market Research
- Check if Polymarket has 5-minute BTC markets
- If not, what's the shortest interval available?
- Alternative: ETH markets, other crypto pairs

### 2. Data Collection
- Need 1-minute BTC price data (Coinbase API)
- Simulate 5-minute intervals historically
- Calculate win rates for different move/time combinations

### 3. Calibration
- Modify `calibrate.py` for 5-minute intervals
- Generate new fair value table
- Validate with backtesting

### 4. Bot Modification
- Change interval from 900s to 300s
- Update time bins in strategy
- Adjust monitoring intervals
- Modify exit logic (shorter timeframes)

### 5. Testing
- Paper trading with historical data
- Live paper trading
- Small live trades

## TECHNICAL CHANGES NEEDED

### In `polymarket-bot`:
```python
# Current (15-minute)
INTERVAL_SECS = 900
ENTRY_WINDOW = (60, 840)  # 1-14 minutes
EXIT_BEFORE_END = 30  # 30 seconds before end

# Proposed (5-minute)  
INTERVAL_SECS = 300
ENTRY_WINDOW = (20, 240)  # 20-240 seconds (20s-4min)
EXIT_BEFORE_END = 10  # 10 seconds before end

# Move bins (smaller moves matter)
MOVE_BINS = [
    (0.01, 0.03, "0.01-0.03"),
    (0.03, 0.06, "0.03-0.06"),
    (0.06, 0.10, "0.06-0.10"),
    (0.10, 999.0, "0.10+"),
]

# Time bins (shorter windows)
ELAPSED_BINS = [
    (20, 60, "20-60"),
    (60, 120, "60-120"),
    (120, 180, "120-180"),
    (180, 241, "180-240"),
]
```

## RISK CONSIDERATIONS

### Higher Frequency = Higher Risk
1. **More trades** → More transaction costs
2. **Tighter spreads needed** → Less room for error
3. **Faster decisions** → More sensitive to latency
4. **Shorter trends** → Less reliable momentum

### Mitigations:
1. **Smaller position sizes** for 5-minute trades
2. **Tighter risk management** (lower loss limits)
3. **Higher minimum edge** requirement
4. **Stricter entry filters**

## PRODUCT PACKAGING

### Option 1: Combined Product
- **"Clawdbot Duo"** - $197
- Includes both 5-minute and 15-minute bots
- Calibrated tables for both timeframes
- Configuration switcher

### Option 2: Separate Products
- **"Clawdbot 15-Minute"** - $97
- **"Clawdbot 5-Minute"** - $147 (premium for higher frequency)
- **"Clawdbot Bundle"** - $197 (both)

### Option 3: SaaS Model
- **"Polymarket Frequency Suite"** - $297/month
- Access to 1, 5, 15, 60-minute strategies
- Real-time signal dashboard
- Performance analytics

## TIMELINE

### Phase 1: Research (2 hours)
- [ ] Check Polymarket for 5-minute markets
- [ ] Analyze historical BTC 1-minute data
- [ ] Estimate potential win rates

### Phase 2: Development (4 hours)
- [ ] Modify calibration script
- [ ] Generate 5-minute fair value table
- [ ] Adapt bot code for 5-minute intervals

### Phase 3: Testing (2 hours)
- [ ] Backtest with historical data
- [ ] Paper trade simulation
- [ ] Validate win rates

### Phase 4: Packaging (2 hours)
- [ ] Create product documentation
- [ ] Build setup guides
- [ ] Prepare marketing materials

**Total:** 10 hours to MVP

## NEXT ACTION

**Immediate:** Check if Polymarket has 5-minute BTC markets via their API or website.

If yes → Proceed with development
If no → Consider alternative (ETH, shorter crypto intervals, or focus on 15-minute only)