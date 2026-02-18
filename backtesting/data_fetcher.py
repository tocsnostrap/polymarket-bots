"""
Fetch and cache BTC historical candle data from the Coinbase public API.

Coinbase "candles" endpoint returns up to 300 candles per request.
We page backward from today to fill the requested number of days,
then persist to CSV so subsequent runs hit cache instead of the network.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

COINBASE_BASE = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"
MAX_CANDLES_PER_REQ = 300
RATE_LIMIT_PAUSE = 0.35  # seconds between requests


async def _fetch_candles_page(
    session: aiohttp.ClientSession,
    granularity: int,
    start_iso: str,
    end_iso: str,
) -> list[list]:
    url = f"{COINBASE_BASE}/products/{PRODUCT}/candles"
    params = {
        "granularity": granularity,
        "start": start_iso,
        "end": end_iso,
    }
    for attempt in range(5):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    wait = 2 ** attempt
                    log.warning("Rate-limited, backing off %ds", wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            wait = 2 ** attempt
            log.warning("Request failed (%s), retry in %ds", exc, wait)
            await asyncio.sleep(wait)
    log.error("Giving up on page %s → %s", start_iso, end_iso)
    return []


async def fetch_btc_candles(
    days: int = 90,
    granularity: int = 60,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of BTC-USD candles with columns:
        timestamp, open, high, low, close, volume
    Granularity is in seconds (60 = 1-minute candles).
    Data is cached to CSV under data/.
    """
    label = f"btc_{granularity // 60}min"
    cache_path = DATA_DIR / f"{label}.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        expected_start = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        if df["timestamp"].min() <= expected_start + pd.Timedelta(hours=2):
            log.info("Using cached %s (%d rows)", cache_path.name, len(df))
            return df
        log.info("Cache stale, re-fetching")

    log.info("Fetching %d days of %s candles from Coinbase …", days, label)
    now = int(time.time())
    total_seconds = days * 86400
    start_ts = now - total_seconds

    page_seconds = MAX_CANDLES_PER_REQ * granularity
    all_candles: list[list] = []

    async with aiohttp.ClientSession() as session:
        cursor = start_ts
        while cursor < now:
            page_end = min(cursor + page_seconds, now)
            s_iso = pd.Timestamp(cursor, unit="s", tz="UTC").isoformat()
            e_iso = pd.Timestamp(page_end, unit="s", tz="UTC").isoformat()
            page = await _fetch_candles_page(session, granularity, s_iso, e_iso)
            if page:
                all_candles.extend(page)
            cursor = page_end
            await asyncio.sleep(RATE_LIMIT_PAUSE)

    if not all_candles:
        raise RuntimeError("Failed to fetch any candle data from Coinbase")

    # Coinbase returns [time, low, high, open, close, volume]
    df = pd.DataFrame(all_candles, columns=["time", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    log.info("Saved %d candles → %s", len(df), cache_path)
    return df


def load_cached_candles(granularity_minutes: int = 1) -> Optional[pd.DataFrame]:
    cache_path = DATA_DIR / f"btc_{granularity_minutes}min.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["timestamp"])
    return None


if __name__ == "__main__":
    df = asyncio.run(fetch_btc_candles(days=90, granularity=60))
    print(f"Fetched {len(df)} 1-minute candles")
    print(df.head())
    print(df.tail())
