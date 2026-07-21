"""
scripts/build_sentiment_cache.py
════════════════════════════════
Build historical sentiment cache using price-derived pseudo-sentiment.

Since real historical RSS feeds aren't available for 2018-2025, this script
generates price-derived pseudo-sentiment as a baseline signal:
  - Positive 5-day return  → positive headline_sentiment
  - High 20-day volatility → high volatility_expectation
  - Strong 60-day momentum → positive sector_momentum
  - ATR-based signal       → regulatory_risk proxy
  - Mean reversion signal  → contrarian_indicator

This gives the agent a noise-correlated sentiment signal that approximates
what real news coverage would have looked like.

Usage:
  # Full build (2018-2025, all tickers)
  python scripts/build_sentiment_cache.py

  # Specific date range
  python scripts/build_sentiment_cache.py --start 2023-01-01 --end 2024-01-01

  # Specific tickers
  python scripts/build_sentiment_cache.py --tickers RELIANCE.NS TCS.NS

  # Resume from interruption (skips already-cached dates)
  python scripts/build_sentiment_cache.py --resume

  # Dry run (count dates, estimate time, don't run)
  python scripts/build_sentiment_cache.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SENTIMENT_AXES = [
    "headline_sentiment",
    "earnings_signal",
    "macro_signal",
    "regulatory_risk",
    "volatility_expectation",
    "fundamental_quality",
    "management_sentiment",
    "sector_momentum",
    "event_catalyst",
    "contrarian_indicator",
]


def _compute_pseudo_sentiment(
    ticker_data: dict,
    tickers: list,
    date: pd.Timestamp,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Generate a pseudo-sentiment vector from price data.

    Returns np.ndarray of shape (n_tickers * 10,) — concatenated per-ticker vectors.
    """
    per_ticker_vectors = []

    for ticker in tickers:
        df = ticker_data.get(ticker)
        if df is None or date not in df.index:
            per_ticker_vectors.append(np.zeros(10, dtype=np.float32))
            continue

        loc = df.index.get_loc(date)

        def safe_ret(lookback: int) -> float:
            if loc < lookback:
                return 0.0
            past = df["Close"].iloc[loc - lookback]
            curr = df["Close"].iloc[loc]
            if past <= 0:
                return 0.0
            return float((curr - past) / past)

        def safe_vol(lookback: int) -> float:
            if loc < lookback:
                return 0.0
            returns = df["Close"].iloc[max(0, loc - lookback):loc + 1].pct_change().dropna()
            return float(returns.std()) if len(returns) > 1 else 0.0

        ret_5d   = safe_ret(5)
        ret_20d  = safe_ret(20)
        ret_60d  = safe_ret(60)
        vol_5d   = safe_vol(5)
        vol_20d  = safe_vol(20)

        # Map price signals to sentiment axes
        headline_sentiment   = float(np.clip(ret_5d * 10, -1.0, 1.0))
        earnings_signal      = float(np.clip(ret_20d * 5, -1.0, 1.0))
        macro_signal         = float(np.clip(ret_60d * 3, -1.0, 1.0))
        regulatory_risk      = float(np.clip(-vol_20d * 20, -1.0, 1.0))   # High vol → negative reg signal
        volatility_expect    = float(np.clip(vol_5d * 20 - 0.5, -1.0, 1.0))  # Centred
        fundamental_quality  = float(np.clip(ret_20d * 8, -1.0, 1.0))
        management_sentiment = float(np.clip(ret_60d * 4, -1.0, 1.0))
        sector_momentum      = float(np.clip(ret_60d * 5, -1.0, 1.0))
        event_catalyst       = float(np.clip(abs(ret_5d) * 15 - 0.5, -1.0, 1.0))  # Large moves = events
        contrarian_indicator = float(np.clip(-ret_20d * 8, -1.0, 1.0))   # Mean reversion

        vec = np.array([
            headline_sentiment, earnings_signal, macro_signal, regulatory_risk,
            volatility_expect, fundamental_quality, management_sentiment,
            sector_momentum, event_catalyst, contrarian_indicator,
        ], dtype=np.float32)

        # Add light noise to simulate real news variability
        if noise_std > 0:
            vec += np.random.normal(0, noise_std, size=vec.shape).astype(np.float32)
        vec = np.clip(vec, -1.0, 1.0)

        per_ticker_vectors.append(vec)

    return np.concatenate(per_ticker_vectors).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build historical sentiment cache")
    parser.add_argument("--start",   default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--tickers", nargs="+", default=None, help="Specific tickers")
    parser.add_argument("--resume",  action="store_true", help="Skip already-cached dates")
    parser.add_argument("--dry-run", action="store_true", help="Count dates, estimate time, exit")
    parser.add_argument("--noise",   type=float, default=0.05, help="Noise std (default: 0.05)")
    parser.add_argument("--config",  default="config/config.yaml", help="Path to config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tickers = args.tickers or config["market"]["tickers"]
    start   = args.start   or config["backtest"]["start_date"]
    end     = args.end     or config["backtest"]["end_date"]

    from llm.sentiment_cache import SentimentCache
    cache = SentimentCache(
        db_path=config.get("paths", {}).get("data_cache", "./data/cache") + "/../sentiment_cache.db",
        budget_inr=config.get("llm", {}).get("budget_inr", 2000.0),
    )

    # Generate all business days in range
    all_dates = pd.bdate_range(start=start, end=end)
    logger.info("Date range: %s → %s (%d business days)", start, end, len(all_dates))
    logger.info("Tickers: %s", tickers)

    if args.resume:
        missing = [str(d.date()) for d in all_dates if not cache.has(str(d.date()))]
        logger.info("Resume mode: %d/%d dates missing from cache", len(missing), len(all_dates))
        dates_to_process = [pd.Timestamp(d) for d in missing]
    else:
        dates_to_process = list(all_dates)

    if args.dry_run:
        secs_per_date = 0.05  # Price-derived pseudo-sentiment is fast
        est_hours = len(dates_to_process) * secs_per_date / 3600
        logger.info(
            "DRY RUN: Would process %d dates (est. %.1f minutes at %.2fs/date)",
            len(dates_to_process), est_hours * 60, secs_per_date,
        )
        return

    # Download market data once for all tickers
    logger.info("Downloading market data for pseudo-sentiment computation...")
    from env.data_loader import NSEDataLoader
    loader = NSEDataLoader(config)
    try:
        raw_data = loader.load(start=start, end=end)
    except Exception as e:
        logger.error("Failed to download market data: %s", e)
        sys.exit(1)

    # Filter to requested tickers
    ticker_data = {t: df for t, df in raw_data.items() if t in tickers}

    vector_dim = len(tickers) * len(SENTIMENT_AXES)
    processed = 0
    errors    = 0
    t0        = time.time()

    for i, date in enumerate(dates_to_process):
        date_str = str(date.date())

        if args.resume and cache.has(date_str):
            continue

        try:
            vec = _compute_pseudo_sentiment(ticker_data, tickers, date, noise_std=args.noise)

            if len(vec) != vector_dim:
                logger.warning("Vector dim mismatch at %s: got %d, expected %d", date_str, len(vec), vector_dim)
                vec = np.zeros(vector_dim, dtype=np.float32)

            cache.put(
                date=date_str,
                vector=vec,
                model="price_derived",
                source_count=len(tickers),
                cost_usd=0.0,
            )
            processed += 1

        except Exception as e:
            logger.warning("Error at %s: %s", date_str, e)
            errors += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate    = processed / max(elapsed, 0.001)
            remain  = (len(dates_to_process) - i - 1) / max(rate, 0.001)
            logger.info(
                "Progress: %d/%d (%.1f%%) | %d errors | %.1f dates/s | ETA %.1f min",
                i + 1, len(dates_to_process),
                (i + 1) / len(dates_to_process) * 100,
                errors, rate, remain / 60,
            )

    # Validate cache
    logger.info("\nValidating cache...")
    stats = cache.stats()
    logger.info("Cache stats: %s", stats)

    # Spot-check a few vectors for NaN
    sample_dates = [str(d.date()) for d in dates_to_process[:10]]
    nan_found = 0
    for d in sample_dates:
        v = cache.get(d)
        if v is not None and np.any(np.isnan(v)):
            nan_found += 1
            logger.warning("NaN in cached vector for %s", d)

    if nan_found == 0:
        logger.info("Validation passed: no NaN in sampled vectors.")
    else:
        logger.warning("Validation found %d vectors with NaN!", nan_found)

    logger.info(
        "Done. Processed %d dates, %d errors. Total cache: %d entries.",
        processed, errors, stats["total_entries"],
    )


if __name__ == "__main__":
    main()
