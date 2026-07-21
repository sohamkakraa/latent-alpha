"""Build a versioned historical sentiment cache from GDELT TimelineTone.

The DOC API is queried in 90-day chunks to retain daily resolution. Per-ticker
components are persisted immediately, so interruption and machine sleep are
safe; rerunning skips completed ticker/date chunks.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.sentiment_cache import SentimentCache
from llm.sentiment_encoder import SENTIMENT_AXES
from precompute_sentiment import TICKER_GDELT_MAP


logger = logging.getLogger(__name__)
API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _ticker_query(ticker: str) -> str:
    names = TICKER_GDELT_MAP.get(ticker, [ticker.replace(".NS", "")])
    return f'"{names[0]}"'


def _chunks(start: pd.Timestamp, end: pd.Timestamp, days: int = 90):
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


def fetch_timeline(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    retries: int = 8,
) -> dict[str, float]:
    params = {
        "query": _ticker_query(ticker),
        "mode": "timelinetone",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
        "timelinesmooth": 0,
    }
    for attempt in range(retries):
        response = requests.get(API_URL, params=params, timeout=180)
        if response.status_code == 429:
            delay = min(30 * (attempt + 1), 180)
            logger.warning("GDELT rate limit; retrying in %ds", delay)
            time.sleep(delay)
            continue
        if response.status_code >= 500:
            time.sleep(min(10 * (attempt + 1), 60))
            continue
        response.raise_for_status()
        payload = response.json()
        resolution = payload.get("query_details", {}).get("date_resolution")
        if resolution != "day":
            raise RuntimeError(
                f"GDELT returned {resolution!r} resolution for "
                f"{ticker} {start.date()}–{end.date()}; expected daily."
            )
        timeline = payload.get("timeline", [])
        records = timeline[0].get("data", []) if timeline else []
        tones = {}
        for record in records:
            date = pd.Timestamp(record["date"][:8]).strftime("%Y-%m-%d")
            tones[date] = float(
                np.clip(float(record["value"]) / 10.0, -1.0, 1.0)
            )
        return tones
    raise RuntimeError(
        f"GDELT failed after {retries} attempts for "
        f"{ticker} {start.date()}–{end.date()}"
    )


def tone_to_axes(tone: float) -> np.ndarray:
    """Map source tone into the documented 10-axis schema."""
    return np.asarray(
        [
            tone,
            tone * 0.4,
            tone * 0.6,
            -abs(tone) * 0.5,
            abs(tone),
            tone * 0.3,
            tone * 0.2,
            tone * 0.5,
            abs(tone) * 0.5,
            -tone * 0.25,
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build daily GDELT TimelineTone sentiment cache"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--request-delay", type=float, default=6.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    with open(args.config) as f:
        config = yaml.safe_load(f)

    start = pd.Timestamp(args.start or config["backtest"]["start_date"])
    end = pd.Timestamp(args.end or config["backtest"]["end_date"])
    dates = pd.bdate_range(start, end)
    date_strings = [date.strftime("%Y-%m-%d") for date in dates]
    tickers = list(config["market"]["tickers"])
    model = config["llm"]["sentiment_cache_model"]
    schema_version = int(config["llm"]["sentiment_schema_version"])

    chunks = list(_chunks(start, end))
    logger.info(
        "Build plan: %d tickers × %d chunks, %d business dates, model=%s",
        len(tickers),
        len(chunks),
        len(dates),
        model,
    )
    if args.dry_run:
        return

    cache = SentimentCache("data/sentiment_cache.db")
    for ticker in tickers:
        completed = cache.component_dates(model=model, ticker=ticker)
        for chunk_start, chunk_end in chunks:
            chunk_dates = [
                date.strftime("%Y-%m-%d")
                for date in pd.bdate_range(chunk_start, chunk_end)
            ]
            if set(chunk_dates).issubset(completed):
                continue
            tones = fetch_timeline(ticker, chunk_start, chunk_end)
            for date in chunk_dates:
                tone = tones.get(date, 0.0)
                cache.put_component(
                    date=date,
                    model=model,
                    ticker=ticker,
                    tone=tone,
                    source_count=int(abs(tone) > 0),
                )
            completed.update(chunk_dates)
            logger.info(
                "%s: cached %s–%s (%d/%d dates)",
                ticker,
                chunk_start.date(),
                chunk_end.date(),
                len(completed),
                len(date_strings),
            )
            time.sleep(max(args.request_delay, 5.0))

    components = cache.load_components(model=model)
    for date in date_strings:
        per_ticker = components.get(date, {})
        vector = np.concatenate(
            [tone_to_axes(per_ticker.get(ticker, 0.0)) for ticker in tickers]
        )
        cache.put(
            date,
            vector,
            model=model,
            source_count=sum(
                int(abs(per_ticker.get(ticker, 0.0)) > 0)
                for ticker in tickers
            ),
            schema_version=schema_version,
            tickers=tickers,
            axes=list(SENTIMENT_AXES),
            model_version=model,
        )

    validation = cache.validate_coverage(
        date_strings,
        model=model,
        vector_dim=len(tickers) * len(SENTIMENT_AXES),
        schema_version=schema_version,
    )
    logger.info("Validated sentiment cache: %s", validation)


if __name__ == "__main__":
    main()
