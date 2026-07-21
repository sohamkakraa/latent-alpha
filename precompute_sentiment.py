"""
precompute_sentiment.py
────────────────────────
Offline script that walks every trading date in the training range,
fetches historical news for that date, runs it through the configured
model, and stores the result in data/sentiment_cache.db.

Run this once. It takes a few hours on a local model. After that,
training reads from the cache at zero cost and near-zero latency.

Usage
-----
  # Gemma 4 local — free, private, recommended
  python precompute_sentiment.py --model local

  # Claude Haiku — ₹78 total, faster
  python precompute_sentiment.py --model haiku

  # Dry run — check coverage without calling any model
  python precompute_sentiment.py --dry-run

  # Resume from where you left off (safe to re-run any time)
  python precompute_sentiment.py --model local

  # Show cache stats
  python precompute_sentiment.py --stats

Model selection guidance
------------------------
  local (Gemma 4):  Free. Requires: brew install ollama && ollama pull gemma4:27b
  haiku:            ₹78 total. Requires: ANTHROPIC_API_KEY in .env
  sonnet:           ₹934 total. Higher quality but expensive for bulk.
  opus:             ₹4,668 — exceeds ₹2,000 budget cap, blocked.

News source for historical dates
---------------------------------
We use GDELT GKG (Global Knowledge Graph) which provides daily bulk files
with sentiment tone and entity data going back to 2015. It's free, covers
Indian companies reasonably well, and requires no API key.

GDELT tone → our sentiment axes mapping is approximate but consistent,
which is what the RL agent needs — not perfect accuracy but stable signal.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import io
import zipfile

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from llm.sentiment_encoder import SENTIMENT_AXES

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger  = logging.getLogger(__name__)
console = Console()

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"

# ── GDELT config ──────────────────────────────────────────────────────────────
GDELT_GKG_URL = "http://data.gdeltproject.org/gkg/{date}.gkg.csv.zip"

# Map our tickers to search terms in GDELT
TICKER_GDELT_MAP = {
    "RELIANCE.NS":   ["Reliance Industries", "Reliance", "RIL"],
    "TCS.NS":        ["Tata Consultancy Services", "TCS"],
    "INFY.NS":       ["Infosys"],
    "HDFCBANK.NS":   ["HDFC Bank", "HDFC"],
    "ICICIBANK.NS":  ["ICICI Bank", "ICICI"],
    "HINDUNILVR.NS": ["Hindustan Unilever", "HUL"],
    "ITC.NS":        ["ITC Limited", "ITC"],
    "KOTAKBANK.NS":  ["Kotak Mahindra", "Kotak"],
    "LT.NS":         ["Larsen Toubro", "L&T"],
    "AXISBANK.NS":   ["Axis Bank"],
}

# Cost per call in USD for cloud models
MODEL_COSTS = {
    "haiku":  0.00061,
    "sonnet": 0.00735,
    "opus":   0.03675,   # blocked — exceeds budget
    "local":  0.0,
    "gdelt":  0.0,
}

BLOCKED_MODELS = {"opus"}   # exceeds ₹2,000 budget cap
USD_INR = 84.0


# ── GDELT news fetcher ────────────────────────────────────────────────────────

def fetch_gdelt_signals(date_str: str, tickers: List[str]) -> Dict[str, dict]:
    """
    Download GDELT GKG data for a specific date and extract tone signals
    for each ticker.

    Returns dict: {ticker: {"tone": float, "article_count": int}}
    tone is in [-100, 100] (GDELT scale) → we normalise to [-1, 1]
    """
    # GDELT GKG daily files use YYYYMMDD format
    gdelt_date = date_str.replace("-", "")
    url = GDELT_GKG_URL.format(date=gdelt_date)

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return {}   # No GDELT file for this date (weekends, holidays)
        resp.raise_for_status()

        # GDELT files are zipped CSVs
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                # GKG columns: DATE, NUMARTS, COUNTS, THEMES, LOCATIONS,
                #              PERSONS, ORGS, TONE, CAMEOEVENTIDS, SOURCES, SOURCEURLS
                df = pd.read_csv(
                    f, sep="\t", header=None, on_bad_lines="skip",
                    usecols=[0, 1, 7],   # DATE, NUMARTS, TONE
                    names=["date", "num_arts", "tone_raw"],
                    encoding="utf-8", encoding_errors="replace",
                )
                # Also need ORGS column for ticker matching
                # Re-read with org column
                f.seek(0) if hasattr(f, 'seek') else None

        # Re-read for org matching — GDELT GKG col 6 = ORGS
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(
                    f, sep="\t", header=None, on_bad_lines="skip",
                    usecols=[0, 1, 6, 7],
                    names=["date", "num_arts", "orgs", "tone_raw"],
                    encoding="utf-8", encoding_errors="replace",
                )

        df["orgs"]     = df["orgs"].fillna("").astype(str)
        df["num_arts"] = pd.to_numeric(df["num_arts"], errors="coerce").fillna(0).astype(int)
        df["tone_raw"] = pd.to_numeric(df["tone_raw"].astype(str).str.split(",").str[0], errors="coerce")

        results = {}
        for ticker in tickers:
            keywords = TICKER_GDELT_MAP.get(ticker, [ticker.replace(".NS", "")])
            mask = df["orgs"].str.contains("|".join(keywords), case=False, na=False)
            relevant = df[mask]
            if len(relevant) == 0:
                results[ticker] = {"tone": 0.0, "article_count": 0}
            else:
                avg_tone = relevant["tone_raw"].mean()
                results[ticker] = {
                    "tone":          float(np.clip(avg_tone / 10.0, -1.0, 1.0)),  # normalise
                    "article_count": int(relevant["num_arts"].sum()),
                }

        return results

    except Exception as e:
        logger.debug("GDELT fetch failed for %s: %s", date_str, e)
        return {}


def gdelt_to_sentiment_vector(
    gdelt_signals: Dict[str, dict],
    tickers: List[str],
    n_axes: int = 10,
) -> np.ndarray:
    """
    Convert GDELT tone signals to a sentiment vector.

    GDELT gives us one tone score per ticker. We expand it across all
    5 sentiment axes with some heuristic differentiation:
      - headline_sentiment:     direct tone mapping
      - earnings_signal:        attenuated (GDELT rarely covers earnings specifically)
      - macro_signal:           slightly attenuated
      - regulatory_risk:        inverted (negative tone → high risk)
      - volatility_expectation: absolute tone (high |tone| → high volatility)
    """
    vector = np.zeros(len(tickers) * n_axes, dtype=np.float32)

    for i, ticker in enumerate(tickers):
        sig  = gdelt_signals.get(ticker, {})
        tone = sig.get("tone", 0.0)

        base = i * n_axes
        vector[base + 0] = tone                            # headline_sentiment
        vector[base + 1] = tone * 0.4                     # earnings_signal (attenuated)
        vector[base + 2] = tone * 0.6                     # macro_signal
        vector[base + 3] = -abs(tone) * 0.5               # regulatory_risk (neg tone = risk)
        vector[base + 4] = abs(tone)                       # volatility_expectation
        if n_axes >= 10:
            vector[base + 5] = tone * 0.3                 # fundamental_quality
            vector[base + 6] = tone * 0.2                 # management_sentiment
            vector[base + 7] = tone * 0.5                 # sector_momentum
            vector[base + 8] = abs(tone) * 0.5            # event_catalyst
            vector[base + 9] = -tone * 0.25               # contrarian_indicator

    return vector


# ── Cloud model caller ────────────────────────────────────────────────────────

def call_cloud_model(
    model_name: str,
    tickers: List[str],
    gdelt_signals: Dict[str, dict],
) -> np.ndarray:
    """
    Call Claude (haiku/sonnet) with GDELT signals as context.
    Used when --model haiku or --model sonnet is specified.
    """
    try:
        import anthropic
        from llm.sentiment_encoder import SYSTEM_PROMPT, SENTIMENT_AXES

        model_map = {
            "haiku":  "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-6",
        }
        claude_model = model_map[model_name]
        client = anthropic.Anthropic()

        # Build a compact prompt from GDELT signals (no full articles needed)
        lines = [f"TICKERS: {', '.join(tickers)}", "", "GDELT TONE SIGNALS (scale -1 to 1):"]
        for ticker in tickers:
            sig = gdelt_signals.get(ticker, {})
            lines.append(f"  {ticker}: tone={sig.get('tone', 0.0):.3f}, articles={sig.get('article_count', 0)}")

        lines.append("")
        lines.append(
            'Return JSON: {"scores": {"TICKER.NS": {"headline_sentiment": 0.0, '
            '"earnings_signal": 0.0, "macro_signal": 0.0, "regulatory_risk": 0.0, '
            '"volatility_expectation": 0.0}}}'
        )

        resp = client.messages.create(
            model=claude_model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n".join(lines)}],
        )
        raw = resp.content[0].text.strip()

        # Parse JSON
        import json
        if "```" in raw:
            raw = raw.split("```")[1][4:] if raw.split("```")[1].startswith("json") else raw.split("```")[1]
        scores_map = json.loads(raw).get("scores", {})

        n_axes = len(SENTIMENT_AXES)
        vector = np.zeros(len(tickers) * n_axes, dtype=np.float32)
        for i, ticker in enumerate(tickers):
            for j, axis in enumerate(SENTIMENT_AXES):
                score = float(scores_map.get(ticker, {}).get(axis, 0.0))
                vector[i * n_axes + j] = float(np.clip(score, -1.0, 1.0))
        return vector

    except Exception as e:
        logger.warning("Cloud model call failed: %s", e)
        return np.zeros(
            len(tickers) * len(SENTIMENT_AXES),
            dtype=np.float32,
        )


def call_local_model(
    tickers: List[str],
    gdelt_signals: Dict[str, dict],
    ollama_model: str = "gemma4:27b",
) -> np.ndarray:
    """Call the local Ollama model with GDELT signals as context."""
    from llm.local_model import OllamaClient

    client = OllamaClient(model=ollama_model)

    # Build mock VerifiedArticle list from GDELT signals
    from llm.verifier import VerifiedArticle
    from llm.news_fetcher import NewsArticle

    articles_by_ticker = {}
    for ticker in tickers:
        sig = gdelt_signals.get(ticker, {})
        if sig.get("article_count", 0) > 0:
            mock_article = NewsArticle(
                url=f"gdelt://{ticker}",
                title=f"GDELT signal for {ticker}: tone={sig['tone']:.2f}",
                summary=(
                    f"Aggregated GDELT news signal for {ticker}. "
                    f"Overall sentiment tone: {sig['tone']:.3f} "
                    f"(scale -1 bearish to +1 bullish). "
                    f"Based on {sig['article_count']} news articles."
                ),
                source="GDELT",
                published_utc=None,
            )
            articles_by_ticker[ticker] = [VerifiedArticle(mock_article, confidence=0.7, corroboration_count=2)]
        else:
            articles_by_ticker[ticker] = []

    return client.get_sentiment(tickers, articles_by_ticker)


# ── Main precompute loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute historical sentiment cache")
    parser.add_argument("--model",       default="gdelt",
                        choices=["gdelt", "local", "haiku", "sonnet"],
                        help="Signal builder (default: direct GDELT tone)")
    parser.add_argument("--ollama-model", default="gemma4:latest",
                        help="Ollama model tag (default: gemma4:27b)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Show what would be computed without calling any model")
    parser.add_argument("--stats",       action="store_true",
                        help="Show cache statistics and exit")
    parser.add_argument("--budget-inr",  type=float, default=2000.0,
                        help="Hard spend cap in INR (default: ₹2,000)")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    from llm.sentiment_cache import SentimentCache
    cache = SentimentCache(
        db_path="data/sentiment_cache.db",
        budget_inr=args.budget_inr,
    )

    if args.stats:
        stats = cache.stats()
        t = Table(title="Sentiment Cache Stats")
        for k, v in stats.items():
            t.add_column(k)
        t.add_row(*[str(v) for v in stats.values()])
        console.print(t)
        return

    # ── Generate all trading dates in range ───────────────────────────────────
    start = pd.Timestamp(config["backtest"]["start_date"])
    end   = pd.Timestamp(config["backtest"]["end_date"])
    all_dates = pd.bdate_range(start=start, end=end)   # business days only
    date_strings = [d.strftime("%Y-%m-%d") for d in all_dates]

    model_id = (
        config["llm"]["sentiment_cache_model"]
        if args.model == "gdelt"
        else f"{args.model}:{args.ollama_model}:schema2"
        if args.model == "local"
        else f"{args.model}:schema2"
    )
    missing  = cache.missing_dates(date_strings, model=model_id)
    tickers  = config["market"]["tickers"]
    n_axes   = config["llm"]["sentiment_dim"]
    cost_per = MODEL_COSTS.get(args.model, 0.0)
    est_cost_inr = len(missing) * cost_per * USD_INR

    console.print(f"\n[bold]Sentiment precomputation[/bold]")
    console.print(f"  Date range:     {start.date()} → {end.date()}")
    console.print(f"  Total dates:    {len(date_strings)}")
    console.print(f"  Already cached: {len(date_strings) - len(missing)}")
    console.print(f"  To compute:     [yellow]{len(missing)}[/yellow]")
    console.print(f"  Model:          [cyan]{model_id}[/cyan]" +
                  (f" ({args.ollama_model})" if args.model == "local" else ""))
    console.print(f"  Est. API cost:  ₹{est_cost_inr:.0f} / budget ₹{args.budget_inr:.0f}")
    console.print(f"  Spent so far:   ₹{cache.total_cost_inr():.2f}\n")

    if args.dry_run:
        console.print("[dim]Dry run — no model calls made.[/dim]")
        return

    if args.model == "local":
        from llm.local_model import OllamaClient
        client = OllamaClient(args.ollama_model)
        if not client.is_available():
            console.print(
                f"[red]Ollama model '{args.ollama_model}' not found.[/red]\n"
                f"Run: [bold]ollama pull {args.ollama_model}[/bold]"
            )
            available = client.list_models()
            if available:
                console.print(f"Available models: {', '.join(available)}")
            return

    errors   = 0
    skipped  = 0  # dates with no GDELT data (weekends, holidays already excluded)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Computing sentiment...", total=len(missing))

        for date_str in missing:
            # ── Budget check ──────────────────────────────────────────────────
            if cache.is_over_budget():
                console.print(
                    f"\n[yellow]⚠ Budget cap of ₹{args.budget_inr:.0f} reached "
                    f"(spent ₹{cache.total_cost_inr():.2f}). Stopping.[/yellow]\n"
                    f"Re-run to continue with --model local (free) or increase --budget-inr."
                )
                break

            progress.update(task, description=f"[dim]{date_str}[/dim]")

            # ── Fetch GDELT signals ───────────────────────────────────────────
            gdelt = fetch_gdelt_signals(date_str, tickers)
            if not gdelt:
                # No GDELT data — store a zero vector (neutral)
                vector = np.zeros(len(tickers) * n_axes, dtype=np.float32)
                cache.put(
                    date_str,
                    vector,
                    model=model_id,
                    source_count=0,
                    cost_usd=0.0,
                    schema_version=config["llm"]["sentiment_schema_version"],
                    tickers=tickers,
                    axes=list(SENTIMENT_AXES),
                    model_version=model_id,
                )
                skipped += 1
                progress.advance(task)
                continue

            # ── Run model ─────────────────────────────────────────────────────
            try:
                if args.model == "gdelt":
                    vector = gdelt_to_sentiment_vector(
                        gdelt, tickers, n_axes
                    )
                elif args.model == "local":
                    vector = call_local_model(tickers, gdelt, args.ollama_model)
                elif args.model in ("haiku", "sonnet"):
                    vector = call_cloud_model(args.model, tickers, gdelt)
                else:
                    vector = gdelt_to_sentiment_vector(gdelt, tickers, n_axes)

                source_count = sum(s.get("article_count", 0) for s in gdelt.values())
                cache.put(
                    date_str, vector,
                    model=model_id,
                    source_count=source_count,
                    cost_usd=cost_per,
                    schema_version=config["llm"]["sentiment_schema_version"],
                    tickers=tickers,
                    axes=list(SENTIMENT_AXES),
                    model_version=model_id,
                )

            except Exception as e:
                logger.warning("Failed on %s: %s", date_str, e)
                errors += 1

            progress.advance(task)
            time.sleep(0.05)   # small delay to avoid hammering Ollama

    # ── Final summary ─────────────────────────────────────────────────────────
    stats = cache.stats()
    try:
        validation = cache.validate_coverage(
            date_strings,
            model=model_id,
            vector_dim=len(tickers) * n_axes,
            schema_version=config["llm"]["sentiment_schema_version"],
        )
    except Exception as exc:
        console.print(f"[red]Cache validation failed: {exc}[/red]")
        raise
    console.print(f"\n[bold green]✓ Done[/bold green]")
    console.print(f"  Cached entries: {stats['total_entries']}")
    console.print(f"  Skipped (no data): {skipped}")
    console.print(f"  Errors: {errors}")
    console.print(f"  Total cost: ₹{stats['total_cost_inr']:.2f} / ₹{args.budget_inr:.0f} budget")
    console.print(f"  Cache size: {stats['db_size_mb']:.1f} MB")
    console.print(
        f"  Non-zero dates: {validation['nonzero_date_rate']:.1%}"
    )
    console.print(f"  Location: [dim]data/sentiment_cache.db[/dim]  ← back this up!")


if __name__ == "__main__":
    main()
