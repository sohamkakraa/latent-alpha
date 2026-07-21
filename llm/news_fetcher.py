"""
llm/news_fetcher.py
────────────────────
Fetches and deduplicates financial news from RSS feeds relevant to
NSE-listed tickers. Articles are normalised, deduplicated by URL,
and filtered by ticker/company name relevance before being passed
to the sentiment encoder.

Design choices:
  - RSS-first: zero API cost, real-time, multiple sources
  - Relevance filtering: only pass articles that mention the ticker
    or company name — reduces LLM hallucination surface
  - TTL cache: identical fetch within the cache window returns
    the same articles (avoids redundant LLM calls during training)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class NewsArticle:
    url: str
    title: str
    summary: str
    source: str
    published_utc: Optional[datetime]
    ticker_relevance: List[str] = field(default_factory=list)  # which tickers this article mentions

    @property
    def uid(self) -> str:
        """Stable unique ID for deduplication."""
        return hashlib.md5(self.url.encode()).hexdigest()

    def to_prompt_snippet(self) -> str:
        """Compact text representation for LLM context window."""
        date_str = self.published_utc.strftime("%Y-%m-%d %H:%M UTC") if self.published_utc else "unknown date"
        return (
            f"SOURCE: {self.source} | DATE: {date_str}\n"
            f"TITLE: {self.title}\n"
            f"SUMMARY: {self.summary[:500]}"
        )


# ── Ticker → company name map ─────────────────────────────────────────────────
# Used to match articles that mention the company name but not the NSE symbol.

TICKER_COMPANY_MAP: Dict[str, List[str]] = {
    "RELIANCE.NS":    ["Reliance", "RIL", "Reliance Industries"],
    "TCS.NS":         ["TCS", "Tata Consultancy"],
    "INFY.NS":        ["Infosys", "INFY"],
    "HDFCBANK.NS":    ["HDFC Bank", "HDFCBANK"],
    "ICICIBANK.NS":   ["ICICI Bank", "ICICIBANK"],
    "HINDUNILVR.NS":  ["HUL", "Hindustan Unilever", "Hindunilvr"],
    "ITC.NS":         ["ITC"],
    "KOTAKBANK.NS":   ["Kotak", "Kotak Mahindra"],
    "LT.NS":          ["L&T", "Larsen", "Larsen & Toubro"],
    "AXISBANK.NS":    ["Axis Bank", "AXISBANK"],
}


# ── Fetcher ───────────────────────────────────────────────────────────────────

class NewsFetcher:
    """
    Fetches, filters, and caches news articles from RSS feeds.

    Parameters
    ----------
    config : full config dict (uses config["llm"]["news_sources"] and cache_ttl)
    """

    def __init__(self, config: dict):
        self.sources     = config["llm"]["news_sources"]
        self.cache_ttl   = config["llm"]["cache_ttl_minutes"] * 60  # seconds
        self.max_articles = config["llm"]["max_articles_per_ticker"]
        self._cache: Dict[str, tuple] = {}  # url → (article, fetched_at)
        self._feed_cache: Dict[str, tuple] = {}  # feed_url → (articles, fetched_at)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_for_ticker(
        self,
        ticker: str,
        max_age_hours: int = 24,
    ) -> List[NewsArticle]:
        """
        Return up to `max_articles` recent articles relevant to `ticker`.

        Parameters
        ----------
        ticker       : NSE ticker, e.g. "RELIANCE.NS"
        max_age_hours: ignore articles older than this

        Returns
        -------
        List of NewsArticle, sorted by recency (newest first)
        """
        all_articles = self._fetch_all_feeds()
        keywords     = TICKER_COMPANY_MAP.get(ticker, [ticker.replace(".NS", "")])
        relevant     = self._filter_by_relevance(all_articles, keywords, ticker, max_age_hours)
        return relevant[:self.max_articles]

    def fetch_for_all_tickers(
        self,
        tickers: List[str],
        max_age_hours: int = 24,
    ) -> Dict[str, List[NewsArticle]]:
        """
        Batch fetch across all tickers. Shares the same underlying RSS pulls.
        """
        all_articles = self._fetch_all_feeds()
        result: Dict[str, List[NewsArticle]] = {}
        for ticker in tickers:
            keywords = TICKER_COMPANY_MAP.get(ticker, [ticker.replace(".NS", "")])
            relevant = self._filter_by_relevance(all_articles, keywords, ticker, max_age_hours)
            result[ticker] = relevant[:self.max_articles]
            logger.debug("Ticker %s: %d relevant articles", ticker, len(result[ticker]))
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_all_feeds(self) -> List[NewsArticle]:
        """
        Pull all configured RSS feeds, deduplicating across sources.
        Results are cached per-feed for `cache_ttl` seconds.
        """
        seen_uids: set = set()
        articles: List[NewsArticle] = []

        for source_cfg in self.sources:
            url  = source_cfg["url"]
            name = source_cfg["name"]
            feed_articles = self._fetch_feed(url, name)
            for a in feed_articles:
                if a.uid not in seen_uids:
                    seen_uids.add(a.uid)
                    articles.append(a)

        logger.info("Total unique articles fetched: %d", len(articles))
        return articles

    def _fetch_feed(self, feed_url: str, source_name: str) -> List[NewsArticle]:
        """Fetch a single RSS feed, using cache if fresh."""
        now = time.time()
        cached = self._feed_cache.get(feed_url)
        if cached is not None:
            articles, fetched_at = cached
            if now - fetched_at < self.cache_ttl:
                return articles

        try:
            feed = feedparser.parse(feed_url)
            articles = []
            for entry in feed.entries:
                article = self._parse_entry(entry, source_name)
                if article:
                    articles.append(article)
            self._feed_cache[feed_url] = (articles, now)
            logger.info("Fetched %d articles from %s", len(articles), source_name)
            return articles

        except Exception as exc:
            logger.warning("Failed to fetch feed %s: %s", feed_url, exc)
            return []

    def _parse_entry(self, entry, source_name: str) -> Optional[NewsArticle]:
        """Parse a feedparser entry into a NewsArticle."""
        try:
            url   = entry.get("link", "")
            title = entry.get("title", "").strip()

            # Extract summary — strip HTML tags
            raw_summary = (
                entry.get("summary", "")
                or entry.get("description", "")
                or ""
            )
            summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ").strip()

            # Parse publish date
            pub = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

            if not url or not title:
                return None

            return NewsArticle(
                url=url,
                title=title,
                summary=summary[:800],
                source=source_name,
                published_utc=pub,
            )
        except Exception as exc:
            logger.debug("Failed to parse feed entry: %s", exc)
            return None

    def _filter_by_relevance(
        self,
        articles: List[NewsArticle],
        keywords: List[str],
        ticker: str,
        max_age_hours: int,
    ) -> List[NewsArticle]:
        """Return articles that mention any keyword, within the age window."""
        now = datetime.now(timezone.utc)
        relevant = []
        for article in articles:
            # Age filter
            if article.published_utc:
                age_hours = (now - article.published_utc).total_seconds() / 3600
                if age_hours > max_age_hours:
                    continue

            # Keyword match (case-insensitive, title + summary)
            text = f"{article.title} {article.summary}".lower()
            if any(kw.lower() in text for kw in keywords):
                article.ticker_relevance.append(ticker)
                relevant.append(article)

        # Sort newest first
        relevant.sort(
            key=lambda a: a.published_utc or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return relevant
