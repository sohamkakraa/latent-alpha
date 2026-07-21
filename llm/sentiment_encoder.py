"""
llm/sentiment_encoder.py
─────────────────────────
Converts verified news articles into a numeric sentiment vector that is
appended to the RL agent's observation at each environment step.

Architecture:
  - We call Claude with a structured prompt asking for a sentiment score
    per ticker on a scale of [-1, 1]
  - The output is a fixed-size vector of length (n_tickers × sentiment_dim)
    where sentiment_dim captures multiple axes: headline sentiment,
    earnings signal, macro signal, regulatory risk, volatility expectation
  - Results are cached by date to avoid redundant API calls during
    historical replay (backtesting)

Why not a fine-tuned financial sentiment model?
  - Claude has broad world knowledge and can reason about _why_ news is
    bullish/bearish, not just classify surface-level tone
  - Structured output via JSON mode gives a reliable vector format
  - Swappable: replace the Anthropic call with any LLM endpoint
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from llm.verifier import VerifiedArticle

logger = logging.getLogger(__name__)


# ── Sentiment dimensions ──────────────────────────────────────────────────────
# Each ticker gets a score on each of these axes.
SENTIMENT_AXES = [
    # Original 5 axes
    "headline_sentiment",       # Overall tone of the news: bearish → bullish
    "earnings_signal",          # Earnings / revenue related signal
    "macro_signal",             # Macro / sector-wide signal
    "regulatory_risk",          # Regulatory / compliance risk
    "volatility_expectation",   # Expected near-term volatility (low → high)
    # Phase 4: 5 specialist analyst perspectives
    "fundamental_quality",      # Balance sheet strength, cash flow, debt levels
    "management_sentiment",     # Management guidance tone, insider activity signals
    "sector_momentum",          # Sector rotation signal, peer comparison
    "event_catalyst",           # M&A, product launches, partnerships, restructuring
    "contrarian_indicator",     # Crowding signal: when consensus bullish, score negative
]
# sentiment_dim in config must match len(SENTIMENT_AXES)


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quantitative financial analyst specialising in Indian equity markets.
You will be given a set of recent news articles and a list of NSE-listed tickers.
Your task is to produce a structured sentiment assessment for each ticker based solely
on the provided news — not on your prior beliefs about the companies.

Rules:
1. Only score a ticker if the news directly mentions it or its sector.
2. If no relevant news exists for a ticker, return 0.0 for all axes.
3. Do NOT hallucinate earnings figures, ratings, or events not in the provided news.
4. Scores must be floats in the range [-1.0, 1.0].
5. Return ONLY valid JSON — no preamble, no explanation.

Score axes:
- headline_sentiment:     -1.0 (very bearish) to +1.0 (very bullish)
- earnings_signal:        -1.0 (miss/guidance cut) to +1.0 (beat/guidance raise)
- macro_signal:           -1.0 (macro headwind) to +1.0 (macro tailwind)
- regulatory_risk:        -1.0 (high regulatory risk) to +1.0 (regulatory clarity/positive)
- volatility_expectation: -1.0 (low volatility expected) to +1.0 (high volatility expected)
- fundamental_quality:    -1.0 (weak balance sheet/cash flow) to +1.0 (strong fundamentals)
- management_sentiment:   -1.0 (negative guidance/insider selling) to +1.0 (positive guidance/buying)
- sector_momentum:        -1.0 (sector rotation out) to +1.0 (sector rotation in / peer outperformance)
- event_catalyst:         -1.0 (negative catalyst: restructuring/loss) to +1.0 (positive: M&A/launch)
- contrarian_indicator:   -1.0 (extreme consensus bullish — fade) to +1.0 (extreme consensus bearish — buy)
"""

def _build_user_prompt(
    tickers: List[str],
    articles_by_ticker: Dict[str, List[VerifiedArticle]],
) -> str:
    lines = ["TICKERS TO SCORE: " + ", ".join(tickers), "", "NEWS ARTICLES:", ""]
    for ticker, articles in articles_by_ticker.items():
        if not articles:
            continue
        lines.append(f"--- {ticker} ---")
        for va in articles[:3]:  # cap at 3 per ticker for token efficiency
            lines.append(va.to_prompt_snippet())
            lines.append("")
    lines.append("")
    axes_example = ", ".join(f'"{a}": 0.0' for a in SENTIMENT_AXES)
    lines.append(
        "Return JSON in exactly this format:\n"
        '{"scores": {"TICKER.NS": {' + axes_example + '}, ...}}'
    )
    return "\n".join(lines)


# ── Encoder ───────────────────────────────────────────────────────────────────

class SentimentEncoder:
    """
    Converts news articles → numeric sentiment observation vector.

    Parameters
    ----------
    config  : full config dict
    tickers : ordered list of tickers (must match environment ticker order)
    """

    def __init__(self, config: dict, tickers: List[str]):
        self.config       = config
        self.tickers      = tickers
        self.n_tickers    = len(tickers)
        self.sentiment_dim = len(SENTIMENT_AXES)
        self.llm_cfg      = config["llm"]
        self.model        = self.llm_cfg.get("live_model", "claude-sonnet-4-6")
        self._cache: Dict[str, np.ndarray] = {}

        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic()
        else:
            self.client = None
            logger.warning(
                "anthropic package not installed — sentiment encoder will return zeros. "
                "Install with: pip install anthropic"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        articles_by_ticker: Dict[str, List[VerifiedArticle]],
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode news into a sentiment vector of shape (n_tickers × sentiment_dim,).

        Parameters
        ----------
        articles_by_ticker : {ticker: [VerifiedArticle, ...]}
        cache_key          : if provided, cache result under this key

        Returns
        -------
        np.ndarray, shape (n_tickers × sentiment_dim,), dtype float32
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        vector = self._call_llm(articles_by_ticker)

        if cache_key:
            self._cache[cache_key] = vector

        return vector

    def encode_zeros(self) -> np.ndarray:
        """Return a zero vector — used when no news is available."""
        return np.zeros(self.n_tickers * self.sentiment_dim, dtype=np.float32)

    def make_sentiment_fn(
        self,
        articles_by_ticker: Dict[str, List[VerifiedArticle]],
    ):
        """
        Returns a callable compatible with NSETradingEnv's `sentiment_fn` parameter.
        The returned function accepts a date and returns a flat sentiment vector.

        Pre-computes the vector once and returns it for any date query.
        Useful during live inference where you want to update sentiment daily.
        """
        vector = self.encode(articles_by_ticker)

        def sentiment_fn(step_date) -> np.ndarray:
            return vector

        return sentiment_fn

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(
        self, articles_by_ticker: Dict[str, List[VerifiedArticle]]
    ) -> np.ndarray:
        """Call the LLM and parse structured sentiment scores."""

        # Check for any non-empty articles
        has_any_articles = any(articles_by_ticker.values())
        if not has_any_articles or self.client is None:
            return self.encode_zeros()

        user_prompt = _build_user_prompt(self.tickers, articles_by_ticker)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_json = response.content[0].text.strip()
            scores_map = self._parse_response(raw_json)
            return self._scores_to_vector(scores_map)

        except Exception as exc:
            logger.error("LLM call failed: %s — returning zero vector.", exc)
            return self.encode_zeros()

    def _parse_response(self, raw: str) -> Dict[str, Dict[str, float]]:
        """Parse JSON response from the LLM."""
        try:
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return data.get("scores", {})
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM JSON response: %s", e)
            return {}

    def _scores_to_vector(
        self, scores_map: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
        """Convert the scores dict to a flat numpy array in ticker order."""
        vector = np.zeros(self.n_tickers * self.sentiment_dim, dtype=np.float32)

        for i, ticker in enumerate(self.tickers):
            ticker_scores = scores_map.get(ticker, {})
            for j, axis in enumerate(SENTIMENT_AXES):
                score = float(ticker_scores.get(axis, 0.0))
                score = float(np.clip(score, -1.0, 1.0))
                vector[i * self.sentiment_dim + j] = score

        return vector
