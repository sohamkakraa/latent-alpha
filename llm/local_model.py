"""
llm/local_model.py
───────────────────
Ollama-based local LLM interface for sentiment encoding.

Runs Gemma 4 27B (or any Ollama model) entirely on-device using
Apple MPS / unified memory. No API cost. No data leaves your machine.

Why Gemma 4 for financial sentiment?
  - 27B parameters handles nuanced financial language well
  - Fits in ~15GB at Q4_K_M quantisation — well within 32GB unified memory
  - Instruction-tuned variant follows JSON output format reliably
  - Fast enough on M1 Pro for overnight offline precomputation

Setup (one-time)
----------------
  brew install ollama
  ollama pull gemma4:27b       # ~15GB download
  ollama serve                 # starts the local API server

Usage
-----
  client = OllamaClient()
  vector = client.get_sentiment(tickers, articles_by_ticker)
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np
import requests

from llm.verifier import VerifiedArticle

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

# Reuse the same axes and system prompt structure as sentiment_encoder.py
SENTIMENT_AXES = [
    # Original 5 axes
    "headline_sentiment",
    "earnings_signal",
    "macro_signal",
    "regulatory_risk",
    "volatility_expectation",
    # Phase 4: specialist analyst perspectives
    "fundamental_quality",
    "management_sentiment",
    "sector_momentum",
    "event_catalyst",
    "contrarian_indicator",
]

SYSTEM_PROMPT = """You are a quantitative financial analyst specialising in Indian equity markets.
Given news articles and NSE tickers, return a JSON sentiment assessment.

Rules:
1. Only score a ticker if the news directly mentions it or its sector.
2. If no relevant news, return 0.0 for all axes.
3. Do NOT hallucinate facts not in the provided news.
4. All scores must be floats in [-1.0, 1.0].
5. Return ONLY valid JSON — no explanation, no markdown.

Score axes:
- headline_sentiment:     -1 (bearish) to +1 (bullish)
- earnings_signal:        -1 (miss) to +1 (beat)
- macro_signal:           -1 (headwind) to +1 (tailwind)
- regulatory_risk:        -1 (high risk) to +1 (clarity/positive)
- volatility_expectation: -1 (calm) to +1 (volatile)
- fundamental_quality:    -1 (weak balance sheet) to +1 (strong fundamentals)
- management_sentiment:   -1 (negative guidance/insider selling) to +1 (positive guidance/buying)
- sector_momentum:        -1 (sector rotation out) to +1 (sector rotation in)
- event_catalyst:         -1 (negative catalyst) to +1 (positive catalyst: M&A/launch)
- contrarian_indicator:   -1 (extreme bullish consensus—fade) to +1 (extreme bearish consensus—buy)

Return format:
{"scores": {"TICKER.NS": {"headline_sentiment": 0.0, "earnings_signal": 0.0, "macro_signal": 0.0, "regulatory_risk": 0.0, "volatility_expectation": 0.0, "fundamental_quality": 0.0, "management_sentiment": 0.0, "sector_momentum": 0.0, "event_catalyst": 0.0, "contrarian_indicator": 0.0}}}"""


def _build_prompt(
    tickers: List[str],
    articles_by_ticker: Dict[str, List[VerifiedArticle]],
) -> str:
    lines = ["TICKERS: " + ", ".join(tickers), "", "NEWS:"]
    for ticker, articles in articles_by_ticker.items():
        if not articles:
            continue
        lines.append(f"\n[{ticker}]")
        for va in articles[:3]:
            lines.append(va.to_prompt_snippet())
    return "\n".join(lines)


class OllamaClient:
    """
    Thin wrapper around the Ollama REST API for local LLM inference.

    Parameters
    ----------
    model       : Ollama model tag, e.g. "gemma4:27b" or "gemma3:27b"
    base_url    : Ollama server URL (default: localhost:11434)
    timeout_sec : Request timeout in seconds (large models can be slow to start)
    """

    def __init__(
        self,
        model: str = "gemma4:latest",
        base_url: str = OLLAMA_BASE_URL,
        timeout_sec: int = 120,
    ):
        self.model   = model
        self.url     = f"{base_url}/api/generate"
        self.timeout = timeout_sec
        self._verify_connection()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_sentiment(
        self,
        tickers: List[str],
        articles_by_ticker: Dict[str, List[VerifiedArticle]],
    ) -> np.ndarray:
        """
        Encode news into a sentiment vector using the local model.

        Returns
        -------
        np.ndarray, shape (n_tickers * len(SENTIMENT_AXES),), dtype float32
        """
        has_articles = any(bool(v) for v in articles_by_ticker.values())
        if not has_articles:
            return self._zero_vector(len(tickers))

        prompt     = _build_prompt(tickers, articles_by_ticker)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        try:
            response = requests.post(
                self.url,
                json={
                    "model":  self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,   # near-deterministic for consistency
                        "num_predict": 512,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
            scores_map = self._parse_response(raw)
            return self._scores_to_vector(tickers, scores_map)

        except requests.exceptions.ConnectionError:
            logger.error(
                "Ollama not running. Start it with: ollama serve\n"
                "Then pull the model: ollama pull %s", self.model
            )
            return self._zero_vector(len(tickers))
        except Exception as exc:
            logger.error("Ollama inference failed: %s", exc)
            return self._zero_vector(len(tickers))

    def is_available(self) -> bool:
        """Return True if Ollama server is reachable and the model is loaded."""
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            tags = [m["name"] for m in r.json().get("models", [])]
            return any(self.model in t for t in tags)
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return list of locally available Ollama models."""
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # ── Private ───────────────────────────────────────────────────────────────

    def _verify_connection(self) -> None:
        try:
            requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        except Exception:
            logger.warning(
                "Ollama server not detected at %s. "
                "Start with: ollama serve", OLLAMA_BASE_URL
            )

    def _parse_response(self, raw: str) -> dict:
        """Extract JSON from model output, handling markdown fences."""
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            # Find the outermost JSON object
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return {}
            return json.loads(raw[start:end]).get("scores", {})
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse model JSON: %s | raw: %s", e, raw[:200])
            return {}

    def _scores_to_vector(
        self, tickers: List[str], scores_map: dict
    ) -> np.ndarray:
        n = len(tickers) * len(SENTIMENT_AXES)
        vector = np.zeros(n, dtype=np.float32)
        for i, ticker in enumerate(tickers):
            ticker_scores = scores_map.get(ticker, {})
            for j, axis in enumerate(SENTIMENT_AXES):
                score = float(ticker_scores.get(axis, 0.0))
                vector[i * len(SENTIMENT_AXES) + j] = float(np.clip(score, -1.0, 1.0))
        return vector

    def _zero_vector(self, n_tickers: int) -> np.ndarray:
        return np.zeros(n_tickers * len(SENTIMENT_AXES), dtype=np.float32)
