"""
llm/debate_analyst.py
─────────────────────
Structured bullish/bearish debate analysis per stock.

Inspired by TradingAgents' multi-agent debate mechanism, but implemented
as a single structured LLM call for efficiency. The LLM argues both sides
for each stock, then produces a net assessment score.

This runs ONLY during backtesting and live inference — not during SAC
training (where it returns zeros for speed). The debate signal gets
registered as a channel in the Ridge meta-model alongside LightGBM
and PatchTST signals.

Usage:
    analyst = DebateAnalyst(config, tickers)
    signal = analyst.analyze(alpha_directions, alpha_convictions, headlines, date)
    # signal.direction  → np.ndarray (n_stocks,)
    # signal.conviction → np.ndarray (n_stocks,)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class DebateSignal:
    """
    Output of the debate analyst for one date.

    direction  : net directional score per stock, range [-1, 1]
    conviction : how confident the debate outcome is, range [0, 1]
    reasoning  : optional human-readable text per ticker (not in obs)
    """
    direction: np.ndarray
    conviction: np.ndarray
    reasoning: Dict[str, str] = field(default_factory=dict)
    consensus_score: float = 0.0

    def to_obs_vector(self) -> np.ndarray:
        """Return [direction | conviction] clipped and concatenated, shape (2*n,)."""
        d = np.clip(self.direction, -3.0, 3.0).astype(np.float32)
        c = np.clip(self.conviction, 0.0, 1.0).astype(np.float32)
        return np.concatenate([d, c])


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_debate_prompt(
    tickers: List[str],
    alpha_directions: np.ndarray,
    alpha_convictions: np.ndarray,
    headlines: Dict[str, List[str]],
) -> str:
    lines = [
        "You are a structured debate analyst for Indian equity markets.",
        "For each stock below, argue BOTH the bullish and bearish case from the",
        "provided news headlines and quantitative signals, then give a net score.",
        "",
        "Quantitative signals (LightGBM alpha model output):",
    ]
    for i, t in enumerate(tickers):
        direction = float(alpha_directions[i]) if i < len(alpha_directions) else 0.0
        conviction = float(alpha_convictions[i]) if i < len(alpha_convictions) else 0.0
        lines.append(f"  {t}: direction={direction:.2f}, conviction={conviction:.2f}")

    lines.extend(["", "News headlines:"])
    for t in tickers:
        hl = headlines.get(t, [])
        if hl:
            lines.append(f"  {t}: " + " | ".join(str(h)[:120] for h in hl[:3]))

    lines.extend([
        "",
        "Rules:",
        "1. For each stock, write 1 sentence bullish case and 1 sentence bearish case.",
        "2. If both cases are strong, set conviction LOW (uncertain).",
        "3. If one side clearly dominates, set conviction HIGH.",
        "4. direction: -1.0 (very bearish) to +1.0 (very bullish)",
        "5. conviction: 0.0 (uncertain) to 1.0 (highly confident)",
        "6. Return ONLY valid JSON, no preamble.",
        "",
        "Return format:",
        '{"debate": {"TICKER.NS": {"bullish": "...", "bearish": "...", '
        '"direction": 0.0, "conviction": 0.0}, ...}}',
    ])
    return "\n".join(lines)


# ── Analyst ───────────────────────────────────────────────────────────────────

class DebateAnalyst:
    """
    Runs structured bullish/bearish debate analysis for a set of stocks.

    Parameters
    ----------
    config  : full config dict
    tickers : ordered list of tickers (must match env ticker order)
    """

    def __init__(self, config: dict, tickers: List[str]):
        self.config = config
        self.tickers = tickers
        self.n_stocks = len(tickers)
        self.debate_cfg = config.get("llm", {}).get("debate", {})
        self.enabled = self.debate_cfg.get("enabled", False)
        self._cache: Dict[str, DebateSignal] = {}

        model_choice = self.debate_cfg.get("model", "local")
        self._backend = model_choice  # "local" or "sonnet"
        self._max_tokens = self.debate_cfg.get("max_tokens", 2048)

        if model_choice != "local" and _ANTHROPIC_AVAILABLE:
            self._client = anthropic.Anthropic()
        else:
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        alpha_directions: np.ndarray,
        alpha_convictions: np.ndarray,
        headlines: Dict[str, List[str]],
        date: str,
    ) -> DebateSignal:
        """
        Run debate analysis for all tickers on a given date.

        Returns cached result if the same date was already analysed.
        Falls back to zero signal when LLM is unavailable.
        """
        cached = self._get_cached(date)
        if cached is not None:
            return cached

        if not self.enabled:
            return self.get_zero_signal()

        prompt = _build_debate_prompt(
            self.tickers, alpha_directions, alpha_convictions, headlines
        )

        try:
            raw = self._call_llm(prompt)
            signal = self._parse_response(raw)
        except Exception as exc:
            logger.warning("DebateAnalyst: LLM call failed (%s) — returning zeros.", exc)
            signal = self.get_zero_signal()

        self._cache[date] = signal
        return signal

    def get_zero_signal(self) -> DebateSignal:
        """Return a zero signal (used as fallback and during SAC training)."""
        return DebateSignal(
            direction=np.zeros(self.n_stocks, dtype=np.float32),
            conviction=np.zeros(self.n_stocks, dtype=np.float32),
        )

    def _get_cached(self, date: str) -> Optional[DebateSignal]:
        return self._cache.get(date)

    # ── LLM backends ─────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        if self._backend == "local":
            return self._call_ollama(prompt)
        else:
            return self._call_anthropic(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        if self._client is None:
            raise RuntimeError("Anthropic client not initialised.")
        resp = self._client.messages.create(
            model=self.config["llm"].get("live_model", "claude-sonnet-4-6"),
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def _call_ollama(self, prompt: str) -> str:
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError("requests not installed.")
        ollama_model = self.config["llm"].get("ollama_model", "gemma4:latest")
        resp = _requests.post(
            "http://localhost:11434/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": self._max_tokens}},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> DebateSignal:
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return self.get_zero_signal()
            data = json.loads(raw[start:end])
            debate = data.get("debate", {})
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("DebateAnalyst: failed to parse JSON: %s", e)
            return self.get_zero_signal()

        directions = np.zeros(self.n_stocks, dtype=np.float32)
        convictions = np.zeros(self.n_stocks, dtype=np.float32)
        reasoning = {}

        for i, ticker in enumerate(self.tickers):
            entry = debate.get(ticker, {})
            if entry:
                d = float(entry.get("direction", 0.0))
                c = float(entry.get("conviction", 0.0))
                directions[i] = float(np.clip(d, -1.0, 1.0))
                convictions[i] = float(np.clip(c, 0.0, 1.0))
                bull = entry.get("bullish", "")
                bear = entry.get("bearish", "")
                if bull or bear:
                    reasoning[ticker] = f"Bullish: {bull} | Bearish: {bear}"

        return DebateSignal(direction=directions, conviction=convictions, reasoning=reasoning)
