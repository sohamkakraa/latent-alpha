"""
alpha/adversarial_filter.py
────────────────────────────
Adversarial conviction filter — devil's advocate for high-conviction signals.

Before the SAC agent sees high-conviction alpha signals, this module runs
a quick LLM "devil's advocate" check: given a bullish case, what is the
strongest counter-argument? If the counter-conviction is high, the alpha
signal conviction is dampened.

Operates as a pass-through filter — only modifies conviction, never flips
direction. Disabled by default; only active during backtesting/live trading,
never during SAC training.

Conviction reduction formula:
    new_conviction = original * (1 - dampening * counter_conviction)
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Set, Tuple

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


# ── Helper functions ──────────────────────────────────────────────────────────

def _select_stocks_to_challenge(
    convictions: np.ndarray,
    threshold: float,
) -> List[int]:
    """Return indices of stocks with conviction above threshold."""
    return [i for i, c in enumerate(convictions) if float(c) > threshold]


def _apply_dampening(
    original_conviction: float,
    counter_conviction: float,
    dampening: float,
) -> float:
    """
    Reduce conviction based on counter-argument strength.

    new = original * (1 - dampening * counter_conviction)
    Always non-negative and never exceeds original.
    """
    reduced = original_conviction * (1.0 - dampening * counter_conviction)
    return float(max(reduced, 0.0))


def _build_devil_prompt(
    tickers: List[str],
    directions: np.ndarray,
    convictions: np.ndarray,
    indices_to_challenge: List[int],
    headlines: Optional[Dict[str, List[str]]] = None,
) -> str:
    lines = [
        "You are a sceptical risk manager reviewing high-conviction trading signals.",
        "For each stock below, provide the STRONGEST counter-argument (devil's advocate).",
        "Then rate your counter-conviction: how strongly does the bearish/bullish counter-case hold?",
        "",
        "Signals to challenge:",
    ]
    for idx in indices_to_challenge:
        t = tickers[idx]
        d = float(directions[idx])
        c = float(convictions[idx])
        bias = "BULLISH" if d > 0 else "BEARISH"
        lines.append(f"  {t}: {bias} direction={d:.2f}, model conviction={c:.2f}")
        if headlines and t in headlines:
            lines.append(f"    Context: " + " | ".join(str(h)[:100] for h in headlines[t][:2]))

    lines.extend([
        "",
        "Rules:",
        "1. Provide 1 sentence counter-argument per stock.",
        "2. counter_conviction: 0.0 = weak counter (signal likely valid), 1.0 = strong counter (signal questionable)",
        "3. Be realistic — not every signal needs challenging.",
        "4. Return ONLY valid JSON, no preamble.",
        "",
        "Return format:",
        '{"counter": {"TICKER.NS": {"argument": "...", "counter_conviction": 0.0}, ...}}',
    ])
    return "\n".join(lines)


# ── Filter ────────────────────────────────────────────────────────────────────

class AdversarialFilter:
    """
    Adversarial conviction filter for alpha signals.

    For stocks with conviction above threshold, queries an LLM for a
    devil's advocate counter-argument and reduces conviction proportionally.

    Parameters
    ----------
    config : full config dict
    tickers : ordered list of tickers (must match alpha signal order)
    """

    def __init__(self, config: dict, tickers: Optional[List[str]] = None):
        self.config = config
        self.tickers = tickers or config["market"]["tickers"]
        self.filter_cfg = config.get("alpha", {}).get("adversarial_filter", {})
        self.enabled = self.filter_cfg.get("enabled", False)
        self.conviction_threshold = self.filter_cfg.get("conviction_threshold", 0.7)
        self.counter_threshold = self.filter_cfg.get("counter_conviction_threshold", 0.7)
        self.dampening = self.filter_cfg.get("dampening", 0.3)
        self._backend = self.filter_cfg.get("model", "local")
        self._cache: Dict[Tuple[str, int], float] = {}

        # Support mock injection for tests
        self._mock_counter_convictions: Optional[Dict[int, float]] = None

        if self._backend != "local" and _ANTHROPIC_AVAILABLE:
            self._client = anthropic.Anthropic()
        else:
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    def filter(
        self,
        alpha_signal,
        date: str,
        headlines: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Apply adversarial filtering to an alpha signal.

        Returns the input unchanged when disabled.
        Only challenges stocks above conviction_threshold.
        Never flips direction — only reduces conviction.

        Parameters
        ----------
        alpha_signal : AlphaSignal (direction, conviction, raw_pred arrays)
        date         : ISO date string for caching
        headlines    : optional {ticker: [headline, ...]} for LLM context
        """
        if not self.enabled:
            return alpha_signal

        indices = _select_stocks_to_challenge(alpha_signal.conviction, self.conviction_threshold)
        if not indices:
            return alpha_signal

        new_conviction = alpha_signal.conviction.copy()

        for idx in indices:
            counter_conv = self._get_counter_conviction(
                alpha_signal, date, idx, indices, headlines
            )
            if counter_conv >= self.counter_threshold:
                new_conviction[idx] = _apply_dampening(
                    float(new_conviction[idx]), counter_conv, self.dampening
                )

        # Build a new signal with updated conviction (direction unchanged)
        import copy
        filtered = copy.copy(alpha_signal)
        filtered.conviction = new_conviction
        return filtered

    def _get_counter_conviction(
        self,
        alpha_signal,
        date: str,
        idx: int,
        all_indices: List[int],
        headlines: Optional[Dict],
    ) -> float:
        """Retrieve counter-conviction from cache or LLM."""
        cached = self._get_cached(date, idx)
        if cached is not None:
            return cached

        # Mock injection for tests
        if self._mock_counter_convictions is not None:
            val = float(self._mock_counter_convictions.get(idx, 0.0))
            self._cache[(date, idx)] = val
            return val

        # Call LLM for all uncached indices in one batch call
        uncached = [i for i in all_indices if self._get_cached(date, i) is None]
        if uncached:
            self._batch_query(alpha_signal, date, uncached, headlines)

        return self._get_cached(date, idx) or 0.0

    def _batch_query(
        self,
        alpha_signal,
        date: str,
        indices: List[int],
        headlines: Optional[Dict],
    ) -> None:
        """Query LLM for counter-convictions for all indices in one call."""
        prompt = _build_devil_prompt(
            self.tickers,
            alpha_signal.direction,
            alpha_signal.conviction,
            indices,
            headlines,
        )
        try:
            raw = self._call_llm(prompt)
            results = self._parse_response(raw)
            for idx in indices:
                ticker = self.tickers[idx]
                counter = results.get(ticker, 0.0)
                self._cache[(date, idx)] = float(np.clip(counter, 0.0, 1.0))
        except Exception as exc:
            logger.warning("AdversarialFilter: LLM call failed (%s) — skipping filter.", exc)
            for idx in indices:
                self._cache[(date, idx)] = 0.0

    def _get_cached(self, date: str, idx: int) -> Optional[float]:
        return self._cache.get((date, idx))

    # ── LLM backends ─────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        if self._backend == "local":
            return self._call_ollama(prompt)
        return self._call_anthropic(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        if self._client is None:
            raise RuntimeError("Anthropic client not initialised.")
        resp = self._client.messages.create(
            model=self.config["llm"].get("live_model", "claude-sonnet-4-6"),
            max_tokens=512,
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
                  "options": {"temperature": 0.1, "num_predict": 512}},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _parse_response(self, raw: str) -> Dict[str, float]:
        """Parse LLM JSON → {ticker: counter_conviction}."""
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return {}
            data = json.loads(raw[start:end])
            counter_data = data.get("counter", {})
            return {t: float(v.get("counter_conviction", 0.0)) for t, v in counter_data.items() if isinstance(v, dict)}
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("AdversarialFilter: JSON parse failed: %s", e)
            return {}
