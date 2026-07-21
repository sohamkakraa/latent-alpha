"""
llm/explainer.py
────────────────
Post-hoc trade explanation using LLM reasoning.

After the SAC agent produces weight deltas, this module interprets
WHY the trade was made based on observable signals. This is purely
for auditability — it does not affect trading decisions.

Key inputs for explanation:
  - Alpha signal direction + conviction per stock
  - Current portfolio weights and proposed deltas
  - Recent sentiment scores (optional)

Output:
  - One-paragraph explanation per significant trade
  - Confidence assessment (does the LLM agree with the trade?)
  - Logged to the backtest trade info dict

Usage:
    explainer = TradeExplainer(config, tickers)
    explanation = explainer.explain(
        weight_deltas=action,
        alpha_direction=alpha_dirs,
        alpha_conviction=alpha_convs,
        portfolio_weights=current_weights,
        date=current_date,
    )
    if explanation:
        info["explanation"] = explanation.explanations
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


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TradeExplanation:
    """Container for LLM-generated trade explanations."""
    date: str
    explanations: Dict[str, str] = field(default_factory=dict)
    llm_agrees: Dict[str, bool] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_significant_trades(
    tickers: List[str],
    deltas: np.ndarray,
    min_delta: float,
) -> Dict[str, float]:
    """Return {ticker: delta} for stocks where |delta| > min_delta."""
    return {
        t: float(deltas[i])
        for i, t in enumerate(tickers)
        if abs(float(deltas[i])) > min_delta
    }


def _build_explanation_prompt(
    tickers: List[str],
    deltas: Dict[str, float],
    alpha_dirs: Dict[str, float],
    alpha_convs: Dict[str, float],
    weights: Dict[str, float],
    date: str,
) -> str:
    lines = [
        f"Date: {date}",
        "A portfolio manager has made the following significant trades.",
        "Explain WHY each trade was likely made given the signals below.",
        "Return your assessment as structured JSON.",
        "",
        "Trades (weight changes):",
    ]
    for t, d in deltas.items():
        direction_str = f"+{d:.1%}" if d >= 0 else f"{d:.1%}"
        lines.append(f"  {t}: {direction_str} (current weight: {weights.get(t, 0):.1%})")

    lines.extend(["", "Quantitative alpha signals:"])
    for t in tickers:
        if t in deltas:
            adir = alpha_dirs.get(t, 0.0)
            aconv = alpha_convs.get(t, 0.0)
            bias = "bullish" if adir > 0.1 else ("bearish" if adir < -0.1 else "neutral")
            lines.append(f"  {t}: alpha={adir:.2f} ({bias}), conviction={aconv:.2f}")

    lines.extend([
        "",
        "Rules:",
        "1. Write one concise sentence explanation per stock.",
        "2. State whether the LLM agrees the trade makes sense (true/false).",
        "3. Confidence: 0.0 = uncertain, 1.0 = highly confident in explanation.",
        "4. Return ONLY valid JSON, no preamble.",
        "",
        "Return format:",
        '{"explanations": {"TICKER.NS": {"text": "...", "agrees": true, "confidence": 0.8}, ...}}',
    ])
    return "\n".join(lines)


# ── Explainer ─────────────────────────────────────────────────────────────────

class TradeExplainer:
    """
    Post-hoc trade explanation using LLM reasoning.

    Only runs when `explainer.enabled: true` in config (disabled by default).
    Returns None immediately when disabled — zero overhead during training.

    Parameters
    ----------
    config  : full config dict
    tickers : ordered list of tickers
    """

    def __init__(self, config: dict, tickers: List[str]):
        self.config = config
        self.tickers = tickers
        self.n_stocks = len(tickers)
        self.explainer_cfg = config.get("llm", {}).get("explainer", {})
        self.enabled = self.explainer_cfg.get("enabled", False)
        self.min_delta = self.explainer_cfg.get("min_delta_to_explain", 0.02)
        self._backend = self.explainer_cfg.get("model", "local")

        if self._backend != "local" and _ANTHROPIC_AVAILABLE:
            self._client = anthropic.Anthropic()
        else:
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        weight_deltas: np.ndarray,
        alpha_direction: np.ndarray,
        alpha_conviction: np.ndarray,
        portfolio_weights: np.ndarray,
        date: str,
    ) -> Optional[TradeExplanation]:
        """
        Explain significant trades for a given step.

        Returns None when disabled or when no significant trades exist.
        """
        if not self.enabled:
            return None

        significant = _filter_significant_trades(self.tickers, weight_deltas, self.min_delta)
        if not significant:
            return None

        alpha_dirs = {t: float(alpha_direction[i]) for i, t in enumerate(self.tickers)}
        alpha_convs = {t: float(alpha_conviction[i]) for i, t in enumerate(self.tickers)}
        weights = {t: float(portfolio_weights[i]) for i, t in enumerate(self.tickers)}

        prompt = _build_explanation_prompt(
            self.tickers, significant, alpha_dirs, alpha_convs, weights, date
        )

        try:
            raw = self._call_llm(prompt)
            return self._parse_response(raw, date)
        except Exception as exc:
            logger.warning("TradeExplainer: LLM call failed (%s).", exc)
            return None

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
            max_tokens=1024,
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
                  "options": {"temperature": 0.1, "num_predict": 1024}},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, raw: str, date: str) -> TradeExplanation:
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return TradeExplanation(date=date)
            data = json.loads(raw[start:end])
            entries = data.get("explanations", {})
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("TradeExplainer: JSON parse failed: %s", e)
            return TradeExplanation(date=date)

        explanations = {}
        llm_agrees = {}
        confidence = {}
        for ticker, entry in entries.items():
            if isinstance(entry, dict):
                explanations[ticker] = str(entry.get("text", ""))
                llm_agrees[ticker] = bool(entry.get("agrees", True))
                confidence[ticker] = float(np.clip(float(entry.get("confidence", 0.5)), 0.0, 1.0))

        return TradeExplanation(
            date=date,
            explanations=explanations,
            llm_agrees=llm_agrees,
            confidence=confidence,
        )
