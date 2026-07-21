"""Tests for post-hoc trade explainer."""
import numpy as np
import pytest


def test_explainer_disabled_is_noop():
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["llm"]["explainer"]["enabled"] = False

    from llm.explainer import TradeExplainer
    tickers = config["market"]["tickers"]
    explainer = TradeExplainer(config, tickers)
    result = explainer.explain(
        weight_deltas=np.zeros(len(tickers)),
        alpha_direction=np.zeros(len(tickers)),
        alpha_conviction=np.zeros(len(tickers)),
        portfolio_weights=np.zeros(len(tickers)),
        date="2024-01-15",
    )
    assert result is None


def test_explainer_filters_small_deltas():
    from llm.explainer import _filter_significant_trades
    tickers = ["A.NS", "B.NS", "C.NS"]
    deltas = np.array([0.05, 0.001, -0.03])
    significant = _filter_significant_trades(tickers, deltas, min_delta=0.02)
    assert "A.NS" in significant
    assert "B.NS" not in significant
    assert "C.NS" in significant


def test_explainer_prompt_structure():
    from llm.explainer import _build_explanation_prompt
    tickers = ["RELIANCE.NS"]
    deltas = {"RELIANCE.NS": 0.04}
    alpha_dirs = {"RELIANCE.NS": 0.7}
    alpha_convs = {"RELIANCE.NS": 0.85}
    weights = {"RELIANCE.NS": 0.15}

    prompt = _build_explanation_prompt(tickers, deltas, alpha_dirs, alpha_convs, weights, "2024-01-15")
    assert "RELIANCE.NS" in prompt
    assert any(s in prompt for s in ["0.04", "4%", "4.0%"])
    assert "bullish" in prompt.lower() or "0.7" in prompt


def test_explanation_dataclass():
    from llm.explainer import TradeExplanation
    exp = TradeExplanation(
        date="2024-01-15",
        explanations={"RELIANCE.NS": "Increased due to strong earnings signal"},
        llm_agrees={"RELIANCE.NS": True},
        confidence={"RELIANCE.NS": 0.8},
    )
    assert exp.date == "2024-01-15"
    assert "RELIANCE.NS" in exp.explanations
    assert exp.llm_agrees["RELIANCE.NS"] is True


def test_explainer_no_significant_trades_returns_none():
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["llm"]["explainer"]["enabled"] = True

    from llm.explainer import TradeExplainer
    tickers = config["market"]["tickers"]
    explainer = TradeExplainer(config, tickers)

    # All deltas below threshold — should return None without calling LLM
    tiny_deltas = np.full(len(tickers), 0.001, dtype=np.float32)
    result = explainer.explain(
        weight_deltas=tiny_deltas,
        alpha_direction=np.zeros(len(tickers)),
        alpha_conviction=np.zeros(len(tickers)),
        portfolio_weights=np.zeros(len(tickers)),
        date="2024-01-15",
    )
    assert result is None
