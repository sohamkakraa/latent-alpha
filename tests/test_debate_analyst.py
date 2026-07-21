"""Tests for debate analyst module."""
import numpy as np
import pytest


def test_debate_signal_dataclass():
    from llm.debate_analyst import DebateSignal
    sig = DebateSignal(
        direction=np.array([0.5, -0.3, 0.1], dtype=np.float32),
        conviction=np.array([0.8, 0.6, 0.4], dtype=np.float32),
        reasoning={"STOCK1": "bullish due to ...", "STOCK2": "bearish because ..."},
    )
    vec = sig.to_obs_vector()
    assert vec.shape == (6,)
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec[:3], sig.direction)
    np.testing.assert_array_equal(vec[3:], sig.conviction)


def test_debate_analyst_fallback():
    from llm.debate_analyst import DebateAnalyst
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    tickers = config["market"]["tickers"]
    analyst = DebateAnalyst(config, tickers)

    signal = analyst.get_zero_signal()
    assert signal.direction.shape == (len(tickers),)
    assert signal.conviction.shape == (len(tickers),)
    assert np.all(signal.direction == 0.0)
    assert np.all(signal.conviction == 0.0)


def test_debate_cache():
    from llm.debate_analyst import DebateAnalyst
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    tickers = config["market"]["tickers"]
    analyst = DebateAnalyst(config, tickers)

    mock_signal = analyst.get_zero_signal()
    analyst._cache["2024-01-15"] = mock_signal

    cached = analyst._get_cached("2024-01-15")
    assert cached is not None
    np.testing.assert_array_equal(cached.direction, mock_signal.direction)


def test_debate_prompt_structure():
    from llm.debate_analyst import _build_debate_prompt
    tickers = ["RELIANCE.NS", "TCS.NS"]
    alpha_directions = np.array([0.5, -0.3])
    alpha_convictions = np.array([0.8, 0.6])
    headlines = {"RELIANCE.NS": ["Q3 beat estimates"], "TCS.NS": ["Weak guidance"]}

    prompt = _build_debate_prompt(tickers, alpha_directions, alpha_convictions, headlines)
    assert "RELIANCE.NS" in prompt
    assert "bullish" in prompt.lower()
    assert "bearish" in prompt.lower()
    assert "0.5" in prompt
    assert "0.8" in prompt


def test_env_debate_dim():
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    debate_dim = config.get("llm", {}).get("debate", {}).get("obs_dim", 0)
    assert debate_dim == 20


def test_debate_signal_obs_vector_clipping():
    from llm.debate_analyst import DebateSignal
    sig = DebateSignal(
        direction=np.array([5.0, -5.0], dtype=np.float32),
        conviction=np.array([1.5, -0.5], dtype=np.float32),
    )
    vec = sig.to_obs_vector()
    assert np.all(vec[:2] >= -3.0) and np.all(vec[:2] <= 3.0)
    assert np.all(vec[2:] >= 0.0) and np.all(vec[2:] <= 1.0)
