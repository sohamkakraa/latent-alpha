"""Tests for versioned, fail-fast sentiment cache use."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from llm.sentiment_cache import CacheValidationError, SentimentCache
from strategy.term_selector import TermSelector


def test_validated_cache_loads_into_memory(tmp_path):
    cache = SentimentCache(tmp_path / "sentiment.db")
    tickers = ["A", "B"]
    axes = ["tone", "risk"]
    dates = ["2024-01-01", "2024-01-02"]
    for index, date in enumerate(dates, 1):
        cache.put(
            date,
            np.full(4, index / 10, dtype=np.float32),
            model="test-v2",
            schema_version=2,
            tickers=tickers,
            axes=axes,
            model_version="test-v2",
        )

    coverage = cache.validate_coverage(
        dates,
        model="test-v2",
        vector_dim=4,
        schema_version=2,
    )
    assert coverage["nonzero_date_rate"] == 1.0

    lookup = cache.make_lookup_fn(
        vector_dim=4,
        model="test-v2",
        schema_version=2,
        training=False,
    )
    np.testing.assert_allclose(
        lookup(pd.Timestamp("2024-01-02")),
        np.full(4, 0.2, dtype=np.float32),
    )


def test_cache_rejects_wrong_shape_and_degenerate_signal(tmp_path):
    cache = SentimentCache(tmp_path / "sentiment.db")
    cache.put(
        "2024-01-01",
        np.zeros(2, dtype=np.float32),
        model="bad-v1",
        schema_version=1,
    )

    with pytest.raises(CacheValidationError, match="dim=2"):
        cache.validate_coverage(
            ["2024-01-01"],
            model="bad-v1",
            vector_dim=4,
            schema_version=2,
        )

    cache.put(
        "2024-01-01",
        np.zeros(4, dtype=np.float32),
        model="zero-v2",
        schema_version=2,
    )
    with pytest.raises(CacheValidationError, match="degenerate"):
        cache.validate_coverage(
            ["2024-01-01"],
            model="zero-v2",
            vector_dim=4,
            schema_version=2,
        )


def test_lookup_requires_explicit_model(tmp_path):
    cache = SentimentCache(tmp_path / "sentiment.db")
    with pytest.raises(ValueError, match="explicit model"):
        cache.make_lookup_fn(vector_dim=4)


def test_term_selector_keeps_train_and_eval_sentiment_separate():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    for section in ("alpha", "patchtst", "gnn", "vae_regime", "meta_model"):
        config[section]["enabled"] = False
    config["llm"]["sentiment_dim"] = 1
    config["llm"]["debate"]["enabled"] = False
    config["backtest"]["training_warmup_days"] = 0

    feature_names = TermSelector(config).loader.get_feature_names()
    dates = pd.bdate_range("2020-01-01", periods=100)
    data = {}
    for ticker in config["market"]["tickers"]:
        data[ticker] = pd.DataFrame(
            {
                "Close": np.linspace(100.0, 110.0, len(dates)),
                **{
                    feature: np.linspace(-1.0, 1.0, len(dates))
                    for feature in feature_names
                },
            },
            index=dates,
        )

    selector = TermSelector(
        config,
        data_cache={"2020-01-01_2021-01-01_medium": data},
    )
    train_fn = lambda date: np.ones(10, dtype=np.float32)
    eval_fn = lambda date: np.full(10, 2.0, dtype=np.float32)
    train_env, eval_env = selector.build_env(
        risk_profile="balanced",
        term="medium",
        train_start="2020-01-01",
        train_end="2021-01-01",
        sentiment_fn=train_fn,
        sentiment_fn_eval=eval_fn,
        normalise_obs=False,
    )

    assert train_env.envs[0].sentiment_fn is train_fn
    assert eval_env.envs[0].sentiment_fn is eval_fn
