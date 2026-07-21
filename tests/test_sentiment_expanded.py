"""Tests for expanded 10-axis sentiment encoder."""
import numpy as np
import pytest


def test_sentiment_axes_count():
    from llm.sentiment_encoder import SENTIMENT_AXES
    assert len(SENTIMENT_AXES) == 10


def test_local_model_axes_match():
    from llm.sentiment_encoder import SENTIMENT_AXES as cloud_axes
    from llm.local_model import SENTIMENT_AXES as local_axes
    assert cloud_axes == local_axes


def test_sentiment_vector_shape():
    from llm.sentiment_encoder import SentimentEncoder
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    tickers = config["market"]["tickers"]
    encoder = SentimentEncoder(config, tickers)
    zeros = encoder.encode_zeros()
    assert zeros.shape == (len(tickers) * 10,)
    assert zeros.dtype == np.float32


def test_scores_to_vector_all_axes():
    from llm.sentiment_encoder import SentimentEncoder, SENTIMENT_AXES
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    tickers = config["market"]["tickers"]
    encoder = SentimentEncoder(config, tickers)

    scores_map = {t: {axis: 0.5 for axis in SENTIMENT_AXES} for t in tickers}
    vec = encoder._scores_to_vector(scores_map)
    assert vec.shape == (len(tickers) * 10,)
    assert np.all(vec == 0.5)


def test_obs_space_dimension():
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    n_stocks = len(config["market"]["tickers"])
    sentiment_dim = config["llm"]["sentiment_dim"]
    assert sentiment_dim == 10

    from env.data_loader import NSEDataLoader
    loader = NSEDataLoader(config)
    n_features = len(loader.get_feature_names())
    alpha_dim = config.get("alpha", {}).get("obs_dim", 0)
    temporal_dim = config.get("patchtst", {}).get("obs_dim", 0)

    expected_obs = (
        n_stocks * n_features
        + n_stocks
        + 2
        + n_stocks
        + n_stocks * sentiment_dim
        + alpha_dim
        + temporal_dim
    )
    print(f"Expected obs_dim (without GNN/VAE/debate): {expected_obs}")
    assert expected_obs > 0


def test_config_sentiment_dim_consistent():
    import yaml
    from llm.sentiment_encoder import SENTIMENT_AXES
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    assert config["llm"]["sentiment_dim"] == len(SENTIMENT_AXES)
