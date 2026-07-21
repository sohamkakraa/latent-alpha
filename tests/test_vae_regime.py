"""Tests for VAE market regime detector."""
import numpy as np
import pytest


def test_regime_signal_dataclass():
    """RegimeSignal has correct interface."""
    from alpha.vae_regime import RegimeSignal
    sig = RegimeSignal(
        embedding=np.random.randn(8).astype(np.float32),
        regime_probs=np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32),
        regime_label="bull_trending",
    )
    vec = sig.to_obs_vector()
    assert vec.shape == (8,)
    assert vec.dtype == np.float32


def test_regime_probs_sum_to_one():
    """Regime probabilities should sum to ~1.0."""
    torch = pytest.importorskip("torch")
    from alpha.vae_regime import RegimeVAE
    model = RegimeVAE(d_input=74, d_hidden=64, d_latent=8, n_regimes=4)
    x = torch.randn(1, 74)
    x_hat, mu, logvar, regime_probs = model(x)
    probs_sum = regime_probs.sum(dim=-1).item()
    assert abs(probs_sum - 1.0) < 1e-5


def test_vae_reconstruction_shape():
    """VAE reconstruction matches input shape."""
    torch = pytest.importorskip("torch")
    from alpha.vae_regime import RegimeVAE
    model = RegimeVAE(d_input=74, d_hidden=64, d_latent=8, n_regimes=4)
    x = torch.randn(32, 74)  # batch of 32
    x_hat, mu, logvar, regime_probs = model(x)
    assert x_hat.shape == (32, 74)
    assert mu.shape == (32, 8)
    assert logvar.shape == (32, 8)
    assert regime_probs.shape == (32, 4)


def test_vae_latent_sampling():
    """Reparameterisation trick: mu is deterministic in eval mode."""
    torch = pytest.importorskip("torch")
    from alpha.vae_regime import RegimeVAE
    model = RegimeVAE(d_input=74, d_hidden=64, d_latent=8, n_regimes=4)
    model.eval()
    x = torch.randn(1, 74)
    _, mu3, _, _ = model(x)
    _, mu4, _, _ = model(x)
    assert torch.allclose(mu3, mu4)


def test_build_market_state_shape():
    """Market state builder produces correct 74-dim vector."""
    from alpha.vae_regime import build_market_state
    import pandas as pd
    n_stocks = 10
    n_days = 100

    tickers = [f"STOCK{i}.NS" for i in range(n_stocks)]
    data = {}
    for t in tickers:
        idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
        df = pd.DataFrame({
            "Close": 100 + np.cumsum(np.random.randn(n_days) * 0.5),
            "High": 101 + np.cumsum(np.random.randn(n_days) * 0.5),
            "Low": 99 + np.cumsum(np.random.randn(n_days) * 0.5),
            "Volume": np.random.randint(1_000_000, 10_000_000, n_days),
        }, index=idx)
        # Add required features
        df["ret_1d"] = df["Close"].pct_change()
        df["ret_5d"] = df["Close"].pct_change(5)
        df["ret_20d"] = df["Close"].pct_change(20)
        df["vol_5d"] = df["ret_1d"].rolling(5).std()
        df["vol_20d"] = df["ret_1d"].rolling(20).std()
        df["rsi_14"] = 50 + np.random.randn(n_days) * 10
        df["macd_hist"] = np.random.randn(n_days) * 0.5
        data[t] = df.dropna()

    date = data[tickers[0]].index[-1]
    state = build_market_state(data, date, tickers)
    assert state.shape == (74,)  # 10 * 7 + 4
    assert state.dtype == np.float32


def test_generate_pseudo_labels():
    """Pseudo labels are valid regime indices."""
    from alpha.vae_regime import generate_pseudo_labels
    import pandas as pd
    n_stocks = 3
    n_days = 100
    tickers = [f"S{i}.NS" for i in range(n_stocks)]
    data = {}
    for t in tickers:
        idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
        df = pd.DataFrame({
            "Close": 100 + np.cumsum(np.random.randn(n_days) * 0.5),
        }, index=idx)
        df["ret_1d"] = df["Close"].pct_change()
        df["ret_20d"] = df["Close"].pct_change(20)
        df["vol_20d"] = df["ret_1d"].rolling(20).std()
        df["trend_strength"] = np.random.randn(n_days)
        data[t] = df.dropna()

    labels = generate_pseudo_labels(data, tickers)
    assert len(labels) > 0
    # Labels should be 0-3
    for date, label in labels.items():
        assert 0 <= label <= 3


def test_regime_detector_train_predict():
    """End-to-end train and predict with mock data."""
    torch = pytest.importorskip("torch")
    from alpha.vae_regime import RegimeDetector
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["vae_regime"]["enabled"] = True
    config["vae_regime"]["epochs"] = 2
    config["vae_regime"]["patience"] = 2

    detector = RegimeDetector(config)
    n_samples = 200
    np.random.seed(42)
    X = np.random.randn(n_samples, 74).astype(np.float32)
    labels = np.random.randint(0, 4, n_samples)

    metrics = detector.train(X, labels)
    assert "recon_loss" in metrics
    assert "kl_loss" in metrics

    signal = detector.predict(X[-1])
    assert signal.embedding.shape == (8,)
    assert signal.regime_probs.shape == (4,)
    assert signal.regime_label in ["bull_trending", "bear_trending", "high_vol_choppy", "low_vol_range"]


def test_config_vae_section():
    """Config has vae_regime section with required keys."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    vae = config.get("vae_regime", {})
    assert "enabled" in vae
    assert "obs_dim" in vae
    assert vae["obs_dim"] == 8
    assert "d_latent" in vae
    assert vae["d_latent"] == 8
    assert "beta_anneal_epochs" in vae
