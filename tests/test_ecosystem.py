"""
tests/test_ecosystem.py
═══════════════════════
Full ecosystem smoke test — validates ALL phases work together end-to-end.

Tests:
  1. All imports succeed (no circular deps, no missing modules)
  2. Config loads and exp B overrides are present
  3. Observation space dimensions match expected (560 with all phases)
  4. Each signal channel produces correct-shape output
  5. Trading env can reset() and step() with all phases wired in
  6. Reward function works with alpha-aware bonus
  7. Meta-model can stack all channel predictions
  8. Short SAC training loop completes without crash (100 steps)
  9. Walk-forward backtester initialises with all phases
 10. End-to-end: build_env → step loop with full obs validation

Run:
  cd latent_alpha
  python -m pytest tests/test_ecosystem.py -v --tb=short
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Fix macOS libomp conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Fixtures ─────────────────────────────────────────────────────────────────

N_STOCKS = 10
N_FEATURES = 35  # 15 base + 20 extended
SENTIMENT_DIM = 10


def _make_config():
    """Return a config dict with exp B overrides and all phases enabled."""
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    return config


def _make_synthetic_data(n_days=300, n_stocks=N_STOCKS):
    """Generate synthetic OHLCV + feature data for testing."""
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
    ][:n_stocks]

    # Feature columns matching data_loader output
    base_features = [
        "log_return", "return_5d", "return_20d", "volatility_20",
        "rsi", "macd", "macd_signal", "macd_diff",
        "bb_width", "bb_pct", "atr", "volume_z",
        "ema_9", "ema_21", "ema_50",
    ]
    extended_features = [
        "ret_2d", "ret_10d", "ret_60d", "roc_5", "roc_20",
        "williams_r", "stoch_k", "stoch_d",
        "vol_ratio_5_20", "parkinson_vol",
        "obv_slope", "vwap_dist", "volume_breakout",
        "adx_14", "cci_20", "trend_strength",
        "rsi_5", "dist_from_52w_high",
        "skewness_20", "kurtosis_20",
    ]
    all_features = base_features + extended_features
    assert len(all_features) == N_FEATURES

    # Alpha feature columns (used by GNN/VAE pipelines)
    alpha_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_5d", "vol_20d", "volume_z_20"]

    data = {}
    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**31)
        price = 1000 + np.cumsum(np.random.randn(n_days) * 5)
        price = np.maximum(price, 100)  # floor

        df = pd.DataFrame(index=dates)
        df["Open"] = price * (1 + np.random.randn(n_days) * 0.002)
        df["High"] = price * (1 + np.abs(np.random.randn(n_days)) * 0.01)
        df["Low"] = price * (1 - np.abs(np.random.randn(n_days)) * 0.01)
        df["Close"] = price
        df["Volume"] = np.random.randint(100000, 10000000, n_days).astype(float)

        # Generate feature columns with realistic ranges
        for feat in all_features:
            if "return" in feat or "ret" in feat or "log_return" in feat:
                df[feat] = np.random.randn(n_days) * 0.02
            elif "rsi" in feat:
                df[feat] = np.random.uniform(20, 80, n_days)
            elif "vol" in feat:
                df[feat] = np.abs(np.random.randn(n_days) * 0.02)
            else:
                df[feat] = np.random.randn(n_days) * 0.5

        # Alpha features for GNN/VAE
        for col in alpha_cols:
            if col not in df.columns:
                df[col] = np.random.randn(n_days) * 0.02

        data[ticker] = df

    return data, tickers, all_features


# ── Test 1: All imports ──────────────────────────────────────────────────────

class TestImports:
    def test_core_imports(self):
        """All core modules import without error."""
        from env.trading_env import NSETradingEnv
        from agent.reward import RiskAwareReward
        from strategy.term_selector import TermSelector

    def test_alpha_imports(self):
        """Phase 1: Alpha layer imports."""
        from alpha.feature_engineering import AlphaFeatureEngine
        from alpha.lightgbm_alpha import LightGBMAlpha
        from alpha.meta_model import RidgeMetaModel, ChannelPrediction

    def test_patchtst_imports(self):
        """Phase 2: PatchTST imports."""
        from alpha.patchtst import PatchTSTAlpha, TemporalSignal
        from alpha.temporal_features import TemporalFeatureBuilder

    def test_gnn_imports(self):
        """Phase 3: GNN imports."""
        from alpha.gnn_alpha import GNNAlpha, GraphSignal

    def test_vae_imports(self):
        """Phase 3: VAE imports."""
        from alpha.vae_regime import RegimeDetector, RegimeSignal, build_market_state, generate_pseudo_labels

    def test_phase4_imports(self):
        """Phase 4: LLM features import."""
        from llm.sentiment_encoder import SentimentEncoder
        from llm.debate_analyst import DebateAnalyst, DebateSignal
        from llm.explainer import TradeExplainer
        from alpha.adversarial_filter import AdversarialFilter


# ── Test 2: Config validation ────────────────────────────────────────────────

class TestConfig:
    def test_config_loads(self):
        config = _make_config()
        assert "market" in config
        assert "sac" in config

    def test_exp_b_overrides_applied(self):
        """Experiment B overrides are baked into base config."""
        config = _make_config()
        assert config["alpha"]["forward_horizon"] == 10, "forward_horizon should be 10 (exp B)"
        assert config["alpha"]["conviction_gate"] == 0.3, "conviction_gate should be 0.3 (exp B)"
        assert config["env"]["turnover_penalty"] == 50.0, "turnover_penalty should be 50 (exp B)"
        assert config["patchtst"]["forward_horizon"] == 10, "patchtst forward_horizon should be 10"

    def test_all_phases_enabled(self):
        config = _make_config()
        assert config["alpha"]["enabled"] is True
        assert config["patchtst"]["enabled"] is True
        assert config["gnn"]["enabled"] is True
        assert config["vae_regime"]["enabled"] is True
        assert config["llm"]["debate"]["enabled"] is True

    def test_obs_dims_consistent(self):
        """Obs dims in config sum to expected total."""
        config = _make_config()
        n_stocks = len(config["market"]["tickers"])
        sentiment_dim = config["llm"]["sentiment_dim"]

        expected = (
            n_stocks * N_FEATURES          # market
            + n_stocks + 2                  # portfolio
            + n_stocks                      # locks
            + n_stocks * sentiment_dim      # sentiment
            + config["alpha"]["obs_dim"]    # lightgbm
            + config["patchtst"]["obs_dim"] # patchtst
            + config["gnn"]["obs_dim"]      # gnn
            + config["vae_regime"]["obs_dim"]  # vae
            + config["llm"]["debate"]["obs_dim"]  # debate
        )
        assert expected == 560, f"Expected 560-dim obs, got {expected}"


# ── Test 3: Signal channel shapes ────────────────────────────────────────────

class TestSignalShapes:
    def test_alpha_signal_shape(self):
        """LightGBM AlphaSignal produces correct obs vector."""
        from alpha.lightgbm_alpha import AlphaSignal
        sig = AlphaSignal(
            direction=np.random.randn(N_STOCKS).astype(np.float32),
            conviction=np.random.uniform(0, 1, N_STOCKS).astype(np.float32),
        )
        obs = sig.to_obs_vector()
        assert obs.shape == (20,), f"AlphaSignal obs should be (20,), got {obs.shape}"

    def test_temporal_signal_shape(self):
        """PatchTST TemporalSignal produces correct obs vector."""
        from alpha.patchtst import TemporalSignal
        sig = TemporalSignal(
            direction=np.random.randn(N_STOCKS).astype(np.float32),
            conviction=np.random.uniform(0, 1, N_STOCKS).astype(np.float32),
            embedding=np.random.randn(N_STOCKS, 16).astype(np.float32),
        )
        obs = sig.to_obs_vector()
        # obs_dim is configured to 20 (direction + conviction only, no embedding in SAC obs)
        assert obs.shape == (20,), f"TemporalSignal obs should be (20,), got {obs.shape}"

    def test_gnn_signal_shape(self):
        """GNN GraphSignal produces correct obs vector."""
        from alpha.gnn_alpha import GraphSignal
        sig = GraphSignal(
            direction=np.random.randn(N_STOCKS).astype(np.float32),
            conviction=np.random.uniform(0, 1, N_STOCKS).astype(np.float32),
            attention_weights=np.random.rand(N_STOCKS, N_STOCKS).astype(np.float32),
        )
        obs = sig.to_obs_vector()
        assert obs.shape == (20,), f"GraphSignal obs should be (20,), got {obs.shape}"

    def test_regime_signal_shape(self):
        """VAE RegimeSignal produces correct obs vector."""
        from alpha.vae_regime import RegimeSignal
        sig = RegimeSignal(
            embedding=np.random.randn(8).astype(np.float32),
            regime_probs=np.array([0.1, 0.6, 0.2, 0.1], dtype=np.float32),
            regime_label="bull_trending",
        )
        obs = sig.to_obs_vector()
        assert obs.shape == (8,), f"RegimeSignal obs should be (8,), got {obs.shape}"

    def test_debate_signal_shape(self):
        """Debate DebateSignal produces correct obs vector."""
        from llm.debate_analyst import DebateSignal
        sig = DebateSignal(
            direction=np.random.randn(N_STOCKS).astype(np.float32),
            conviction=np.random.uniform(0, 1, N_STOCKS).astype(np.float32),
            consensus_score=0.7,
            reasoning={},
        )
        obs = sig.to_obs_vector()
        assert obs.shape == (20,), f"DebateSignal obs should be (20,), got {obs.shape}"


# ── Test 4: Trading env with all phases ──────────────────────────────────────

class TestTradingEnvFullPipeline:
    @pytest.fixture
    def env_setup(self):
        """Create a trading env with all signal functions mocked."""
        from env.trading_env import NSETradingEnv
        from alpha.lightgbm_alpha import AlphaSignal
        from alpha.patchtst import TemporalSignal
        from alpha.gnn_alpha import GraphSignal
        from alpha.vae_regime import RegimeSignal
        from llm.debate_analyst import DebateSignal

        config = _make_config()
        data, tickers, features = _make_synthetic_data()

        # Mock signal functions
        def alpha_fn(date):
            return AlphaSignal(
                direction=np.random.randn(N_STOCKS).astype(np.float32) * 0.5,
                conviction=np.random.uniform(0.1, 0.9, N_STOCKS).astype(np.float32),
            )

        def temporal_fn(date):
            return TemporalSignal(
                direction=np.random.randn(N_STOCKS).astype(np.float32) * 0.3,
                conviction=np.random.uniform(0.2, 0.8, N_STOCKS).astype(np.float32),
                embedding=np.random.randn(N_STOCKS, 16).astype(np.float32),
            )

        def gnn_fn(date):
            return GraphSignal(
                direction=np.random.randn(N_STOCKS).astype(np.float32) * 0.4,
                conviction=np.random.uniform(0.1, 0.7, N_STOCKS).astype(np.float32),
                attention_weights=np.random.rand(N_STOCKS, N_STOCKS).astype(np.float32),
            )

        def regime_fn(date):
            return RegimeSignal(
                embedding=np.random.randn(8).astype(np.float32),
                regime_probs=np.random.dirichlet([1, 1, 1, 1]).astype(np.float32),
                regime_label="bull_trending",
            )

        def debate_fn(date):
            return DebateSignal(
                direction=np.random.randn(N_STOCKS).astype(np.float32) * 0.2,
                conviction=np.random.uniform(0.3, 0.6, N_STOCKS).astype(np.float32),
                consensus_score=0.65,
                reasoning={},
            )

        def sentiment_fn(date):
            return np.random.randn(N_STOCKS * SENTIMENT_DIM).astype(np.float32) * 0.1

        env = NSETradingEnv(
            data=data,
            feature_names=features,
            config=config,
            risk_profile="balanced",
            term="medium",
            sentiment_fn=sentiment_fn,
            alpha_fn=alpha_fn,
            temporal_fn=temporal_fn,
            gnn_fn=gnn_fn,
            regime_fn=regime_fn,
            debate_fn=debate_fn,
        )
        return env

    def test_obs_space_dimension(self, env_setup):
        """Observation space matches expected 560 dims."""
        env = env_setup
        assert env.observation_space.shape == (560,), \
            f"Obs space should be (560,), got {env.observation_space.shape}"

    def test_action_space_dimension(self, env_setup):
        env = env_setup
        assert env.action_space.shape == (N_STOCKS,)

    def test_reset_returns_valid_obs(self, env_setup):
        env = env_setup
        obs, info = env.reset()
        assert obs.shape == (560,), f"Reset obs shape: {obs.shape}"
        assert not np.any(np.isnan(obs)), "Obs contains NaN after reset"
        assert np.all(np.abs(obs) <= 10.0), "Obs not clipped to [-10, 10]"

    def test_step_returns_valid(self, env_setup):
        env = env_setup
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, done, truncated, info = env.step(action)

        assert obs2.shape == (560,), f"Step obs shape: {obs2.shape}"
        assert not np.any(np.isnan(obs2)), "Obs contains NaN after step"
        assert isinstance(reward, float), "Reward should be float"
        assert "portfolio_value" in info
        assert "weights" in info

    def test_multi_step_episode(self, env_setup):
        """Run 50 steps without crash."""
        env = env_setup
        obs, _ = env.reset()
        total_reward = 0.0

        for i in range(50):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        assert info["portfolio_value"] > 0, "Portfolio value should be positive"

    def test_conviction_gate_applied(self, env_setup):
        """Conviction gate should zero out low-conviction signals."""
        config = _make_config()
        gate = config["alpha"].get("conviction_gate", 0)
        assert gate == 0.3, f"Conviction gate should be 0.3, got {gate}"
        # The gate is applied in term_selector's alpha_fn closure, not in the env directly.
        # This test just validates the config is set correctly.


# ── Test 5: Reward function with alpha bonus ─────────────────────────────────

class TestRewardWithAlpha:
    def test_reward_computes_with_alpha(self):
        """Reward function handles alpha_signal parameter."""
        from agent.reward import RiskAwareReward
        from alpha.lightgbm_alpha import AlphaSignal

        config = _make_config()
        # Enable alpha bonus for this test
        config["env"]["alpha_reward_bonus"] = 0.05
        config["env"]["alpha_conviction_threshold"] = 0.5

        reward_fn = RiskAwareReward(config, "balanced")
        reward_fn.reset()

        # First step to set prev_weights
        r1 = reward_fn.compute(
            step_return=0.01,
            portfolio_values=[1000000, 1010000],
            transaction_cost=50.0,
            current_weights=np.array([0.1, 0.05, -0.02, 0, 0, 0, 0, 0, 0, 0]),
        )

        alpha_sig = AlphaSignal(
            direction=np.array([1, 1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            conviction=np.array([0.8, 0.3, 0.9, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        )

        r2 = reward_fn.compute(
            step_return=0.005,
            portfolio_values=[1000000, 1010000, 1015000],
            transaction_cost=30.0,
            current_weights=np.array([0.15, 0.08, -0.05, 0, 0, 0, 0, 0, 0, 0]),
            alpha_signal=alpha_sig,
        )

        assert isinstance(r2, float)
        assert not np.isnan(r2)


# ── Test 6: Meta-model with all channels ─────────────────────────────────────

class TestMetaModelAllChannels:
    def test_register_all_channels(self):
        """Meta-model can register all 5 channel types."""
        from alpha.meta_model import RidgeMetaModel, ChannelPrediction

        config = _make_config()
        meta = RidgeMetaModel(config)
        meta.register_channel("lightgbm")
        meta.register_channel("patchtst")
        meta.register_channel("gnn")
        meta.register_channel("vae_regime")
        meta.register_channel("debate")

        assert len(meta.get_channels()) == 5

    def test_combine_all_channels(self):
        """Meta-model combines predictions from all channels."""
        from alpha.meta_model import RidgeMetaModel, ChannelPrediction

        config = _make_config()
        meta = RidgeMetaModel(config)

        channels = ["lightgbm", "patchtst", "gnn", "vae_regime", "debate"]
        for ch in channels:
            meta.register_channel(ch)

        preds = [
            ChannelPrediction(
                name=ch,
                direction=np.random.randn(N_STOCKS).astype(np.float32),
                conviction=np.random.uniform(0, 1, N_STOCKS).astype(np.float32),
            )
            for ch in channels
        ]

        combined_dir, combined_conv = meta.combine(preds)
        assert combined_dir.shape == (N_STOCKS,)
        assert combined_conv.shape == (N_STOCKS,)
        assert np.all(np.abs(combined_dir) <= 3.0), "Direction should be clipped to [-3, 3]"
        assert np.all(combined_conv >= 0) and np.all(combined_conv <= 1.0)


# ── Test 7: GNN standalone ───────────────────────────────────────────────────

class TestGNNStandalone:
    def test_gnn_train_and_predict(self):
        """GNN trains on synthetic data and produces GraphSignal."""
        from alpha.gnn_alpha import GNNAlpha

        config = _make_config()
        model = GNNAlpha(config)

        n_samples = 100
        X = np.random.randn(n_samples, N_STOCKS, 10).astype(np.float32)
        returns = np.random.randn(n_samples, N_STOCKS).astype(np.float32) * 0.02
        y = np.random.randn(n_samples, N_STOCKS).astype(np.float32) * 0.02

        metrics = model.train(X, returns, y)
        assert "train_loss" in metrics or "val_loss" in metrics

        # Predict
        node_feats = np.random.randn(N_STOCKS, 10).astype(np.float32)
        returns_window = np.random.randn(60, N_STOCKS).astype(np.float32) * 0.02
        signal = model.predict(node_feats, returns_window)

        assert signal.direction.shape == (N_STOCKS,)
        assert signal.conviction.shape == (N_STOCKS,)


# ── Test 8: VAE standalone ───────────────────────────────────────────────────

class TestVAEStandalone:
    def test_vae_train_and_predict(self):
        """VAE trains on synthetic market states and classifies regimes."""
        from alpha.vae_regime import RegimeDetector

        config = _make_config()
        detector = RegimeDetector(config)

        n_samples = 200
        d_input = config["vae_regime"]["d_input"]  # 74
        X = np.random.randn(n_samples, d_input).astype(np.float32)
        labels = np.random.randint(0, 4, n_samples).astype(np.int64)

        metrics = detector.train(X, labels)
        assert "recon_loss" in metrics or "best_epoch" in metrics

        # Predict
        state = np.random.randn(d_input).astype(np.float32)
        signal = detector.predict(state)
        assert signal.regime_label in ["bull_trending", "bear_trending", "high_vol_choppy", "low_vol_range"]
        assert signal.embedding.shape == (8,)


# ── Test 9: SAC short training loop ─────────────────────────────────────────

class TestSACSmoke:
    def test_sac_trains_with_full_obs(self):
        """SAC can train for 100 steps on the full 560-dim obs space."""
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
        from env.trading_env import NSETradingEnv
        from alpha.lightgbm_alpha import AlphaSignal
        from alpha.patchtst import TemporalSignal
        from alpha.gnn_alpha import GraphSignal
        from alpha.vae_regime import RegimeSignal
        from llm.debate_analyst import DebateSignal

        config = _make_config()
        data, tickers, features = _make_synthetic_data(n_days=200)

        def alpha_fn(d):
            return AlphaSignal(np.random.randn(N_STOCKS).astype(np.float32) * 0.3,
                               np.random.uniform(0, 1, N_STOCKS).astype(np.float32))

        def temporal_fn(d):
            return TemporalSignal(np.random.randn(N_STOCKS).astype(np.float32) * 0.2,
                                  np.random.uniform(0, 1, N_STOCKS).astype(np.float32),
                                  np.random.randn(N_STOCKS, 16).astype(np.float32))

        def gnn_fn(d):
            return GraphSignal(np.random.randn(N_STOCKS).astype(np.float32) * 0.2,
                               np.random.uniform(0, 1, N_STOCKS).astype(np.float32),
                               np.random.rand(N_STOCKS, N_STOCKS).astype(np.float32))

        def regime_fn(d):
            return RegimeSignal(
                                embedding=np.random.randn(8).astype(np.float32),
                                regime_probs=np.array([0.1, 0.6, 0.2, 0.1], dtype=np.float32),
                                regime_label="bull_trending")

        def debate_fn(d):
            return DebateSignal(np.random.randn(N_STOCKS).astype(np.float32) * 0.1,
                                np.random.uniform(0.3, 0.6, N_STOCKS).astype(np.float32),
                                0.65, {})

        def sentiment_fn(d):
            return np.random.randn(N_STOCKS * SENTIMENT_DIM).astype(np.float32) * 0.1

        def make_env():
            return NSETradingEnv(
                data=data, feature_names=features, config=config,
                risk_profile="balanced", term="medium",
                sentiment_fn=sentiment_fn, alpha_fn=alpha_fn,
                temporal_fn=temporal_fn, gnn_fn=gnn_fn,
                regime_fn=regime_fn, debate_fn=debate_fn,
            )

        vec_env = DummyVecEnv([make_env])

        model = SAC(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            buffer_size=1000,
            batch_size=32,
            learning_starts=50,
            policy_kwargs={"net_arch": config["sac"]["net_arch"]},
            verbose=0,
            device="cpu",
        )

        # 100 steps — just verify it doesn't crash
        model.learn(total_timesteps=100)

        # Verify it can predict
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1, N_STOCKS)


# ── Test 10: Adversarial filter + explainer (Phase 4) ────────────────────────

class TestPhase4Integration:
    def test_adversarial_filter_with_alpha(self):
        """Adversarial filter can process an AlphaSignal."""
        from alpha.adversarial_filter import AdversarialFilter
        from alpha.lightgbm_alpha import AlphaSignal

        config = _make_config()
        tickers = config["market"]["tickers"]

        filt = AdversarialFilter(config, tickers)

        signal = AlphaSignal(
            direction=np.array([0.8, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            conviction=np.array([0.9, 0.8, 0.2, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        )

        filtered = filt.filter(signal, "2024-01-15")
        assert filtered.direction.shape == (N_STOCKS,)
        assert filtered.conviction.shape == (N_STOCKS,)

    def test_explainer_doesnt_crash(self):
        """Trade explainer handles a basic call."""
        from llm.explainer import TradeExplainer

        config = _make_config()
        tickers = config["market"]["tickers"]

        explainer = TradeExplainer(config, tickers)

        result = explainer.explain(
            weight_deltas=np.array([0.05, -0.03, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            alpha_direction=np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            alpha_conviction=np.array([0.8, 0.7, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            portfolio_weights=np.zeros(N_STOCKS, dtype=np.float32),
            date="2024-01-15",
        )
        # Result may be None if no trades exceed min_delta — that's fine
        # Just verify it doesn't throw


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
