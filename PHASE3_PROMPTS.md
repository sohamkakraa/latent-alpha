# Phase 3: GNN Cross-Stock Dependencies + VAE Regime Detection

> Adds two new alpha channels to the meta-model stack.
> Each prompt is self-contained. Run them in order (1→2→3).
> All changes are gated behind config flags — existing training is unaffected.
> These can be built while Phase 2 experiments are still running.

---

## Current Architecture (for context)

```
Market data (OHLCV × 10 stocks)
        │
        ├─→ AlphaFeatureEngine (54 features) ─→ LightGBM ─→ AlphaSignal (dir + conv)
        │                                                          │
        ├─→ TemporalFeatureBuilder (sequences) ─→ PatchTST ─→ TemporalSignal (dir + conv)
        │                                                          │
        │                                              ┌───────────┘
        │                                              ▼
        │                                     Ridge Meta-Model
        │                                              │
        └─→ SAC obs = [market(350) + portfolio(12) + locks(10) + sentiment(50) + alpha(20) + temporal(20)]
                                                       │
                                                  SAC Agent (512-dim obs → delta weights)
```

**After Phase 3:**

```
        ├─→ LightGBM ─→ AlphaSignal ─────────────────────────┐
        ├─→ PatchTST ─→ TemporalSignal ──────────────────────┤
        ├─→ GNN ─→ GraphSignal (cross-stock structure) ──────┤ → Meta-Model → SAC obs
        ├─→ VAE ─→ RegimeSignal (market regime embedding) ───┘
```

**Key files in the codebase:**

- `alpha/lightgbm_alpha.py` — AlphaSignal dataclass (direction, conviction, raw_pred, to_obs_vector())
- `alpha/patchtst.py` — TemporalSignal dataclass (direction, conviction, embedding, to_obs_vector())
- `alpha/meta_model.py` — RidgeMetaModel with ChannelPrediction, register_channel(), combine()
- `alpha/feature_engineering.py` — AlphaFeatureEngine, 54 features across 8 groups, cross-stock features (ret_rank, vol_rank, mom_rank, ret_vs_mean, sector_rel_mom, dispersion_contrib)
- `alpha/temporal_features.py` — TemporalFeatureBuilder, sliding window sequences for PatchTST
- `env/trading_env.py` — NSETradingEnv, observation space concatenation: [market, portfolio, locks, sentiment, alpha, temporal]
- `env/data_loader.py` — NSEDataLoader, yfinance data, technical indicators, EXTENDED_FEATURE_NAMES
- `strategy/term_selector.py` — TermSelector, builds alpha/temporal pipelines, creates alpha_fn/temporal_fn closures
- `config/config.yaml` — all hyperparameters; alpha.obs_dim=20, patchtst.obs_dim=20, sac.net_arch=[384,384]
- `agent/reward.py` — RiskAwareReward with 6 components (raw return, Sharpe, DD, TC, turnover, holding, alpha bonus)

**10 tickers:** RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, KOTAKBANK.NS, LT.NS, AXISBANK.NS

**Current obs space:** ~512 dims (varies by feature count)

---

## Prompt 1: GNN Cross-Stock Dependency Model

### Context

The latent.alpha system currently treats each stock's alpha signal independently. The LightGBM model has 6 cross-stock features (ret_rank, vol_rank, mom_rank, ret_vs_mean, sector_rel_mom, dispersion_contrib), but these are simple rank-based — they don't model the actual dependency structure between stocks.

In Indian markets, stocks are highly interconnected: banking stocks (HDFC, ICICI, Kotak, Axis) move together; IT stocks (TCS, Infy) correlate; Reliance has idiosyncratic exposure. A GNN can learn this structure from return correlations and propagate signals across the graph, so that when one banking stock shows alpha, the model can infer implications for other banking stocks.

### Task

Create `alpha/gnn_alpha.py` — a Graph Neural Network module that:

1. Builds a dynamic adjacency matrix from rolling return correlations between stocks
2. Runs message-passing (Graph Attention Network) to propagate signals across the stock graph
3. Produces a `GraphSignal` dataclass with per-stock direction and conviction
4. Registers as a channel in the Ridge meta-model

### Architecture

```
Input per stock:
  - LightGBM direction + conviction (2 dims)
  - PatchTST direction + conviction (2 dims)
  - Return features: ret_1d, ret_5d, ret_20d (3 dims)
  - Volatility: vol_5d, vol_20d (2 dims)
  - Volume: volume_z_20 (1 dim)
  → 10 dims per node

Graph structure:
  - Nodes: 10 stocks
  - Edges: weighted by rolling 60-day return correlation
  - Edge threshold: |corr| > 0.3 (sparsify — keep only meaningful relationships)
  - Adjacency updated per timestep (dynamic graph)

GNN layers:
  - 2-layer Graph Attention Network (GAT)
  - d_hidden = 32 (compact — only 10 nodes)
  - n_heads = 4
  - Output: per-node embedding → linear head → (direction, conviction)

Training:
  - Same purged walk-forward CV as LightGBM
  - Target: forward N-day return (same as alpha model)
  - Loss: MSE on direction + calibration loss on conviction
  - Optimizer: AdamW, lr=1e-3, cosine annealing
  - Epochs: 30, patience: 8
```

### Files to create

**`alpha/gnn_alpha.py`:**

```python
"""
alpha/gnn_alpha.py
══════════════════
Graph Neural Network for cross-stock dependency modelling.

Learns the correlation structure between NSE stocks and propagates
alpha signals across the graph. When HDFC Bank shows bullish alpha,
the GNN can learn to propagate a dampened bullish signal to ICICI,
Kotak, and Axis based on historical co-movement patterns.

Architecture:
  - Dynamic adjacency: rolling 60-day return correlation, thresholded
  - 2-layer GAT (Graph Attention Network) with 4 heads
  - Per-node input: alpha signals + return/vol/volume features (10 dims)
  - Per-node output: direction + conviction (2 dims)

The GAT attention mechanism learns WHICH stock relationships matter
for alpha propagation, not just that they're correlated. This is
strictly more powerful than the simple cross-stock rank features
in the LightGBM model.

Key design decisions:
  - PyTorch only (no PyG dependency) — implement message passing manually.
    This keeps the dependency footprint small and avoids PyG's complex
    installation on macOS ARM.
  - Small model (~5K params) — 10 nodes is tiny, so we don't need a deep GNN.
  - Dynamic adjacency — recomputed each timestep from rolling correlations.
    Static sector assignments would miss regime-dependent correlation shifts.
  - Trained on same labels as LightGBM for direct comparability.
"""
```

The module should contain:
- `GraphSignal` dataclass with `direction`, `conviction`, `attention_weights` (for interpretability), and `to_obs_vector()` → `[direction | conviction]` of shape `(2 * n_stocks,)`
- `GATLayer` — single GAT attention layer (PyTorch nn.Module). Implements multi-head attention over the adjacency graph. Inputs: node features (n_nodes, d_in), adjacency (n_nodes, n_nodes). Output: (n_nodes, d_out).
- `GNNModel` — 2-layer GAT with prediction heads (PyTorch nn.Module). Input: (n_nodes, d_in=10), adjacency. Output heads: direction_head (n_nodes, 1), conviction_head (n_nodes, 1).
- `build_adjacency(returns_df, window=60, threshold=0.3)` — function that computes rolling correlation matrix and thresholds it. Returns adjacency tensor of shape (n_stocks, n_stocks).
- `GNNAlpha` — wrapper class (like LightGBMAlpha / PatchTSTAlpha). Has `train(data, labels)`, `predict(node_features, adjacency)` → `GraphSignal`, `save/load`.
- Graceful fallback when PyTorch is missing (`HAS_TORCH` flag).

### Files to modify

1. **`config/config.yaml`** — add new section:
   ```yaml
   # ── v3.1 Phase 3: GNN cross-stock model ─────────────────────────────────────
   gnn:
     enabled: false                    # Set true to activate GNN channel
     obs_dim: 20                       # 2 × n_stocks (direction + conviction)

     # Graph construction
     correlation_window: 60            # Rolling window for adjacency computation
     edge_threshold: 0.3               # |corr| > threshold to create edge
     self_loops: true                  # Include self-connections

     # GAT architecture
     d_input: 10                       # Per-node feature dim
     d_hidden: 32                      # Hidden dimension
     n_heads: 4                        # Attention heads
     n_layers: 2                       # GAT depth
     dropout: 0.1

     # Training
     learning_rate: 1.0e-3
     epochs: 30
     batch_size: 64
     patience: 8
     forward_horizon: 5                # Match alpha model target
   ```

2. **`alpha/__init__.py`** — add conditional import:
   ```python
   try:
       from alpha.gnn_alpha import GNNAlpha, GraphSignal
   except ImportError:
       pass  # PyTorch not available
   ```

3. **Do NOT modify `trading_env.py` or `term_selector.py` yet** — that's the integration step in Prompt 3. For now, the GNN is a standalone module with its own train/predict interface.

### Tests to write

Create `tests/test_gnn_alpha.py`:

```python
"""Tests for GNN cross-stock dependency model."""
import numpy as np
import pytest

def test_graph_signal_dataclass():
    """GraphSignal has correct interface."""
    from alpha.gnn_alpha import GraphSignal
    sig = GraphSignal(
        direction=np.array([0.5, -0.3, 0.1], dtype=np.float32),
        conviction=np.array([0.8, 0.6, 0.4], dtype=np.float32),
    )
    vec = sig.to_obs_vector()
    assert vec.shape == (6,)  # 2 * 3 stocks
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec[:3], np.clip(sig.direction, -3, 3))
    np.testing.assert_array_equal(vec[3:], np.clip(sig.conviction, 0, 1))

def test_build_adjacency_shape():
    """Adjacency matrix has correct shape and properties."""
    from alpha.gnn_alpha import build_adjacency
    # Create mock return data: 100 days × 5 stocks
    np.random.seed(42)
    returns = np.random.randn(100, 5) * 0.02
    # Make stocks 0 and 1 correlated
    returns[:, 1] = returns[:, 0] * 0.8 + np.random.randn(100) * 0.01

    adj = build_adjacency(returns, window=60, threshold=0.3)
    assert adj.shape == (5, 5)
    # Diagonal should be 1 (self-loops)
    np.testing.assert_array_almost_equal(np.diag(adj), np.ones(5))
    # Should be symmetric
    np.testing.assert_array_almost_equal(adj, adj.T)
    # Stocks 0 and 1 should be connected (high correlation)
    assert adj[0, 1] > 0

def test_build_adjacency_threshold():
    """Edges below threshold are zeroed out."""
    from alpha.gnn_alpha import build_adjacency
    np.random.seed(42)
    # Uncorrelated stocks
    returns = np.random.randn(100, 3) * 0.02
    adj = build_adjacency(returns, window=60, threshold=0.5)
    # Off-diagonal should be mostly zero for uncorrelated stocks
    off_diag = adj[~np.eye(3, dtype=bool)]
    assert np.sum(off_diag > 0) <= 2  # at most a couple of spurious edges

def test_gat_layer_forward():
    """GAT layer produces correct output shape."""
    torch = pytest.importorskip("torch")
    from alpha.gnn_alpha import GATLayer
    layer = GATLayer(d_in=10, d_out=32, n_heads=4, dropout=0.0)
    x = torch.randn(10, 10)  # 10 nodes, 10 features
    adj = torch.eye(10) + torch.randn(10, 10).abs() * 0.5
    adj = (adj > 0.3).float()
    out = layer(x, adj)
    assert out.shape == (10, 32)  # n_nodes × d_out

def test_gnn_model_forward():
    """Full GNN model produces direction and conviction."""
    torch = pytest.importorskip("torch")
    from alpha.gnn_alpha import GNNModel
    model = GNNModel(d_input=10, d_hidden=32, n_heads=4, n_layers=2, n_stocks=10)
    x = torch.randn(10, 10)
    adj = torch.eye(10)
    direction, conviction = model(x, adj)
    assert direction.shape == (10,)
    assert conviction.shape == (10,)
    # Conviction should be in [0, 1] (sigmoid output)
    assert torch.all(conviction >= 0) and torch.all(conviction <= 1)

def test_gnn_alpha_fallback():
    """GNNAlpha returns zero signal when PyTorch unavailable."""
    from alpha.gnn_alpha import GNNAlpha
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["gnn"]["enabled"] = True

    gnn = GNNAlpha(config)
    signal = gnn.get_zero_signal(n_stocks=10)
    assert signal.direction.shape == (10,)
    assert np.all(signal.direction == 0)
    assert np.all(signal.conviction == 0)

def test_gnn_alpha_train_predict():
    """End-to-end train and predict with mock data."""
    torch = pytest.importorskip("torch")
    from alpha.gnn_alpha import GNNAlpha
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["gnn"]["enabled"] = True
    config["gnn"]["epochs"] = 2  # fast test
    config["gnn"]["patience"] = 2

    gnn = GNNAlpha(config)
    n_stocks = 10
    n_samples = 200

    # Mock node features: (n_samples, n_stocks, d_input)
    np.random.seed(42)
    X = np.random.randn(n_samples, n_stocks, 10).astype(np.float32)
    # Mock returns for adjacency: (n_samples, n_stocks)
    returns = np.random.randn(n_samples, n_stocks).astype(np.float32) * 0.02
    # Mock labels: forward returns (n_samples, n_stocks)
    y = np.random.randn(n_samples, n_stocks).astype(np.float32) * 0.01

    metrics = gnn.train(X, returns, y)
    assert "train_loss" in metrics
    assert "val_loss" in metrics

    # Predict on single timestep
    signal = gnn.predict(X[-1], returns[-60:])
    assert signal.direction.shape == (n_stocks,)
    assert signal.conviction.shape == (n_stocks,)

def test_adjacency_dynamic():
    """Adjacency changes when correlation structure changes."""
    from alpha.gnn_alpha import build_adjacency
    np.random.seed(42)
    # Period 1: stocks 0,1 correlated
    returns_1 = np.random.randn(60, 3) * 0.02
    returns_1[:, 1] = returns_1[:, 0] * 0.9 + np.random.randn(60) * 0.005
    adj_1 = build_adjacency(returns_1, window=60, threshold=0.3)

    # Period 2: stocks 0,2 correlated instead
    returns_2 = np.random.randn(60, 3) * 0.02
    returns_2[:, 2] = returns_2[:, 0] * 0.9 + np.random.randn(60) * 0.005
    adj_2 = build_adjacency(returns_2, window=60, threshold=0.3)

    # Adjacencies should differ
    assert not np.allclose(adj_1, adj_2)

def test_config_gnn_section():
    """Config has gnn section with required keys."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    gnn = config.get("gnn", {})
    assert "enabled" in gnn
    assert "obs_dim" in gnn
    assert gnn["obs_dim"] == 20
    assert "correlation_window" in gnn
    assert "d_hidden" in gnn
    assert "n_heads" in gnn
```

### Verification checklist

- [ ] `GraphSignal.to_obs_vector()` returns shape `(2 * n_stocks,)` with proper clipping
- [ ] `build_adjacency()` produces symmetric matrix with self-loops, thresholded edges
- [ ] `GATLayer` implements multi-head attention over adjacency
- [ ] `GNNModel` forward pass: (n_stocks, d_in) + adjacency → (direction, conviction)
- [ ] `GNNAlpha.train()` runs purged walk-forward CV, returns metrics
- [ ] `GNNAlpha.predict()` returns `GraphSignal`
- [ ] Graceful fallback when PyTorch missing
- [ ] Save/load serialisation works
- [ ] Config has `gnn:` section with all hyperparameters
- [ ] All tests pass
- [ ] No existing files modified except config.yaml and alpha/__init__.py

---

## Prompt 2: VAE Market Regime Detector

### Context

The latent.alpha system's alpha models (LightGBM, PatchTST) produce per-stock signals but have no explicit awareness of the overall market regime. In a bull market, momentum signals work well; in a bear market, mean-reversion signals dominate; in a choppy sideways market, both fail. A VAE can learn to compress the market's state into a low-dimensional regime embedding that the SAC agent uses to modulate its strategy.

### Task

Create `alpha/vae_regime.py` — a Variational Autoencoder that:

1. Takes a snapshot of the entire market state (cross-stock features for all 10 stocks)
2. Encodes it into a low-dimensional latent space (regime embedding)
3. Produces a `RegimeSignal` dataclass with the regime embedding + decoded regime probabilities
4. Registers as a channel in the meta-model (or directly appended to SAC obs)

### Architecture

```
Input: market state snapshot
  - Per-stock: ret_1d, ret_5d, ret_20d, vol_5d, vol_20d, rsi_14, macd_hist (7 dims)
  - Cross-stock: mean_ret, ret_dispersion, correlation_mean, vol_of_vol (4 dims)
  → Total: 10 * 7 + 4 = 74 dims per timestep

Encoder:
  - Linear(74, 64) → ReLU → Linear(64, 32) → ReLU
  - mu_head: Linear(32, 8)      # regime embedding mean
  - logvar_head: Linear(32, 8)  # regime embedding log-variance

Latent space:
  - z ~ N(mu, exp(logvar/2))    # 8-dim regime embedding
  - Reparameterisation trick for training

Decoder (reconstruction):
  - Linear(8, 32) → ReLU → Linear(32, 64) → ReLU → Linear(64, 74)
  - Reconstructs input market state

Regime classifier head (auxiliary):
  - Linear(8, 4) → Softmax
  - 4 regimes: bull_trending, bear_trending, high_vol_choppy, low_vol_range
  - Trained with pseudo-labels derived from rolling metrics:
    * bull_trending: ret_20d > 0 AND vol_20d < median_vol AND trend_strength > 0
    * bear_trending: ret_20d < 0 AND vol_20d < median_vol AND trend_strength < 0
    * high_vol_choppy: vol_20d > 75th percentile
    * low_vol_range: vol_20d < 25th percentile AND |ret_20d| < 1%

Loss:
  - reconstruction_loss = MSE(input, decoded)
  - kl_divergence = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
  - regime_loss = CrossEntropy(regime_pred, pseudo_label)  [when labels available]
  - total = reconstruction_loss + beta * kl_divergence + gamma * regime_loss
  - beta annealing: starts at 0.0, linearly increases to 1.0 over first 10 epochs

Training:
  - AdamW, lr=1e-3, cosine annealing
  - Epochs: 50, patience: 10
  - Batch size: 128
```

### Files to create

**`alpha/vae_regime.py`:**

```python
"""
alpha/vae_regime.py
═══════════════════
Variational Autoencoder for market regime detection.

Compresses the full market state into an 8-dimensional latent embedding
that captures the current market regime. The SAC agent uses this to
modulate its strategy — e.g., increase cash in high-vol regimes,
lean into momentum in trending regimes.

Why a VAE instead of a simple classifier?
  - The latent space captures regime TRANSITIONS smoothly. A hard
    classifier snaps from "bull" to "bear" in one step; the VAE
    embedding moves continuously through latent space.
  - The reconstruction objective forces the model to learn a
    compressed representation that preserves market state information,
    not just regime labels.
  - KL regularisation prevents the latent space from collapsing
    to discrete clusters, maintaining smooth interpolation.

The auxiliary regime classifier head provides interpretability
(you can check which regime the VAE thinks we're in) but the SAC
agent receives the raw 8-dim embedding, not the regime labels.

Key design decisions:
  - 8-dim latent space: small enough to not overwhelm SAC obs,
    large enough to capture nuanced regime characteristics.
  - Beta annealing: prevents KL collapse in early training.
  - Pseudo-labels from rolling metrics: no manual regime annotation needed.
  - Cross-stock aggregate features: the VAE sees the market as a whole,
    not individual stocks.
"""
```

The module should contain:
- `RegimeSignal` dataclass with `embedding` (8-dim), `regime_probs` (4-dim: bull/bear/choppy/range), `regime_label` (str), and `to_obs_vector()` → embedding of shape `(latent_dim,)`
- `VAEEncoder` — PyTorch nn.Module: input → hidden → (mu, logvar)
- `VAEDecoder` — PyTorch nn.Module: z → hidden → reconstructed input
- `RegimeVAE` — full model (PyTorch nn.Module) with encoder, decoder, regime head. Has `encode(x) → (mu, logvar)`, `decode(z) → x_hat`, `forward(x) → (x_hat, mu, logvar, regime_probs)`, `get_regime(x) → RegimeSignal`
- `build_market_state(data, date, tickers)` — function that builds the 74-dim input vector from current market data
- `generate_pseudo_labels(data, tickers)` — function that computes regime labels from rolling metrics
- `RegimeDetector` — wrapper class (like GNNAlpha). Has `train(X, pseudo_labels)`, `predict(market_state) → RegimeSignal`, `save/load`
- Graceful fallback when PyTorch missing (`HAS_TORCH` flag)

### Files to modify

1. **`config/config.yaml`** — add new section:
   ```yaml
   # ── v3.1 Phase 3: VAE regime detection ──────────────────────────────────────
   vae_regime:
     enabled: false                    # Set true to activate regime channel
     obs_dim: 8                        # Latent embedding dimension

     # Architecture
     d_input: 74                       # 10 stocks × 7 features + 4 cross-stock
     d_hidden: 64                      # Encoder/decoder hidden dim
     d_latent: 8                       # Latent space dimension
     n_regimes: 4                      # Number of regime categories

     # Training
     learning_rate: 1.0e-3
     epochs: 50
     batch_size: 128
     patience: 10
     beta_start: 0.0                   # KL weight start (annealed to 1.0)
     beta_end: 1.0                     # KL weight end
     beta_anneal_epochs: 10            # Epochs to anneal beta
     gamma: 0.5                        # Regime classification loss weight
   ```

2. **`alpha/__init__.py`** — add conditional import:
   ```python
   try:
       from alpha.vae_regime import RegimeDetector, RegimeSignal
   except ImportError:
       pass
   ```

3. **Do NOT modify `trading_env.py` or `term_selector.py` yet** — that's Prompt 3.

### Tests to write

Create `tests/test_vae_regime.py`:

```python
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
    """Reparameterisation trick produces different samples with same (mu, logvar)."""
    torch = pytest.importorskip("torch")
    from alpha.vae_regime import RegimeVAE
    model = RegimeVAE(d_input=74, d_hidden=64, d_latent=8, n_regimes=4)
    model.train()  # training mode enables stochastic sampling
    x = torch.randn(1, 74)
    _, mu1, _, _ = model(x)
    _, mu2, _, _ = model(x)
    # In eval mode, mu should be deterministic
    model.eval()
    _, mu3, _, _ = model(x)
    _, mu4, _, _ = model(x)
    assert torch.allclose(mu3, mu4)

def test_build_market_state_shape():
    """Market state builder produces correct 74-dim vector."""
    from alpha.vae_regime import build_market_state
    import pandas as pd
    n_stocks = 10
    n_days = 100

    # Mock data: dict of {ticker: DataFrame}
    tickers = [f"STOCK{i}.NS" for i in range(n_stocks)]
    data = {}
    for t in tickers:
        idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
        df = pd.DataFrame({
            "Close": 100 + np.cumsum(np.random.randn(n_days) * 0.5),
            "High": 101 + np.cumsum(np.random.randn(n_days) * 0.5),
            "Low": 99 + np.cumsum(np.random.randn(n_days) * 0.5),
            "Volume": np.random.randint(1e6, 1e7, n_days),
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
```

### Verification checklist

- [ ] `RegimeSignal.to_obs_vector()` returns shape `(d_latent,)` = `(8,)`
- [ ] `RegimeVAE` forward pass: (batch, 74) → (x_hat, mu, logvar, regime_probs)
- [ ] Regime probs sum to 1.0
- [ ] Reparameterisation trick: stochastic in train mode, deterministic in eval
- [ ] Beta annealing works (KL weight increases from 0 to 1 over N epochs)
- [ ] `build_market_state()` extracts 74-dim vector from market data
- [ ] `generate_pseudo_labels()` produces valid regime labels from rolling metrics
- [ ] `RegimeDetector` train/predict cycle works end-to-end
- [ ] Graceful fallback when PyTorch missing
- [ ] Save/load serialisation works
- [ ] Config has `vae_regime:` section
- [ ] All tests pass

---

## Prompt 3: Integration — Wire GNN + VAE into SAC Pipeline

### Context

After Prompts 1 and 2, we have two new standalone modules:
- `alpha/gnn_alpha.py` — GNN with GraphSignal (direction + conviction per stock)
- `alpha/vae_regime.py` — VAE with RegimeSignal (8-dim regime embedding)

These need to be integrated into the SAC observation space and the walk-forward backtest pipeline, just like LightGBM and PatchTST were integrated in Phase 1 and 2.

### Task

Wire GNN and VAE into the existing pipeline:

1. Add `gnn_fn` and `regime_fn` parameters to `NSETradingEnv`
2. Include their outputs in the observation vector
3. Build the GNN and VAE pipelines in `TermSelector`
4. Update the observation space calculation
5. Register both as channels in the Ridge meta-model

### Files to modify

1. **`env/trading_env.py`**

   Add to `__init__()`:
   ```python
   # v3.1 Phase 3: GNN cross-stock signal
   self.gnn_fn = gnn_fn
   self.gnn_dim = config.get("gnn", {}).get("obs_dim", 0) if config.get("gnn", {}).get("enabled", False) else 0

   # v3.1 Phase 3: VAE regime embedding
   self.regime_fn = regime_fn
   self.regime_dim = config.get("vae_regime", {}).get("obs_dim", 0) if config.get("vae_regime", {}).get("enabled", False) else 0
   ```

   Update `obs_dim` calculation to include `+ self.gnn_dim + self.regime_dim`.

   Update `_get_observation()` to call `gnn_fn(date)` and `regime_fn(date)`, concatenating their obs vectors.

   Update the `__init__` signature to accept `gnn_fn=None, regime_fn=None`.

2. **`strategy/term_selector.py`**

   Add `_build_gnn_pipeline()` method — similar to `_build_alpha_pipeline()`:
   - Build node features from alpha signals + return/vol features
   - Compute adjacency from return correlations
   - Train GNN on forward return labels
   - Return `gnn_fn(date)` closure

   Add `_build_regime_pipeline()` method:
   - Build market state vectors for each date
   - Generate pseudo-labels
   - Train VAE
   - Return `regime_fn(date)` closure

   Wire both into `build_env()` and `build_backtest_env()`, passing them to the env constructor.

   Add lazy imports for GNN and VAE modules at the top (like `_import_alpha()` and `_import_patchtst()`).

3. **`config/config.yaml`** — update `sac:` section:
   ```yaml
   net_arch: [512, 512]  # Accommodate new obs dims (gnn: +20, vae: +8 → total ~540)
   ```

### Tests to write

Create `tests/test_phase3_integration.py`:

```python
"""Integration tests for Phase 3 (GNN + VAE) in the SAC pipeline."""
import numpy as np
import pytest

def test_obs_space_with_gnn_vae():
    """Observation space includes GNN and VAE dimensions when enabled."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    n_stocks = len(config["market"]["tickers"])

    # Enable both
    config["gnn"]["enabled"] = True
    config["vae_regime"]["enabled"] = True

    from env.data_loader import NSEDataLoader
    loader = NSEDataLoader(config)
    n_features = len(loader.get_feature_names())
    sentiment_dim = config["llm"]["sentiment_dim"]
    alpha_dim = config.get("alpha", {}).get("obs_dim", 0)
    temporal_dim = config.get("patchtst", {}).get("obs_dim", 0)
    gnn_dim = config["gnn"]["obs_dim"]
    vae_dim = config["vae_regime"]["obs_dim"]

    expected_obs = (
        n_stocks * n_features
        + n_stocks + 2 + n_stocks
        + n_stocks * sentiment_dim
        + alpha_dim + temporal_dim
        + gnn_dim + vae_dim
    )
    print(f"Phase 3 obs_dim: {expected_obs}")
    assert gnn_dim == 20
    assert vae_dim == 8
    assert expected_obs > 500

def test_env_accepts_gnn_regime_fn():
    """TradingEnv constructor accepts gnn_fn and regime_fn without error."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["gnn"]["enabled"] = True
    config["vae_regime"]["enabled"] = True

    # Check that the env signature accepts the new params
    import inspect
    from env.trading_env import NSETradingEnv
    sig = inspect.signature(NSETradingEnv.__init__)
    params = list(sig.parameters.keys())
    assert "gnn_fn" in params
    assert "regime_fn" in params

def test_env_obs_includes_gnn_vae_zeros():
    """When gnn_fn/regime_fn are None, obs still has correct shape with zeros."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["gnn"]["enabled"] = True
    config["vae_regime"]["enabled"] = True

    # The obs should include zero-padded GNN and VAE dims
    n_stocks = len(config["market"]["tickers"])
    gnn_dim = config["gnn"]["obs_dim"]
    vae_dim = config["vae_regime"]["obs_dim"]

    # GNN and VAE dims should be present even when fns are None
    assert gnn_dim == 20
    assert vae_dim == 8

def test_backward_compatibility():
    """With GNN and VAE disabled, obs space matches Phase 2."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["gnn"]["enabled"] = False
    config["vae_regime"]["enabled"] = False

    n_stocks = len(config["market"]["tickers"])
    from env.data_loader import NSEDataLoader
    loader = NSEDataLoader(config)
    n_features = len(loader.get_feature_names())
    sentiment_dim = config["llm"]["sentiment_dim"]
    alpha_dim = config.get("alpha", {}).get("obs_dim", 0)
    temporal_dim = config.get("patchtst", {}).get("obs_dim", 0)

    phase2_obs = (
        n_stocks * n_features
        + n_stocks + 2 + n_stocks
        + n_stocks * sentiment_dim
        + alpha_dim + temporal_dim
    )
    # Should match original Phase 2 obs (no GNN/VAE dims added when disabled)
    print(f"Phase 2 obs_dim (backward compat): {phase2_obs}")
    assert phase2_obs > 400

def test_term_selector_has_gnn_regime_builders():
    """TermSelector has _build_gnn_pipeline and _build_regime_pipeline methods."""
    from strategy.term_selector import TermSelector
    assert hasattr(TermSelector, "_build_gnn_pipeline")
    assert hasattr(TermSelector, "_build_regime_pipeline")
```

### Verification checklist

- [ ] `NSETradingEnv.__init__` accepts `gnn_fn=None` and `regime_fn=None`
- [ ] `gnn_dim` and `regime_dim` are 0 when their respective modules are disabled
- [ ] Observation vector concatenation: `[market, portfolio, locks, sentiment, alpha, temporal, gnn, regime]`
- [ ] `observation_space` shape calculation includes GNN and VAE dims
- [ ] `_get_observation()` calls `gnn_fn(date)` and `regime_fn(date)` with try/except fallback
- [ ] `TermSelector._build_gnn_pipeline()` trains GNN and returns closure
- [ ] `TermSelector._build_regime_pipeline()` trains VAE and returns closure
- [ ] Both pipelines are called in `build_env()` and `build_backtest_env()`
- [ ] `sac.net_arch` updated to `[512, 512]` in config
- [ ] Backward compatible: with `gnn.enabled: false` and `vae_regime.enabled: false`, obs space matches Phase 2
- [ ] All tests pass
- [ ] Training smoke test passes: `KMP_DUPLICATE_LIB_OK=TRUE python3 train.py --algo sac --risk balanced --term medium --timesteps 10000`

---

## Running all Phase 3 tests

```bash
cd latent_alpha
python -m pytest tests/test_gnn_alpha.py tests/test_vae_regime.py tests/test_phase3_integration.py -v --tb=short
```

## Implementation order

1. **Prompt 1** (GNN) — standalone module, no pipeline changes
2. **Prompt 2** (VAE) — standalone module, no pipeline changes
3. **Prompt 3** (Integration) — wires both into env + term_selector

Prompts 1 and 2 are independent and can be done in parallel.
Prompt 3 depends on both being complete.

## Post-Phase 3 training

Once all three prompts are implemented and tests pass:

```bash
# Smoke test (5 minutes)
caffeinate -dims bash -c 'KMP_DUPLICATE_LIB_OK=TRUE python3 train.py --algo sac --risk balanced --term medium --timesteps 10000'

# Full training with GNN + VAE enabled
# Edit config.yaml: set gnn.enabled: true, vae_regime.enabled: true
caffeinate -dims bash -c 'KMP_DUPLICATE_LIB_OK=TRUE python3 -m experiments.runner --timesteps 1000000'
```
