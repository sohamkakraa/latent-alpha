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
