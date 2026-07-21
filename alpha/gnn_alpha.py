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

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    logger.warning(
        "PyTorch not installed. GNNAlpha will produce zero signals."
    )
    HAS_TORCH = False


# ── GraphSignal (output) ─────────────────────────────────────────────────────

@dataclass
class GraphSignal:
    """Per-stock signal from the GNN cross-dependency model."""

    direction: np.ndarray        # (n_stocks,) predicted return direction
    conviction: np.ndarray       # (n_stocks,) confidence in [0, 1]
    attention_weights: Optional[np.ndarray] = field(default=None, repr=False)

    def to_obs_vector(self) -> np.ndarray:
        """Flatten for SAC obs: [direction_clipped | conviction_clipped]."""
        return np.concatenate([
            np.clip(self.direction, -3, 3),
            np.clip(self.conviction, 0, 1),
        ]).astype(np.float32)

    @property
    def n_stocks(self) -> int:
        return len(self.direction)


# ── Graph construction ────────────────────────────────────────────────────────

def build_adjacency(
    returns: np.ndarray,
    window: int = 60,
    threshold: float = 0.3,
    self_loops: bool = True,
) -> np.ndarray:
    """
    Build dynamic adjacency matrix from rolling return correlations.

    Parameters
    ----------
    returns   : (n_days, n_stocks) float array
    window    : rolling window — use last `window` rows
    threshold : |corr| > threshold creates an edge
    self_loops: include self-connections (diagonal = 1)

    Returns
    -------
    adj : (n_stocks, n_stocks) symmetric float32 adjacency matrix
    """
    n_stocks = returns.shape[1]

    # Use last `window` rows
    r = returns[-window:] if len(returns) > window else returns

    if len(r) < 2:
        return np.eye(n_stocks, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(r.T)  # (n_stocks, n_stocks)

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    adj = (np.abs(corr) > threshold).astype(np.float32)
    adj = np.maximum(adj, adj.T)  # ensure symmetry

    if self_loops:
        np.fill_diagonal(adj, 1.0)

    return adj.astype(np.float32)


# ── PyTorch model components ──────────────────────────────────────────────────

if HAS_TORCH:

    class GATLayer(nn.Module):
        """
        Single Graph Attention Network layer (manual implementation, no PyG).

        Implements multi-head attention over the adjacency graph.
        Input:  x (n_nodes, d_in), adj (n_nodes, n_nodes)
        Output: (n_nodes, d_out)
        """

        def __init__(
            self,
            d_in: int,
            d_out: int,
            n_heads: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            assert d_out % n_heads == 0, (
                f"d_out {d_out} must be divisible by n_heads {n_heads}"
            )
            self.d_in = d_in
            self.d_out = d_out
            self.n_heads = n_heads
            self.d_head = d_out // n_heads

            # Shared linear projection for all heads
            self.W = nn.Linear(d_in, d_out, bias=False)

            # Attention vector per head: a_k applied to [h_i || h_j]
            self.a = nn.Parameter(torch.empty(n_heads, 2 * self.d_head))
            nn.init.xavier_uniform_(self.a.unsqueeze(0))

            self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_out)

            # Residual projection when dimensions differ
            self.residual = (
                nn.Linear(d_in, d_out, bias=False)
                if d_in != d_out
                else nn.Identity()
            )

        def forward(
            self, x: "torch.Tensor", adj: "torch.Tensor"
        ) -> "torch.Tensor":
            """
            x   : (n_nodes, d_in)
            adj : (n_nodes, n_nodes)
            returns : (n_nodes, d_out)
            """
            n_nodes = x.shape[0]

            # (n_nodes, d_out) → (n_nodes, n_heads, d_head)
            h = self.W(x).view(n_nodes, self.n_heads, self.d_head)

            # Expand for all (i,j) pairs
            h_i = h.unsqueeze(1).expand(n_nodes, n_nodes, self.n_heads, self.d_head)
            h_j = h.unsqueeze(0).expand(n_nodes, n_nodes, self.n_heads, self.d_head)

            # Attention scores: (n_nodes, n_nodes, n_heads)
            h_cat = torch.cat([h_i, h_j], dim=-1)
            e = (h_cat * self.a[None, None, :, :]).sum(dim=-1)
            e = self.leaky_relu(e)

            # Mask non-edges with -inf
            mask = (adj > 0).unsqueeze(-1)
            e = torch.where(mask, e, torch.full_like(e, -1e9))

            # Softmax over "from" nodes (dim=1), then dropout
            attn = torch.softmax(e, dim=1)   # (n_nodes, n_nodes, n_heads)
            attn = self.dropout(attn)

            # Aggregate: weighted sum of h_j → (n_nodes, n_heads, d_head)
            out = (attn.unsqueeze(-1) * h_j).sum(dim=1)

            # Concat heads → (n_nodes, d_out)
            out = out.reshape(n_nodes, self.d_out)

            return self.norm(out + self.residual(x))

    class GNNModel(nn.Module):
        """
        2-layer Graph Attention Network with direction and conviction heads.

        Input:  (n_stocks, d_input), adjacency (n_stocks, n_stocks)
        Output: direction (n_stocks,), conviction (n_stocks,)
        """

        def __init__(
            self,
            d_input: int = 10,
            d_hidden: int = 32,
            n_heads: int = 4,
            n_layers: int = 2,
            n_stocks: int = 10,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_layers = n_layers

            layers = []
            in_dim = d_input
            for _ in range(n_layers):
                layers.append(GATLayer(in_dim, d_hidden, n_heads, dropout))
                in_dim = d_hidden
            self.gat_layers = nn.ModuleList(layers)

            self.relu = nn.ReLU()
            self.direction_head = nn.Linear(d_hidden, 1)
            self.conviction_head = nn.Sequential(
                nn.Linear(d_hidden, 1),
                nn.Sigmoid(),
            )

        def forward(
            self, x: "torch.Tensor", adj: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            x   : (n_stocks, d_input)
            adj : (n_stocks, n_stocks)
            returns : direction (n_stocks,), conviction (n_stocks,)
            """
            h = x
            for i, layer in enumerate(self.gat_layers):
                h = layer(h, adj)
                if i < self.n_layers - 1:
                    h = self.relu(h)

            direction = self.direction_head(h).squeeze(-1)
            conviction = self.conviction_head(h).squeeze(-1)
            return direction, conviction


# ── High-level GNN wrapper ────────────────────────────────────────────────────

class GNNAlpha:
    """
    GNN alpha model — wrapper for training, prediction, and pipeline integration.

    Usage
    -----
    model = GNNAlpha(config)
    metrics = model.train(X, returns, y)
    signal = model.predict(node_features, returns_window)  # → GraphSignal
    """

    def __init__(self, config: dict):
        self.cfg = config
        gnn_cfg = config.get("gnn", {})

        self.enabled = gnn_cfg.get("enabled", False)
        self.epochs = gnn_cfg.get("epochs", 30)
        self.batch_size = gnn_cfg.get("batch_size", 64)
        self.patience = gnn_cfg.get("patience", 8)

        self._model = None
        self._is_trained = False
        self._n_stocks = 10
        self._d_input = 10
        self._device = "cpu"
        self._best_state: Optional[dict] = None

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_zero_signal(self, n_stocks: int) -> GraphSignal:
        """Return a zero signal (used as fallback when model unavailable)."""
        return GraphSignal(
            direction=np.zeros(n_stocks, dtype=np.float32),
            conviction=np.zeros(n_stocks, dtype=np.float32),
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        returns: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train GNN on forward return labels with dynamic adjacency.

        Parameters
        ----------
        X       : (n_samples, n_stocks, d_input) node features per timestep
        returns : (n_samples, n_stocks) raw returns for adjacency computation
        y       : (n_samples, n_stocks) forward return labels

        Returns
        -------
        Dict with train_loss, val_loss, best_epoch
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available — GNN training skipped.")
            self._is_trained = True
            return {"train_loss": 0.0, "val_loss": 0.0, "best_epoch": 0}

        n_samples, n_stocks, d_input = X.shape
        self._n_stocks = n_stocks
        self._d_input = d_input

        gnn_cfg = self.cfg.get("gnn", {})
        window = gnn_cfg.get("correlation_window", 60)
        threshold = gnn_cfg.get("edge_threshold", 0.3)

        # Pre-compute adjacency for each sample
        adjs = []
        for i in range(n_samples):
            start = max(0, i - window)
            adj = build_adjacency(returns[start:i + 1], window=window, threshold=threshold)
            adjs.append(adj)

        # Chronological 80/20 split
        split = int(n_samples * 0.8)
        X_train, X_val = X[:split], X[split:]
        adjs_train, adjs_val = adjs[:split], adjs[split:]
        y_train, y_val = y[:split], y[split:]

        # Build model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = GNNModel(
            d_input=d_input,
            d_hidden=gnn_cfg.get("d_hidden", 32),
            n_heads=gnn_cfg.get("n_heads", 4),
            n_layers=gnn_cfg.get("n_layers", 2),
            n_stocks=n_stocks,
            dropout=gnn_cfg.get("dropout", 0.1),
        ).to(self._device)

        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info("GNN: %d parameters (%.1fK)", n_params, n_params / 1000)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=gnn_cfg.get("learning_rate", 1e-3),
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self._device)
        y_train_t = torch.FloatTensor(y_train).to(self._device)
        adjs_train_t = [torch.FloatTensor(a).to(self._device) for a in adjs_train]

        X_val_t = torch.FloatTensor(X_val).to(self._device)
        y_val_t = torch.FloatTensor(y_val).to(self._device)
        adjs_val_t = [torch.FloatTensor(a).to(self._device) for a in adjs_val]

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        avg_train_loss = 0.0

        for epoch in range(self.epochs):
            # ── Train ─────────────────────────────────────────────────────
            self._model.train()
            perm = np.random.permutation(len(X_train))
            epoch_loss_sum = 0.0
            n_batches = 0

            for batch_start in range(0, len(X_train), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(X_train))
                batch_idx = perm[batch_start:batch_end]

                optimizer.zero_grad()
                losses = []

                for idx in batch_idx:
                    xi = X_train_t[idx]
                    adj_i = adjs_train_t[idx]
                    yi = y_train_t[idx]

                    direction, conviction = self._model(xi, adj_i)

                    yi_mean = yi.mean()
                    yi_std = yi.std() + 1e-8
                    yi_norm = ((yi - yi_mean) / yi_std).clamp(-3, 3)

                    dir_loss = F.mse_loss(direction, yi_norm)
                    agreement = torch.sigmoid(direction.detach() * yi_norm)
                    conv_loss = F.mse_loss(conviction, agreement)

                    losses.append(dir_loss + 0.5 * conv_loss)

                if losses:
                    batch_loss = sum(losses) / len(losses)
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss_sum += batch_loss.item()
                    n_batches += 1

            avg_train_loss = epoch_loss_sum / max(n_batches, 1)

            # ── Validate ──────────────────────────────────────────────────
            self._model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for i in range(len(X_val)):
                    direction, _ = self._model(X_val_t[i], adjs_val_t[i])
                    yi = y_val_t[i]
                    yi_norm = ((yi - yi.mean()) / (yi.std() + 1e-8)).clamp(-3, 3)
                    val_loss_sum += F.mse_loss(direction, yi_norm).item()
            val_loss = val_loss_sum / max(len(X_val), 1)

            scheduler.step()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(
                    "GNN epoch %d/%d: train=%.4f, val=%.4f",
                    epoch, self.epochs, avg_train_loss, val_loss,
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self._best_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("GNN early stopping at epoch %d (best: %d)", epoch, best_epoch)
                    break

        # Restore best
        if self._best_state is not None:
            self._model.load_state_dict(self._best_state)
            self._model.to(self._device)

        self._is_trained = True
        logger.info(
            "GNN trained: best_val_loss=%.4f, best_epoch=%d, params=%dK",
            best_val_loss, best_epoch, n_params // 1000,
        )

        return {
            "train_loss": avg_train_loss,
            "val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        node_features: np.ndarray,
        returns_window: np.ndarray,
    ) -> GraphSignal:
        """
        Generate GraphSignal for current timestep.

        Parameters
        ----------
        node_features  : (n_stocks, d_input) per-node features
        returns_window : (n_days, n_stocks) recent returns for adjacency

        Returns
        -------
        GraphSignal
        """
        n_stocks = node_features.shape[0]

        if not HAS_TORCH or not self._is_trained or self._model is None:
            return self.get_zero_signal(n_stocks)

        gnn_cfg = self.cfg.get("gnn", {})
        window = gnn_cfg.get("correlation_window", 60)
        threshold = gnn_cfg.get("edge_threshold", 0.3)
        adj = build_adjacency(returns_window, window=window, threshold=threshold)

        self._model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(node_features).to(self._device)
            adj_t = torch.FloatTensor(adj).to(self._device)
            direction_t, conviction_t = self._model(x_t, adj_t)

        direction_np = direction_t.cpu().numpy()
        conviction_np = conviction_t.cpu().numpy()

        # Standardize direction (cross-sectional)
        d_mean = direction_np.mean()
        d_std = direction_np.std()
        if d_std > 0:
            direction = np.clip((direction_np - d_mean) / d_std, -3, 3)
        else:
            direction = np.zeros(n_stocks)

        return GraphSignal(
            direction=direction.astype(np.float32),
            conviction=conviction_np.astype(np.float32),
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights and metadata."""
        if not HAS_TORCH or self._model is None:
            logger.warning("No PyTorch model to save.")
            return

        gnn_cfg = self.cfg.get("gnn", {})
        state = {
            "model_state": self._model.state_dict(),
            "n_stocks": self._n_stocks,
            "d_input": self._d_input,
            "config": {
                "d_hidden": gnn_cfg.get("d_hidden", 32),
                "n_heads": gnn_cfg.get("n_heads", 4),
                "n_layers": gnn_cfg.get("n_layers", 2),
                "dropout": gnn_cfg.get("dropout", 0.1),
            },
            "is_trained": self._is_trained,
        }
        torch.save(state, path)
        logger.info("GNN saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available — cannot load GNN.")
            return

        state = torch.load(path, map_location=self._device)
        self._n_stocks = state["n_stocks"]
        self._d_input = state["d_input"]
        cfg = state["config"]

        self._model = GNNModel(
            d_input=self._d_input,
            d_hidden=cfg["d_hidden"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            n_stocks=self._n_stocks,
            dropout=cfg.get("dropout", 0.1),
        ).to(self._device)
        self._model.load_state_dict(state["model_state"])
        self._is_trained = state["is_trained"]
        logger.info("GNN loaded from %s", path)
