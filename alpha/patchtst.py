"""
alpha/patchtst.py
═════════════════
PatchTST (Patch Time Series Transformer) for long-range temporal encoding.

Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
       (Nie et al., 2023 — ICLR 2023)

Architecture overview:
  1. Channel-independent design: each stock's feature series is processed
     independently through the same transformer. This prevents cross-stock
     attention from dominating and lets the model learn universal temporal
     patterns (e.g., momentum decay, mean-reversion cycles).

  2. Patching: instead of attending to individual timesteps, we split the
     lookback window into fixed-length patches. This:
     - Reduces sequence length by patch_len/stride ×
     - Captures local semantic information within each patch
     - Dramatically improves computational efficiency

  3. Output: A temporal embedding vector per stock that captures long-range
     patterns over the context window (default 120 trading days ≈ 6 months).
     This embedding is:
     - Fed to the meta-model as a second stacking channel
     - Injected into the SAC observation space

Key design decisions:
  - Context window: 120 days (6 months) — captures medium-term momentum,
    earnings cycles, and seasonal patterns
  - Patch length: 5 days (one trading week) — natural market rhythm
  - Stride: 5 (non-overlapping patches) — 24 patches per sequence
  - d_model: 64 — compact embedding (financial data is noisy; smaller
    models regularise better)
  - Prediction: outputs both a forward return prediction (for the meta-model)
    and a temporal embedding (for the SAC observation)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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
        "PyTorch not installed. Install with: pip install torch. "
        "PatchTST will fall back to zero embeddings."
    )
    HAS_TORCH = False


# ── PatchTST Signal (output) ────────────────────────────────────────────────

@dataclass
class TemporalSignal:
    """
    Output of PatchTST for each stock at a given timestep.

    Attributes
    ----------
    embedding    : np.ndarray (n_stocks, embed_dim) — temporal context embedding
    direction    : np.ndarray (n_stocks,) — predicted return direction
    conviction   : np.ndarray (n_stocks,) — model confidence [0, 1]
    """
    embedding: np.ndarray
    direction: np.ndarray
    conviction: np.ndarray

    def to_obs_vector(self) -> np.ndarray:
        """Flatten for SAC observation: [direction | conviction] only.

        The raw embedding is NOT included — SAC can't learn to interpret
        high-dimensional transformer outputs directly. The embedding is
        still available via self.embedding for the meta-model pipeline.
        """
        return np.concatenate([
            self.direction,
            self.conviction,
        ]).astype(np.float32)

    @property
    def n_stocks(self) -> int:
        return len(self.direction)


# ── PyTorch model components ─────────────────────────────────────────────────

if HAS_TORCH:

    class PatchEmbedding(nn.Module):
        """
        Splits a time series into patches and projects to d_model.

        Input:  (batch, n_features, seq_len)
        Output: (batch, n_patches, d_model)
        """

        def __init__(
            self,
            n_features: int,
            patch_len: int,
            stride: int,
            d_model: int,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride

            # Linear projection: each patch of (n_features × patch_len) → d_model
            self.projection = nn.Linear(n_features * patch_len, d_model)
            self.dropout = nn.Dropout(dropout)

            # Learnable position encoding for patches
            max_patches = 512  # Support up to ~2500 day context
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_patches, d_model) * 0.02
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (batch, n_features, seq_len)
            returns: (batch, n_patches, d_model)
            """
            batch, n_feat, seq_len = x.shape

            # Unfold into patches: (batch, n_features, n_patches, patch_len)
            patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
            n_patches = patches.shape[2]

            # Reshape: (batch, n_patches, n_features * patch_len)
            patches = patches.permute(0, 2, 1, 3).reshape(batch, n_patches, -1)

            # Project to d_model
            out = self.projection(patches)
            out = out + self.pos_encoding[:, :n_patches, :]
            out = self.dropout(out)

            return out

    class TransformerEncoderBlock(nn.Module):
        """
        Standard transformer encoder block with pre-norm (more stable for
        financial time series where input distributions shift over time).
        """

        def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Pre-norm attention
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out

            # Pre-norm feedforward
            normed = self.norm2(x)
            x = x + self.ff(normed)

            return x

    class PatchTSTModel(nn.Module):
        """
        PatchTST: Channel-independent patch-based transformer for time series.

        Processes each stock independently through the same transformer weights.
        Outputs a temporal embedding and a forward return prediction per stock.

        Parameters
        ----------
        n_features     : number of input features per timestep
        context_len    : number of timesteps in the lookback window
        patch_len      : length of each patch (timesteps)
        stride         : stride between patches
        d_model        : transformer hidden dimension
        n_heads        : number of attention heads
        n_layers       : number of transformer encoder layers
        d_ff           : feedforward hidden dimension
        embed_dim      : output embedding dimension per stock
        dropout        : dropout rate
        """

        def __init__(
            self,
            n_features: int,
            context_len: int = 120,
            patch_len: int = 5,
            stride: int = 5,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 3,
            d_ff: int = 128,
            embed_dim: int = 16,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.n_features = n_features
            self.context_len = context_len
            self.patch_len = patch_len
            self.stride = stride
            self.d_model = d_model
            self.embed_dim = embed_dim

            # Number of patches
            self.n_patches = (context_len - patch_len) // stride + 1

            # Patch embedding
            self.patch_embed = PatchEmbedding(
                n_features=n_features,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                dropout=dropout,
            )

            # Transformer encoder stack
            self.encoder = nn.Sequential(*[
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])

            # Final layer norm
            self.norm = nn.LayerNorm(d_model)

            # Output heads
            # 1. Temporal embedding (for SAC observation)
            self.embed_head = nn.Linear(d_model, embed_dim)

            # 2. Forward return prediction (for meta-model)
            self.pred_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

            # 3. Uncertainty head (for conviction estimation)
            self.uncertainty_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Softplus(),  # Ensures positive uncertainty
            )

            self._init_weights()

        def _init_weights(self):
            """Xavier uniform initialisation for stable training."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Parameters
            ----------
            x : (batch, n_features, context_len) — channel-independent input

            Returns
            -------
            embedding   : (batch, embed_dim) — temporal context embedding
            prediction  : (batch, 1) — forward return prediction
            uncertainty : (batch, 1) — prediction uncertainty (positive)
            """
            # Patch + positional embedding
            patches = self.patch_embed(x)  # (batch, n_patches, d_model)

            # Transformer encoder
            encoded = self.encoder(patches)  # (batch, n_patches, d_model)
            encoded = self.norm(encoded)

            # Global average pooling over patches
            pooled = encoded.mean(dim=1)  # (batch, d_model)

            # Output heads
            embedding = self.embed_head(pooled)       # (batch, embed_dim)
            prediction = self.pred_head(pooled)        # (batch, 1)
            uncertainty = self.uncertainty_head(pooled)  # (batch, 1)

            return embedding, prediction, uncertainty


# ── High-level PatchTST wrapper ──────────────────────────────────────────────

class PatchTSTAlpha:
    """
    PatchTST alpha model — wrapper that handles training, prediction,
    and integration with the meta-model pipeline.

    Usage
    -----
    model = PatchTSTAlpha(config)
    metrics = model.train(sequences, labels)
    signal = model.predict(current_sequences)  # → TemporalSignal
    """

    def __init__(self, config: dict):
        self.cfg = config
        patchtst_cfg = config.get("patchtst", {})

        self.context_len = patchtst_cfg.get("context_len", 120)
        self.patch_len = patchtst_cfg.get("patch_len", 5)
        self.stride = patchtst_cfg.get("stride", 5)
        self.d_model = patchtst_cfg.get("d_model", 64)
        self.n_heads = patchtst_cfg.get("n_heads", 4)
        self.n_layers = patchtst_cfg.get("n_layers", 3)
        self.d_ff = patchtst_cfg.get("d_ff", 128)
        self.embed_dim = patchtst_cfg.get("embed_dim", 16)
        self.dropout = patchtst_cfg.get("dropout", 0.1)
        self.lr = patchtst_cfg.get("learning_rate", 1e-3)
        self.epochs = patchtst_cfg.get("epochs", 50)
        self.batch_size = patchtst_cfg.get("batch_size", 64)
        self.patience = patchtst_cfg.get("patience", 10)

        self.n_features: int = 0
        self._model = None
        self._device = "cpu"
        self._is_trained = False

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        val_sequences: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train PatchTST on windowed time series sequences.

        Parameters
        ----------
        sequences     : (n_samples, n_features, context_len)
        labels        : (n_samples,) forward returns
        val_sequences : optional validation sequences
        val_labels    : optional validation labels

        Returns
        -------
        Dict with training metrics {train_loss, val_loss, ic, best_epoch}
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available — PatchTST training skipped.")
            self._is_trained = True
            return {"train_loss": 0, "val_loss": 0, "ic": 0, "best_epoch": 0}

        n_samples, self.n_features, seq_len = sequences.shape
        assert seq_len == self.context_len, (
            f"Sequence length {seq_len} != context_len {self.context_len}"
        )

        # Auto-detect device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("PatchTST training on %s, %d samples", self._device, n_samples)

        # Build model
        self._model = PatchTSTModel(
            n_features=self.n_features,
            context_len=self.context_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
        ).to(self._device)

        # Log param count
        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info("PatchTST: %d parameters (%.1fK)", n_params, n_params / 1000)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        # Convert to tensors
        X_train = torch.FloatTensor(sequences).to(self._device)
        y_train = torch.FloatTensor(labels).to(self._device)

        if val_sequences is not None:
            X_val = torch.FloatTensor(val_sequences).to(self._device)
            y_val = torch.FloatTensor(val_labels).to(self._device)
        else:
            # Use last 20% as validation
            split = int(n_samples * 0.8)
            X_val = X_train[split:]
            y_val = y_train[split:]
            X_train = X_train[:split]
            y_train = y_train[:split]

        # Training loop
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.epochs):
            # ── Train ────────────────────────────────────────────────────
            self._model.train()
            train_losses = []

            # Shuffle
            perm = torch.randperm(len(X_train))
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_shuffled[i:i + self.batch_size]
                batch_y = y_shuffled[i:i + self.batch_size]

                optimizer.zero_grad()
                _, pred, uncertainty = self._model(batch_X)
                pred = pred.squeeze(-1)
                uncertainty = uncertainty.squeeze(-1)

                # Gaussian NLL loss: penalise both prediction error and
                # miscalibrated uncertainty
                loss = self._gaussian_nll_loss(pred, batch_y, uncertainty)
                loss.backward()

                # Gradient clipping (financial data has fat tails)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()

            # ── Validate ─────────────────────────────────────────────────
            self._model.eval()
            with torch.no_grad():
                _, val_pred, val_unc = self._model(X_val)
                val_pred = val_pred.squeeze(-1)
                val_unc = val_unc.squeeze(-1)
                val_loss = self._gaussian_nll_loss(val_pred, y_val, val_unc).item()

            avg_train_loss = np.mean(train_losses)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(
                    "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, lr=%.2e",
                    epoch, self.epochs, avg_train_loss, val_loss,
                    scheduler.get_last_lr()[0],
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                self._best_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d (best: %d)", epoch, best_epoch)
                    break

        # Restore best model
        if hasattr(self, "_best_state"):
            self._model.load_state_dict(self._best_state)
            self._model.to(self._device)

        # Compute IC on validation set
        self._model.eval()
        with torch.no_grad():
            _, val_pred, _ = self._model(X_val)
            val_pred_np = val_pred.squeeze(-1).cpu().numpy()
            val_y_np = y_val.cpu().numpy()

        ic = float(np.corrcoef(val_pred_np, val_y_np)[0, 1]) if len(val_pred_np) > 1 else 0

        self._is_trained = True

        logger.info(
            "PatchTST trained: best_val_loss=%.4f, IC=%.4f, best_epoch=%d",
            best_val_loss, ic, best_epoch,
        )
        return {
            "train_loss": avg_train_loss,
            "val_loss": best_val_loss,
            "ic": ic,
            "best_epoch": best_epoch,
            "n_params": n_params,
        }

    @staticmethod
    def _gaussian_nll_loss(
        pred: "torch.Tensor", target: "torch.Tensor", uncertainty: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Gaussian negative log-likelihood loss.
        Encourages the model to output calibrated uncertainty estimates.

        loss = 0.5 * (log(σ²) + (y - ŷ)² / σ²)
        """
        variance = uncertainty ** 2 + 1e-6  # Numerical stability
        loss = 0.5 * (torch.log(variance) + (target - pred) ** 2 / variance)
        return loss.mean()

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, sequences: np.ndarray) -> TemporalSignal:
        """
        Generate TemporalSignal for a batch of stock sequences.

        Parameters
        ----------
        sequences : (n_stocks, n_features, context_len)

        Returns
        -------
        TemporalSignal with embedding, direction, and conviction
        """
        n_stocks = sequences.shape[0]

        if not HAS_TORCH or not self._is_trained or self._model is None:
            return TemporalSignal(
                embedding=np.zeros((n_stocks, self.embed_dim), dtype=np.float32),
                direction=np.zeros(n_stocks, dtype=np.float32),
                conviction=np.full(n_stocks, 0.5, dtype=np.float32),
            )

        self._model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(sequences).to(self._device)
            embedding, pred, uncertainty = self._model(X)

            embedding_np = embedding.cpu().numpy()
            pred_np = pred.squeeze(-1).cpu().numpy()
            unc_np = uncertainty.squeeze(-1).cpu().numpy()

        # Standardise direction
        pred_mean = pred_np.mean()
        pred_std = pred_np.std()
        if pred_std > 0:
            direction = (pred_np - pred_mean) / pred_std
        else:
            direction = np.zeros(n_stocks)
        direction = np.clip(direction, -3, 3)

        # Conviction from uncertainty: lower uncertainty → higher conviction
        max_unc = unc_np.max() if unc_np.max() > 0 else 1.0
        conviction = np.clip(1.0 - unc_np / max_unc, 0.1, 1.0)

        return TemporalSignal(
            embedding=embedding_np.astype(np.float32),
            direction=direction.astype(np.float32),
            conviction=conviction.astype(np.float32),
        )

    def predict_single_step(
        self,
        sequence_data: Dict[str, np.ndarray],
    ) -> TemporalSignal:
        """
        Convenience: predict from a dict of {ticker: (n_features, context_len)}.
        """
        configured = self.cfg.get("market", {}).get("tickers", [])
        tickers = [ticker for ticker in configured if ticker in sequence_data]
        tickers.extend(
            ticker for ticker in sequence_data if ticker not in tickers
        )
        sequences = np.stack([sequence_data[t] for t in tickers])
        return self.predict(sequences)

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model weights and config."""
        if not HAS_TORCH or self._model is None:
            logger.warning("No PyTorch model to save.")
            return

        state = {
            "model_state": self._model.state_dict(),
            "n_features": self.n_features,
            "config": {
                "context_len": self.context_len,
                "patch_len": self.patch_len,
                "stride": self.stride,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "d_ff": self.d_ff,
                "embed_dim": self.embed_dim,
                "dropout": self.dropout,
            },
            "is_trained": self._is_trained,
        }
        torch.save(state, path)
        logger.info("PatchTST saved to %s", path)

    def load(self, path: str):
        """Load model weights."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available — cannot load PatchTST.")
            return

        state = torch.load(path, map_location=self._device)
        self.n_features = state["n_features"]
        cfg = state["config"]

        self._model = PatchTSTModel(
            n_features=self.n_features, **cfg
        ).to(self._device)
        self._model.load_state_dict(state["model_state"])
        self._is_trained = state["is_trained"]
        logger.info("PatchTST loaded from %s", path)
