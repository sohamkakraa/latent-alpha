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

from __future__ import annotations

import logging
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
        "PyTorch not installed. RegimeDetector will produce zero embeddings."
    )
    HAS_TORCH = False

REGIME_LABELS = ["bull_trending", "bear_trending", "high_vol_choppy", "low_vol_range"]

# Per-stock features extracted from market data (7 dims per stock). Each tuple
# lists equivalent names from the alpha panel and the base NSEDataLoader panel.
_STOCK_FEATURES = [
    ("ret_1d", "log_return"),
    ("ret_5d", "return_5d"),
    ("ret_20d", "return_20d"),
    ("vol_5d",),
    ("vol_20d", "volatility_20"),
    ("rsi_14", "rsi"),
    ("macd_hist", "macd_diff"),
]


# ── RegimeSignal (output) ─────────────────────────────────────────────────────

@dataclass
class RegimeSignal:
    """
    Market regime signal from the VAE.

    Attributes
    ----------
    embedding    : np.ndarray (d_latent,) — continuous regime embedding
    regime_probs : np.ndarray (n_regimes,) — soft regime probabilities
    regime_label : str — most likely regime name
    """

    embedding: np.ndarray
    regime_probs: np.ndarray
    regime_label: str

    def to_obs_vector(self) -> np.ndarray:
        """Return latent embedding for SAC observation."""
        return self.embedding.astype(np.float32)


# ── PyTorch model components ──────────────────────────────────────────────────

if HAS_TORCH:

    class VAEEncoder(nn.Module):
        """Input → hidden → (mu, logvar)."""

        def __init__(self, d_input: int = 74, d_hidden: int = 64, d_latent: int = 8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_input, d_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_hidden, 32),
                nn.ReLU(),
            )
            self.mu_head = nn.Linear(32, d_latent)
            self.logvar_head = nn.Linear(32, d_latent)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            h = self.net(x)
            return self.mu_head(h), self.logvar_head(h)

    class VAEDecoder(nn.Module):
        """z → hidden → reconstructed input."""

        def __init__(self, d_latent: int = 8, d_hidden: int = 64, d_output: int = 74):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_latent, 32),
                nn.ReLU(),
                nn.Linear(32, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_output),
            )

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.net(z)

    class RegimeVAE(nn.Module):
        """
        Full VAE: encoder + decoder + regime classifier head.

        forward(x) → (x_hat, mu, logvar, regime_probs)
        """

        def __init__(
            self,
            d_input: int = 74,
            d_hidden: int = 64,
            d_latent: int = 8,
            n_regimes: int = 4,
        ):
            super().__init__()
            self.encoder = VAEEncoder(d_input, d_hidden, d_latent)
            self.decoder = VAEDecoder(d_latent, d_hidden, d_input)
            self.regime_head = nn.Linear(d_latent, n_regimes)  # outputs logits
            self.d_latent = d_latent
            self.n_regimes = n_regimes

        def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            return self.encoder(x)

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.decoder(z)

        def reparameterize(
            self, mu: "torch.Tensor", logvar: "torch.Tensor"
        ) -> "torch.Tensor":
            if self.training:
                std = torch.exp(0.5 * logvar)
                return mu + std * torch.randn_like(std)
            return mu

        def forward(
            self, x: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_hat = self.decode(z)
            regime_logits = self.regime_head(z)
            regime_probs = F.softmax(regime_logits, dim=-1)
            return x_hat, mu, logvar, regime_probs

        def get_regime(self, x_np: np.ndarray) -> "RegimeSignal":
            """Single-sample inference → RegimeSignal."""
            self.eval()
            with torch.no_grad():
                x_t = torch.FloatTensor(x_np)
                if x_t.dim() == 1:
                    x_t = x_t.unsqueeze(0)
                x_hat, mu, logvar, regime_probs = self.forward(x_t)
                embedding = mu.squeeze(0).cpu().numpy()
                probs = regime_probs.squeeze(0).cpu().numpy()

            label = REGIME_LABELS[int(np.argmax(probs))]
            return RegimeSignal(
                embedding=embedding.astype(np.float32),
                regime_probs=probs.astype(np.float32),
                regime_label=label,
            )


# ── Market state builder ──────────────────────────────────────────────────────

def build_market_state(
    data: Dict,
    date,
    tickers: List[str],
) -> np.ndarray:
    """
    Build 74-dim market state vector: 10 stocks × 7 features + 4 cross-stock.

    Parameters
    ----------
    data    : dict {ticker: DataFrame} with feature columns
    date    : the date index to extract
    tickers : ordered list of ticker symbols

    Returns
    -------
    state : (74,) float32 array
    """
    per_stock = []
    rets_1d = []
    vols_20d = []

    for ticker in tickers:
        df = data.get(ticker)
        if df is not None and date in df.index:
            row = df.loc[date]
            feats = []
            for aliases in _STOCK_FEATURES:
                feature_name = next(
                    (name for name in aliases if name in df.columns),
                    None,
                )
                v = (
                    row.get(feature_name, 0.0)
                    if feature_name is not None and hasattr(row, "get")
                    else 0.0
                )
                value = float(v) if v is not None else 0.0
                feats.append(value if np.isfinite(value) else 0.0)
            per_stock.extend(feats)
            rets_1d.append(feats[0])   # ret_1d
            vols_20d.append(feats[4])  # vol_20d
        else:
            per_stock.extend([0.0] * 7)
            rets_1d.append(0.0)
            vols_20d.append(0.0)

    rets_arr = np.array(rets_1d, dtype=np.float32)
    vols_arr = np.array(vols_20d, dtype=np.float32)

    mean_ret = float(np.nanmean(rets_arr))
    ret_dispersion = float(np.nanstd(rets_arr))

    # Rolling cross-stock correlation mean (last 20 days)
    correlation_mean = 0.0
    try:
        import pandas as pd
        rets_cols = []
        for ticker in tickers:
            df = data.get(ticker)
            if df is not None and date in df.index and "ret_1d" in df.columns:
                idx_loc = df.index.get_loc(date)
                if isinstance(idx_loc, slice):
                    idx_loc = idx_loc.stop - 1
                elif hasattr(idx_loc, "__len__"):
                    idx_loc = int(np.where(idx_loc)[0][-1])
                start = max(0, idx_loc - 20)
                col = df["ret_1d"].iloc[start:idx_loc + 1].values
                col = col[~np.isnan(col)]
                if len(col) > 2:
                    rets_cols.append(col)
        if len(rets_cols) >= 2:
            min_len = min(len(c) for c in rets_cols)
            rets_matrix = np.array([c[-min_len:] for c in rets_cols])
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(rets_matrix)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            n = len(rets_cols)
            mask = ~np.eye(n, dtype=bool)
            if mask.any():
                correlation_mean = float(np.mean(corr_matrix[mask]))
    except Exception:
        correlation_mean = 0.0

    vol_of_vol = float(np.nanstd(vols_arr[vols_arr > 0])) if np.any(vols_arr > 0) else 0.0

    cross_stock = np.array(
        [mean_ret, ret_dispersion, correlation_mean, vol_of_vol], dtype=np.float32
    )

    state = np.concatenate([np.array(per_stock, dtype=np.float32), cross_stock])
    return state.astype(np.float32)


# ── Pseudo-label generator ────────────────────────────────────────────────────

def generate_pseudo_labels(data: Dict, tickers: List[str]) -> Dict:
    """
    Generate regime pseudo-labels from rolling market metrics.

    Regime assignment (priority order):
      2 (high_vol_choppy) : vol_20d > 75th percentile
      3 (low_vol_range)   : vol_20d < 25th percentile AND |ret_20d| < 1%
      0 (bull_trending)   : ret_20d > 0 AND vol < median AND trend > 0
      1 (bear_trending)   : ret_20d < 0 AND vol < median AND trend < 0
      default             : 0 if ret_20d >= 0 else 1

    Returns
    -------
    dict {date: int} where int ∈ {0, 1, 2, 3}
    """
    # Find common dates
    common_dates = None
    for ticker in tickers:
        if ticker in data:
            dates = set(data[ticker].index)
            common_dates = dates if common_dates is None else common_dates & dates
    if not common_dates:
        return {}
    common_dates = sorted(common_dates)

    # Collect aggregate metrics per date
    all_ret20 = []
    all_vol20 = []
    all_trend = []

    for date in common_dates:
        rets, vols, trends = [], [], []
        for ticker in tickers:
            df = data.get(ticker)
            if df is None or date not in df.index:
                continue
            row = df.loc[date]
            for aliases, arr in [
                (("ret_20d", "return_20d"), rets),
                (("vol_20d", "volatility_20"), vols),
                (("trend_strength",), trends),
            ]:
                feature_name = next(
                    (name for name in aliases if name in df.columns),
                    None,
                )
                v = (
                    row.get(feature_name, 0.0)
                    if feature_name is not None and hasattr(row, "get")
                    else 0.0
                )
                value = float(v) if v is not None else 0.0
                arr.append(value if np.isfinite(value) else 0.0)
        all_ret20.append(float(np.nanmean(rets)) if rets else 0.0)
        all_vol20.append(float(np.nanmean(vols)) if vols else 0.0)
        all_trend.append(float(np.nanmean(trends)) if trends else 0.0)

    all_ret20 = np.array(all_ret20)
    all_vol20 = np.array(all_vol20)
    all_trend = np.array(all_trend)

    valid_vol = all_vol20[all_vol20 > 0]
    if len(valid_vol) == 0:
        return {date: 0 for date in common_dates}

    median_vol = float(np.median(valid_vol))
    vol_75 = float(np.percentile(valid_vol, 75))
    vol_25 = float(np.percentile(valid_vol, 25))

    labels = {}
    for i, date in enumerate(common_dates):
        ret20 = all_ret20[i]
        vol20 = all_vol20[i]
        trend = all_trend[i]

        if vol20 > vol_75:
            label = 2  # high_vol_choppy
        elif vol20 < vol_25 and abs(ret20) < 0.01:
            label = 3  # low_vol_range
        elif ret20 > 0 and vol20 < median_vol and trend > 0:
            label = 0  # bull_trending
        elif ret20 < 0 and vol20 < median_vol and trend < 0:
            label = 1  # bear_trending
        else:
            label = 0 if ret20 >= 0 else 1

        labels[date] = label

    return labels


# ── High-level RegimeDetector wrapper ────────────────────────────────────────

class RegimeDetector:
    """
    VAE-based market regime detector.

    Usage
    -----
    detector = RegimeDetector(config)
    metrics = detector.train(X, pseudo_labels)
    signal = detector.predict(market_state)  # → RegimeSignal
    """

    def __init__(self, config: dict):
        self.cfg = config
        vae_cfg = config.get("vae_regime", {})

        self.d_input = vae_cfg.get("d_input", 74)
        self.d_hidden = vae_cfg.get("d_hidden", 64)
        self.d_latent = vae_cfg.get("d_latent", 8)
        self.n_regimes = vae_cfg.get("n_regimes", 4)
        self.lr = vae_cfg.get("learning_rate", 1e-3)
        self.epochs = vae_cfg.get("epochs", 50)
        self.batch_size = vae_cfg.get("batch_size", 128)
        self.patience = vae_cfg.get("patience", 10)
        self.beta_start = vae_cfg.get("beta_start", 0.0)
        self.beta_end = vae_cfg.get("beta_end", 1.0)
        self.beta_anneal_epochs = vae_cfg.get("beta_anneal_epochs", 10)
        self.gamma = vae_cfg.get("gamma", 0.5)

        self._model = None
        self._is_trained = False
        self._device = "cpu"
        self._best_state: Optional[dict] = None

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train VAE on market state vectors with regime pseudo-labels.

        Parameters
        ----------
        X      : (n_samples, d_input) market state vectors
        labels : (n_samples,) integer pseudo-labels {0,1,2,3}

        Returns
        -------
        Dict with recon_loss, kl_loss, regime_loss, best_epoch
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available — VAE training skipped.")
            self._is_trained = True
            return {"recon_loss": 0.0, "kl_loss": 0.0, "regime_loss": 0.0, "best_epoch": 0}

        n_samples = len(X)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = RegimeVAE(
            d_input=self.d_input,
            d_hidden=self.d_hidden,
            d_latent=self.d_latent,
            n_regimes=self.n_regimes,
        ).to(self._device)

        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info("VAE: %d parameters (%.1fK)", n_params, n_params / 1000)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        # Chronological 80/20 split
        split = int(n_samples * 0.8)
        X_train = torch.FloatTensor(X[:split]).to(self._device)
        X_val = torch.FloatTensor(X[split:]).to(self._device)
        labels_train = torch.LongTensor(labels[:split]).to(self._device)
        labels_val = torch.LongTensor(labels[split:]).to(self._device)

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        avg_recon = avg_kl = avg_regime = 0.0

        for epoch in range(self.epochs):
            # Beta annealing
            beta = self.beta_start + (self.beta_end - self.beta_start) * min(
                epoch / max(1, self.beta_anneal_epochs), 1.0
            )

            # ── Train ─────────────────────────────────────────────────────
            self._model.train()
            perm = torch.randperm(len(X_train))
            epoch_recon = epoch_kl = epoch_regime = 0.0
            n_batches = 0

            for i in range(0, len(X_train), self.batch_size):
                idx = perm[i:i + self.batch_size]
                x_batch = X_train[idx]
                lb_batch = labels_train[idx]

                optimizer.zero_grad()
                x_hat, mu, logvar, regime_probs = self._model(x_batch)

                recon_loss = F.mse_loss(x_hat, x_batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                regime_loss = F.nll_loss(
                    torch.log(regime_probs.clamp(min=1e-8)), lb_batch
                )

                total = recon_loss + beta * kl_loss + self.gamma * regime_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                epoch_regime += regime_loss.item()
                n_batches += 1

            if n_batches > 0:
                avg_recon = epoch_recon / n_batches
                avg_kl = epoch_kl / n_batches
                avg_regime = epoch_regime / n_batches

            # ── Validate ──────────────────────────────────────────────────
            self._model.eval()
            with torch.no_grad():
                xh, mu_v, lv_v, rp_v = self._model(X_val)
                val_recon = F.mse_loss(xh, X_val).item()
                val_kl = (-0.5 * torch.mean(1 + lv_v - mu_v.pow(2) - lv_v.exp())).item()
                val_loss = val_recon + beta * val_kl

            scheduler.step()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(
                    "VAE epoch %d/%d: recon=%.4f kl=%.4f regime=%.4f beta=%.2f",
                    epoch, self.epochs, avg_recon, avg_kl, avg_regime, beta,
                )

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
                    logger.info("VAE early stopping at epoch %d (best: %d)", epoch, best_epoch)
                    break

        if self._best_state is not None:
            self._model.load_state_dict(self._best_state)
            self._model.to(self._device)

        self._is_trained = True
        logger.info(
            "VAE trained: best_val_loss=%.4f, best_epoch=%d", best_val_loss, best_epoch
        )

        return {
            "recon_loss": avg_recon,
            "kl_loss": avg_kl,
            "regime_loss": avg_regime,
            "best_epoch": best_epoch,
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, market_state: np.ndarray) -> RegimeSignal:
        """
        Encode market state into a RegimeSignal.

        Parameters
        ----------
        market_state : (d_input,) float32 array

        Returns
        -------
        RegimeSignal
        """
        if not HAS_TORCH or not self._is_trained or self._model is None:
            return RegimeSignal(
                embedding=np.zeros(self.d_latent, dtype=np.float32),
                regime_probs=np.full(self.n_regimes, 1.0 / self.n_regimes, dtype=np.float32),
                regime_label="bull_trending",
            )
        return self._model.get_regime(market_state)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights."""
        if not HAS_TORCH or self._model is None:
            return
        state = {
            "model_state": self._model.state_dict(),
            "config": {
                "d_input": self.d_input,
                "d_hidden": self.d_hidden,
                "d_latent": self.d_latent,
                "n_regimes": self.n_regimes,
            },
            "is_trained": self._is_trained,
        }
        torch.save(state, path)
        logger.info("VAE saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights."""
        if not HAS_TORCH:
            return
        state = torch.load(path, map_location=self._device)
        cfg = state["config"]
        self._model = RegimeVAE(**cfg).to(self._device)
        self._model.load_state_dict(state["model_state"])
        self._is_trained = state["is_trained"]
        logger.info("VAE loaded from %s", path)
