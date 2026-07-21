"""
alpha/ — Multi-model alpha prediction layer (v3.1 Phase 1-3)
═════════════════════════════════════════════════════════════
Modules:
  feature_engineering  – 54 features: technical, cross-stock, statistical
  lightgbm_alpha       – LightGBM model with purged walk-forward CV
  patchtst             – PatchTST transformer for long-range temporal encoding
  temporal_features    – Sliding-window sequence builder for PatchTST
  meta_model           – Ridge stacking for multi-model ensemble
  gnn_alpha            – GAT graph neural network for cross-stock dependencies
  vae_regime           – VAE for market regime detection and embedding
"""

from alpha.feature_engineering import AlphaFeatureEngine
from alpha.lightgbm_alpha import LightGBMAlpha
from alpha.patchtst import PatchTSTAlpha, TemporalSignal
from alpha.temporal_features import TemporalFeatureBuilder
from alpha.meta_model import RidgeMetaModel

try:
    from alpha.gnn_alpha import GNNAlpha, GraphSignal
except ImportError:
    pass  # PyTorch not available

try:
    from alpha.vae_regime import RegimeDetector, RegimeSignal
except ImportError:
    pass  # PyTorch not available

try:
    from alpha.adversarial_filter import AdversarialFilter
except ImportError:
    pass

__all__ = [
    "AlphaFeatureEngine", "LightGBMAlpha",
    "PatchTSTAlpha", "TemporalSignal", "TemporalFeatureBuilder",
    "RidgeMetaModel",
    "GNNAlpha", "GraphSignal",
    "RegimeDetector", "RegimeSignal",
    "AdversarialFilter",
]
