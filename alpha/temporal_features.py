"""
alpha/temporal_features.py
══════════════════════════
Builds sliding-window sequences from OHLCV + indicator data for PatchTST input.

Responsibilities:
  1. Construct (n_features, context_len) matrices per stock per date
  2. Build training sets with forward return labels
  3. Handle edge cases: insufficient history, missing dates, padding
  4. Date-aligned sequence retrieval for real-time prediction

Design decisions:
  - Features used are the base 15 from data_loader + extended 20 from v3.1.
    This gives PatchTST a rich multivariate input (35 channels).
  - Sequences are NOT z-scored here — the data_loader already applies
    rolling z-score. PatchTST sees normalised features.
  - For dates near the start of the series (< context_len history),
    we zero-pad on the left. This is a natural choice since the transformer
    can learn to attend only to valid patches.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalFeatureBuilder:
    """
    Constructs windowed sequences for PatchTST from preprocessed stock data.

    Usage
    -----
    builder = TemporalFeatureBuilder(config, feature_names)
    sequences, labels, dates = builder.build_training_set(data, forward_horizon=5)
    seq = builder.get_sequence(data, ticker, date)  # single stock, single date
    """

    def __init__(self, config: dict, feature_names: List[str]):
        self.cfg = config
        self.feature_names = feature_names
        self.n_features = len(feature_names)

        patchtst_cfg = config.get("patchtst", {})
        self.context_len = patchtst_cfg.get("context_len", 120)

    # ── Training set construction ────────────────────────────────────────────

    def build_training_set(
        self,
        data: Dict[str, pd.DataFrame],
        forward_horizon: int = 5,
        min_history: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, pd.Timestamp]]]:
        """
        Build (sequences, labels) for PatchTST training.

        For each (ticker, date) pair where sufficient history exists,
        creates a feature sequence and a forward return label.

        Parameters
        ----------
        data             : dict of {ticker: DataFrame} with feature columns
        forward_horizon  : days ahead for return label
        min_history      : minimum history required (skip if less)

        Returns
        -------
        sequences : np.ndarray of shape (n_samples, n_features, context_len)
        labels    : np.ndarray of shape (n_samples,) — forward log returns
        meta      : list of (ticker, date) tuples for each sample
        """
        all_sequences = []
        all_labels = []
        all_meta = []

        for ticker, df in data.items():
            if "Close" not in df.columns:
                continue

            # Ensure feature columns exist
            avail_features = [f for f in self.feature_names if f in df.columns]
            if len(avail_features) < self.n_features * 0.5:
                logger.warning(
                    "%s: Only %d/%d features available — skipping.",
                    ticker, len(avail_features), self.n_features,
                )
                continue

            feature_matrix = df[self.feature_names].values  # (n_dates, n_features)
            close_prices = df["Close"].values
            dates = df.index

            n_dates = len(dates)

            # Forward returns for labels
            for i in range(min_history, n_dates - forward_horizon):
                # Build sequence: lookback window ending at date i
                seq = self._extract_sequence(feature_matrix, i)
                if seq is None:
                    continue

                # Forward return label
                fwd_ret = np.log(close_prices[i + forward_horizon] / close_prices[i])

                all_sequences.append(seq)
                all_labels.append(fwd_ret)
                all_meta.append((ticker, dates[i]))

        if not all_sequences:
            raise ValueError("No valid sequences built. Check data and feature availability.")

        sequences = np.stack(all_sequences)  # (n_samples, n_features, context_len)
        labels = np.array(all_labels)

        logger.info(
            "Temporal training set: %d sequences, %d features × %d timesteps, "
            "labels mean=%.4f std=%.4f",
            len(sequences), self.n_features, self.context_len,
            labels.mean(), labels.std(),
        )
        return sequences, labels, all_meta

    def build_validation_split(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        meta: List[Tuple[str, pd.Timestamp]],
        val_fraction: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into train/val by date across every ticker.

        Returns (X_train, y_train, X_val, y_val)
        """
        if len(meta) != len(sequences):
            raise ValueError("Temporal metadata must align one-to-one with sequences.")

        unique_dates = sorted({pd.Timestamp(date) for _, date in meta})
        if len(unique_dates) < 3:
            raise ValueError("Need at least three unique dates for temporal validation.")

        split_date_idx = max(
            1,
            min(
                int(len(unique_dates) * (1 - val_fraction)),
                len(unique_dates) - 1,
            ),
        )
        horizon = int(
            self.cfg.get("patchtst", {}).get("forward_horizon", 5)
        )
        train_end_idx = max(0, split_date_idx - horizon)
        train_dates = set(unique_dates[:train_end_idx])
        val_dates = set(unique_dates[split_date_idx:])

        train_idx = [
            i for i, (_, date) in enumerate(meta)
            if pd.Timestamp(date) in train_dates
        ]
        val_idx = [
            i for i, (_, date) in enumerate(meta)
            if pd.Timestamp(date) in val_dates
        ]
        if not train_idx or not val_idx:
            raise ValueError("Temporal date split produced an empty train or validation set.")

        # Keep deterministic chronological order within each partition.
        train_idx.sort(key=lambda i: (pd.Timestamp(meta[i][1]), meta[i][0]))
        val_idx.sort(key=lambda i: (pd.Timestamp(meta[i][1]), meta[i][0]))
        return (
            sequences[train_idx],
            labels[train_idx],
            sequences[val_idx],
            labels[val_idx],
        )

    # ── Single-step prediction support ───────────────────────────────────────

    def get_sequences_for_date(
        self,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> Dict[str, np.ndarray]:
        """
        Get the feature sequence ending at `date` for each stock.

        Returns dict of {ticker: (n_features, context_len)} arrays.
        Stocks with insufficient history get zero-padded sequences.
        """
        result = {}
        for ticker, df in data.items():
            if date not in df.index:
                # Zero sequence for missing dates
                result[ticker] = np.zeros(
                    (self.n_features, self.context_len), dtype=np.float32
                )
                continue

            date_idx = df.index.get_loc(date)
            feature_matrix = df[self.feature_names].values
            seq = self._extract_sequence(feature_matrix, date_idx)

            if seq is None:
                seq = np.zeros((self.n_features, self.context_len), dtype=np.float32)

            result[ticker] = seq

        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    def _extract_sequence(
        self, feature_matrix: np.ndarray, end_idx: int
    ) -> Optional[np.ndarray]:
        """
        Extract a (n_features, context_len) sequence ending at end_idx.

        Left-zero-pads if insufficient history.
        """
        n_dates, n_features = feature_matrix.shape
        start_idx = end_idx - self.context_len + 1

        if start_idx >= 0:
            # Full window available
            seq = feature_matrix[start_idx:end_idx + 1].T  # (n_features, context_len)
        else:
            # Partial window: zero-pad on the left
            available = feature_matrix[:end_idx + 1].T  # (n_features, available_len)
            pad_len = self.context_len - available.shape[1]
            seq = np.concatenate([
                np.zeros((n_features, pad_len), dtype=np.float32),
                available,
            ], axis=1)

        # Replace NaN with 0
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        return seq.astype(np.float32)
