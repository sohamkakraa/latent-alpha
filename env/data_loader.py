"""
env/data_loader.py
──────────────────
Downloads and preprocesses historical NSE OHLCV data via yfinance.
Computes all technical indicators and returns normalised feature matrices
ready to be consumed by the trading environment.

Key design decisions:
  - Data is cached locally to avoid repeated API calls during training.
  - Indicators are computed once and stored alongside raw prices.
  - Normalisation uses rolling z-score (mean/std over lookback window)
    rather than global min-max, so future data cannot leak into past observations.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import ta

logger = logging.getLogger(__name__)


# ── v3.1 Extended feature names (added to the original 15) ──────────────────
# These are computed by compute_extended_indicators() and appended alongside
# the original features. The alpha model uses its own 50+ feature set via
# AlphaFeatureEngine, but the SAC observation also benefits from richer signals.

EXTENDED_FEATURE_NAMES = [
    # Additional momentum
    "ret_2d", "ret_10d", "ret_60d", "roc_5", "roc_20",
    "williams_r", "stoch_k", "stoch_d",
    # Additional volatility
    "vol_ratio_5_20", "parkinson_vol",
    # Volume anomalies
    "obv_slope", "vwap_dist", "volume_breakout",
    # Trend
    "adx_14", "cci_20", "trend_strength",
    # Mean reversion
    "rsi_5", "dist_from_52w_high",
    # Rolling stats
    "skewness_20", "kurtosis_20",
]


# ── Indicator computation ─────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Append technical indicators to a raw OHLCV dataframe.

    Parameters
    ----------
    df  : DataFrame with columns [Open, High, Low, Close, Volume]
    cfg : indicators section from config.yaml

    Returns
    -------
    DataFrame with additional indicator columns (NaN rows dropped)
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    volume = df["Volume"]

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(
        close, window=cfg.get("rsi_period", 14)
    ).rsi()

    # MACD
    macd_obj = ta.trend.MACD(
        close,
        window_slow=cfg.get("macd_slow", 26),
        window_fast=cfg.get("macd_fast", 12),
        window_sign=cfg.get("macd_signal", 9),
    )
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()

    # Bollinger Bands
    bb_obj = ta.volatility.BollingerBands(
        close,
        window=cfg.get("bb_period", 20),
        window_dev=cfg.get("bb_std", 2.0),
    )
    df["bb_upper"]  = bb_obj.bollinger_hband()
    df["bb_lower"]  = bb_obj.bollinger_lband()
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / close
    df["bb_pct"]    = bb_obj.bollinger_pband()   # % position within bands

    # ATR (normalised by close)
    df["atr"] = ta.volatility.AverageTrueRange(
        high, low, close, window=cfg.get("atr_period", 14)
    ).average_true_range() / close

    # EMAs
    for period in cfg.get("ema_periods", [9, 21, 50]):
        ema = ta.trend.EMAIndicator(close, window=period).ema_indicator()
        df[f"ema_{period}"] = (ema - close) / close   # Normalised distance from price

    # Volume z-score (rolling 20-period)
    vol_mean = volume.rolling(20).mean()
    vol_std  = volume.rolling(20).std().replace(0, 1)
    df["volume_z"] = (volume - vol_mean) / vol_std

    # Returns (log)
    df["log_return"]    = np.log(close / close.shift(1))
    df["return_5d"]     = np.log(close / close.shift(5))
    df["return_20d"]    = np.log(close / close.shift(20))
    df["volatility_20"] = df["log_return"].rolling(20).std()

    df.dropna(inplace=True)
    return df


def rolling_zscore(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Normalise all numeric columns using a rolling z-score.
    Uses only past data — no look-ahead bias.
    Price columns (Open/High/Low/Close/Volume) are excluded from this
    normalisation since they are not directly fed to the agent; only
    derived features are.
    """
    exclude = {"Open", "High", "Low", "Close", "Volume"}
    feature_cols = [c for c in df.columns if c not in exclude]

    means = df[feature_cols].rolling(window, min_periods=1).mean()
    stds  = df[feature_cols].rolling(window, min_periods=1).std().replace(0, 1)

    df[feature_cols] = (df[feature_cols] - means) / stds
    return df


def compute_extended_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Append v3.1 extended technical indicators to a DataFrame that already
    has the base indicators from compute_indicators().

    These additional 20 features enrich the SAC observation space and are
    also used by the alpha model pipeline.

    Parameters
    ----------
    df  : DataFrame with OHLCV + base indicator columns
    cfg : indicators section from config.yaml

    Returns
    -------
    DataFrame with extended indicator columns appended
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    log_ret = np.log(close / close.shift(1))

    # ── Additional momentum ──────────────────────────────────────────────
    df["ret_2d"] = np.log(close / close.shift(2))
    df["ret_10d"] = np.log(close / close.shift(10))
    df["ret_60d"] = np.log(close / close.shift(60))
    df["roc_5"] = (close - close.shift(5)) / close.shift(5)
    df["roc_20"] = (close - close.shift(20)) / close.shift(20)

    df["williams_r"] = ta.momentum.WilliamsRIndicator(
        high, low, close, lbp=14
    ).williams_r() / 100.0

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch() / 100.0
    df["stoch_d"] = stoch.stoch_signal() / 100.0

    # ── Additional volatility ────────────────────────────────────────────
    vol_5d = log_ret.rolling(5).std()
    vol_20d = log_ret.rolling(20).std()
    # Retain canonical names for downstream GNN/VAE channels even though
    # these helper columns are not part of the base SAC feature list.
    df["vol_5d"] = vol_5d
    df["vol_20d"] = vol_20d
    df["vol_ratio_5_20"] = vol_5d / vol_20d.replace(0, np.nan)

    log_hl = np.log(high / low)
    df["parkinson_vol"] = log_hl.rolling(20).apply(
        lambda x: np.sqrt((1 / (4 * np.log(2))) * np.mean(x**2)),
        raw=True,
    )

    # ── Volume anomalies ─────────────────────────────────────────────────
    vol_mean_20 = volume.rolling(20).mean()

    obv = (np.sign(log_ret) * volume).cumsum()
    df["obv_slope"] = obv.rolling(5).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 5 else 0,
        raw=True,
    ) / vol_mean_20.replace(0, 1)

    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).rolling(20).sum()
    cum_vol = volume.rolling(20).sum().replace(0, 1)
    vwap = cum_tp_vol / cum_vol
    df["vwap_dist"] = (close - vwap) / vwap

    df["volume_breakout"] = (volume > 2 * vol_mean_20).astype(float)

    # ── Trend ────────────────────────────────────────────────────────────
    adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx_14"] = adx_obj.adx() / 100.0

    df["cci_20"] = ta.trend.CCIIndicator(
        high, low, close, window=20
    ).cci() / 200.0

    ema_50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["trend_strength"] = df["adx_14"] * np.sign(close - ema_50)

    # ── Mean reversion ───────────────────────────────────────────────────
    df["rsi_5"] = ta.momentum.RSIIndicator(close, window=5).rsi() / 100.0

    high_52w = close.rolling(252, min_periods=60).max()
    df["dist_from_52w_high"] = (close - high_52w) / high_52w

    # ── Rolling statistical ──────────────────────────────────────────────
    df["skewness_20"] = log_ret.rolling(20).skew()
    df["kurtosis_20"] = log_ret.rolling(20).kurt()

    return df


# ── Data loading ──────────────────────────────────────────────────────────────

class NSEDataLoader:
    """
    Downloads, caches, and preprocesses NSE stock data for the trading
    environment.

    Usage
    -----
    loader = NSEDataLoader(config)
    data   = loader.load(start="2018-01-01", end="2024-01-01")
    # data is a dict: {ticker: pd.DataFrame}
    """

    def __init__(self, config: dict):
        self.cfg        = config
        self.tickers    = config["market"]["tickers"]
        self.ind_cfg    = config["indicators"]
        self.cache_dir  = Path(config["paths"]["data_cache"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # v3.1: Enable extended features if alpha config is present
        self.use_extended = config.get("alpha", {}).get("enabled", False)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        start: str,
        end: str,
        term: str = "medium",
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return preprocessed OHLCV + indicator DataFrames for all tickers.

        Parameters
        ----------
        start     : ISO date string, e.g. "2018-01-01"
        end       : ISO date string, e.g. "2024-01-01"
        term      : "short" | "medium" | "long" — determines bar interval
        use_cache : Load from disk cache if available

        Returns
        -------
        dict mapping ticker → DataFrame (cleaned, indicators computed, normalised)
        """
        interval = self.cfg["terms"][term]["interval"]
        cache_key = self._cache_key(start, end, interval)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        if use_cache and cache_path.exists():
            logger.info("Loading data from cache: %s", cache_path)
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        def _fetch_one(ticker: str) -> Optional[pd.DataFrame]:
            df = self._download(ticker, start, end, interval)
            if df is None or len(df) < 50:
                return None
            df = compute_indicators(df, self.ind_cfg)
            if self.use_extended:
                df = compute_extended_indicators(df, self.ind_cfg)
            return rolling_zscore(df)

        data: Dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            try:
                df = _fetch_one(ticker)
                if df is None:
                    logger.warning("Insufficient data for %s — will retry.", ticker)
                    continue
                data[ticker] = df
                logger.info("Loaded %s: %d bars", ticker, len(df))
            except Exception as exc:
                logger.error("Failed to load %s: %s", ticker, exc)

        # Retry missing tickers. yfinance fetches are flaky; a transient miss
        # silently shrinks the universe, which changes n_stocks → obs_dim and
        # corrupts train/eval shape alignment (a dropped ticker crashed a fold's
        # eval with VecNormalize shape mismatch 492 != 398).
        missing = [t for t in self.tickers if t not in data]
        for attempt in range(3):
            if not missing:
                break
            logger.warning(
                "Retrying %d missing ticker(s), attempt %d/3: %s",
                len(missing), attempt + 1, missing,
            )
            time.sleep(2 * (attempt + 1))
            for ticker in list(missing):
                try:
                    df = _fetch_one(ticker)
                    if df is not None:
                        data[ticker] = df
                        logger.info("Recovered %s: %d bars", ticker, len(df))
                except Exception as exc:
                    logger.error("Retry failed for %s: %s", ticker, exc)
            missing = [t for t in self.tickers if t not in data]

        if missing:
            # Never cache or return a partial universe — a shrunk n_stocks yields
            # a mismatched obs_dim that crashes VecNormalize at eval time.
            raise RuntimeError(
                f"Incomplete universe after retries: missing {missing}. "
                f"Refusing to cache/return a partial dataset (would corrupt obs_dim)."
            )

        if not data:
            raise RuntimeError("No data loaded — check tickers and date range.")

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Data cached to %s", cache_path)

        return data

    def align_dates(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Trim all ticker DataFrames to the intersection of their date indices
        so every ticker has the same number of rows. Required for the
        multi-stock environment step logic.
        """
        common = None
        for df in data.values():
            idx = set(df.index)
            common = idx if common is None else common & idx
        common = sorted(common)
        return {ticker: df.loc[common] for ticker, df in data.items()}

    def load_benchmark(
        self,
        start: str,
        end: str,
        term: str = "medium",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load raw adjusted OHLCV for the configured benchmark index."""
        ticker = self.cfg["market"]["benchmark"]
        interval = self.cfg["terms"][term]["interval"]
        raw = f"benchmark_{ticker}_{start}_{end}_{interval}"
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        path = self.cache_dir / f"{key}_benchmark.pkl"
        if use_cache and path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        frame = self._download(ticker, start, end, interval)
        if frame is None or frame.empty:
            raise RuntimeError(
                f"No benchmark data for {ticker} in {start}–{end}."
            )
        with open(path, "wb") as f:
            pickle.dump(frame, f)
        return frame

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of per-stock feature columns fed to the agent."""
        base = [
            "log_return", "return_5d", "return_20d", "volatility_20",
            "rsi", "macd", "macd_signal", "macd_diff",
            "bb_width", "bb_pct", "atr",
            "volume_z",
        ] + [f"ema_{p}" for p in self.ind_cfg.get("ema_periods", [9, 21, 50])]
        if self.use_extended:
            base = base + EXTENDED_FEATURE_NAMES
        return base

    @staticmethod
    def fingerprint(data: Dict[str, pd.DataFrame]) -> str:
        """Content hash the market panel used by a fold."""
        digest = hashlib.sha256()
        for ticker in sorted(data):
            frame = data[ticker]
            columns = [
                column
                for column in ("Open", "High", "Low", "Close", "Volume")
                if column in frame.columns
            ]
            digest.update(ticker.encode("utf-8"))
            digest.update("|".join(columns).encode("utf-8"))
            hashed = pd.util.hash_pandas_object(
                frame[columns],
                index=True,
                categorize=False,
            )
            digest.update(hashed.to_numpy(dtype=np.uint64).tobytes())
        return digest.hexdigest()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _download(
        self, ticker: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Download raw OHLCV from Yahoo Finance."""
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df

    def _cache_key(self, start: str, end: str, interval: str) -> str:
        tickers_str = "_".join(sorted(self.tickers))
        raw = f"{tickers_str}_{start}_{end}_{interval}"
        base_hash = hashlib.md5(raw.encode()).hexdigest()[:12]

        # Include feature-engineering config so cache auto-invalidates when
        # indicator/feature settings change for the same date range.
        feature_cfg = {
            "feature_schema_version": 2,
            "use_extended": self.use_extended,
            "ema_periods": self.ind_cfg.get("ema_periods", []),
            "indicators": self.ind_cfg,
        }
        feature_hash = hashlib.md5(
            json.dumps(feature_cfg, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{base_hash}_{feature_hash}"
