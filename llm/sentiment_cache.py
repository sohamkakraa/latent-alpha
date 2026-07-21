"""
llm/sentiment_cache.py
───────────────────────
SQLite-backed persistent cache for historical sentiment vectors.

Every sentiment vector computed — whether by Claude or a local model —
is stored here keyed by (date, model). Training lookups hit the cache
first; the LLM is only called for cache misses.

Schema
------
  sentiment_cache
    date         TEXT     — ISO date, e.g. "2020-03-23"
    model        TEXT     — model that produced this vector
    vector_json  TEXT     — JSON array of floats, length n_tickers * sentiment_dim
    source_count INTEGER  — number of news articles used
    cost_usd     REAL     — API cost for this call (0.0 for local models)
    created_at   TEXT     — ISO datetime when this row was inserted

  cost_ledger
    date         TEXT
    model        TEXT
    cost_usd     REAL
    created_at   TEXT

The cache is stored in data/sentiment_cache.db — commit this file to
your repo (or back it up separately) so you never pay to recompute it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# USD → INR rate used for budget display (update if needed)
USD_INR_RATE = 84.0


class CacheValidationError(RuntimeError):
    """Raised when sentiment data is unsafe for an experiment."""


class SentimentCache:
    """
    Persistent SQLite cache for sentiment vectors.

    Parameters
    ----------
    db_path     : path to the SQLite file (created if it doesn't exist)
    budget_inr  : hard spend cap in INR — precompute will stop if exceeded
    """

    def __init__(
        self,
        db_path: str | Path = "data/sentiment_cache.db",
        budget_inr: float = 2000.0,
    ):
        self.db_path    = Path(db_path)
        self.budget_inr = budget_inr
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        date: str,
        model: str = "any",
        vector_dim: int | None = None,
        schema_version: int | None = None,
    ) -> Optional[np.ndarray]:
        """
        Retrieve a cached sentiment vector for a given date.

        Parameters
        ----------
        date  : ISO date string, e.g. "2020-03-23"
        model : if "any", return the first available vector for that date

        Returns
        -------
        np.ndarray or None if not cached
        """
        with self._conn() as conn:
            if model == "any":
                row = conn.execute(
                    """SELECT vector_json, vector_dim, schema_version
                       FROM sentiment_cache WHERE date = ? LIMIT 1""",
                    (date,),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT vector_json, vector_dim, schema_version
                       FROM sentiment_cache WHERE date = ? AND model = ?""",
                    (date, model),
                ).fetchone()

        if row is None:
            return None
        vector = np.array(json.loads(row[0]), dtype=np.float32)
        stored_dim = int(row[1] or len(vector))
        stored_schema = int(row[2] or 1)
        if vector_dim is not None and (
            stored_dim != vector_dim or len(vector) != vector_dim
        ):
            raise CacheValidationError(
                f"Sentiment vector for {date}/{model} has dimension "
                f"{len(vector)} (metadata={stored_dim}), expected {vector_dim}."
            )
        if schema_version is not None and stored_schema != schema_version:
            raise CacheValidationError(
                f"Sentiment vector for {date}/{model} uses schema "
                f"{stored_schema}, expected {schema_version}."
            )
        return vector

    def put(
        self,
        date: str,
        vector: np.ndarray,
        model: str,
        source_count: int = 0,
        cost_usd: float = 0.0,
        schema_version: int = 2,
        tickers: Optional[List[str]] = None,
        axes: Optional[List[str]] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Store a sentiment vector, logging the API cost."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sentiment_cache
                   (date, model, vector_json, source_count, cost_usd, created_at,
                    schema_version, vector_dim, tickers_json, axes_json,
                    model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    date,
                    model,
                    json.dumps(vector.tolist()),
                    source_count,
                    cost_usd,
                    datetime.utcnow().isoformat(),
                    int(schema_version),
                    int(len(vector)),
                    json.dumps(tickers or []),
                    json.dumps(axes or []),
                    model_version or model,
                ),
            )
            conn.execute(
                """INSERT INTO cost_ledger (date, model, cost_usd, created_at)
                   VALUES (?, ?, ?, ?)""",
                (date, model, cost_usd, datetime.utcnow().isoformat()),
            )

    def has(self, date: str, model: str = "any") -> bool:
        """Return True if a vector exists for this date."""
        return self.get(date, model) is not None

    def missing_dates(self, all_dates: List[str], model: str = "any") -> List[str]:
        """Return dates from all_dates that are not yet in the cache."""
        return [d for d in all_dates if not self.has(d, model)]

    def total_cost_usd(self) -> float:
        """Sum of all API costs recorded in the ledger."""
        with self._conn() as conn:
            row = conn.execute("SELECT SUM(cost_usd) FROM cost_ledger").fetchone()
        return float(row[0] or 0.0)

    def total_cost_inr(self) -> float:
        return self.total_cost_usd() * USD_INR_RATE

    def budget_remaining_inr(self) -> float:
        return self.budget_inr - self.total_cost_inr()

    def is_over_budget(self) -> bool:
        return self.total_cost_inr() >= self.budget_inr

    def stats(self, model: str | None = None) -> dict:
        """Return cache statistics."""
        with self._conn() as conn:
            where = " WHERE model = ?" if model is not None else ""
            params = (model,) if model is not None else ()
            total, min_date, max_date, distinct_dates = conn.execute(
                f"""SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT date)
                    FROM sentiment_cache{where}""",
                params,
            ).fetchone()
            by_model = conn.execute(
                "SELECT model, COUNT(*) FROM sentiment_cache GROUP BY model"
            ).fetchall()
            dimensions = conn.execute(
                f"""SELECT COALESCE(vector_dim, json_array_length(vector_json)),
                           COUNT(*)
                    FROM sentiment_cache{where}
                    GROUP BY COALESCE(vector_dim, json_array_length(vector_json))""",
                params,
            ).fetchall()
            schemas = conn.execute(
                f"""SELECT COALESCE(schema_version, 1), COUNT(*)
                    FROM sentiment_cache{where}
                    GROUP BY COALESCE(schema_version, 1)""",
                params,
            ).fetchall()
        return {
            "total_entries":   total,
            "distinct_dates":  distinct_dates,
            "min_date":        min_date,
            "max_date":        max_date,
            "by_model":        dict(by_model),
            "vector_dimensions": {int(k): v for k, v in dimensions},
            "schema_versions": {int(k): v for k, v in schemas},
            "total_cost_usd":  round(self.total_cost_usd(), 4),
            "total_cost_inr":  round(self.total_cost_inr(), 2),
            "budget_inr":      self.budget_inr,
            "remaining_inr":   round(self.budget_remaining_inr(), 2),
            "db_size_mb":      round(self.db_path.stat().st_size / 1e6, 2),
        }

    def load_vectors(
        self,
        *,
        model: str,
        vector_dim: int,
        schema_version: int,
    ) -> Dict[str, np.ndarray]:
        """Load one validated model/schema arm into memory."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT date, vector_json, vector_dim, schema_version
                   FROM sentiment_cache WHERE model = ? ORDER BY date""",
                (model,),
            ).fetchall()
        vectors: Dict[str, np.ndarray] = {}
        for date, raw, stored_dim, stored_schema in rows:
            vector = np.asarray(json.loads(raw), dtype=np.float32)
            if (
                len(vector) != vector_dim
                or int(stored_dim or len(vector)) != vector_dim
                or int(stored_schema or 1) != schema_version
            ):
                raise CacheValidationError(
                    f"Incompatible sentiment row {date}/{model}: "
                    f"dim={len(vector)}, schema={stored_schema or 1}; "
                    f"expected dim={vector_dim}, schema={schema_version}."
                )
            vectors[date] = vector
        return vectors

    def validate_coverage(
        self,
        required_dates: List[str],
        *,
        model: str,
        vector_dim: int,
        schema_version: int = 2,
        min_nonzero_rate: float = 0.01,
    ) -> dict:
        """Fail fast on missing, stale, malformed, or degenerate cache data."""
        vectors = self.load_vectors(
            model=model,
            vector_dim=vector_dim,
            schema_version=schema_version,
        )
        missing = sorted(set(required_dates) - set(vectors))
        if missing:
            raise CacheValidationError(
                f"Sentiment cache model={model} is missing {len(missing)} "
                f"required dates ({missing[0]} … {missing[-1]})."
            )
        selected = [vectors[date] for date in required_dates]
        if not selected:
            raise CacheValidationError("No required sentiment dates were supplied.")
        matrix = np.stack(selected)
        if not np.all(np.isfinite(matrix)):
            raise CacheValidationError(
                f"Sentiment cache model={model} contains NaN/Inf values."
            )
        nonzero_rate = float(np.mean(np.any(np.abs(matrix) > 1e-12, axis=1)))
        if nonzero_rate < min_nonzero_rate:
            raise CacheValidationError(
                f"Sentiment cache model={model} is degenerate: only "
                f"{nonzero_rate:.1%} non-zero dates; required "
                f"{min_nonzero_rate:.1%}."
            )
        return {
            "model": model,
            "required_dates": len(required_dates),
            "vector_dim": vector_dim,
            "schema_version": schema_version,
            "nonzero_date_rate": nonzero_rate,
            "min_date": min(required_dates),
            "max_date": max(required_dates),
        }

    def put_component(
        self,
        *,
        date: str,
        model: str,
        ticker: str,
        tone: float,
        source_count: int = 0,
    ) -> None:
        """Persist one ticker/date source component for resumable cache builds."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sentiment_components
                   (date, model, ticker, tone, source_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    date,
                    model,
                    ticker,
                    float(tone),
                    int(source_count),
                    datetime.utcnow().isoformat(),
                ),
            )

    def component_dates(self, *, model: str, ticker: str) -> set[str]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT date FROM sentiment_components
                   WHERE model = ? AND ticker = ?""",
                (model, ticker),
            ).fetchall()
        return {row[0] for row in rows}

    def load_components(self, *, model: str) -> Dict[str, Dict[str, float]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT date, ticker, tone FROM sentiment_components
                   WHERE model = ? ORDER BY date, ticker""",
                (model,),
            ).fetchall()
        result: Dict[str, Dict[str, float]] = {}
        for date, ticker, tone in rows:
            result.setdefault(date, {})[ticker] = float(tone)
        return result

    def make_lookup_fn(
        self,
        vector_dim: int,
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        training: bool = False,
        model: str = "any",
        schema_version: int = 2,
        seed: int = 42,
    ):
        """
        Return a callable(date) → np.ndarray for use as sentiment_fn.

        Parameters
        ----------
        vector_dim   : expected output length (n_tickers * sentiment_dim)
        noise_std    : Gaussian noise std added to vectors (training only)
        dropout_prob : probability of zeroing the entire vector (training only)
        training     : if True, apply noise and dropout augmentation
        """
        if model == "any":
            raise ValueError(
                "Sentiment lookup requires an explicit model/version identifier."
            )
        vectors = self.load_vectors(
            model=model,
            vector_dim=vector_dim,
            schema_version=schema_version,
        )
        rng = np.random.default_rng(seed)

        def lookup(step_date) -> np.ndarray:
            date_str = str(step_date)[:10]
            vector = vectors.get(date_str)

            if vector is None:
                # No data for this date — return zeros (neutral sentiment)
                return np.zeros(vector_dim, dtype=np.float32)

            vector = vector.copy()

            if training:
                # ── Noise injection ───────────────────────────────────────────
                # Real news is noisy: misattributed articles, delayed feeds,
                # conflicting signals. Training on augmented sentiment prevents
                # the agent from over-relying on perfect sentiment signals.
                if dropout_prob > 0 and rng.random() < dropout_prob:
                    # Simulate a "no news day" — entire vector dropped
                    return np.zeros(vector_dim, dtype=np.float32)

                if noise_std > 0:
                    # Additive Gaussian noise, then re-clip to [-1, 1]
                    noise = rng.normal(
                        0, noise_std, size=vector.shape
                    ).astype(np.float32)
                    vector = np.clip(vector + noise, -1.0, 1.0)

            return vector

        return lookup

    # ── Private ───────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    date         TEXT NOT NULL,
                    model        TEXT NOT NULL,
                    vector_json  TEXT NOT NULL,
                    source_count INTEGER DEFAULT 0,
                    cost_usd     REAL    DEFAULT 0.0,
                    created_at   TEXT,
                    schema_version INTEGER DEFAULT 1,
                    vector_dim     INTEGER,
                    tickers_json   TEXT,
                    axes_json      TEXT,
                    model_version  TEXT,
                    PRIMARY KEY (date, model)
                )
            """)
            existing_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(sentiment_cache)"
                ).fetchall()
            }
            migrations = {
                "schema_version": "INTEGER DEFAULT 1",
                "vector_dim": "INTEGER",
                "tickers_json": "TEXT",
                "axes_json": "TEXT",
                "model_version": "TEXT",
            }
            for column, definition in migrations.items():
                if column not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE sentiment_cache "
                        f"ADD COLUMN {column} {definition}"
                    )
            conn.execute(
                """UPDATE sentiment_cache
                   SET vector_dim = json_array_length(vector_json)
                   WHERE vector_dim IS NULL"""
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_ledger (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    date       TEXT,
                    model      TEXT,
                    cost_usd   REAL,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_components (
                    date         TEXT NOT NULL,
                    model        TEXT NOT NULL,
                    ticker       TEXT NOT NULL,
                    tone         REAL NOT NULL,
                    source_count INTEGER DEFAULT 0,
                    created_at   TEXT,
                    PRIMARY KEY (date, model, ticker)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON sentiment_cache(date)")
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_component_model_ticker
                   ON sentiment_components(model, ticker, date)"""
            )
