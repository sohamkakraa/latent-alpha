"""
backtest/diagnostics.py
═══════════════════════
Per-fold diagnostic collection and reporting for walk-forward backtests.

Collects:
  - Per-fold return curve (daily portfolio values)
  - Per-fold transaction cost breakdown
  - Alpha channel hit rates per fold (% of alpha signals that were correct)
  - Regime distribution per fold (how much time in each regime)
  - Weight stability metrics per fold (mean turnover, hold period distribution)
  - Drawdown timeline per fold

Produces:
  - JSON file with all per-fold metrics
  - Summary CSV comparing folds
  - diagnostics_summary.txt with human-readable analysis
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FoldDiagnostics:
    """All per-fold diagnostic metrics."""

    fold_id:        int
    train_start:    str
    train_end:      str
    test_start:     str
    test_end:       str

    daily_values:   List[float] = field(default_factory=list)
    daily_returns:  List[float] = field(default_factory=list)

    transaction_costs:  float = 0.0
    alpha_hit_rate:     float = 0.0   # Fraction of correct direction calls
    mean_turnover:      float = 0.0   # Average weight change per step
    regime_distribution: Dict[int, float] = field(default_factory=dict)

    max_drawdown_pct:   float = 0.0
    sharpe:             float = 0.0
    calmar:             float = 0.0
    annual_return_pct:  float = 0.0
    total_steps:        int   = 0


class DiagnosticCollector:
    """
    Collects metrics during a backtest fold.

    Usage:
        collector = DiagnosticCollector(fold_id=0, ...)
        for each step:
            collector.record_step(info)
            collector.record_alpha_prediction(date, direction, actual_return)
            collector.record_regime(date, regime_label)
        diagnostics = collector.finalise()
    """

    def __init__(
        self,
        fold_id: int,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ):
        self.fold_id     = fold_id
        self.train_start = train_start
        self.train_end   = train_end
        self.test_start  = test_start
        self.test_end    = test_end

        self._values:       List[float] = []
        self._tc:           List[float] = []
        self._weights:      List[List[float]] = []
        self._alpha_calls:  List[tuple] = []   # (direction, actual_return)
        self._regime_calls: List[int]   = []

    def record_step(self, info_dict: dict) -> None:
        """Called after each env.step(). info_dict is the info returned by the env."""
        self._values.append(float(info_dict.get("portfolio_value", 0.0)))
        self._tc.append(float(info_dict.get("transaction_cost", 0.0)))
        w = info_dict.get("weights", [])
        if w:
            self._weights.append([float(x) for x in w])

    def record_alpha_prediction(
        self, date: str, direction: float, actual_return: float
    ) -> None:
        """Track alpha direction vs. actual outcome."""
        self._alpha_calls.append((float(direction), float(actual_return)))

    def record_regime(self, date: str, regime_label: int) -> None:
        """Track regime label at each step."""
        self._regime_calls.append(int(regime_label))

    def finalise(self) -> FoldDiagnostics:
        """Compute all summary metrics and return FoldDiagnostics."""
        daily_values  = self._values
        daily_returns = []
        if len(daily_values) >= 2:
            daily_returns = [
                (daily_values[i] - daily_values[i - 1]) / max(daily_values[i - 1], 1.0)
                for i in range(1, len(daily_values))
            ]

        total_tc = float(sum(self._tc))

        # Alpha hit rate
        alpha_hit_rate = 0.0
        if self._alpha_calls:
            hits = sum(
                1 for direction, actual in self._alpha_calls
                if (direction > 0 and actual > 0) or (direction < 0 and actual < 0)
            )
            alpha_hit_rate = hits / len(self._alpha_calls)

        # Mean turnover
        mean_turnover = 0.0
        if len(self._weights) >= 2:
            turnovers = [
                float(np.mean(np.abs(np.array(self._weights[i]) - np.array(self._weights[i - 1]))))
                for i in range(1, len(self._weights))
            ]
            mean_turnover = float(np.mean(turnovers)) if turnovers else 0.0

        # Regime distribution
        regime_dist: Dict[int, float] = {}
        if self._regime_calls:
            total = len(self._regime_calls)
            for label in set(self._regime_calls):
                regime_dist[label] = self._regime_calls.count(label) / total

        # Risk metrics
        max_dd = 0.0
        if daily_values:
            arr = np.array(daily_values, dtype=np.float64)
            peak = np.maximum.accumulate(arr)
            dd   = (peak - arr) / np.maximum(peak, 1.0)
            max_dd = float(dd.max() * 100)

        sharpe = 0.0
        ann_ret = 0.0
        calmar  = 0.0
        if daily_returns:
            rets = np.array(daily_returns, dtype=np.float64)
            mean_r = float(np.mean(rets))
            std_r  = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
            sharpe  = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
            ann_ret = ((1 + mean_r) ** 252 - 1) * 100
            calmar  = ann_ret / max_dd if max_dd > 0 else 0.0

        return FoldDiagnostics(
            fold_id=self.fold_id,
            train_start=self.train_start,
            train_end=self.train_end,
            test_start=self.test_start,
            test_end=self.test_end,
            daily_values=daily_values,
            daily_returns=daily_returns,
            transaction_costs=total_tc,
            alpha_hit_rate=alpha_hit_rate,
            mean_turnover=mean_turnover,
            regime_distribution=regime_dist,
            max_drawdown_pct=round(max_dd, 3),
            sharpe=round(sharpe, 4),
            calmar=round(calmar, 4),
            annual_return_pct=round(ann_ret, 3),
            total_steps=len(daily_values),
        )


def write_diagnostics_report(
    folds: List[FoldDiagnostics],
    output_dir: str,
) -> None:
    """
    Write diagnostic reports to output_dir:
      - fold_diagnostics.json (raw data)
      - fold_comparison.csv   (key metrics side by side)
      - diagnostics_summary.txt (human-readable analysis)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON — all raw data
    json_data = []
    for fd in folds:
        row = {
            "fold_id":           fd.fold_id,
            "train_start":       fd.train_start,
            "train_end":         fd.train_end,
            "test_start":        fd.test_start,
            "test_end":          fd.test_end,
            "total_steps":       fd.total_steps,
            "annual_return_pct": fd.annual_return_pct,
            "sharpe":            fd.sharpe,
            "calmar":            fd.calmar,
            "max_drawdown_pct":  fd.max_drawdown_pct,
            "transaction_costs": fd.transaction_costs,
            "alpha_hit_rate":    round(fd.alpha_hit_rate, 4),
            "mean_turnover":     round(fd.mean_turnover, 6),
            "regime_distribution": {str(k): v for k, v in fd.regime_distribution.items()},
        }
        json_data.append(row)

    with open(out / "fold_diagnostics.json", "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("Wrote fold_diagnostics.json")

    # CSV — key metrics
    try:
        import pandas as pd
        df = pd.DataFrame(json_data).drop(columns=["regime_distribution"], errors="ignore")
        df.to_csv(out / "fold_comparison.csv", index=False)
        logger.info("Wrote fold_comparison.csv")
    except ImportError:
        pass

    # Text summary
    if folds:
        valid = [fd for fd in folds if fd.total_steps > 5]
        best_fold  = max(valid, key=lambda fd: fd.sharpe, default=None) if valid else None
        worst_fold = min(valid, key=lambda fd: fd.sharpe, default=None) if valid else None

        mean_sharpe  = float(np.mean([fd.sharpe for fd in valid])) if valid else 0.0
        mean_ret     = float(np.mean([fd.annual_return_pct for fd in valid])) if valid else 0.0
        mean_dd      = float(np.mean([fd.max_drawdown_pct for fd in valid])) if valid else 0.0
        mean_tc      = float(np.mean([fd.transaction_costs for fd in valid])) if valid else 0.0
        mean_hit     = float(np.mean([fd.alpha_hit_rate for fd in valid])) if valid else 0.0

        lines = [
            "Walk-Forward Diagnostics Summary",
            "=" * 50,
            f"Total folds:   {len(folds)} ({len(valid)} valid)",
            f"Mean Sharpe:   {mean_sharpe:.3f}",
            f"Mean Return:   {mean_ret:.2f}%",
            f"Mean Drawdown: {mean_dd:.2f}%",
            f"Mean TC:       ₹{mean_tc:,.0f} per fold",
            f"Alpha hit rate: {mean_hit:.1%}",
            "",
        ]

        if best_fold:
            lines += [
                f"Best fold:  #{best_fold.fold_id} ({best_fold.test_start}→{best_fold.test_end})",
                f"  Sharpe={best_fold.sharpe:.3f}, Return={best_fold.annual_return_pct:.2f}%",
            ]
        if worst_fold:
            lines += [
                f"Worst fold: #{worst_fold.fold_id} ({worst_fold.test_start}→{worst_fold.test_end})",
                f"  Sharpe={worst_fold.sharpe:.3f}, Return={worst_fold.annual_return_pct:.2f}%",
                "",
            ]

        # TC as % of gross profit
        lines.append("Transaction cost as % of gross profit per fold:")
        for fd in valid:
            gross = max(fd.annual_return_pct, 0.01)
            tc_ratio = fd.transaction_costs / (gross / 100 * 1_000_000) * 100 if gross > 0 else 0
            lines.append(f"  Fold {fd.fold_id}: {tc_ratio:.1f}%")

        with open(out / "diagnostics_summary.txt", "w") as f:
            f.write("\n".join(lines))
        logger.info("Wrote diagnostics_summary.txt")
