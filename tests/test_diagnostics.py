"""Tests for walk-forward diagnostics (P1 Prompt 7)."""
import numpy as np
import pytest


def test_diagnostic_collector_records():
    """DiagnosticCollector should accumulate step records."""
    from backtest.diagnostics import DiagnosticCollector

    collector = DiagnosticCollector(
        fold_id=0,
        train_start="2018-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-06-30",
    )

    for i in range(100):
        collector.record_step({
            "portfolio_value": 1_000_000 + i * 100,
            "transaction_cost": 50.0,
            "weights": [0.1] * 10,
        })

    result = collector.finalise()
    assert result.fold_id == 0
    assert len(result.daily_values) == 100
    assert result.transaction_costs > 0


def test_alpha_hit_rate_calculation():
    """Alpha hit rate should be correct for known predictions."""
    from backtest.diagnostics import DiagnosticCollector

    collector = DiagnosticCollector(
        fold_id=0,
        train_start="2020-01-01", train_end="2021-01-01",
        test_start="2021-01-01", test_end="2021-06-30",
    )

    # 7 correct (direction=1, actual > 0), 3 wrong (direction=-1, actual > 0)
    for i in range(10):
        direction = 1.0 if i < 7 else -1.0
        actual = 0.01
        collector.record_alpha_prediction("2021-01-01", direction, actual)

    result = collector.finalise()
    assert result.alpha_hit_rate == pytest.approx(0.7, abs=0.01)


def test_regime_distribution_sums_to_one():
    """Regime fractions should sum to 1."""
    from backtest.diagnostics import DiagnosticCollector

    collector = DiagnosticCollector(
        fold_id=0,
        train_start="2020-01-01", train_end="2021-01-01",
        test_start="2021-01-01", test_end="2021-06-30",
    )

    for i in range(100):
        collector.record_regime("2021-01-01", regime_label=i % 4)

    result = collector.finalise()
    total = sum(result.regime_distribution.values())
    assert total == pytest.approx(1.0, abs=0.01)


def test_fold_diagnostics_fields():
    """FoldDiagnostics should have all required fields."""
    from backtest.diagnostics import DiagnosticCollector

    collector = DiagnosticCollector(
        fold_id=1,
        train_start="2019-01-01", train_end="2020-12-31",
        test_start="2021-01-01", test_end="2021-12-31",
    )

    for i in range(50):
        collector.record_step({
            "portfolio_value": 1_000_000 + i * 200,
            "transaction_cost": 25.0,
            "weights": [0.05] * 10,
        })

    result = collector.finalise()
    assert hasattr(result, "fold_id")
    assert hasattr(result, "daily_values")
    assert hasattr(result, "daily_returns")
    assert hasattr(result, "transaction_costs")
    assert hasattr(result, "alpha_hit_rate")
    assert hasattr(result, "mean_turnover")
    assert hasattr(result, "regime_distribution")
    assert hasattr(result, "max_drawdown_pct")
    assert hasattr(result, "sharpe")
    assert hasattr(result, "total_steps")
    assert result.total_steps == 50


def test_write_diagnostics_report(tmp_path):
    """write_diagnostics_report should produce output files."""
    from backtest.diagnostics import DiagnosticCollector, write_diagnostics_report

    folds = []
    for fold_id in range(3):
        collector = DiagnosticCollector(
            fold_id=fold_id,
            train_start="2018-01-01", train_end="2019-12-31",
            test_start="2020-01-01", test_end="2020-06-30",
        )
        for i in range(100):
            val = 1_000_000 * (1 + np.random.randn() * 0.001)
            collector.record_step({"portfolio_value": val, "transaction_cost": 10.0})
        folds.append(collector.finalise())

    write_diagnostics_report(folds, str(tmp_path))

    assert (tmp_path / "fold_diagnostics.json").exists()
    assert (tmp_path / "diagnostics_summary.txt").exists()
