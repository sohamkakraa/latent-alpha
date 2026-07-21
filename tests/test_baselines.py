"""Tests for cost-aware classical comparison strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from backtest.baselines import evaluate_market_baselines, performance_metrics


def test_performance_metrics_include_risk_and_cost_fields():
    metrics = performance_metrics(
        [1_000_000, 1_010_000, 1_005_000, 1_020_000],
        total_transaction_cost=123.0,
        mean_turnover=0.2,
    )
    assert metrics["annualised_sharpe"] != 0
    assert metrics["max_drawdown_pct"] > 0
    assert metrics["total_transaction_cost"] == 123.0
    assert metrics["mean_turnover"] == 0.2


def test_market_baselines_share_cost_model_and_causal_dates():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["market"]["tickers"] = ["A", "B", "C"]
    dates = list(pd.bdate_range("2024-01-01", periods=30))
    data = {}
    for index, ticker in enumerate(config["market"]["tickers"]):
        close = 100 + np.arange(len(dates)) * (index + 1)
        data[ticker] = pd.DataFrame(
            {
                "Close": close,
                "return_20d": np.full(len(dates), index, dtype=float),
            },
            index=dates,
        )

    results = evaluate_market_baselines(
        data=data,
        dates=dates,
        config=config,
    )

    assert results["cash"]["cumulative_return_pct"] == 0.0
    assert results["equal_weight_buy_hold"]["cumulative_return_pct"] > 0
    assert (
        results["equal_weight_rebalanced"]["total_transaction_cost"]
        >= results["equal_weight_buy_hold"]["total_transaction_cost"]
    )
    assert results["momentum_top3"]["total_steps"] == len(dates) - 2
