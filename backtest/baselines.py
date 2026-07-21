"""Cost-aware classical baselines evaluated on the same causal clock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class BaselineRun:
    name: str
    values: list[float]
    total_transaction_cost: float
    mean_turnover: float


def performance_metrics(
    values: Iterable[float],
    *,
    annualisation_factor: int = 252,
    total_transaction_cost: float = 0.0,
    mean_turnover: float = 0.0,
) -> dict:
    values_array = np.asarray(list(values), dtype=np.float64)
    if len(values_array) < 2:
        return {"total_steps": len(values_array)}
    returns = np.diff(values_array) / np.maximum(values_array[:-1], 1.0)
    mean_return = float(np.mean(returns))
    std_return = (
        float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    )
    sharpe = (
        mean_return / std_return * np.sqrt(annualisation_factor)
        if std_return > 1e-10
        else 0.0
    )
    downside = returns[returns < 0]
    downside_std = (
        float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    )
    sortino = (
        mean_return / downside_std * np.sqrt(annualisation_factor)
        if downside_std > 1e-10
        else 0.0
    )
    peaks = np.maximum.accumulate(values_array)
    drawdowns = (peaks - values_array) / np.maximum(peaks, 1.0)
    max_drawdown = float(np.max(drawdowns)) * 100.0
    n_years = len(returns) / annualisation_factor
    cumulative = float(values_array[-1] / values_array[0] - 1.0)
    annual_return = (
        float((values_array[-1] / values_array[0]) ** (1 / n_years) - 1)
        if n_years > 0.01
        else cumulative
    )
    return {
        "cumulative_return_pct": round(cumulative * 100.0, 4),
        "annual_return_pct": round(annual_return * 100.0, 4),
        "annualised_sharpe": round(float(sharpe), 4),
        "annualised_sortino": round(float(sortino), 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "calmar_ratio": round(
            annual_return * 100.0 / max_drawdown
            if max_drawdown > 0.01
            else 0.0,
            4,
        ),
        "total_transaction_cost": round(
            float(total_transaction_cost), 4
        ),
        "mean_turnover": round(float(mean_turnover), 6),
        "total_steps": len(values_array) - 1,
        "n_years": round(n_years, 3),
    }


def simulate_target_weights(
    *,
    name: str,
    data: Dict[str, pd.DataFrame],
    dates: list[pd.Timestamp],
    target_fn: Callable[[pd.Timestamp, np.ndarray], np.ndarray],
    initial_capital: float,
    transaction_cost_pct: float,
    slippage_pct: float,
    rebalance_each_step: bool = True,
    execution_lag_bars: int = 1,
) -> BaselineRun:
    tickers = list(data)
    n_assets = len(tickers)
    positions = np.zeros(n_assets, dtype=np.float64)
    weights = np.zeros(n_assets, dtype=np.float64)
    cash = float(initial_capital)
    portfolio_value = float(initial_capital)
    values = [portfolio_value]
    total_cost = 0.0
    turnovers: list[float] = []

    def prices(date):
        return np.asarray(
            [float(data[ticker].loc[date, "Close"]) for ticker in tickers],
            dtype=np.float64,
        )

    execution_lag_bars = max(int(execution_lag_bars), 1)
    for execution_idx in range(execution_lag_bars, len(dates) - 1):
        signal_date = dates[execution_idx - execution_lag_bars]
        execution_date = dates[execution_idx]
        current_prices = prices(execution_date)
        portfolio_value = cash + float(np.sum(positions * current_prices))

        if execution_idx == 1 or rebalance_each_step:
            target = np.asarray(
                target_fn(signal_date, weights.copy()),
                dtype=np.float64,
            )
            if target.shape != (n_assets,):
                raise ValueError(
                    f"{name} target shape {target.shape} != ({n_assets},)"
                )
            target = np.clip(target, 0.0, 1.0)
            gross = float(np.sum(target))
            if gross > 1.0:
                target /= gross
            target_values = target * portfolio_value
            target_shares = target_values / np.maximum(
                current_prices, 1e-8
            )
            delta_shares = target_shares - positions
            fills = current_prices * (
                1 + np.sign(delta_shares) * slippage_pct
            )
            notional = np.abs(delta_shares * fills)
            cost = float(np.sum(notional * transaction_cost_pct))
            cash -= float(np.sum(delta_shares * fills)) + cost
            positions = target_shares
            turnovers.append(float(np.sum(np.abs(target - weights))))
            total_cost += cost

        next_prices = prices(dates[execution_idx + 1])
        portfolio_value = cash + float(np.sum(positions * next_prices))
        weights = (
            positions * next_prices / portfolio_value
            if portfolio_value > 1e-8
            else np.zeros(n_assets)
        )
        values.append(portfolio_value)

    return BaselineRun(
        name=name,
        values=values,
        total_transaction_cost=total_cost,
        mean_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
    )


def evaluate_market_baselines(
    *,
    data: Dict[str, pd.DataFrame],
    dates: list[pd.Timestamp],
    config: dict,
    alpha_fn=None,
    annualisation_factor: int = 252,
) -> dict:
    n_assets = len(data)
    equal_weights = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

    def equal_target(date, current):
        return equal_weights

    def momentum_target(date, current):
        scores = []
        for ticker in data:
            frame = data[ticker]
            value = (
                frame.loc[date, "return_20d"]
                if date in frame.index and "return_20d" in frame.columns
                else 0.0
            )
            scores.append(float(value) if np.isfinite(value) else 0.0)
        scores_array = np.asarray(scores)
        winners = np.argsort(scores_array)[-min(3, n_assets):]
        target = np.zeros(n_assets, dtype=np.float64)
        target[winners] = 1.0 / len(winners)
        return target

    def alpha_target(date, current):
        if alpha_fn is None:
            return np.zeros(n_assets, dtype=np.float64)
        signal = alpha_fn(date)
        score = np.asarray(signal.direction) * np.asarray(signal.conviction)
        positive = np.flatnonzero(score > 0)
        if not len(positive):
            return np.zeros(n_assets, dtype=np.float64)
        winners = positive[np.argsort(score[positive])[-min(3, len(positive)):]]
        target = np.zeros(n_assets, dtype=np.float64)
        target[winners] = 1.0 / len(winners)
        return target

    env = config["env"]
    common = {
        "data": data,
        "dates": dates,
        "initial_capital": env["initial_capital"],
        "transaction_cost_pct": env["transaction_cost_pct"],
        "slippage_pct": env["slippage_pct"],
        "execution_lag_bars": env.get("execution_lag_bars", 1),
    }
    lag = max(int(env.get("execution_lag_bars", 1)), 1)
    runs = [
        BaselineRun(
            "cash",
            [env["initial_capital"]] * (len(dates) - lag),
            0,
            0,
        ),
        simulate_target_weights(
            name="equal_weight_buy_hold",
            target_fn=equal_target,
            rebalance_each_step=False,
            **common,
        ),
        simulate_target_weights(
            name="equal_weight_rebalanced",
            target_fn=equal_target,
            **common,
        ),
        simulate_target_weights(
            name="momentum_top3",
            target_fn=momentum_target,
            **common,
        ),
    ]
    if alpha_fn is not None:
        runs.append(
            simulate_target_weights(
                name="alpha_top3",
                target_fn=alpha_target,
                **common,
            )
        )
    return {
        run.name: performance_metrics(
            run.values,
            annualisation_factor=annualisation_factor,
            total_transaction_cost=run.total_transaction_cost,
            mean_turnover=run.mean_turnover,
        )
        for run in runs
    }
