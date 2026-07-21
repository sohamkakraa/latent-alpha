"""
backtest/runner.py
──────────────────
Walk-forward backtesting engine.

Walk-forward validation works like this:
  - Training window:   N months of data → train the agent
  - Test window:       M months immediately after → evaluate (never seen during train)
  - Slide forward by M months, repeat

This avoids the look-ahead bias of a single train/test split and
produces a realistic out-of-sample performance curve.

Output:
  - Per-fold metrics: Sharpe, Calmar, max drawdown, annual return vs benchmark
  - Aggregate metrics across all folds
  - Trade log: every step with portfolio value, actions, and costs
  - Matplotlib equity curve saved to results_dir
"""

from __future__ import annotations

import copy
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from agent.reward import RewardAggregator
from env.trading_env import NSETradingEnv
from strategy.term_selector import TermSelector

logger = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    risk_profile: str
    term: str
    metrics: dict = field(default_factory=dict)
    trade_log: List[dict] = field(default_factory=list)


@dataclass
class BacktestResult:
    risk_profile: str
    term: str
    folds: List[FoldResult] = field(default_factory=list)

    def aggregate(self) -> dict:
        """Aggregate metrics across all folds."""
        if not self.folds:
            return {}

        valid = [f for f in self.folds if f.metrics.get("total_steps", 0) > 5]
        if not valid:
            return {"n_folds": len(self.folds), "note": "all folds had insufficient data"}

        all_returns = [f.metrics.get("annual_return_pct", f.metrics.get("cumulative_return_pct", 0)) for f in valid]
        all_sharpe  = [f.metrics.get("annualised_sharpe", 0) for f in valid]
        all_dd      = [f.metrics.get("max_drawdown_pct", 0) for f in valid]
        all_calmar  = [f.metrics.get("calmar_ratio", 0) for f in valid]

        aggregate = {
            "n_folds":                len(self.folds),
            "n_valid_folds":          len(valid),
            "mean_annual_return_pct": round(float(np.mean(all_returns)), 2),
            "std_annual_return_pct":  round(float(np.std(all_returns)), 2),
            "mean_sharpe":            round(float(np.mean(all_sharpe)), 3),
            "std_sharpe":             round(float(np.std(all_sharpe)), 3),
            "worst_fold_sharpe":      round(float(np.min(all_sharpe)), 3),
            "mean_max_drawdown_pct":  round(float(np.mean(all_dd)), 2),
            "mean_calmar":            round(float(np.mean(all_calmar)), 3),
            # Median is the timestep-grid selection metric (robust to fold outliers
            # like a single regime-crash fold dragging the mean).
            "median_annual_return_pct": round(float(np.median(all_returns)), 2),
            "median_sharpe":            round(float(np.median(all_sharpe)), 3),
            "median_calmar":            round(float(np.median(all_calmar)), 3),
            "risk_profile":           self.risk_profile,
            "term":                   self.term,
        }
        baseline_names = sorted(
            {
                name
                for fold in valid
                for name in fold.metrics.get("baselines", {})
            }
        )
        if baseline_names:
            aggregate["baselines"] = {}
            for name in baseline_names:
                rows = [
                    fold.metrics["baselines"][name]
                    for fold in valid
                    if name in fold.metrics.get("baselines", {})
                ]
                sharpes = [
                    row.get("annualised_sharpe", 0.0) for row in rows
                ]
                returns = [
                    row.get("annual_return_pct", 0.0) for row in rows
                ]
                drawdowns = [
                    row.get("max_drawdown_pct", 0.0) for row in rows
                ]
                aggregate["baselines"][name] = {
                    "n_folds": len(rows),
                    "mean_sharpe": round(float(np.mean(sharpes)), 3),
                    "median_sharpe": round(
                        float(np.median(sharpes)), 3
                    ),
                    "mean_annual_return_pct": round(
                        float(np.mean(returns)), 2
                    ),
                    "mean_max_drawdown_pct": round(
                        float(np.mean(drawdowns)), 2
                    ),
                }
            benchmark = aggregate["baselines"].get("benchmark")
            if benchmark is not None:
                aggregate["sharpe_excess_vs_benchmark"] = round(
                    aggregate["median_sharpe"]
                    - benchmark["median_sharpe"],
                    3,
                )
        return aggregate

    def equity_curve(self) -> pd.Series:
        """Concatenate per-fold portfolio values into a single equity curve."""
        all_values = []
        for fold in self.folds:
            for record in fold.trade_log:
                all_values.append({
                    "date":            record.get("date"),
                    "portfolio_value": record.get("portfolio_value"),
                    "fold":            fold.fold_idx,
                })
        if not all_values:
            return pd.Series(dtype=float)
        df = pd.DataFrame(all_values)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df["portfolio_value"]


# ── Walk-forward engine ───────────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Runs walk-forward backtesting for a given risk profile and term.

    Parameters
    ----------
    config       : full config dict
    risk_profile : "conservative" | "balanced" | "aggressive"
    term         : "short" | "medium" | "long"
    sentiment_fn : optional LLM sentiment callable
    """

    def __init__(
        self,
        config: dict,
        risk_profile: str = "balanced",
        term: str = "medium",
        algo: str = "sac",
        sentiment_fn=None,
        sentiment_fn_eval=None,
        timesteps_override: int = None,
        explain_trades: bool = False,
        start_fold: int = 0,
        seed: int = 42,
        run_id: str | None = None,
        device: str = "cpu",
        fold_indices: list[int] | None = None,
    ):
        self.config              = copy.deepcopy(config)
        self.risk_profile        = risk_profile
        self.term                = term
        self.algo                = algo
        self.seed                = int(seed)
        self.device              = device
        self.fold_indices        = (
            tuple(sorted(set(fold_indices)))
            if fold_indices is not None
            else None
        )
        self.sentiment_fn        = sentiment_fn
        # Eval/test sentiment fn — clean (no noise/dropout) for a fair OOS read.
        # Falls back to the train fn when not supplied.
        self.sentiment_fn_eval   = sentiment_fn_eval if sentiment_fn_eval is not None else sentiment_fn
        self.timesteps_override  = timesteps_override
        self.start_fold          = start_fold
        self.bt_cfg              = self.config["backtest"]
        self.config.setdefault("experiment", {}).update(
            {
                "base_seed": self.seed,
                "sentiment_enabled": sentiment_fn is not None,
                "fold_indices": self.fold_indices,
            }
        )

        from experiments.provenance import (
            build_run_identity,
            write_run_manifest,
        )

        repo_root = Path(__file__).resolve().parents[1]
        generated_id, manifest = build_run_identity(
            self.config,
            risk_profile=risk_profile,
            term=term,
            algorithm=algo,
            seed=self.seed,
            timesteps=(
                timesteps_override
                or self.config.get(algo, {}).get("total_timesteps")
            ),
            device=self.device,
            repo_root=repo_root,
        )
        self.run_id = run_id or generated_id
        if run_id is not None:
            manifest["run_id"] = run_id
        self.identity_sha256 = manifest["identity_sha256"]

        results_base = Path(self.config["paths"]["results_dir"])
        self.results_dir = results_base / "runs" / self.run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.config["paths"]["results_dir"] = str(self.results_dir)
        self.config["paths"]["model_dir"] = str(self.results_dir / "models")
        self.config["paths"]["logs_dir"] = str(self.results_dir / "logs")
        write_run_manifest(self.results_dir, manifest, self.config)

        self.selector            = TermSelector(self.config)
        self._last_agent         = None

        # Phase 4: post-hoc explainability
        self._explainer = None
        explainer_cfg = config.get("llm", {}).get("explainer", {})
        if explain_trades or explainer_cfg.get("enabled", False):
            from llm.explainer import TradeExplainer
            tickers = config["market"]["tickers"]
            self._explainer = TradeExplainer(config, tickers)

    def run(self, retrain_each_fold: bool = True) -> BacktestResult:
        """
        Execute walk-forward backtest.

        Parameters
        ----------
        retrain_each_fold : if True, retrain from scratch each fold.
                            if False, the same pre-trained model is evaluated
                            (faster but less realistic).

        Returns
        -------
        BacktestResult with per-fold and aggregate metrics
        """
        if not retrain_each_fold:
            raise NotImplementedError(
                "run(retrain_each_fold=False) is disabled: channel pipelines are fold-scoped "
                "and cannot be safely reused across folds yet. Use retrain_each_fold=True."
            )

        folds = self._generate_folds()
        result = BacktestResult(risk_profile=self.risk_profile, term=self.term)

        for fold_idx, (train_s, train_e, test_s, test_e) in enumerate(folds):
            if (
                self.fold_indices is not None
                and fold_idx not in self.fold_indices
            ):
                continue
            # Durable resume: a fold persisted by a prior (possibly crashed) run is
            # reused as-is, so an interrupted walk-forward never re-burns completed
            # folds. See _fold_state_dir for the per-run signature.
            cached = self._load_fold(fold_idx)
            if cached is not None:
                logger.info(
                    "Fold %d/%d cached — reusing persisted metrics: %s",
                    fold_idx + 1, len(folds), cached.metrics,
                )
                result.folds.append(cached)
                continue

            if fold_idx < self.start_fold:
                logger.warning(
                    "Fold %d/%d skipped (start_fold=%d) with no persisted state — "
                    "excluded from aggregate.",
                    fold_idx + 1, len(folds), self.start_fold,
                )
                continue

            logger.info(
                "Fold %d/%d | Train: %s → %s | Test: %s → %s",
                fold_idx + 1, len(folds), train_s, train_e, test_s, test_e,
            )

            fold_result = self._run_fold(
                fold_idx=fold_idx,
                train_start=train_s, train_end=train_e,
                test_start=test_s,   test_end=test_e,
                retrain=retrain_each_fold,
            )
            result.folds.append(fold_result)

            logger.info("Fold %d metrics: %s", fold_idx + 1, fold_result.metrics)
            # Persist immediately so a crash in a later fold keeps this one.
            self._save_fold(fold_result)
            # Only now is it safe to discard mid-training recovery artifacts.
            if (
                self._last_agent is not None
                and hasattr(self._last_agent, "clear_periodic_checkpoints")
            ):
                self._last_agent.clear_periodic_checkpoints()

        agg = result.aggregate()
        logger.info("Backtest complete. Aggregate: %s", agg)
        self._save_results(result)
        return result

    # ── Fold execution ────────────────────────────────────────────────────────

    def _run_fold(
        self,
        fold_idx: int,
        train_start: str, train_end: str,
        test_start: str,  test_end: str,
        retrain: bool,
    ) -> FoldResult:
        fold = FoldResult(
            fold_idx=fold_idx,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            risk_profile=self.risk_profile,
            term=self.term,
        )
        fold_seed = self.seed + fold_idx * 1009
        self._seed_fold(fold_seed)

        # Build and (optionally) train agent
        agent = self._make_agent(fold_seed)
        self._last_agent = agent
        train_data_fingerprint = None

        if retrain:
            # Train first — this caches the trained channel fns on the selector
            # so build_backtest_env can reuse them (avoids retraining on warmup-only data)
            train_env, eval_env = self.selector.build_env(
                risk_profile=self.risk_profile,
                term=self.term,
                train_start=train_start,
                train_end=train_end,
                sentiment_fn=self.sentiment_fn,
                sentiment_fn_eval=self.sentiment_fn_eval,
            )
            train_data_fingerprint = getattr(
                self.selector, "_last_data_fingerprint", None
            )
            # resume=True lets the agent pick up a periodic checkpoint if one
            # exists from a crashed prior run. Safe because periodic checkpoints
            # are namespaced per (risk_profile, term), and `agent.save()` after
            # `learn()` writes the canonical fold artifact regardless.
            agent.train(
                train_env, eval_env=eval_env, resume=True,
                total_timesteps=self.timesteps_override,
            )
        else:
            # Use the single trained model for all folds — fast evaluation mode
            if not agent._checkpoint_exists():
                raise FileNotFoundError(
                    f"No trained model found at {agent.model_path}.zip\n"
                    f"Train first with: python train.py --risk {self.risk_profile} --term {self.term}"
                )

        # Build test environment after train (selector may have cached trained fns)
        test_env = self.selector.build_backtest_env(  # uses clean eval sentiment
            risk_profile=self.risk_profile,
            term=self.term,
            start=test_start,
            end=test_end,
            train_start=train_start,
            train_end=train_end,
            sentiment_fn=self.sentiment_fn_eval,
        )
        test_data_fingerprint = getattr(
            self.selector, "_last_data_fingerprint", None
        )
        test_vec_env = DummyVecEnv([lambda: test_env])
        agent.load(raw_env=test_vec_env)
        # SAC attaches the training VecNormalize statistics to its model during
        # load(). Roll out through that wrapper; predicting from test_vec_env
        # directly would feed raw observations to a policy trained on normalized
        # inputs.
        rollout_env = test_vec_env
        model = getattr(agent, "model", None)
        if model is not None and hasattr(model, "get_env"):
            model_env = model.get_env()
            if model_env is not None:
                rollout_env = model_env

        # ── Evaluation loop ───────────────────────────────────────────────────
        obs = rollout_env.reset()
        aggregator = RewardAggregator()
        done = False
        prev_weights = np.zeros(len(self.config["market"]["tickers"]))

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            action_for_info = action[0] if np.ndim(action) > 1 else action
            obs, rewards, dones, infos = rollout_env.step(action)
            done = bool(dones[0])
            reward = float(rewards[0]) if np.ndim(rewards) > 0 else float(rewards)
            info = infos[0]
            previous_value = (
                fold.trade_log[-1]["portfolio_value"]
                if fold.trade_log
                else self.config["env"]["initial_capital"]
            )
            aggregator.record(
                reward,
                step_return=(
                    info["portfolio_value"] - previous_value
                ) / max(previous_value, 1.0),
            )

            # Phase 4: post-hoc explanation (no-op when explainer is None/disabled)
            if self._explainer is not None:
                try:
                    explanation = self._explainer.explain(
                        weight_deltas=np.array(action_for_info, dtype=np.float32),
                        alpha_direction=np.zeros(len(prev_weights), dtype=np.float32),
                        alpha_conviction=np.zeros(len(prev_weights), dtype=np.float32),
                        portfolio_weights=prev_weights,
                        date=str(info.get("date", "")),
                    )
                    if explanation is not None:
                        info["explanation"] = explanation.explanations
                        info["llm_agrees"] = explanation.llm_agrees
                except Exception:
                    pass
            prev_weights = np.array(info.get("weights", prev_weights), dtype=np.float32)
            fold.trade_log.append(info)

        # Compute metrics
        port_values = [self.config["env"]["initial_capital"]] + [
            r["portfolio_value"] for r in fold.trade_log
        ]
        fold.metrics = self._compute_metrics(port_values, aggregator)
        fold.metrics["seed"] = fold_seed
        fold.metrics["train_data_sha256"] = train_data_fingerprint
        fold.metrics["test_data_sha256"] = test_data_fingerprint
        fold.metrics["total_transaction_cost"] = round(
            float(
                sum(
                    record.get("transaction_cost", 0.0)
                    for record in fold.trade_log
                )
            ),
            4,
        )

        if hasattr(test_env, "data") and hasattr(test_env, "_dates"):
            from backtest.baselines import (
                evaluate_market_baselines,
                performance_metrics,
                simulate_target_weights,
            )

            ann_factor = {
                "short": 252 * 7,
                "medium": 252,
                "long": 52,
            }.get(self.term, 252)
            baselines = evaluate_market_baselines(
                data=test_env.data,
                dates=list(test_env._dates),
                config=self.config,
                alpha_fn=getattr(test_env, "alpha_fn", None),
                annualisation_factor=ann_factor,
            )
            try:
                benchmark = self.selector.loader.load_benchmark(
                    test_start,
                    test_end,
                    self.term,
                )
                benchmark_dates = [
                    date
                    for date in test_env._dates
                    if date in benchmark.index
                ]
                if len(benchmark_dates) >= 3:
                    run = simulate_target_weights(
                        name="benchmark",
                        data={
                            self.config["market"]["benchmark"]: benchmark
                        },
                        dates=benchmark_dates,
                        target_fn=lambda date, current: np.ones(1),
                        initial_capital=self.config["env"]["initial_capital"],
                        transaction_cost_pct=self.config["env"][
                            "transaction_cost_pct"
                        ],
                        slippage_pct=self.config["env"]["slippage_pct"],
                        execution_lag_bars=self.config["env"].get(
                            "execution_lag_bars", 1
                        ),
                        rebalance_each_step=False,
                    )
                    baselines["benchmark"] = performance_metrics(
                        run.values,
                        annualisation_factor=ann_factor,
                        total_transaction_cost=run.total_transaction_cost,
                        mean_turnover=run.mean_turnover,
                    )
            except Exception as exc:
                logger.warning("Benchmark evaluation unavailable: %s", exc)
            fold.metrics["baselines"] = baselines
        return fold

    def _make_agent(self, seed: int | None = None):
        """Return the appropriate agent based on self.algo."""
        agent_seed = self.seed if seed is None else seed
        if self.algo == "sac":
            from agent.sac_agent import LatentAlphaSACAgent
            return LatentAlphaSACAgent(
                self.config,
                self.risk_profile,
                self.term,
                device=self.device,
                seed=agent_seed,
            )
        else:
            from agent.ppo_agent import LatentAlphaAgent
            return LatentAlphaAgent(
                self.config,
                self.risk_profile,
                self.term,
                device=self.device,
                seed=agent_seed,
            )

    def _seed_fold(self, seed: int) -> None:
        """Seed every stochastic library before constructing fold models."""
        random.seed(seed)
        np.random.seed(seed)
        self.config.setdefault("experiment", {})["seed"] = int(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        portfolio_values: List[float],
        aggregator: RewardAggregator,
    ) -> dict:
        if len(portfolio_values) < 2:
            return {"total_steps": len(portfolio_values), "note": "insufficient data"}

        values  = np.array(portfolio_values, dtype=np.float64)
        returns = np.diff(values) / np.maximum(values[:-1], 1.0)

        # Annualise based on term bar frequency
        # medium = daily bars → 252 trading days/year
        # short  = hourly bars → 252 * 6.5 hours/day
        # long   = weekly bars → 52 weeks/year
        freq_map   = {"short": 252 * 7, "medium": 252, "long": 52}
        ann_factor = freq_map.get(self.term, 252)

        mean_r = float(np.mean(returns))
        std_r  = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = float(mean_r / std_r * np.sqrt(ann_factor)) if std_r > 1e-10 else 0.0

        cumulative_return = float((values[-1] / values[0]) - 1.0) * 100.0

        # Max drawdown
        peak   = np.maximum.accumulate(values)
        dd     = (peak - values) / np.maximum(peak, 1.0)
        max_dd = float(np.max(dd)) * 100.0

        # Annualised return (CAGR approximation)
        n_years       = len(returns) / ann_factor
        annual_return = (
            float((values[-1] / values[0]) ** (1.0 / n_years) - 1.0) * 100.0
            if n_years > 0.01 else cumulative_return
        )

        # Calmar: annualised return / max drawdown
        calmar = (annual_return / max_dd) if max_dd > 0.01 else 0.0

        return {
            "cumulative_return_pct": round(cumulative_return, 4),
            "annual_return_pct":     round(annual_return, 4),
            "annualised_sharpe":     round(sharpe, 4),
            "max_drawdown_pct":      round(max_dd, 4),
            "calmar_ratio":          round(calmar, 4),
            "total_steps":           len(portfolio_values),
            "n_years":               round(n_years, 3),
        }

    # ── Date fold generation ──────────────────────────────────────────────────

    def _generate_folds(self) -> List[tuple]:
        """
        Generate (train_start, train_end, test_start, test_end) tuples.
        """
        start = pd.Timestamp(self.bt_cfg["start_date"])
        end   = pd.Timestamp(self.bt_cfg["end_date"])
        train_months = self.bt_cfg["walk_forward_train_months"]
        test_months  = self.bt_cfg["walk_forward_test_months"]

        folds = []
        cursor = start
        while True:
            train_end   = cursor + pd.DateOffset(months=train_months)
            test_start  = train_end
            test_end    = test_start + pd.DateOffset(months=test_months)
            if test_end > end:
                break
            folds.append((
                cursor.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))
            # Slide the whole window forward by the TEST window, not the train
            # window. cursor += test_months gives a rolling walk-forward with
            # overlapping train windows and one OOS evaluation per test period.
            # (Previously `cursor = test_start` jumped forward by train_months,
            # collapsing the run to only 2 folds for a 36mo/6mo config.)
            cursor = cursor + pd.DateOffset(months=test_months)

        logger.info("Generated %d walk-forward folds.", len(folds))
        return folds

    # ── Persistence ───────────────────────────────────────────────────────────

    def _fold_state_dir(self) -> Path:
        """
        Per-run directory for completed fold state.

        The parent run directory is already keyed by the full scientific config,
        seed, algorithm, code commit, and dirty-worktree hash.
        """
        return self.results_dir / "fold_checkpoints"

    def _save_fold(self, fold: FoldResult) -> None:
        """Persist a completed fold's metrics + window (lightweight; no trade_log)."""
        d = self._fold_state_dir()
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "fold_idx":     fold.fold_idx,
            "train_start":  fold.train_start, "train_end": fold.train_end,
            "test_start":   fold.test_start,  "test_end":  fold.test_end,
            "risk_profile": fold.risk_profile, "term":     fold.term,
            "metrics":      fold.metrics,
            "run_id":       self.run_id,
            "identity_sha256": self.identity_sha256,
            "algorithm":    self.algo,
            "base_seed":    self.seed,
        }
        path = d / f"fold_{fold.fold_idx}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Persisted fold %d state → %s", fold.fold_idx + 1, path)

    def _load_fold(self, fold_idx: int) -> Optional[FoldResult]:
        """Load a persisted fold if present. trade_log is not restored (metrics only)."""
        path = self._fold_state_dir() / f"fold_{fold_idx}.json"
        if not path.exists():
            return None
        with open(path) as f:
            p = json.load(f)
        if (
            p.get("run_id") != self.run_id
            or p.get("identity_sha256") != self.identity_sha256
            or p.get("algorithm") != self.algo
            or int(p.get("base_seed", -1)) != self.seed
        ):
            logger.warning("Ignoring incompatible fold state: %s", path)
            return None
        return FoldResult(
            fold_idx=p["fold_idx"],
            train_start=p["train_start"], train_end=p["train_end"],
            test_start=p["test_start"],   test_end=p["test_end"],
            risk_profile=p["risk_profile"], term=p["term"],
            metrics=p["metrics"],
        )

    def _save_results(self, result: BacktestResult) -> None:
        """Save aggregate metrics and equity curve to results_dir."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix    = f"{self.risk_profile}_{self.term}_{timestamp}"

        # Save aggregate metrics as CSV
        agg_df = pd.DataFrame([result.aggregate()])
        agg_path = self.results_dir / f"{prefix}_aggregate.csv"
        agg_df.to_csv(agg_path, index=False)

        # Save full trade log
        all_trades = []
        for fold in result.folds:
            for record in fold.trade_log:
                record["fold"] = fold.fold_idx
                all_trades.append(record)
        if all_trades:
            trades_df  = pd.DataFrame(all_trades)
            trades_path = self.results_dir / f"{prefix}_trades.csv"
            trades_df.to_csv(trades_path, index=False)

        logger.info("Results saved to %s", self.results_dir)
        self._plot_equity_curve(result, prefix)

    def _plot_equity_curve(self, result: BacktestResult, prefix: str) -> None:
        """Save a simple equity curve plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            curve = result.equity_curve()
            if curve.empty:
                return

            fig, ax = plt.subplots(figsize=(12, 5))
            curve.plot(ax=ax, color="#0F6E56", linewidth=1.5, label="Portfolio value")
            ax.axhline(
                self.config["env"]["initial_capital"],
                color="#888888", linestyle="--", linewidth=0.8, label="Initial capital",
            )
            agg = result.aggregate()
            ax.set_title(
                f"latent.alpha | {self.risk_profile} / {self.term} | "
                f"Sharpe: {agg.get('mean_sharpe', 0):.2f} | "
                f"Max DD: {agg.get('mean_max_drawdown_pct', 0):.1f}%",
                fontsize=11,
            )
            ax.set_ylabel("Portfolio value (₹)")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()

            plot_path = self.results_dir / f"{prefix}_equity_curve.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info("Equity curve saved to %s", plot_path)

        except Exception as exc:
            logger.warning("Could not plot equity curve: %s", exc)
