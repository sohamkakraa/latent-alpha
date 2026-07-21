# latent.alpha

Python research and trading stack — SAC reinforcement learning agent for equities.

## Layout

```text
.
├── agent/        # SAC agent, adaptive entropy, reward shaping
├── alpha/        # Feature engineering, LightGBM, meta-model, VAE regime
├── backtest/     # Backtesting runner and diagnostics
├── config/       # config.yaml
├── data/         # Market data cache and sentiment DB
├── env/          # Trading environment and data loader
├── experiments/  # Hyperparameter sweep configs and runner
├── llm/          # LLM debate analyst and sentiment cache
├── models/       # Saved model artifacts
├── results/      # Experiment outputs
├── scripts/      # Utility scripts (night trainer, sentiment cache builder)
├── strategy/     # Multi-horizon strategy and term selector
├── tests/        # pytest test suite
├── train.py      # Training entry point
├── simulate.py   # Simulation entry point
└── backtest.py   # Backtest entry point
```

## Reproducible setup

```bash
uv sync --frozen --python 3.11
uv run pytest
```

`uv.lock` pins the complete environment, including LightGBM. Every experiment
writes a resolved config and manifest containing its code/config hash, seed,
dependency versions, device, and data fingerprints.

## Research contract

- Observations and external signals available through bar `t-1` execute at bar
  `t`; the strategy never trades on the close used to construct its features.
- Supervised models fit only inside each outer training window. Their inference
  functions are rebound to causal OOS feature panels for the following test
  window.
- Walk-forward tests use rolling 36-month train / 6-month test windows.
- Folds 0–6 are development data. Folds 7–10 are an untouched lockbox.
- The primary selection score is median OOS Sharpe penalized for fold/seed
  instability, drawdown, and benchmark underperformance.
- Results are compared with cash, equal-weight, momentum, direct-alpha, and
  Nifty baselines under the same costs and execution lag.

Historical A/B/C metrics predate these controls and are not comparable.

## Commands

```bash
# Fast causal smoke test
python -m experiments.runner \
  --experiments market_only --timesteps 1000 --folds 0 --seeds 42

# Persistent architecture search (safe to restart)
caffeinate -dims python -m experiments.successive_halving \
  --mode architecture --name architecture_v1 --device cpu

# Tune the selected architecture
caffeinate -dims python -m experiments.successive_halving \
  --mode hyperparameters --name hyperparameters_v1 \
  --architecture <architecture-winner> --device cpu

# Build versioned historical GDELT sentiment (safe to restart)
caffeinate -dims python scripts/build_gdelt_timeline_cache.py

# Sentiment is opt-in and fails before training if coverage/schema is invalid
python backtest.py --with-sentiment --seed 42
```

Long runs write fold state immediately. SAC recovery checkpoints include the
policy, `VecNormalize`, replay buffer, and RNG-seeded run identity. A stopped
fold is never reported as complete.
