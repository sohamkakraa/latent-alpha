# Causal, Reproducible Alpha Optimization

## Status

This is a paused work-in-progress checkpoint intended for continuation on
another machine.

- The original A/B/C results and the partial `architecture_v1` search are
  non-authoritative.
- Long-running architecture and sentiment jobs were stopped.
- Completed GDELT components and experiment fold state are resumable local
  artifacts, but are intentionally not part of Git.
- The implementation includes substantial causal, provenance, baseline,
  sentiment-validation, and orchestration work, but independent review found
  additional blockers listed below.

Do not restart ranked searches until the P0 items are resolved and the complete
test/smoke suite passes.

## Objective

Select a system using robust out-of-sample performance, not a lucky fold:

- Primary signal: median OOS Sharpe.
- Penalties: fold/seed instability, drawdown, turnover, benchmark return
  shortfall, and economically inert exposure.
- Guardrails: meaningful net return, positive severe-stress performance,
  complete benchmark coverage, and no failed/missing replicate.
- Stopping rule: stop after two consecutive search rounds improve the robust
  composite by less than 3%, or all remaining candidates fail a guardrail.

## Architectural invariants

### 1. Causal execution clock

Use one explicit clock across market features, portfolio state, sentiment,
execution, reward, and logs:

1. Observe market and portfolio state available at close `t-1`.
2. Produce the action after close `t-1`.
3. Execute at open `t` (or another explicitly modeled later price).
4. Mark to market at close `t`.
5. Make the close-`t` state available only to the next decision.

Portfolio weights, cash ratio, stop-loss state, and portfolio value must not
reveal the same close used for execution. Stop liquidation is queued for the
next executable price.

### 2. Nested model training

- Outer walk-forward: 36-month training / 6-month test windows.
- Development folds: 0–6.
- Untouched lockbox folds: 7–10.
- LightGBM, PatchTST, and GNN must produce purged, date-indexed OOF predictions
  for RL training and Ridge stacking.
- Final refitted channel models are used only for inner eval and outer OOS.
- A failed channel fit clears current-fold state and cannot reuse a prior fold.
- Ridge inference requires the complete registered channel set and exact shapes.

### 3. Economically valid portfolio accounting

- Cash-equity mode is long-only and caps effective leverage at 1.0.
- Execute sells first, then scale buys to cash remaining after slippage and fees.
- Reward turnover uses actual traded notional or explicit pre/post-trade
  weights, not passive price drift.
- First-step drawdown uses initial capital as the starting peak.
- Holding locks begin on entry in long-only mode.
- Benchmark and all baselines use the identical causal clock and costs.

### 4. Fold-scoped state and fail-closed evaluation

- Model, policy, replay, normalization, stop sentinel, and checkpoints are
  namespaced by immutable run and fold identity.
- A fold never resumes another fold's policy or replay buffer.
- Missing `VecNormalize` state is fatal for a normalized policy.
- Resume is either exact (including RNG/environment state) or the interrupted
  fold restarts and is labeled non-equivalent.
- Keep only the newest complete atomic checkpoint generation.

### 5. Immutable provenance

Run identity must include:

- Resolved scientific config and candidate overrides.
- Code and untracked-source content hash.
- `uv.lock` and concrete dependency/runtime identity.
- Market-data content/calendar hash.
- Sentiment schema/content/builder hash.
- Algorithm, device, seed, fold set, stage plan, and timestep budget.

Existing manifests and fold state are verified, never overwritten or reused
under a mismatched identity. Search/output directories use exclusive process
locks.

### 6. Sentiment contract

- Historical training fails closed without a compatible historical cache.
  Current live news is never broadcast over historical dates.
- Canonical schema identity includes ordered tickers, ordered axes, source/model
  version, query aliases, transformation version, and availability policy.
- Validation checks each ticker/axis across the full date range for coverage,
  variance, finite values, and source quality.
- Daily sentiment is available only after the prior session; hourly/weekly
  modes require frequency-appropriate timestamp aggregation.
- GDELT chunk state distinguishes no coverage from failed/incomplete responses.
- Sentiment OFF and ON are explicit, isolated experiment arms.

### 7. Search and promotion rules

- Successive-halving state is keyed by an immutable search-spec hash.
- Every expected seed/fold must complete with finite metrics before ranking.
- Failed or partial candidates cannot advance.
- Required benchmark failure aborts ranked evaluation.
- Score includes meaningful return/exposure and benchmark return shortfall.
- Final validation loads the exact frozen winning config.
- Cost/delay stress evaluates the same trained policy; it does not retrain under
  each stress scenario.
- Promotion requires all seeds, folds, and scenarios to pass.

## Work already implemented

The current WIP includes:

- Exact sample dates for purged LightGBM CV.
- Cross-sectional rank correction.
- Date-wide purged PatchTST split.
- Correct GNN forward-return target and reduced nonzero node schema.
- VAE feature aliases.
- OOS channel rebinding to test-date panels.
- Separate noisy training and clean eval/test sentiment functions.
- Versioned sentiment/cache validation scaffolding and resumable GDELT builder.
- Run manifests, config/code identity, selected folds, seed propagation, and
  fold persistence.
- Cost-aware cash, equal-weight, momentum, direct-alpha, and Nifty baselines.
- Immutable experiment-arm definitions.
- Successive-halving and lockbox orchestration scaffolding.
- Locked dependencies in `uv.lock`.
- Regression tests for the implemented causal/split/cache/provenance behavior.

These are foundations, not proof that the remaining pipeline is valid.

## Required continuation order

### P0: Correctness before optimization

1. Replace the remaining same-close portfolio-state path with
   previous-close observation → next-open execution → same-day close valuation.
2. Implement true purged OOF channel predictions for RL and Ridge.
3. Fix cash-safe long-only execution, reward turnover, initial drawdown, and
   entry holding locks.
4. Purge GNN validation labels and derive adjacency/regime labels from raw,
   unnormalized returns.
5. Clear all fitted state per fold and require complete meta-channel inference.
6. Make missing normalization, benchmark data, or requested folds fatal.
7. Repair lagged buy-and-hold initialization.

### P1: Provenance and durable execution

1. Add market/sentiment/lock/dependency hashes to run identity.
2. Namespace checkpoints and stop sentinels by run and fold.
3. Add exact search-spec/state validation and process locks.
4. Reject incomplete stages and failed replicates.
5. Make night/background shutdown wait for confirmed atomic checkpointing.

### P2: Sentiment

1. Enforce ordered schema and builder identities.
2. Remove historical live-news fallback.
3. Add chunk-quality manifests and per-ticker/per-axis validation.
4. Finish the GDELT cache, then run identical OFF/ON ablations.

### P3: Model selection

1. Run fast causal smokes.
2. Run strong non-RL baselines.
3. Screen market-only SAC and one channel at a time.
4. Run multi-seed, representative-fold successive halving.
5. Tune the winning architecture.
6. Freeze the resolved winner.

### P4: Lockbox

1. Evaluate the frozen policy on folds 7–10 with at least three seeds.
2. Evaluate the same policy under 1.5× and 2× costs and an extra-bar delay.
3. Report stitched OOS return, Sharpe, Sortino, Calmar, drawdown, turnover,
   costs, benchmark alpha/information ratio, dispersion, and confidence bounds.
4. Promote only if every economic and completeness guardrail passes.

## Resume on another machine

```bash
git clone <repository-url>
cd latent-alpha
uv sync --frozen --python 3.11
uv run pytest -q
```

Local caches, models, results, and partial GDELT components are excluded from
Git. Copy them separately only if desired; otherwise rebuild them.

Do not resume `architecture_v1`. Start a new search name after the P0–P2 fixes:

```bash
caffeinate -dims python -m experiments.successive_halving \
  --mode architecture --name architecture_v2 --device cpu
```
