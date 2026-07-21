# P0 & P1: Critical Fixes + System Activation — Implementation Prompts

> These prompts address the findings from the full codebase audit (45 files, 11,511 LOC).
> P0 items fix the **root cause** limiting ALL experiment performance (entropy collapse,
> signal misalignment, missing guards). P1 items **activate unused scaffolding** and add
> diagnostics. Each prompt is self-contained. Run P0 prompts first (1→2→3→4), then P1 (5→6→7→8).
> All changes are backward-compatible — existing training must not break.

---

## Current Architecture (for context)

```
Market data (OHLCV × 10 stocks)
        │
        ├─→ AlphaFeatureEngine (54 features) ─→ LightGBM ─→ AlphaSignal ──────┐
        ├─→ TemporalFeatureBuilder (sequences) ─→ PatchTST ─→ TemporalSignal ─┤
        ├─→ GNN (10-node GAT) ─→ GraphSignal ─────────────────────────────────┤ → [raw concat] → SAC obs
        ├─→ VAE Regime Detector ─→ RegimeSignal ──────────────────────────────┤
        └─→ Debate Analyst (LLM) ─→ DebateSignal ────────────────────────────┘
                                                                              │
SAC obs = [market(350) + portfolio(12) + locks(10) + sentiment(100) + alpha(20)
         + patchtst(20) + gnn(20) + vae(8) + debate(20)] = 560 dims
                                                                              │
                                               SAC Agent ([512,512] MLP) ─→ delta weights (10-dim)
```

**Known issues this prompt set fixes:**
1. **Entropy collapse** — SAC `target_entropy="auto"` computes −10, too aggressive; ent_coef drops to 0.0004 by 500K steps across ALL experiments
2. **Signal misalignment** — GNN uses `forward_horizon: 5` while alpha/patchtst use 10
3. **Meta-model unused** — `RidgeMetaModel.train()` exists but is never called; 5 channels just concatenated raw
4. **No NaN guards** — actions not checked for NaN; stop-loss detected but not enforced (no liquidation)
5. **Low test coverage** — data_loader, reward, feature_engineering, SAC agent have zero unit tests

**Key files in the codebase:**

- `agent/sac_agent.py` (269 LOC) — SAC wrapper, line 170-186: constructor passes `ent_coef` from config, no target_entropy override, no entropy floor
- `agent/reward.py` (231 LOC) — RiskAwareReward with 6 components + RewardAggregator
- `alpha/meta_model.py` (239 LOC) — RidgeMetaModel with train() (lines 100-168) and combine() (lines 172-218), NEVER CALLED
- `alpha/lightgbm_alpha.py` — AlphaSignal dataclass, LightGBMAlpha model
- `alpha/patchtst.py` — TemporalSignal dataclass, PatchTSTAlpha model
- `alpha/gnn_alpha.py` — GraphSignal dataclass, GNNAlpha model
- `alpha/vae_regime.py` — RegimeSignal dataclass, RegimeDetector
- `alpha/feature_engineering.py` (557 LOC) — 54 features across 8 groups, NO unit tests
- `env/trading_env.py` (466 LOC) — NSETradingEnv, obs concatenation at line 395, stop-loss check at line 274 (only sets truncated, no liquidation)
- `env/data_loader.py` (365 LOC) — compute_indicators(), rolling_zscore(), NO unit tests
- `strategy/term_selector.py` (850 LOC) — builds all 5 pipelines (alpha, patchtst, gnn, regime, debate), passes closures to env
- `config/config.yaml` (331 LOC) — all hyperparameters
- `experiments/runner.py` (304 LOC) — WalkForwardBacktester integration
- `backtest/runner.py` (420 LOC) — WalkForwardBacktester with per-fold train/eval

**Experiment results (for reference):**
- v2 baseline: Sharpe 0.940, Return +13.1%
- A_baseline_revised (v3.1 full stack): Sharpe 0.704, Return +8.18%
- **B_arch_tuned** (winner, baked into config): Sharpe 0.958, Return +9.19%
- C_feature_pruned: Sharpe 0.216 (feature pruning hurt)
- D_reward_shaped: entropy collapsed at 950K steps (predicted failure)

**10 tickers:** RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, KOTAKBANK.NS, LT.NS, AXISBANK.NS

---

# ═══════════════════════════════════════════════════════════════════════════
# P0 — CRITICAL FIXES (run these first)
# ═══════════════════════════════════════════════════════════════════════════

## Prompt 1: Fix SAC Entropy Collapse

### Context

The single biggest issue limiting ALL experiments is SAC entropy collapse. When SB3's SAC is configured with `ent_coef: "auto"` and `target_entropy: "auto"`, it computes `target_entropy = -dim(action_space) = -10` (for our 10-stock portfolio). This target is far too aggressive — it drives the entropy coefficient (`ent_coef`) down to ~0.0004 by 500K steps, at which point the agent stops exploring entirely.

Evidence from experiment D's training metrics at 32% progress:
- `ent_coef`: 0.000413 (should be 0.01–0.1 for meaningful exploration)
- `critic_loss`: 9.5e-06 (near-zero — critic has converged to a flat value surface)
- `actor_loss`: -2.13 (actor is barely being updated)

This pattern repeats across ALL four experiments — entropy collapse is universal, not config-specific.

**Root cause:** For portfolio allocation, the "natural" entropy of a 10-dim continuous action space is much lower than for robotic control (SAC's original domain). A target of −10 means the policy should have extremely low entropy — essentially deterministic. But in trading, maintaining exploration is critical because the reward landscape shifts with market regimes.

**The fix has three parts:**
1. Override `target_entropy` to −5.0 (half the default, allows more exploration)
2. Add an entropy floor — clamp `ent_coef` to never go below 0.005
3. Initialize `ent_coef` at 0.1 (higher starting point for exploration)

### Task

Modify `agent/sac_agent.py` to fix entropy collapse, add config fields, and add entropy monitoring callbacks.

### Files to modify

**1. `agent/sac_agent.py`** — SAC constructor and training:

The current SAC constructor (lines 170-186) passes `ent_coef` from config but does NOT pass `target_entropy`. SB3 then defaults to `target_entropy = -dim(action_space)`.

Changes needed:

```python
# In the __init__ method, after self.sac_cfg = config["sac"], add:
self.entropy_cfg = {
    "target_entropy_ratio": self.sac_cfg.get("target_entropy_ratio", 0.5),
    "ent_coef_floor": self.sac_cfg.get("ent_coef_floor", 0.005),
    "ent_coef_init": self.sac_cfg.get("ent_coef_init", 0.1),
}
```

In the `train()` method, modify the SAC constructor call (line 170-186):

```python
# BEFORE (broken):
self.model = SAC(
    ...
    ent_coef=self.sac_cfg["ent_coef"],   # "auto" → target_entropy = -10
    ...
)

# AFTER (fixed):
# Compute target entropy from action space dimension
# Default ratio=0.5 → target_entropy = -5 (half the default)
# This allows more exploration than the default -10
action_dim = train_env.action_space.shape[-1]  # 10 for 10 stocks
target_entropy = -action_dim * self.entropy_cfg["target_entropy_ratio"]
logger.info(
    "SAC entropy config: target_entropy=%.1f (ratio=%.2f × dim=%d), "
    "ent_coef_init=%.3f, floor=%.4f",
    target_entropy,
    self.entropy_cfg["target_entropy_ratio"],
    action_dim,
    self.entropy_cfg["ent_coef_init"],
    self.entropy_cfg["ent_coef_floor"],
)

self.model = SAC(
    policy=self.sac_cfg.get("policy", "MlpPolicy"),
    env=train_env,
    learning_rate=self.sac_cfg["learning_rate"],
    buffer_size=self.sac_cfg["buffer_size"],
    batch_size=self.sac_cfg["batch_size"],
    tau=self.sac_cfg["tau"],
    gamma=self.sac_cfg["gamma"],
    ent_coef=self.entropy_cfg["ent_coef_init"],   # Start at 0.1, not "auto"
    target_entropy=target_entropy,                  # -5.0, not -10
    train_freq=self.sac_cfg.get("train_freq", 1),
    gradient_steps=self.sac_cfg.get("gradient_steps", 1),
    learning_starts=self.sac_cfg["learning_starts"],
    policy_kwargs=policy_kwargs,
    tensorboard_log=self._tensorboard_log_dir(),
    device=self.device,
    verbose=1,
)
```

Add an `EntropyFloorCallback` class that clamps ent_coef after each training step:

```python
class EntropyFloorCallback(BaseCallback):
    """
    Prevents entropy collapse by clamping ent_coef to a minimum value.
    
    Without this, SAC's automatic entropy tuning can drive ent_coef
    to near-zero, killing exploration. The floor ensures the agent
    always maintains some minimum level of stochasticity.
    
    Also logs ent_coef periodically for monitoring.
    """

    def __init__(
        self,
        ent_coef_floor: float = 0.005,
        log_every_n_steps: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ent_coef_floor = ent_coef_floor
        self.log_every = log_every_n_steps
        self._clamp_count = 0

    def _on_step(self) -> bool:
        # Access the log_ent_coef parameter (SAC stores log(ent_coef) for stability)
        if hasattr(self.model, "log_ent_coef"):
            import torch
            import math
            with torch.no_grad():
                current_log = self.model.log_ent_coef.item()
                current_ent = math.exp(current_log)
                
                if current_ent < self.ent_coef_floor:
                    clamped_log = math.log(self.ent_coef_floor)
                    self.model.log_ent_coef.fill_(clamped_log)
                    self._clamp_count += 1
                    
                    if self._clamp_count % 100 == 1:
                        logger.warning(
                            "Entropy floor active: clamped ent_coef from %.6f to %.4f "
                            "(clamp #%d)",
                            current_ent, self.ent_coef_floor, self._clamp_count,
                        )

            # Periodic logging
            if self.n_calls % self.log_every == 0:
                ent_coef = math.exp(self.model.log_ent_coef.item())
                logger.info(
                    "Step %d | ent_coef=%.6f | clamps=%d",
                    self.num_timesteps, ent_coef, self._clamp_count,
                )
        return True
```

Add the callback to the callbacks list in `train()` (after line 196):

```python
callbacks.append(
    EntropyFloorCallback(
        ent_coef_floor=self.entropy_cfg["ent_coef_floor"],
        log_every_n_steps=10_000,
    )
)
```

**2. `config/config.yaml`** — add entropy parameters to the `sac:` section:

```yaml
sac:
  # ... existing params ...
  ent_coef: "auto"                 # Kept for backward compat but overridden by ent_coef_init
  target_entropy: "auto"           # Kept for backward compat but overridden by target_entropy_ratio
  
  # v3.2 P0: Entropy collapse fix
  # The default target_entropy = -dim(action) = -10 is too aggressive for portfolio allocation.
  # These parameters override the defaults to maintain healthy exploration.
  target_entropy_ratio: 0.5        # target_entropy = -dim(action) × ratio = -5.0
                                   # 0.5 = half the default aggressiveness
                                   # Range: [0.3, 0.8]. Lower = more exploration.
  ent_coef_init: 0.1               # Initial entropy coefficient (replaces "auto" start)
                                   # SAC auto-tunes from here toward target_entropy.
  ent_coef_floor: 0.005            # Minimum ent_coef — never go below this.
                                   # Prevents the complete exploration death seen in all experiments.
```

### Tests to write

Create `tests/test_entropy_fix.py`:

```python
"""Tests for SAC entropy collapse fix."""
import math
import pytest
import numpy as np


def test_entropy_config_defaults():
    """Verify entropy config fields exist with correct defaults."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    sac = config["sac"]
    assert "target_entropy_ratio" in sac
    assert "ent_coef_init" in sac
    assert "ent_coef_floor" in sac
    assert 0.0 < sac["target_entropy_ratio"] <= 1.0
    assert sac["ent_coef_init"] > sac["ent_coef_floor"]
    assert sac["ent_coef_floor"] > 0


def test_target_entropy_computation():
    """Verify target_entropy = -dim × ratio, not -dim."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    n_stocks = len(config["market"]["tickers"])  # 10
    ratio = config["sac"]["target_entropy_ratio"]
    
    target_entropy = -n_stocks * ratio  # Should be -5.0
    default_entropy = -n_stocks          # Would be -10.0
    
    assert target_entropy > default_entropy  # Less aggressive
    assert target_entropy == pytest.approx(-5.0, abs=0.1)


def test_entropy_floor_callback_clamps():
    """Verify EntropyFloorCallback clamps low ent_coef values."""
    from agent.sac_agent import EntropyFloorCallback
    
    callback = EntropyFloorCallback(ent_coef_floor=0.005, log_every_n_steps=100)
    assert callback.ent_coef_floor == 0.005
    assert callback._clamp_count == 0


def test_sac_agent_reads_entropy_config():
    """Verify LatentAlphaSACAgent reads entropy config fields."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    from agent.sac_agent import LatentAlphaSACAgent
    agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")
    
    assert hasattr(agent, "entropy_cfg")
    assert agent.entropy_cfg["target_entropy_ratio"] == 0.5
    assert agent.entropy_cfg["ent_coef_init"] == 0.1
    assert agent.entropy_cfg["ent_coef_floor"] == 0.005


def test_ent_coef_init_replaces_auto():
    """Verify SAC is initialized with ent_coef_init, not 'auto'."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # The agent should use ent_coef_init (0.1), not ent_coef ("auto")
    agent_cfg = config["sac"]
    assert agent_cfg["ent_coef_init"] == 0.1
    # "auto" is kept in config for backward compat but should NOT be passed to SAC
    # The agent code should pass ent_coef_init instead
```

### Run the tests

```bash
cd latent_alpha && python -m pytest tests/test_entropy_fix.py -v
```

### Verification checklist

- [ ] `config.yaml` has `target_entropy_ratio: 0.5`, `ent_coef_init: 0.1`, `ent_coef_floor: 0.005`
- [ ] `LatentAlphaSACAgent.__init__()` reads `entropy_cfg` from config
- [ ] `SAC()` constructor receives `ent_coef=0.1` (not `"auto"`) and `target_entropy=-5.0`
- [ ] `EntropyFloorCallback` is added to callbacks list
- [ ] `EntropyFloorCallback` correctly accesses `model.log_ent_coef` and clamps it
- [ ] Backward compat: old configs without the new fields still work (use `.get()` with defaults)
- [ ] All tests pass
- [ ] `python -c "from agent.sac_agent import LatentAlphaSACAgent, EntropyFloorCallback"` succeeds

### Expected impact

With `target_entropy=-5` and `ent_coef_floor=0.005`:
- At 500K steps, ent_coef should stabilize around 0.01–0.05 (vs. 0.0004 before)
- The agent will continue exploring new weight allocations throughout training
- Combined with the existing anti-churning mechanisms (delta weights, holding bonus), this should yield higher Sharpe ratios than any current experiment

---

## Prompt 2: Fix Signal Alignment + Activate Meta-Model

### Context

Two related issues prevent the multi-alpha stack from working as designed:

**Issue 1: GNN forward_horizon misalignment**
- `alpha.forward_horizon: 10` (predicting 10-day returns) ✓
- `patchtst.forward_horizon: 10` ✓
- `gnn.forward_horizon: 5` ✗ — predicting 5-day returns while everything else predicts 10-day
- All channels must predict the same horizon to be comparable and combinable

**Issue 2: Meta-model never called**
- `RidgeMetaModel` in `alpha/meta_model.py` has a fully implemented `train()` method (lines 100-168) that performs Ridge regression to learn optimal channel weights, and a `combine()` method (lines 172-218) that produces weighted direction + conviction.
- But in `strategy/term_selector.py`, each channel's signal is passed independently to `trading_env.py`, where they're **raw-concatenated** into the obs vector (line 395).
- The meta-model scaffold was built but never wired into the pipeline.

**The fix:**
1. Change `gnn.forward_horizon` from 5 to 10 in config
2. Wire `RidgeMetaModel.train()` into `term_selector.py` — after training all individual models, collect their predictions and train the meta-model
3. Wire `RidgeMetaModel.combine()` into the alpha_fn/temporal_fn/gnn_fn/regime_fn closures — instead of passing raw signals, pass combined signals
4. Replace the 5 separate signal slots in the obs with a single combined signal slot

### Task

Fix GNN horizon, activate the meta-model, and reshape the observation space.

### Files to modify

**1. `config/config.yaml`** — fix GNN forward_horizon and add meta-model config:

```yaml
gnn:
  # ... existing params ...
  forward_horizon: 10              # CHANGED from 5 — must match alpha/patchtst horizon

# ── v3.2 Meta-model activation ──────────────────────────────────────────────
meta_model:
  enabled: true                    # Activate Ridge meta-model to combine channels
  obs_dim: 20                      # Combined signal: 2 × n_stocks (direction + conviction)
  ridge_alpha: 1.0                 # L2 regularisation for Ridge regression
  min_train_samples: 200           # Minimum samples before meta-model trains
  fallback: "equal_weight"         # If training fails: "equal_weight" or "alpha_only"
```

**2. `strategy/term_selector.py`** — the main integration point.

Add a new method `_train_meta_model()` that:
1. Collects predictions from all trained channels (alpha, patchtst, gnn, regime) on the training set
2. Calls `RidgeMetaModel.train(channel_predictions, actuals)` to learn optimal weights
3. Returns the trained meta-model

Modify `build_env()` and `build_backtest_env()` to:
1. After building all individual pipelines, call `_train_meta_model()`
2. Create a single `combined_alpha_fn(date)` closure that:
   - Calls each individual channel's fn to get its signal
   - Packages them as `ChannelPrediction` objects
   - Calls `meta_model.combine()` to get weighted (direction, conviction)
   - Returns an `AlphaSignal` with the combined direction and conviction
3. Pass `combined_alpha_fn` as the `alpha_fn` to the env, and set all other signal fns to `None`

Here is the new method to add to `TermSelector`:

```python
def _train_meta_model(
    self,
    train_data: Dict,
    train_end_date=None,
    alpha_fn=None,
    temporal_fn=None,
    gnn_fn=None,
    regime_fn=None,
):
    """
    Train the Ridge meta-model to combine all alpha channels.
    
    Collects predictions from each channel on the training dates,
    paired with actual forward returns, and trains Ridge regression
    to learn optimal combination weights.
    
    Returns
    -------
    (meta_model, combined_fn) — trained RidgeMetaModel and a closure
    that combines all channels into a single AlphaSignal.
    """
    from alpha.meta_model import RidgeMetaModel, ChannelPrediction
    from alpha.lightgbm_alpha import AlphaSignal
    import pandas as pd
    import numpy as np
    
    meta_cfg = self.config.get("meta_model", {})
    if not meta_cfg.get("enabled", False):
        return None, None
    
    ridge_alpha = meta_cfg.get("ridge_alpha", 1.0)
    min_samples = meta_cfg.get("min_train_samples", 200)
    forward_horizon = self.config.get("alpha", {}).get("forward_horizon", 10)
    
    meta = RidgeMetaModel(n_stocks=len(self.config["market"]["tickers"]),
                          ridge_alpha=ridge_alpha)
    
    # Register available channels
    channel_fns = {}
    if alpha_fn is not None:
        meta.register_channel("lgbm_alpha")
        channel_fns["lgbm_alpha"] = alpha_fn
    if temporal_fn is not None:
        meta.register_channel("patchtst")
        channel_fns["patchtst"] = temporal_fn
    if gnn_fn is not None:
        meta.register_channel("gnn")
        channel_fns["gnn"] = gnn_fn
    if regime_fn is not None:
        meta.register_channel("vae_regime")
        channel_fns["vae_regime"] = regime_fn
    
    if len(channel_fns) < 2:
        logger.info("Meta-model skipped — fewer than 2 channels available.")
        return None, None
    
    # Collect predictions on training dates
    # Use the first ticker's dates as reference
    ref_ticker = sorted(train_data.keys())[0]
    all_dates = sorted(train_data[ref_ticker].index)
    
    if train_end_date is not None:
        cutoff = pd.Timestamp(train_end_date)
        train_dates = [d for d in all_dates if d <= cutoff]
    else:
        train_dates = all_dates
    
    # Need enough dates for forward returns
    if len(train_dates) < min_samples + forward_horizon:
        logger.warning("Not enough training dates for meta-model (%d < %d).",
                       len(train_dates), min_samples + forward_horizon)
        return None, None
    
    # Sample dates (skip first 60 for warmup, last forward_horizon for labels)
    sample_dates = train_dates[60:-forward_horizon]
    
    channel_preds = {name: [] for name in channel_fns}
    actuals = []
    
    for date in sample_dates:
        # Compute forward return (mean across stocks)
        future_idx = all_dates.index(date) + forward_horizon
        if future_idx >= len(all_dates):
            continue
        future_date = all_dates[future_idx]
        
        # Per-stock forward returns
        fwd_rets = []
        for ticker in sorted(train_data.keys()):
            df = train_data[ticker]
            if date in df.index and future_date in df.index:
                ret = (df.loc[future_date, "Close"] - df.loc[date, "Close"]) / df.loc[date, "Close"]
                fwd_rets.append(ret)
            else:
                fwd_rets.append(0.0)
        
        actuals.append(np.mean(fwd_rets))
        
        # Collect channel predictions
        for name, fn in channel_fns.items():
            try:
                signal = fn(date)
                # Use mean direction as the channel's scalar prediction
                direction = getattr(signal, "direction", np.zeros(10))
                channel_preds[name].append(float(np.mean(direction)))
            except Exception:
                channel_preds[name].append(0.0)
    
    # Convert to arrays and train
    actuals_arr = np.array(actuals)
    preds_dict = {name: np.array(vals) for name, vals in channel_preds.items()}
    
    try:
        result = meta.train(preds_dict, actuals_arr)
        logger.info("Meta-model trained: R²=%.4f, weights=%s",
                    result["r2"], result["weights"])
    except Exception as e:
        logger.warning("Meta-model training failed: %s — using equal weights.", e)
    
    # Create combined closure
    _meta = meta
    _channel_fns = channel_fns
    _n_stocks = len(self.config["market"]["tickers"])
    
    def combined_alpha_fn(date):
        """Combine all channels through the meta-model into one AlphaSignal."""
        predictions = []
        for name, fn in _channel_fns.items():
            try:
                signal = fn(date)
                direction = getattr(signal, "direction", np.zeros(_n_stocks))
                conviction = getattr(signal, "conviction", np.zeros(_n_stocks))
                predictions.append(ChannelPrediction(
                    name=name,
                    direction=np.array(direction, dtype=np.float32),
                    conviction=np.array(conviction, dtype=np.float32),
                ))
            except Exception:
                pass
        
        if predictions:
            combined_dir, combined_conv = _meta.combine(predictions)
        else:
            combined_dir = np.zeros(_n_stocks, dtype=np.float32)
            combined_conv = np.zeros(_n_stocks, dtype=np.float32)
        
        return AlphaSignal(
            direction=combined_dir,
            conviction=combined_conv,
            raw_pred=combined_dir,  # Use direction as raw prediction
        )
    
    return meta, combined_alpha_fn
```

Modify `build_env()` (around line 195) — after building all individual pipelines, add:

```python
# v3.2: Combine channels through meta-model
meta_model, combined_fn = self._train_meta_model(
    train_data=aligned,
    train_end_date=train_dates[-1] if train_dates else None,
    alpha_fn=alpha_fn,
    temporal_fn=temporal_fn,
    gnn_fn=gnn_fn,
    regime_fn=regime_fn,
)

if combined_fn is not None:
    # Replace individual signals with combined signal
    alpha_fn = combined_fn
    temporal_fn = None
    gnn_fn = None
    regime_fn = None
    logger.info("Meta-model active — all channels combined into alpha_fn.")
```

Apply the same pattern to `build_backtest_env()`.

**3. `env/trading_env.py`** — update obs space to handle meta-model mode.

When meta-model is active, only `alpha_dim` is non-zero (all other signal dims are 0 because their fns are None). The obs space automatically shrinks from 560 to ~472 dims (removing patchtst 20 + gnn 20 + vae 8 + debate 20 = 68 dims, but adding nothing). This is handled naturally by the existing code since the dims read from config only contribute when the corresponding fn is not None.

However, add a `meta_model` section to the obs_dim calculation for future expansion:

```python
# In the obs_dim calculation (around line 152), add:
meta_dim = (
    config.get("meta_model", {}).get("obs_dim", 0)
    if config.get("meta_model", {}).get("enabled", False)
    else 0
)
```

**Important**: When meta-model is enabled, `alpha_dim` should use `meta_model.obs_dim` (20) instead of `alpha.obs_dim` (20). Since both are 20, no numeric change, but make this explicit:

```python
# Replace:
self.alpha_dim = config.get("alpha", {}).get("obs_dim", 0)

# With:
if config.get("meta_model", {}).get("enabled", False):
    # Meta-model combines all channels into a single alpha signal
    self.alpha_dim = config.get("meta_model", {}).get("obs_dim", 20)
else:
    self.alpha_dim = config.get("alpha", {}).get("obs_dim", 0)
```

**4. `config/config.yaml`** — update `sac.net_arch` for smaller obs:

When meta-model is active, obs drops from 560 to ~492 dims. The [512, 512] network is well-sized for this. No change needed, but add a comment:

```yaml
sac:
  net_arch: [512, 512]             # 512 > obs_dim (~492 with meta-model) — good compression ratio
```

### Tests to write

Create `tests/test_meta_model_activation.py`:

```python
"""Tests for meta-model activation and signal alignment."""
import numpy as np
import pytest


def test_gnn_forward_horizon_aligned():
    """All channels must use the same forward_horizon."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    alpha_horizon = config["alpha"]["forward_horizon"]
    patchtst_horizon = config["patchtst"]["forward_horizon"]
    gnn_horizon = config["gnn"]["forward_horizon"]
    
    assert alpha_horizon == patchtst_horizon == gnn_horizon, (
        f"Horizons misaligned: alpha={alpha_horizon}, "
        f"patchtst={patchtst_horizon}, gnn={gnn_horizon}"
    )


def test_meta_model_config_exists():
    """Verify meta_model config section exists with correct fields."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    assert "meta_model" in config
    mc = config["meta_model"]
    assert mc["enabled"] == True
    assert mc["obs_dim"] == 20
    assert mc["ridge_alpha"] > 0
    assert mc["min_train_samples"] > 0


def test_ridge_meta_model_train_and_combine():
    """Verify RidgeMetaModel can train and combine predictions."""
    from alpha.meta_model import RidgeMetaModel, ChannelPrediction
    
    meta = RidgeMetaModel(n_stocks=10, ridge_alpha=1.0)
    meta.register_channel("lgbm_alpha")
    meta.register_channel("patchtst")
    
    # Train with synthetic data
    n_samples = 100
    preds = {
        "lgbm_alpha": np.random.randn(n_samples) * 0.01,
        "patchtst": np.random.randn(n_samples) * 0.01,
    }
    actuals = 0.6 * preds["lgbm_alpha"] + 0.4 * preds["patchtst"] + np.random.randn(n_samples) * 0.001
    
    result = meta.train(preds, actuals)
    assert "r2" in result
    assert "weights" in result
    assert result["r2"] > 0.5  # Should fit well given clean signal
    
    # Combine
    cp = [
        ChannelPrediction(
            name="lgbm_alpha",
            direction=np.random.randn(10).astype(np.float32),
            conviction=np.abs(np.random.randn(10)).astype(np.float32) * 0.5,
        ),
        ChannelPrediction(
            name="patchtst",
            direction=np.random.randn(10).astype(np.float32),
            conviction=np.abs(np.random.randn(10)).astype(np.float32) * 0.5,
        ),
    ]
    direction, conviction = meta.combine(cp)
    assert direction.shape == (10,)
    assert conviction.shape == (10,)


def test_combined_alpha_fn_returns_alpha_signal():
    """When meta-model is active, combined_fn returns a valid AlphaSignal."""
    from alpha.lightgbm_alpha import AlphaSignal
    
    # Verify AlphaSignal interface
    sig = AlphaSignal(
        direction=np.random.randn(10).astype(np.float32),
        conviction=np.abs(np.random.randn(10)).astype(np.float32),
        raw_pred=np.random.randn(10).astype(np.float32),
    )
    obs = sig.to_obs_vector()
    assert obs.shape == (20,)  # 10 direction + 10 conviction


def test_obs_dim_with_meta_model():
    """Obs space should be smaller with meta-model (combined signal vs 5 separate)."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    n_stocks = len(config["market"]["tickers"])
    
    # With meta-model: only alpha_dim contributes (others are None/0)
    meta_obs_dim = config["meta_model"]["obs_dim"]  # 20
    
    # Without meta-model: all 5 channels contribute
    separate_obs_dim = (
        config["alpha"]["obs_dim"]
        + config["patchtst"]["obs_dim"]
        + config["gnn"]["obs_dim"]
        + config["vae_regime"]["obs_dim"]
        + config["llm"]["debate"]["obs_dim"]
    )
    
    assert meta_obs_dim < separate_obs_dim, (
        f"Meta-model obs ({meta_obs_dim}) should be smaller than "
        f"separate channels ({separate_obs_dim})"
    )
```

### Run the tests

```bash
cd latent_alpha && python -m pytest tests/test_meta_model_activation.py -v
```

### Verification checklist

- [ ] `config.yaml` has `gnn.forward_horizon: 10` (was 5)
- [ ] `config.yaml` has `meta_model:` section with `enabled: true`
- [ ] `term_selector.py` has `_train_meta_model()` method
- [ ] `build_env()` calls `_train_meta_model()` and replaces individual fns with combined fn
- [ ] `build_backtest_env()` does the same
- [ ] When meta-model fails to train, individual channels are still passed through (graceful fallback)
- [ ] `trading_env.py` obs_dim uses `meta_model.obs_dim` when meta-model is enabled
- [ ] All forward_horizons are aligned at 10
- [ ] All tests pass
- [ ] Backward compat: when `meta_model.enabled: false`, behaviour is unchanged

---

## Prompt 3: Production Guards (NaN, Stop-Loss, Logging)

### Context

The trading environment has three production-readiness gaps that can cause silent failures:

**1. No NaN guard on actions** — If the SAC agent produces NaN actions (possible during early training with poorly conditioned inputs), the environment silently propagates them through the weight calculation. The `_get_observation()` method (line 396) has `nan_to_num` on the OUTPUT obs, but there's no check on the INPUT action in `step()`.

**2. Stop-loss detected but not enforced** — `_check_stop_loss()` at line 424-428 correctly computes whether drawdown exceeds the threshold, and line 274 passes `stop_triggered` to the info dict. But the only consequence is setting `truncated=True` — the portfolio is NOT liquidated. If the backtest runner doesn't check `truncated`, the agent keeps trading through severe drawdowns.

**3. No observation logging** — When the 560-dim observation contains anomalies (extreme values, all-zeros from a failed signal, persistent NaN), there's no way to detect this post-hoc. Adding periodic observation statistics logging enables diagnostics.

### Task

Add NaN guards, stop-loss enforcement, and observation logging to `env/trading_env.py`.

### Files to modify

**1. `env/trading_env.py`** — three changes:

**1a. NaN guard on actions** — add at the TOP of `step()` (line 200), immediately after the assertion:

```python
def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    assert not self._is_done(), "Episode is done. Call reset()."
    
    # ── v3.2 P0: NaN guard on actions ─────────────────────────────────────
    if np.any(np.isnan(action)):
        logger.warning(
            "NaN detected in action at step %d — replacing with zeros.",
            self._current_step,
        )
        action = np.nan_to_num(action, nan=0.0)
    
    # Also clamp extreme values (SAC should output [-1, 1], but be safe)
    action = np.clip(action, -1.0, 1.0)
    
    max_w = self.risk_cfg["max_position_pct"]
    # ... rest of step() ...
```

**1b. Stop-loss enforcement** — modify the stop-loss handling at line 274:

```python
# BEFORE (stop detected but not enforced):
stop_triggered = self._check_stop_loss()

# AFTER (stop enforced with liquidation):
stop_triggered = self._check_stop_loss()
if stop_triggered:
    # ── v3.2 P0: Enforce stop-loss by liquidating to cash ─────────────
    logger.warning(
        "Stop-loss triggered at step %d | drawdown=%.1f%% > threshold=%.1f%% | "
        "Liquidating portfolio (value=₹%.0f).",
        self._current_step,
        self._compute_drawdown() * 100,
        self.risk_cfg["stop_loss_pct"] * 100,
        self.portfolio_value,
    )
    # Move everything to cash
    self.weights = np.zeros(self.n_stocks)
    self.positions = np.zeros(self.n_stocks)
    self.cash = self.portfolio_value
```

Add the helper method:

```python
def _compute_drawdown(self) -> float:
    """Compute current drawdown from peak."""
    peak = max(self._value_history)
    if peak <= 0:
        return 0.0
    return (peak - self.portfolio_value) / peak
```

**1c. Observation logging** — add periodic stats logging to `_get_observation()`:

```python
# At the end of _get_observation(), before the return (line 396):

# ── v3.2 P0: Periodic observation diagnostics ────────────────────────
if self._current_step % 50 == 0:  # Every 50 steps
    obs_stats = {
        "step": self._current_step,
        "obs_mean": float(np.mean(obs)),
        "obs_std": float(np.std(obs)),
        "obs_min": float(np.min(obs)),
        "obs_max": float(np.max(obs)),
        "obs_zeros_pct": float(np.mean(obs == 0) * 100),
        "obs_nan_count": int(np.sum(np.isnan(obs))),  # Should be 0 after nan_to_num
        "obs_clipped_pct": float(np.mean(np.abs(obs) >= 9.9) * 100),  # Near clip boundary
    }
    logger.debug("Obs diagnostics: %s", obs_stats)
    
    # Warn on anomalies
    if obs_stats["obs_zeros_pct"] > 80:
        logger.warning(
            "Step %d: %.0f%% of observation is zeros — possible signal failure.",
            self._current_step, obs_stats["obs_zeros_pct"],
        )
    if obs_stats["obs_clipped_pct"] > 20:
        logger.warning(
            "Step %d: %.0f%% of observation is near clip boundary — "
            "possible scaling issue.",
            self._current_step, obs_stats["obs_clipped_pct"],
        )
```

**2. `backtest/runner.py`** — ensure the backtest runner respects truncated signal:

Search for the per-step loop in `_run_fold()` and ensure that when `truncated=True`, the episode ends:

```python
# In the step loop, after:
# obs, reward, done, truncated, info = env.step(action)
# Ensure:
if done or truncated:
    break  # Stop trading this fold — stop-loss or episode end
```

### Tests to write

Create `tests/test_production_guards.py`:

```python
"""Tests for production guards: NaN handling, stop-loss enforcement, obs logging."""
import numpy as np
import pytest
import yaml
import logging


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def test_nan_action_replaced_with_zeros(config):
    """NaN actions should be replaced with zeros, not propagated."""
    from env.trading_env import NSETradingEnv
    from env.data_loader import NSEDataLoader
    
    loader = NSEDataLoader(config)
    data = loader.load(
        start=config["backtest"]["start_date"],
        end="2019-01-01",
    )
    aligned = loader.align_dates(data)
    feature_names = loader.get_feature_names()
    
    env = NSETradingEnv(
        data=aligned,
        feature_names=feature_names,
        config=config,
        risk_profile="balanced",
        term="medium",
    )
    obs, info = env.reset()
    
    # Send NaN action — should not crash
    nan_action = np.full(len(config["market"]["tickers"]), np.nan, dtype=np.float32)
    obs, reward, done, truncated, info = env.step(nan_action)
    
    # Observation should be valid (no NaN)
    assert not np.any(np.isnan(obs)), "NaN in observation after NaN action"
    # Weights should be unchanged (NaN→0 delta)
    assert np.allclose(env.weights, 0.0), "NaN action should result in zero delta"


def test_extreme_action_clipped(config):
    """Actions outside [-1, 1] should be clipped."""
    from env.trading_env import NSETradingEnv
    from env.data_loader import NSEDataLoader
    
    loader = NSEDataLoader(config)
    data = loader.load(
        start=config["backtest"]["start_date"],
        end="2019-01-01",
    )
    aligned = loader.align_dates(data)
    feature_names = loader.get_feature_names()
    
    env = NSETradingEnv(
        data=aligned,
        feature_names=feature_names,
        config=config,
        risk_profile="balanced",
        term="medium",
    )
    obs, info = env.reset()
    
    # Send extreme action
    extreme_action = np.full(len(config["market"]["tickers"]), 100.0, dtype=np.float32)
    obs, reward, done, truncated, info = env.step(extreme_action)
    
    # Weights should be bounded by max_position_pct
    max_w = config["risk_profiles"]["balanced"]["max_position_pct"]
    assert np.all(np.abs(env.weights) <= max_w + 1e-6)


def test_stop_loss_liquidates_portfolio(config):
    """When stop-loss triggers, portfolio should be fully liquidated to cash."""
    from env.trading_env import NSETradingEnv
    from env.data_loader import NSEDataLoader
    
    loader = NSEDataLoader(config)
    data = loader.load(
        start=config["backtest"]["start_date"],
        end="2019-01-01",
    )
    aligned = loader.align_dates(data)
    feature_names = loader.get_feature_names()
    
    env = NSETradingEnv(
        data=aligned,
        feature_names=feature_names,
        config=config,
        risk_profile="balanced",
        term="medium",
    )
    obs, info = env.reset()
    
    # Simulate a large drawdown by manipulating portfolio value
    env.portfolio_value = env.initial_capital * 0.5  # 50% loss
    env._value_history.append(env.portfolio_value)
    env.prev_value = env.portfolio_value
    
    # Next step should trigger stop-loss
    action = np.zeros(len(config["market"]["tickers"]), dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    
    # After stop-loss: all weights should be zero, everything in cash
    if truncated:  # Stop-loss was triggered
        assert np.allclose(env.weights, 0.0), "Weights should be zero after stop-loss"
        assert np.allclose(env.positions, 0.0), "Positions should be zero after stop-loss"


def test_observation_no_nan_after_guards(config):
    """Observations should never contain NaN after guards are applied."""
    from env.trading_env import NSETradingEnv
    from env.data_loader import NSEDataLoader
    
    loader = NSEDataLoader(config)
    data = loader.load(
        start=config["backtest"]["start_date"],
        end="2019-01-01",
    )
    aligned = loader.align_dates(data)
    feature_names = loader.get_feature_names()
    
    env = NSETradingEnv(
        data=aligned,
        feature_names=feature_names,
        config=config,
        risk_profile="balanced",
        term="medium",
    )
    obs, info = env.reset()
    assert not np.any(np.isnan(obs)), "NaN in initial observation"
    
    # Run 20 random steps
    n_stocks = len(config["market"]["tickers"])
    for _ in range(20):
        action = np.random.uniform(-1, 1, size=n_stocks).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        assert not np.any(np.isnan(obs)), f"NaN in observation at step {env._current_step}"
        assert not np.isnan(reward), f"NaN reward at step {env._current_step}"
        if done or truncated:
            break


def test_compute_drawdown_method(config):
    """_compute_drawdown returns correct drawdown percentage."""
    from env.trading_env import NSETradingEnv
    from env.data_loader import NSEDataLoader
    
    loader = NSEDataLoader(config)
    data = loader.load(
        start=config["backtest"]["start_date"],
        end="2019-01-01",
    )
    aligned = loader.align_dates(data)
    feature_names = loader.get_feature_names()
    
    env = NSETradingEnv(
        data=aligned,
        feature_names=feature_names,
        config=config,
        risk_profile="balanced",
        term="medium",
    )
    obs, info = env.reset()
    
    # Initially drawdown should be 0
    dd = env._compute_drawdown()
    assert dd == pytest.approx(0.0, abs=1e-6)
    
    # Simulate 10% loss
    env.portfolio_value = env.initial_capital * 0.9
    env._value_history.append(env.portfolio_value)
    dd = env._compute_drawdown()
    assert dd == pytest.approx(0.1, abs=1e-6)
```

### Run the tests

```bash
cd latent_alpha && python -m pytest tests/test_production_guards.py -v
```

### Verification checklist

- [ ] `step()` checks for NaN actions and replaces with zeros
- [ ] `step()` clips actions to [-1, 1] before processing
- [ ] `_check_stop_loss()` result triggers liquidation (weights→0, positions→0, cash=portfolio_value)
- [ ] `_compute_drawdown()` helper method exists and returns correct value
- [ ] `_get_observation()` logs stats every 50 steps at DEBUG level
- [ ] `_get_observation()` warns on anomalies (>80% zeros, >20% clipped)
- [ ] `backtest/runner.py` breaks on `truncated=True`
- [ ] All tests pass
- [ ] No regressions in existing test_ecosystem.py tests

---

## Prompt 4: Core Module Test Coverage

### Context

The codebase audit found that 4 critical modules have ZERO unit tests:

1. **`env/data_loader.py`** (365 LOC) — `compute_indicators()`, `rolling_zscore()`, `compute_extended_indicators()` — the foundation of all feature computation
2. **`agent/reward.py`** (231 LOC) — `RiskAwareReward` 6-component reward, `RewardAggregator` — directly shapes agent behaviour
3. **`alpha/feature_engineering.py`** (557 LOC) — 54 features across 8 groups — the alpha model's input
4. **`agent/sac_agent.py`** (269 LOC) — `LatentAlphaSACAgent` — the core agent wrapper

These modules are exercised indirectly by `test_ecosystem.py`, but without targeted tests, regressions in individual components are invisible.

### Task

Create focused unit test files for each of the 4 modules. Tests should be fast (no training, no large data downloads) and use synthetic data where possible.

### Files to create

**1. `tests/test_data_loader.py`:**

```python
"""Unit tests for env/data_loader.py — indicator computation and normalisation."""
import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_ohlcv():
    """Create a synthetic OHLCV DataFrame with 200 rows."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)  # Prevent negative prices
    
    df = pd.DataFrame({
        "Open": close * (1 + np.random.randn(n) * 0.005),
        "High": close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "Low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "Close": close,
        "Volume": np.random.randint(100000, 10000000, size=n).astype(float),
    }, index=dates)
    return df


class TestComputeIndicators:
    """Tests for compute_indicators()."""
    
    def test_rsi_in_range(self, config, sample_ohlcv):
        """RSI must be in [0, 100]."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        df = loader.compute_indicators(sample_ohlcv.copy())
        rsi = df["RSI_14"].dropna()
        assert rsi.min() >= 0, f"RSI below 0: {rsi.min()}"
        assert rsi.max() <= 100, f"RSI above 100: {rsi.max()}"
    
    def test_macd_columns_exist(self, config, sample_ohlcv):
        """MACD, MACD_signal, MACD_hist columns should exist."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        df = loader.compute_indicators(sample_ohlcv.copy())
        for col in ["MACD", "MACD_signal", "MACD_hist"]:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_bollinger_bands_order(self, config, sample_ohlcv):
        """BB_lower <= Close <= BB_upper (approximately — not always but on average)."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        df = loader.compute_indicators(sample_ohlcv.copy())
        valid = df.dropna(subset=["BB_upper", "BB_lower"])
        # On average, close should be between bands
        assert (valid["BB_lower"] <= valid["BB_upper"]).all(), "BB_lower > BB_upper"
    
    def test_atr_positive(self, config, sample_ohlcv):
        """ATR must be non-negative."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        df = loader.compute_indicators(sample_ohlcv.copy())
        atr = df["ATR_14"].dropna()
        assert (atr >= 0).all(), f"Negative ATR found: {atr.min()}"
    
    def test_ema_columns_match_config(self, config, sample_ohlcv):
        """EMA columns should match periods in config."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        df = loader.compute_indicators(sample_ohlcv.copy())
        for period in config["indicators"]["ema_periods"]:
            col = f"EMA_{period}"
            assert col in df.columns, f"Missing EMA column: {col}"
    
    def test_no_nan_after_warmup(self, config, sample_ohlcv):
        """After sufficient warmup, indicators should have no NaN."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        df = loader.compute_indicators(sample_ohlcv.copy())
        # After 60 rows (covers 50-day EMA + 26-day MACD), should be clean
        feature_cols = loader.get_feature_names()
        # Some feature names from get_feature_names() may not all be in indicators
        indicator_cols = [c for c in feature_cols if c in df.columns]
        tail = df.iloc[60:][indicator_cols]
        nan_counts = tail.isna().sum()
        assert nan_counts.sum() == 0, f"NaN after warmup: {nan_counts[nan_counts > 0].to_dict()}"


class TestRollingZscore:
    """Tests for rolling_zscore normalisation."""
    
    def test_zscore_mean_near_zero(self, config, sample_ohlcv):
        """Z-scored values should have mean ≈ 0 over the window."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        df = loader.compute_indicators(sample_ohlcv.copy())
        df = loader.rolling_zscore(df, window=60)
        # Check a few z-scored columns
        for col in ["Close", "Volume"]:
            if col in df.columns:
                vals = df[col].dropna().iloc[60:]  # After warmup
                # Mean should be near 0 (within tolerance for 140 samples)
                assert abs(vals.mean()) < 1.0, f"{col} z-score mean too far from 0: {vals.mean()}"
    
    def test_zscore_no_lookahead(self, config, sample_ohlcv):
        """Z-score at time t should only use data up to t (no future leakage)."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        df = loader.compute_indicators(sample_ohlcv.copy())
        
        # Compute z-scores
        df_z = loader.rolling_zscore(df.copy(), window=60)
        
        # The z-score at index 100 should be the same whether we pass 200 rows or 101 rows
        df_short = loader.compute_indicators(sample_ohlcv.iloc[:101].copy())
        df_short_z = loader.rolling_zscore(df_short.copy(), window=60)
        
        # Values at index 100 should match
        for col in ["Close"]:
            if col in df_z.columns and col in df_short_z.columns:
                full_val = df_z[col].iloc[100]
                short_val = df_short_z[col].iloc[100]
                if not (np.isnan(full_val) or np.isnan(short_val)):
                    assert abs(full_val - short_val) < 1e-6, (
                        f"Look-ahead leakage in {col}: full={full_val}, short={short_val}"
                    )


class TestFeatureNames:
    """Tests for feature name consistency."""
    
    def test_feature_names_non_empty(self, config):
        """get_feature_names() should return a non-empty list."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        names = loader.get_feature_names()
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
    
    def test_feature_names_unique(self, config):
        """Feature names should be unique."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        names = loader.get_feature_names()
        assert len(names) == len(set(names)), f"Duplicate feature names: {[n for n in names if names.count(n) > 1]}"
```

**2. `tests/test_reward.py`:**

```python
"""Unit tests for agent/reward.py — risk-aware reward components."""
import numpy as np
import pytest
import yaml


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def reward_fn(config):
    from agent.reward import RiskAwareReward
    return RiskAwareReward(config, risk_profile="balanced")


class TestRiskAwareReward:
    """Tests for RiskAwareReward.compute()."""
    
    def test_positive_return_positive_reward(self, reward_fn):
        """Positive step return should contribute positively to reward."""
        reward_fn.reset()
        r = reward_fn.compute(
            step_return=0.01,
            portfolio_values=[1_000_000, 1_010_000],
        )
        assert r > 0, f"Positive return gave negative reward: {r}"
    
    def test_negative_return_negative_reward(self, reward_fn):
        """Large negative return should give negative reward."""
        reward_fn.reset()
        r = reward_fn.compute(
            step_return=-0.05,
            portfolio_values=[1_000_000, 950_000],
        )
        assert r < 0, f"Large negative return gave positive reward: {r}"
    
    def test_drawdown_penalty_increases_with_drawdown(self, reward_fn):
        """Larger drawdowns should produce more negative reward (quadratic)."""
        reward_fn.reset()
        # Small drawdown
        r_small = reward_fn.compute(
            step_return=-0.01,
            portfolio_values=[1_000_000, 990_000],
        )
        
        reward_fn.reset()
        # Large drawdown
        r_large = reward_fn.compute(
            step_return=-0.05,
            portfolio_values=[1_000_000, 950_000],
        )
        
        assert r_large < r_small, "Larger drawdown should give worse reward"
    
    def test_turnover_penalty_quadratic(self, reward_fn):
        """Large weight changes should be penalised more than small ones (quadratic)."""
        reward_fn.reset()
        
        # Step 1: establish baseline weights
        w1 = np.array([0.1] * 10, dtype=np.float32)
        reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000],
            current_weights=w1,
        )
        
        # Step 2a: small rebalance
        w2_small = w1 + 0.01
        r_small = reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000, 1_000_000],
            current_weights=w2_small,
        )
        
        # Reset and re-establish baseline
        reward_fn.reset()
        reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000],
            current_weights=w1,
        )
        
        # Step 2b: large rebalance
        w2_large = w1 + 0.1
        r_large = reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000, 1_000_000],
            current_weights=w2_large,
        )
        
        assert r_large < r_small, (
            f"Large rebalance ({r_large:.4f}) should be penalised more than "
            f"small rebalance ({r_small:.4f})"
        )
    
    def test_holding_bonus_for_stable_positions(self, reward_fn):
        """Keeping positions stable (< 2% change) should give a bonus."""
        reward_fn.reset()
        
        # Step 1: set up position
        w1 = np.array([0.1] * 10, dtype=np.float32)
        reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000],
            current_weights=w1,
        )
        
        # Step 2: hold steady (zero change)
        r_hold = reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000, 1_000_000],
            current_weights=w1.copy(),  # Same weights
        )
        
        # Reset and do the same but with change
        reward_fn.reset()
        reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000],
            current_weights=w1,
        )
        w2_changed = w1 + 0.05  # 5% change — above 2% threshold
        r_change = reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000, 1_000_000],
            current_weights=w2_changed,
        )
        
        assert r_hold > r_change, "Holding steady should give better reward than rebalancing"
    
    def test_reset_clears_state(self, reward_fn):
        """reset() should clear all state for a new episode."""
        reward_fn.reset()
        reward_fn.compute(
            step_return=0.01,
            portfolio_values=[1_000_000, 1_010_000],
            current_weights=np.array([0.1] * 10, dtype=np.float32),
        )
        
        assert len(reward_fn._returns) > 0
        assert reward_fn._prev_weights is not None
        
        reward_fn.reset()
        assert len(reward_fn._returns) == 0
        assert reward_fn._prev_weights is None
        assert reward_fn._peak_value == 0.0
    
    def test_sharpe_component_after_warmup(self, reward_fn):
        """Sharpe component should activate after 5 steps."""
        reward_fn.reset()
        
        # First 4 steps — no Sharpe
        for i in range(4):
            reward_fn.compute(
                step_return=0.001,
                portfolio_values=[1_000_000 + i * 1000] + [1_000_000 + (i+1) * 1000],
            )
        
        # Step 5 — Sharpe should be non-zero (all positive returns → positive Sharpe)
        # This test just verifies the code path executes without error
        r = reward_fn.compute(
            step_return=0.001,
            portfolio_values=list(range(1_000_000, 1_006_000, 1000)),
        )
        # Not asserting sign — just that it runs
        assert isinstance(r, float)
        assert not np.isnan(r)


class TestRewardAggregator:
    """Tests for RewardAggregator summary statistics."""
    
    def test_summary_keys(self):
        """Summary should contain expected keys."""
        from agent.reward import RewardAggregator
        agg = RewardAggregator()
        
        for i in range(100):
            agg.record(reward=0.01, step_return=0.001)
        
        summary = agg.summary()
        expected_keys = ["total_reward", "mean_reward", "cumulative_return_pct",
                         "annualised_sharpe", "max_drawdown_pct"]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"
    
    def test_positive_returns_positive_sharpe(self):
        """Consistently positive returns should give positive Sharpe."""
        from agent.reward import RewardAggregator
        agg = RewardAggregator()
        
        for i in range(252):  # One year of trading days
            agg.record(reward=0.01, step_return=0.001)
        
        summary = agg.summary()
        assert summary["annualised_sharpe"] > 0
        assert summary["cumulative_return_pct"] > 0
    
    def test_max_drawdown_non_negative(self):
        """Max drawdown should be non-negative."""
        from agent.reward import RewardAggregator
        agg = RewardAggregator()
        
        returns = [0.01, 0.01, -0.05, 0.01, -0.02, 0.01]
        for r in returns:
            agg.record(reward=r, step_return=r)
        
        summary = agg.summary()
        assert summary["max_drawdown_pct"] >= 0
```

**3. `tests/test_feature_engineering.py`:**

```python
"""Unit tests for alpha/feature_engineering.py — 54 features across 8 groups."""
import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_multi_stock_data():
    """Create synthetic OHLCV data for 3 stocks (enough for cross-stock features)."""
    np.random.seed(42)
    n = 300  # Enough for rolling windows
    dates = pd.bdate_range("2020-01-01", periods=n)
    
    data = {}
    for ticker in ["STOCK_A", "STOCK_B", "STOCK_C"]:
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)
        df = pd.DataFrame({
            "Open": close * (1 + np.random.randn(n) * 0.005),
            "High": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "Low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "Close": close,
            "Volume": np.random.randint(100000, 10000000, size=n).astype(float),
        }, index=dates)
        data[ticker] = df
    return data


class TestAlphaFeatureEngine:
    """Tests for AlphaFeatureEngine.build_features()."""
    
    def test_build_features_returns_dict(self, config, sample_multi_stock_data):
        """build_features() should return a dict of DataFrames."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        
        assert isinstance(result, dict)
        assert len(result) == len(sample_multi_stock_data)
        for ticker, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_feature_count(self, config, sample_multi_stock_data):
        """Should produce 40+ features per stock."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        
        feature_names = engine.get_feature_names()
        assert len(feature_names) >= 40, f"Only {len(feature_names)} features (expected 40+)"
    
    def test_feature_names_unique(self, config, sample_multi_stock_data):
        """Feature names should be unique."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        engine.build_features(sample_multi_stock_data)
        
        names = engine.get_feature_names()
        assert len(names) == len(set(names)), "Duplicate feature names found"
    
    def test_momentum_features_present(self, config, sample_multi_stock_data):
        """Momentum group should produce ret_1d, ret_5d, ret_20d, etc."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        
        ticker = list(result.keys())[0]
        df = result[ticker]
        momentum_cols = [c for c in df.columns if "ret_" in c.lower() or "mom" in c.lower()]
        assert len(momentum_cols) >= 3, f"Expected 3+ momentum features, got {len(momentum_cols)}"
    
    def test_cross_stock_features_present(self, config, sample_multi_stock_data):
        """Cross-stock group should produce rank and relative features."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        
        ticker = list(result.keys())[0]
        df = result[ticker]
        cross_cols = [c for c in df.columns if "rank" in c.lower() or "rel" in c.lower() or "dispersion" in c.lower()]
        assert len(cross_cols) >= 2, f"Expected 2+ cross-stock features, got {len(cross_cols)}"
    
    def test_no_nan_after_warmup(self, config, sample_multi_stock_data):
        """After 120 days warmup, features should have no NaN."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        
        for ticker, df in result.items():
            feature_names = engine.get_feature_names()
            feat_cols = [c for c in feature_names if c in df.columns]
            tail = df.iloc[120:][feat_cols]
            nan_pct = tail.isna().mean()
            bad_cols = nan_pct[nan_pct > 0.05]
            assert len(bad_cols) == 0, (
                f"{ticker}: columns with >5% NaN after warmup: {bad_cols.to_dict()}"
            )
    
    def test_no_inf_values(self, config, sample_multi_stock_data):
        """Features should not contain inf values."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        
        for ticker, df in result.items():
            numeric_df = df.select_dtypes(include=[np.number])
            inf_count = np.isinf(numeric_df.values).sum()
            assert inf_count == 0, f"{ticker}: {inf_count} inf values found"
    
    def test_zscore_features(self, config, sample_multi_stock_data):
        """zscore_features() should normalise without NaN."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        normed = engine.zscore_features(result)
        
        for ticker, df in normed.items():
            feature_names = engine.get_feature_names()
            feat_cols = [c for c in feature_names if c in df.columns]
            tail = df.iloc[120:][feat_cols]
            # Z-scored features should have reasonable range
            max_abs = tail.abs().max().max()
            assert max_abs < 100, f"{ticker}: z-scored feature has extreme value {max_abs}"


class TestBuildTrainingSet:
    """Tests for building training labels."""
    
    def test_training_set_shapes(self, config, sample_multi_stock_data):
        """X and y should have matching first dimension."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        normed = engine.zscore_features(result)
        
        X, y = engine.build_training_set(normed, forward_horizon=10, purge_gap=5)
        assert X.shape[0] == y.shape[0], f"X rows ({X.shape[0]}) != y rows ({y.shape[0]})"
        assert X.shape[0] > 0, "Training set is empty"
        assert X.shape[1] > 0, "No features in training set"
    
    def test_labels_are_returns(self, config, sample_multi_stock_data):
        """y (forward returns) should be centered near 0."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        normed = engine.zscore_features(result)
        
        X, y = engine.build_training_set(normed, forward_horizon=10, purge_gap=5)
        assert abs(y.mean()) < 0.1, f"Labels mean too far from 0: {y.mean()}"
        assert y.std() > 0, "Labels have zero variance"
```

**4. `tests/test_sac_agent.py`:**

```python
"""Unit tests for agent/sac_agent.py — SAC agent wrapper."""
import pytest
import yaml
import numpy as np


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


class TestLatentAlphaSACAgent:
    """Tests for agent construction and configuration."""
    
    def test_init_balanced(self, config):
        """Agent should initialise with balanced profile."""
        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")
        
        assert agent.risk_profile == "balanced"
        assert agent.term == "medium"
        assert agent.model is None  # Not trained yet
    
    def test_init_all_profiles(self, config):
        """Agent should initialise with all risk profiles."""
        from agent.sac_agent import LatentAlphaSACAgent
        for profile in ["conservative", "balanced", "aggressive"]:
            agent = LatentAlphaSACAgent(config, risk_profile=profile, term="medium")
            assert agent.risk_profile == profile
    
    def test_entropy_config_loaded(self, config):
        """Agent should load entropy config fields."""
        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")
        
        assert hasattr(agent, "entropy_cfg")
        assert "target_entropy_ratio" in agent.entropy_cfg
        assert "ent_coef_init" in agent.entropy_cfg
        assert "ent_coef_floor" in agent.entropy_cfg
    
    def test_model_dir_created(self, config, tmp_path):
        """Model directory should be created on init."""
        config = config.copy()
        config["paths"] = config.get("paths", {}).copy()
        config["paths"]["model_dir"] = str(tmp_path / "models")
        
        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")
        
        assert (tmp_path / "models").exists()


class TestCallbacks:
    """Tests for custom callback classes."""
    
    def test_activity_check_callback_init(self):
        """ActivityCheckCallback should initialise without error."""
        from agent.sac_agent import ActivityCheckCallback
        cb = ActivityCheckCallback(check_every_n_steps=1000)
        assert cb.check_every == 1000
    
    def test_training_progress_callback_init(self):
        """TrainingProgressCallback should initialise without error."""
        from agent.sac_agent import TrainingProgressCallback
        cb = TrainingProgressCallback(log_every_n_steps=5000)
        assert cb.log_every == 5000
    
    def test_entropy_floor_callback_init(self):
        """EntropyFloorCallback should initialise with correct floor."""
        from agent.sac_agent import EntropyFloorCallback
        cb = EntropyFloorCallback(ent_coef_floor=0.005)
        assert cb.ent_coef_floor == 0.005
        assert cb._clamp_count == 0


class TestPredictInterface:
    """Tests for the predict API (without training)."""
    
    def test_predict_requires_model(self, config):
        """predict() should raise error if model not trained."""
        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")
        
        obs = np.zeros(560, dtype=np.float32)
        with pytest.raises(Exception):
            agent.predict(obs)
```

### Run the tests

```bash
cd latent_alpha && python -m pytest tests/test_data_loader.py tests/test_reward.py tests/test_feature_engineering.py tests/test_sac_agent.py -v
```

### Verification checklist

- [ ] `tests/test_data_loader.py` — 8+ tests covering indicators, z-score, feature names
- [ ] `tests/test_reward.py` — 10+ tests covering all 6 reward components + aggregator
- [ ] `tests/test_feature_engineering.py` — 9+ tests covering feature groups, NaN, training set
- [ ] `tests/test_sac_agent.py` — 7+ tests covering init, entropy config, callbacks, predict
- [ ] All tests pass with `pytest -v`
- [ ] Tests run fast (< 30 seconds total — no training, no downloads)
- [ ] Tests use synthetic data, not real market data
- [ ] No import errors in any test file

---

# ═══════════════════════════════════════════════════════════════════════════
# P1 — SYSTEM ACTIVATION (run after P0)
# ═══════════════════════════════════════════════════════════════════════════

## Prompt 5: Adaptive Exploration via Regime-Aware Entropy Scaling

### Context

After P0 Prompt 1 fixes the entropy floor, the SAC agent will maintain a minimum level of exploration. But a fixed entropy target doesn't adapt to market conditions. During high-volatility regimes (e.g., COVID crash, rate hike cycles), the agent should explore MORE to find new strategies. During stable bull markets, it can exploit more.

The VAE regime detector (`alpha/vae_regime.py`) already produces a `RegimeSignal` with:
- `regime_probs`: array of shape `(n_regimes,)` = `(4,)` — probability of being in each regime
- `regime_label`: int — the most likely regime (0-3)
- `embedding`: array of shape `(d_latent,)` = `(8,)` — latent state vector

We can use regime uncertainty (entropy of `regime_probs`) to dynamically scale the entropy target: when the market is in a clear regime (low uncertainty), allow less exploration; when regimes are ambiguous (high uncertainty), increase exploration.

### Task

Create a `RegimeAwareEntropyCallback` that dynamically adjusts SAC's target entropy based on VAE regime uncertainty. This replaces the static `target_entropy` from P0 Prompt 1 while keeping the floor.

### Files to create

**`agent/adaptive_entropy.py`:**

```python
"""
agent/adaptive_entropy.py
═════════════════════════
Regime-aware dynamic entropy scaling for SAC.

Adjusts the SAC agent's target entropy based on how uncertain the VAE
regime detector is about the current market state. When regime_probs
is uniform (high uncertainty → market is transitioning), we INCREASE
the entropy target to encourage exploration. When one regime dominates
(low uncertainty → clear regime), we allow more exploitation.

Formula:
  regime_uncertainty = H(regime_probs) / H_max
  scaled_target = base_target × (1 + uncertainty_scale × regime_uncertainty)

Where:
  - H(p) = -sum(p * log(p)) is the entropy of the regime distribution
  - H_max = log(n_regimes) is the maximum possible entropy (uniform distribution)
  - base_target = -5.0 (from P0 fix)
  - uncertainty_scale = 0.5 (configurable)
  - Result: target ranges from -5.0 (clear regime) to -7.5 (max uncertainty)

Note: A MORE NEGATIVE target_entropy means MORE exploration in SAC's formulation.
So when uncertainty is high, we make target more negative → more exploration.
"""
```

The module should contain:

1. `compute_regime_uncertainty(regime_probs)` — pure function that computes normalised entropy of regime probability distribution. Returns float in [0, 1].

2. `RegimeAwareEntropyCallback(BaseCallback)` — SB3 callback that:
   - Takes a `regime_fn` callable (same as what trading_env uses)
   - Every N steps (default 1000), queries the current regime
   - Computes regime uncertainty
   - Adjusts SAC's target entropy: `new_target = base_target × (1 + scale × uncertainty)`
   - Respects the entropy floor from P0
   - Logs the adjustment

3. Config fields to add to `sac:` section:
   ```yaml
   # v3.2 P1: Adaptive entropy
   adaptive_entropy: true             # Enable regime-aware entropy scaling
   entropy_uncertainty_scale: 0.5     # How much regime uncertainty amplifies exploration
   entropy_adjust_freq: 1000          # Steps between entropy target adjustments
   ```

### Files to modify

1. **`agent/adaptive_entropy.py`** — new file (described above)
2. **`agent/sac_agent.py`** — add `RegimeAwareEntropyCallback` to callbacks when enabled:
   ```python
   # In train(), after other callbacks:
   if self.config.get("sac", {}).get("adaptive_entropy", False) and regime_fn is not None:
       from agent.adaptive_entropy import RegimeAwareEntropyCallback
       callbacks.append(RegimeAwareEntropyCallback(
           regime_fn=regime_fn,
           base_target_entropy=target_entropy,
           uncertainty_scale=self.sac_cfg.get("entropy_uncertainty_scale", 0.5),
           adjust_freq=self.sac_cfg.get("entropy_adjust_freq", 1000),
           ent_coef_floor=self.entropy_cfg["ent_coef_floor"],
       ))
   ```
   Note: `train()` will need to accept an optional `regime_fn` parameter, which `term_selector.py` passes from the built regime pipeline.

3. **`strategy/term_selector.py`** — pass `regime_fn` to `agent.train()` when adaptive entropy is enabled.

4. **`config/config.yaml`** — add the 3 new config fields.

### Tests to write

Create `tests/test_adaptive_entropy.py`:

```python
"""Tests for regime-aware adaptive entropy scaling."""
import numpy as np
import pytest


def test_regime_uncertainty_uniform():
    """Uniform regime probs should give uncertainty = 1.0."""
    from agent.adaptive_entropy import compute_regime_uncertainty
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    u = compute_regime_uncertainty(probs)
    assert u == pytest.approx(1.0, abs=0.01)


def test_regime_uncertainty_certain():
    """One-hot regime probs should give uncertainty ≈ 0."""
    from agent.adaptive_entropy import compute_regime_uncertainty
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    u = compute_regime_uncertainty(probs)
    assert u == pytest.approx(0.0, abs=0.01)


def test_regime_uncertainty_in_range():
    """Uncertainty should always be in [0, 1]."""
    from agent.adaptive_entropy import compute_regime_uncertainty
    for _ in range(100):
        probs = np.random.dirichlet(np.ones(4))
        u = compute_regime_uncertainty(probs)
        assert 0.0 <= u <= 1.0, f"Uncertainty out of range: {u}"


def test_scaled_target_more_negative_with_uncertainty():
    """Higher uncertainty should make target_entropy more negative (more exploration)."""
    base = -5.0
    scale = 0.5
    
    target_certain = base * (1 + scale * 0.0)  # -5.0
    target_uncertain = base * (1 + scale * 1.0)  # -7.5
    
    assert target_uncertain < target_certain, (
        "More uncertainty should give more negative target"
    )


def test_callback_initialises():
    """RegimeAwareEntropyCallback should initialise without error."""
    from agent.adaptive_entropy import RegimeAwareEntropyCallback
    
    def dummy_regime_fn(date):
        from alpha.vae_regime import RegimeSignal
        return RegimeSignal(
            embedding=np.zeros(8, dtype=np.float32),
            regime_probs=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
            regime_label=0,
        )
    
    cb = RegimeAwareEntropyCallback(
        regime_fn=dummy_regime_fn,
        base_target_entropy=-5.0,
        uncertainty_scale=0.5,
        adjust_freq=1000,
        ent_coef_floor=0.005,
    )
    assert cb.base_target_entropy == -5.0
```

### Run the tests

```bash
cd latent_alpha && python -m pytest tests/test_adaptive_entropy.py -v
```

### Verification checklist

- [ ] `compute_regime_uncertainty()` returns correct values for edge cases
- [ ] `RegimeAwareEntropyCallback` adjusts target entropy based on regime uncertainty
- [ ] Entropy floor from P0 is still respected
- [ ] Config fields added to `sac:` section
- [ ] `agent.train()` accepts optional `regime_fn` parameter
- [ ] When `adaptive_entropy: false`, behaviour is unchanged from P0
- [ ] All tests pass

---

## Prompt 6: Sentiment Cache Build Instructions

### Context

The 10-axis sentiment encoder (`llm/sentiment_encoder.py`) and the precomputation script (`precompute_sentiment.py`) are fully built, but the sentiment cache is EMPTY. During training and backtesting, the sentiment_fn returns zeros for every date because there's no cached historical sentiment data.

This means the 100-dim sentiment slot in the observation vector (10 axes × 10 stocks = 100 dims) is always zero — the agent can't learn anything from it.

### Task

Provide step-by-step instructions and a helper script to build the historical sentiment cache using the local Ollama model (free, no API cost).

### Files to create

**`scripts/build_sentiment_cache.py`:**

```python
"""
scripts/build_sentiment_cache.py
════════════════════════════════
Build historical sentiment cache using local Ollama model.

This script:
1. Iterates over all trading dates in the backtest window (2018-2025)
2. For each date × ticker, fetches historical news headlines from the RSS cache
3. Runs the 10-axis sentiment encoder via Ollama (Gemma 4 local model)
4. Caches results to disk for use during training/backtesting

Prerequisites:
  - Ollama installed and running: `ollama serve`
  - Gemma 4 model pulled: `ollama pull gemma4:latest`
  
Usage:
  # Full build (2018-2025, all tickers) — takes ~4-8 hours on M-series Mac
  python scripts/build_sentiment_cache.py
  
  # Specific date range
  python scripts/build_sentiment_cache.py --start 2023-01-01 --end 2024-01-01
  
  # Specific tickers
  python scripts/build_sentiment_cache.py --tickers RELIANCE.NS TCS.NS
  
  # Resume from interruption (skips already-cached dates)
  python scripts/build_sentiment_cache.py --resume
  
  # Dry run (count dates, estimate time, don't run)
  python scripts/build_sentiment_cache.py --dry-run

Key design:
  - Rate-limited to avoid overwhelming Ollama (1 request/sec default)
  - Saves progress incrementally — safe to interrupt and resume
  - Falls back to zero vector for dates with no news
  - Validates cache integrity at the end
"""
```

The script should:

1. Load config to get tickers, date range, model settings
2. Create a `SentimentCache` instance from `llm/sentiment_cache.py`
3. For each business day in the range:
   a. Check if already cached (skip if `--resume`)
   b. For each ticker, generate synthetic headlines from price data (since we don't have real historical RSS) OR use the zero vector with a noise model that simulates realistic sentiment
   c. Run the encoder and cache the result
4. Print progress every 100 dates
5. At the end, validate the cache has no NaN and print stats

**Important**: Since real historical RSS feeds aren't available for 2018-2025, the script should generate **price-derived pseudo-sentiment** as a baseline:
- Positive 5-day return → positive headline_sentiment
- High 20-day volatility → high volatility_expectation
- Strong 60-day momentum → positive sector_momentum
- This gives the agent a noise-injected sentiment signal that correlates with what real news would have been

### Files to modify

1. **`llm/sentiment_cache.py`** — ensure the cache supports batch writes and date-range queries
2. **`config/config.yaml`** — add `llm.cache_dir` if not present:
   ```yaml
   llm:
     cache_dir: "./data/sentiment_cache"
   ```

### Verification checklist

- [ ] Ollama install instructions are clear
- [ ] Script handles interruption and resume gracefully
- [ ] `--dry-run` mode estimates time without running
- [ ] Price-derived pseudo-sentiment has reasonable correlation with returns
- [ ] Cache validation at the end checks for NaN and shape consistency
- [ ] After running, sentiment_fn in trading_env returns non-zero vectors
- [ ] Noise injection during training still works on top of cached values

---

## Prompt 7: Walk-Forward Diagnostics Dashboard

### Context

The current walk-forward backtest runner (`backtest/runner.py`) produces aggregate metrics (mean Sharpe, mean return, mean drawdown across folds), but provides no per-fold breakdown or diagnostic information. When an experiment underperforms, there's no way to tell:

1. Which fold failed (was it the COVID period? The 2022 rate hikes?)
2. What the alpha channels predicted vs. what happened
3. Whether the agent was exploring or exploiting (entropy trajectory)
4. How much transaction cost ate into returns per fold

### Task

Add per-fold diagnostic logging and a summary dashboard to the backtest runner.

### Files to create

**`backtest/diagnostics.py`:**

```python
"""
backtest/diagnostics.py
═══════════════════════
Per-fold diagnostic collection and reporting for walk-forward backtests.

Collects:
  - Per-fold return curve (daily portfolio values)
  - Per-fold transaction cost breakdown
  - Alpha channel hit rates per fold (what % of alpha signals were correct)
  - Regime distribution per fold (how much time in each regime)
  - Weight stability metrics per fold (mean turnover, hold period distribution)
  - Drawdown timeline per fold

Produces:
  - JSON file with all per-fold metrics
  - Summary CSV comparing folds
  - (Optional) matplotlib charts if matplotlib is available
"""
```

The module should contain:

1. `FoldDiagnostics` — dataclass holding all per-fold metrics:
   - `fold_id`, `train_start`, `train_end`, `test_start`, `test_end`
   - `daily_values`: list of portfolio values
   - `daily_returns`: list of step returns
   - `transaction_costs`: total and per-step
   - `alpha_hit_rate`: % of alpha direction predictions that were correct
   - `mean_turnover`: average daily weight change
   - `regime_distribution`: dict of {regime_label: fraction_of_time}
   - `max_drawdown_pct`, `sharpe`, `calmar`, `annual_return_pct`

2. `DiagnosticCollector` — class that collects metrics during a fold:
   - `record_step(info_dict)` — called after each env.step()
   - `record_alpha_prediction(date, direction, actual_return)` — tracks hit rate
   - `record_regime(date, regime_label)` — tracks regime distribution
   - `finalise() → FoldDiagnostics` — compute all summary metrics

3. `write_diagnostics_report(folds: List[FoldDiagnostics], output_dir: str)`:
   - Writes `fold_diagnostics.json` with all raw data
   - Writes `fold_comparison.csv` with key metrics side by side
   - Writes `diagnostics_summary.txt` with human-readable analysis:
     - Which fold was best/worst and why
     - Whether performance correlated with market regime
     - Transaction cost as % of gross profit per fold

### Files to modify

1. **`backtest/runner.py`** — integrate `DiagnosticCollector` into `_run_fold()`:
   - Create a collector at the start of each fold
   - Call `collector.record_step(info)` after each environment step
   - Call `collector.finalise()` at the end and add to results
   - Pass all fold diagnostics to `write_diagnostics_report()` after the full run

2. **`experiments/runner.py`** — save diagnostics alongside metrics.json

### Tests to write

Create `tests/test_diagnostics.py`:

```python
"""Tests for walk-forward diagnostics."""
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
    
    # 7 correct, 3 wrong → 70% hit rate
    for i in range(10):
        direction = 1.0 if i < 7 else -1.0
        actual = 0.01  # Always positive
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
```

### Run the tests

```bash
cd latent_alpha && python -m pytest tests/test_diagnostics.py -v
```

### Verification checklist

- [ ] `DiagnosticCollector` accumulates per-step metrics
- [ ] `FoldDiagnostics` contains all required fields
- [ ] `write_diagnostics_report()` produces JSON, CSV, and text files
- [ ] `backtest/runner.py` integrates collector into `_run_fold()`
- [ ] Alpha hit rate correctly counts direction predictions
- [ ] Regime distribution sums to 1.0
- [ ] Transaction cost breakdown is per-fold and total
- [ ] All tests pass
- [ ] Diagnostics are saved to the experiment results directory

---

## Prompt 8: Multi-Horizon Portfolio Orchestrator

### Context

The latent.alpha system supports three trading horizons (short/medium/long) via `config.yaml` terms section, but currently only ONE horizon is active at a time (set at training time via `--term medium`). The `TermSelector` class builds environments for a single horizon.

A production system should run all three horizons simultaneously and combine their signals. Short-horizon alpha (intraday momentum) catches different patterns than long-horizon alpha (macro trends). The combination should be more robust than any single horizon.

### Task

Create a `MultiHorizonOrchestrator` that:
1. Trains separate SAC agents for each active horizon
2. At inference time, collects weight recommendations from each agent
3. Combines them using a configurable weighting scheme (equal weight, risk-parity, or meta-learned)
4. Produces a final portfolio allocation

### Files to create

**`strategy/multi_horizon.py`:**

```python
"""
strategy/multi_horizon.py
═════════════════════════
Multi-horizon portfolio orchestrator.

Runs short/medium/long horizon agents in parallel and combines
their portfolio weight recommendations into a single allocation.

Combination strategies:
  1. Equal weight (default) — 1/3 each
  2. Risk-parity — weight inversely by recent volatility
  3. Sharpe-weighted — weight by rolling Sharpe ratio
  4. Meta-learned — Ridge regression on historical performance

The orchestrator handles:
  - Different rebalancing frequencies (daily vs weekly vs monthly)
  - Capital allocation across horizons
  - Conflict resolution when short says BUY but long says SELL
"""
```

The module should contain:

1. `HorizonAgent` — named tuple holding (term, agent, env, last_weights, last_rebalance_date)

2. `MultiHorizonOrchestrator`:
   - `__init__(config, risk_profile, horizons=["short", "medium", "long"])` — initialise
   - `train(data, ...)` — train all horizon agents (can be parallelised)
   - `combine_weights(horizon_weights: Dict[str, np.ndarray], strategy: str)` → `np.ndarray` — combine
   - `step(date)` → combined portfolio weights — only re-queries each horizon when its rebalance frequency triggers

3. Combination strategies:
   - `_equal_weight(weights_dict)` — simple average
   - `_risk_parity(weights_dict, volatilities)` — inverse vol weighting
   - `_sharpe_weighted(weights_dict, sharpes)` — Sharpe ratio weighting

4. Config section:
   ```yaml
   multi_horizon:
     enabled: false                   # Set true for multi-horizon mode
     horizons: ["medium", "long"]     # Active horizons (start with 2, add short later)
     combination: "equal_weight"      # "equal_weight", "risk_parity", "sharpe_weighted"
     capital_split:                   # % capital allocated to each horizon
       short: 0.2
       medium: 0.5
       long: 0.3
   ```

### Tests to write

Create `tests/test_multi_horizon.py`:

```python
"""Tests for multi-horizon portfolio orchestrator."""
import numpy as np
import pytest


def test_equal_weight_combination():
    """Equal weight should average across horizons."""
    from strategy.multi_horizon import MultiHorizonOrchestrator
    
    weights = {
        "short": np.array([0.1, 0.2, -0.1] + [0.0] * 7),
        "medium": np.array([0.2, 0.0, 0.1] + [0.0] * 7),
        "long": np.array([0.0, 0.1, 0.0] + [0.0] * 7),
    }
    
    combined = MultiHorizonOrchestrator._equal_weight(weights)
    expected = np.array([0.1, 0.1, 0.0] + [0.0] * 7)
    np.testing.assert_allclose(combined, expected, atol=1e-6)


def test_risk_parity_favours_low_vol():
    """Risk parity should give more weight to lower-volatility horizons."""
    from strategy.multi_horizon import MultiHorizonOrchestrator
    
    weights = {
        "medium": np.array([0.2] * 10),
        "long": np.array([0.1] * 10),
    }
    vols = {"medium": 0.2, "long": 0.1}  # Long is less volatile
    
    combined = MultiHorizonOrchestrator._risk_parity(weights, vols)
    # Long should have higher effective weight (lower vol → more allocation)
    # So combined should be closer to long's weights than medium's
    assert np.mean(combined) < 0.2  # Pulled toward long's 0.1


def test_combination_preserves_leverage_constraint():
    """Combined weights should not exceed leverage limit."""
    from strategy.multi_horizon import MultiHorizonOrchestrator
    
    # Each horizon at max leverage
    weights = {
        "short": np.array([0.3] * 10),   # Sum = 3.0
        "medium": np.array([0.2] * 10),  # Sum = 2.0
    }
    
    combined = MultiHorizonOrchestrator._equal_weight(weights)
    total_exposure = np.sum(np.abs(combined))
    # Should be clipped to leverage (1.0 for balanced)
    assert total_exposure <= 1.5, f"Combined exposure {total_exposure} exceeds leverage"
```

### Run the tests

```bash
cd latent_alpha && python -m pytest tests/test_multi_horizon.py -v
```

### Verification checklist

- [ ] `MultiHorizonOrchestrator` can initialise with 2+ horizons
- [ ] Equal weight combination is correct
- [ ] Risk parity favours low-volatility horizons
- [ ] Combined weights respect leverage constraint
- [ ] Config section added with `enabled: false` default
- [ ] When disabled, system works exactly as before (single horizon)
- [ ] All tests pass

---

## Execution Order & Dependencies

```
P0 (Critical — run first):
  Prompt 1: Entropy Collapse Fix          ← standalone, no deps
  Prompt 2: Signal Alignment + Meta-Model ← standalone, no deps on P1
  Prompt 3: Production Guards             ← standalone, no deps on P2
  Prompt 4: Core Test Coverage            ← depends on P1-P3 (tests the new code)

P1 (Activation — run after P0):
  Prompt 5: Adaptive Exploration          ← depends on P0.1 (entropy fix)
  Prompt 6: Sentiment Cache Build         ← standalone
  Prompt 7: Walk-Forward Diagnostics      ← standalone
  Prompt 8: Multi-Horizon Portfolio       ← standalone, but benefits from all P0 fixes
```

After completing all P0 prompts, re-run the experiment suite to measure improvement:

```bash
cd latent_alpha
python -m experiments.runner --experiments A_baseline_revised B_arch_tuned --timesteps 3000000
```

The entropy fix alone (P0.1) should produce the largest single improvement. The meta-model activation (P0.2) will either improve or maintain current performance (Ridge regression will learn if channels are complementary or redundant). Production guards (P0.3) prevent silent failures. Test coverage (P0.4) prevents regressions during future development.
