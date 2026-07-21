# Phase 4: Multi-Perspective LLM Integration — Implementation Prompts

> Inspired by TauricResearch/TradingAgents' multi-agent architecture.
> Each prompt is self-contained. Run them in order (1→2→3→4).
> All changes must be backward-compatible — existing training must not break.

---

## Prompt 1: Expanded Multi-Axis Sentiment Encoder

### Context

The latent.alpha trading system has an existing LLM sentiment layer in `llm/sentiment_encoder.py` that produces a 5-dimensional sentiment vector per ticker (headline_sentiment, earnings_signal, macro_signal, regulatory_risk, volatility_expectation). This is fed into the SAC agent's observation space via `sentiment_fn`.

The system also has:
- `llm/local_model.py` — Ollama-based local inference (Gemma 4)
- `llm/news_fetcher.py` — RSS-based news fetching with deduplication
- `llm/verifier.py` — Cross-source news verification (VerifiedArticle dataclass)
- `llm/sentiment_cache.py` — Caching layer
- `precompute_sentiment.py` — Offline precomputation script for historical sentiment

The config lives in `config/config.yaml` under the `llm:` section. Currently `sentiment_dim: 5`.

### Task

Expand the sentiment encoder from 5 axes to 10 axes, adding **specialist analyst dimensions** inspired by TradingAgents' multi-agent approach. Instead of spawning separate LLM agents, we encode each "analyst perspective" as additional numeric axes in a single structured LLM call (much cheaper and faster).

**New axes to add (5 new, keeping the original 5):**

```python
# Existing 5
"headline_sentiment",       # Overall tone
"earnings_signal",          # Earnings/revenue signal
"macro_signal",             # Macro/sector signal
"regulatory_risk",          # Regulatory risk
"volatility_expectation",   # Expected volatility

# New 5 — specialist analyst perspectives
"fundamental_quality",      # Balance sheet strength, cash flow, debt levels
"management_sentiment",     # Management guidance tone, insider activity
"sector_momentum",          # Sector rotation signal, peer comparison
"event_catalyst",           # M&A, product launches, partnerships, restructuring
"contrarian_indicator",     # Crowding signal — when everyone is bullish, score negative
```

### Files to modify

1. **`llm/sentiment_encoder.py`**
   - Update `SENTIMENT_AXES` list to include all 10 axes
   - Update `SYSTEM_PROMPT` to include scoring instructions for the 5 new axes
   - Update `_build_user_prompt()` to request the expanded format
   - No changes needed to `SentimentEncoder` class logic — it's already generic over `SENTIMENT_AXES`

2. **`llm/local_model.py`**
   - Update the duplicated `SENTIMENT_AXES` list to match
   - Update `SYSTEM_PROMPT` to match
   - The `_scores_to_vector()` method already works generically

3. **`config/config.yaml`**
   - Change `sentiment_dim: 5` to `sentiment_dim: 10`

4. **`env/trading_env.py`**
   - No code changes needed — it reads `sentiment_dim` from config dynamically
   - But the observation space will grow by `n_stocks * 5` = 50 dims
   - Verify the `observation_space` calculation still works

5. **`config/config.yaml` → `sac:` section**
   - Bump `net_arch` from `[384, 384]` to `[512, 512]` to handle the larger obs space (462 + 50 = 512 dims)

### Tests to write

Create `tests/test_sentiment_expanded.py`:

```python
"""Tests for expanded 10-axis sentiment encoder."""
import numpy as np
import pytest

def test_sentiment_axes_count():
    """Verify we have exactly 10 sentiment axes."""
    from llm.sentiment_encoder import SENTIMENT_AXES
    assert len(SENTIMENT_AXES) == 10

def test_local_model_axes_match():
    """Verify local_model.py axes match sentiment_encoder.py axes."""
    from llm.sentiment_encoder import SENTIMENT_AXES as cloud_axes
    from llm.local_model import SENTIMENT_AXES as local_axes
    assert cloud_axes == local_axes

def test_sentiment_vector_shape():
    """Verify encode_zeros produces correct shape."""
    from llm.sentiment_encoder import SentimentEncoder
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    tickers = config["market"]["tickers"]
    encoder = SentimentEncoder(config, tickers)
    zeros = encoder.encode_zeros()
    assert zeros.shape == (len(tickers) * 10,)
    assert zeros.dtype == np.float32

def test_scores_to_vector_all_axes():
    """Verify _scores_to_vector handles all 10 axes."""
    from llm.sentiment_encoder import SentimentEncoder, SENTIMENT_AXES
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    tickers = config["market"]["tickers"]
    encoder = SentimentEncoder(config, tickers)

    # Build a mock scores_map with all 10 axes
    scores_map = {}
    for t in tickers:
        scores_map[t] = {axis: 0.5 for axis in SENTIMENT_AXES}

    vec = encoder._scores_to_vector(scores_map)
    assert vec.shape == (len(tickers) * 10,)
    assert np.all(vec == 0.5)

def test_obs_space_dimension():
    """Verify total observation space accounts for expanded sentiment."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    n_stocks = len(config["market"]["tickers"])
    sentiment_dim = config["llm"]["sentiment_dim"]
    assert sentiment_dim == 10

    # Compute expected obs_dim (matching trading_env.py formula)
    from env.data_loader import NSEDataLoader
    loader = NSEDataLoader(config)
    n_features = len(loader.get_feature_names())
    alpha_dim = config.get("alpha", {}).get("obs_dim", 0)
    temporal_dim = config.get("patchtst", {}).get("obs_dim", 0)

    expected_obs = (
        n_stocks * n_features       # market features
        + n_stocks                  # current weights
        + 2                         # portfolio value, cash ratio
        + n_stocks                  # holding lock countdown
        + n_stocks * sentiment_dim  # LLM sentiment
        + alpha_dim                 # alpha signal
        + temporal_dim              # temporal signal
    )
    print(f"Expected obs_dim: {expected_obs}")
    assert expected_obs > 0

def test_config_sentiment_dim_consistent():
    """Config sentiment_dim must match len(SENTIMENT_AXES)."""
    import yaml
    from llm.sentiment_encoder import SENTIMENT_AXES
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    assert config["llm"]["sentiment_dim"] == len(SENTIMENT_AXES)
```

### Run the tests

```bash
cd latent_alpha && python -m pytest tests/test_sentiment_expanded.py -v
```

### Verification checklist

- [ ] `SENTIMENT_AXES` has exactly 10 entries in both `sentiment_encoder.py` and `local_model.py`
- [ ] `config.yaml` has `sentiment_dim: 10`
- [ ] System prompts in both files include scoring instructions for all 10 axes
- [ ] `precompute_sentiment.py` still works (it reads axes dynamically)
- [ ] All tests pass
- [ ] No import errors when running `python -c "from llm.sentiment_encoder import SentimentEncoder"`

---

## Prompt 2: Debate-Based Pre-Trade Analysis Signal

### Context

The latent.alpha system uses a SAC (Soft Actor-Critic) reinforcement learning agent for portfolio management. It currently has two alpha signal sources feeding into the SAC observation space:

1. **LightGBM** cross-sectional alpha (direction + conviction per stock) — `alpha/lightgbm_alpha.py`
2. **PatchTST** temporal encoder (direction + conviction per stock) — `alpha/patchtst.py`

Both are combined via a Ridge meta-model (`alpha/meta_model.py`) and injected into the env via `alpha_fn` and `temporal_fn` callables in `env/trading_env.py`.

The meta-model has a `ChannelPrediction` dataclass and supports dynamic channel registration.

Inspired by TradingAgents' bullish/bearish debate mechanism, we want to add a **debate signal** — a lightweight LLM-based pre-trade analysis that runs a structured bullish vs bearish assessment for each stock, producing a numeric "debate score" that gets registered as a new channel in the meta-model.

### Task

Create `llm/debate_analyst.py` — a module that:

1. Takes the current alpha signals (LightGBM direction + conviction) and recent news for each stock
2. Runs a **single LLM call** with a structured prompt that asks the model to argue BOTH the bullish and bearish case for each stock, then score the net assessment
3. Produces a `DebateSignal` dataclass with `direction` and `conviction` arrays (same interface as `AlphaSignal` / `TemporalSignal`)
4. Can be registered as a channel in the `RidgeMetaModel`
5. Has a **fallback mode** that returns zeros when no LLM is available (for training — the LLM debate only runs during backtesting/live, not during SAC training steps)
6. Results are cached by date to avoid redundant API calls

### Architecture

```
                    ┌──────────────────────────┐
                    │  Alpha signals (LGB+TST)  │
                    │  + Recent news headlines  │
                    └─────────┬────────────────┘
                              ▼
                    ┌──────────────────────────┐
                    │  Single LLM call:        │
                    │  "For each stock, argue  │
                    │   bullish case, bearish  │
                    │   case, then score -1→+1"│
                    └─────────┬────────────────┘
                              ▼
                    ┌──────────────────────────┐
                    │  DebateSignal            │
                    │  direction[n_stocks]     │
                    │  conviction[n_stocks]    │
                    │  reasoning[n_stocks]     │
                    └──────────────────────────┘
```

### Files to create

**`llm/debate_analyst.py`:**

```python
"""
llm/debate_analyst.py
─────────────────────
Structured bullish/bearish debate analysis per stock.

Inspired by TradingAgents' multi-agent debate mechanism, but implemented
as a single structured LLM call for efficiency. The LLM argues both sides
for each stock, then produces a net assessment score.

This runs ONLY during backtesting and live inference — not during SAC
training (where it returns zeros for speed). The debate signal gets
registered as a channel in the Ridge meta-model alongside LightGBM
and PatchTST signals.

Usage:
    analyst = DebateAnalyst(config, tickers)
    signal = analyst.analyze(alpha_signals, news_headlines, date)
    # signal.direction  → np.ndarray (n_stocks,)
    # signal.conviction → np.ndarray (n_stocks,)
"""
```

Key design decisions:
- **Single LLM call, not multi-agent** — avoids the latency/cost of TradingAgents' approach. One structured prompt that forces bullish + bearish reasoning for each stock.
- **Conviction derived from argument balance** — if bullish and bearish cases are both strong, conviction is LOW (uncertain). If one side clearly dominates, conviction is HIGH.
- **The prompt includes the quantitative alpha signals** — the LLM can see "LightGBM says bullish with 0.8 conviction" and reason about whether to agree or disagree. This is the key insight: the LLM acts as a qualitative check on the quantitative model.
- **Cache by date** — same date with same inputs returns cached result.
- **Supports both Anthropic API and Ollama** — use `config["llm"]["precompute_model"]` to select.

The `DebateSignal` dataclass must have a `to_obs_vector()` method returning `[direction | conviction]` — shape `(2 * n_stocks,)`.

### Files to modify

1. **`config/config.yaml`** — add under `llm:`:
   ```yaml
   debate:
     enabled: true
     obs_dim: 20              # 2 × n_stocks (direction + conviction)
     cache_ttl_minutes: 1440  # Cache debate results for 24h
     model: "local"           # "local" for Ollama, "sonnet" for Claude
     max_tokens: 2048
   ```

2. **`env/trading_env.py`** — add `debate_fn=None` parameter alongside `alpha_fn` and `temporal_fn`. Add `self.debate_dim` from config. Include in observation concatenation and `observation_space` calculation.

3. **`strategy/term_selector.py`** — build debate pipeline in `_build_alpha_pipeline()` AFTER alpha model is trained. Create debate_fn closure that calls the debate analyst. Pass it to the env constructors.

4. **`alpha/__init__.py`** — export DebateSignal

### Tests to write

Create `tests/test_debate_analyst.py`:

```python
"""Tests for debate analyst module."""
import numpy as np
import pytest

def test_debate_signal_dataclass():
    """DebateSignal has correct interface."""
    from llm.debate_analyst import DebateSignal
    sig = DebateSignal(
        direction=np.array([0.5, -0.3, 0.1], dtype=np.float32),
        conviction=np.array([0.8, 0.6, 0.4], dtype=np.float32),
        reasoning={"STOCK1": "bullish due to ...", "STOCK2": "bearish because ..."},
    )
    vec = sig.to_obs_vector()
    assert vec.shape == (6,)  # 2 * 3 stocks
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec[:3], sig.direction)
    np.testing.assert_array_equal(vec[3:], sig.conviction)

def test_debate_analyst_fallback():
    """When LLM unavailable, returns zero signal."""
    from llm.debate_analyst import DebateAnalyst
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    tickers = config["market"]["tickers"]
    analyst = DebateAnalyst(config, tickers)

    # Force fallback mode
    signal = analyst.get_zero_signal()
    assert signal.direction.shape == (len(tickers),)
    assert signal.conviction.shape == (len(tickers),)
    assert np.all(signal.direction == 0.0)
    assert np.all(signal.conviction == 0.0)

def test_debate_cache():
    """Debate results are cached by date."""
    from llm.debate_analyst import DebateAnalyst
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    tickers = config["market"]["tickers"]
    analyst = DebateAnalyst(config, tickers)

    # Cache a mock result
    mock_signal = analyst.get_zero_signal()
    analyst._cache["2024-01-15"] = mock_signal

    # Retrieve from cache
    cached = analyst._get_cached("2024-01-15")
    assert cached is not None
    np.testing.assert_array_equal(cached.direction, mock_signal.direction)

def test_debate_prompt_structure():
    """Verify prompt includes alpha signals and structures debate."""
    from llm.debate_analyst import DebateAnalyst, _build_debate_prompt
    tickers = ["RELIANCE.NS", "TCS.NS"]
    alpha_directions = np.array([0.5, -0.3])
    alpha_convictions = np.array([0.8, 0.6])
    headlines = {"RELIANCE.NS": ["Q3 beat estimates"], "TCS.NS": ["Weak guidance"]}

    prompt = _build_debate_prompt(tickers, alpha_directions, alpha_convictions, headlines)
    assert "RELIANCE.NS" in prompt
    assert "bullish" in prompt.lower()
    assert "bearish" in prompt.lower()
    assert "0.5" in prompt   # alpha direction included
    assert "0.8" in prompt   # alpha conviction included

def test_env_debate_dim():
    """Trading env accepts debate_fn and includes debate_dim in obs space."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    debate_dim = config.get("llm", {}).get("debate", {}).get("obs_dim", 0)
    assert debate_dim == 20  # 2 × 10 stocks

def test_debate_signal_obs_vector_clipping():
    """Direction should be clipped to [-3, 3], conviction to [0, 1]."""
    from llm.debate_analyst import DebateSignal
    sig = DebateSignal(
        direction=np.array([5.0, -5.0], dtype=np.float32),
        conviction=np.array([1.5, -0.5], dtype=np.float32),
    )
    vec = sig.to_obs_vector()
    assert np.all(vec[:2] >= -3.0) and np.all(vec[:2] <= 3.0)
    assert np.all(vec[2:] >= 0.0) and np.all(vec[2:] <= 1.0)
```

### Verification checklist

- [ ] `DebateSignal.to_obs_vector()` returns shape `(2 * n_stocks,)` with direction clipped to [-3,3] and conviction to [0,1]
- [ ] `DebateAnalyst` has both Anthropic and Ollama backends
- [ ] Fallback mode returns zeros (for use during SAC training)
- [ ] Cache by date string works
- [ ] `_build_debate_prompt()` includes alpha signals + headlines
- [ ] `trading_env.py` includes `debate_dim` in observation space
- [ ] `term_selector.py` builds debate pipeline and passes `debate_fn` to env
- [ ] Config has `llm.debate` section
- [ ] All tests pass
- [ ] Training still works with `debate_fn=None` (backward compatible)

---

## Prompt 3: Post-Hoc Explainability Layer

### Context

The latent.alpha SAC agent is a black box — it takes a 512-dim observation and outputs portfolio weight deltas, but we can't see WHY it made a particular trade. This makes it hard to audit, debug, and trust.

We want to add a post-hoc explainability module that, after the SAC agent makes a decision, uses an LLM to interpret the decision based on the observable inputs (alpha signals, sentiment, portfolio state).

This does NOT change the trading logic. It's purely an observation/logging layer.

### Task

Create `llm/explainer.py` — a module that:

1. Takes the current observation breakdown (alpha signals, temporal signals, sentiment, portfolio weights, proposed weight deltas)
2. Calls an LLM to produce a concise human-readable explanation of the trade
3. Logs the explanation alongside the trade in the backtest trade log
4. Can be disabled via config for fast training runs

### Architecture

The explainer runs AFTER `env.step()` returns, using the `info` dict plus the alpha/temporal signals. It does NOT slow down training — it only runs during backtesting when `explain_trades` is enabled.

### Files to create

**`llm/explainer.py`:**

```python
"""
llm/explainer.py
────────────────
Post-hoc trade explanation using LLM reasoning.

After the SAC agent produces weight deltas, this module interprets
WHY the trade was made based on observable signals. This is purely
for auditability — it does not affect trading decisions.

Key inputs for explanation:
  - Alpha signal direction + conviction per stock
  - PatchTST temporal signal
  - Top contributing features from LightGBM importance
  - Current portfolio weights and proposed deltas
  - Recent sentiment scores

Output:
  - One-paragraph explanation per significant trade
  - Confidence assessment (does the LLM agree with the trade?)
  - Logged to the backtest trade CSV

Usage:
    explainer = TradeExplainer(config, tickers)
    explanation = explainer.explain(
        weight_deltas=action,
        alpha_signal=alpha_sig,
        temporal_signal=temporal_sig,
        sentiment=sentiment_vec,
        portfolio_weights=current_weights,
        date=current_date,
    )
    # explanation.text → "Increased RELIANCE position (+3.2%) driven by..."
    # explanation.llm_agrees → True/False
"""
```

Key design decisions:
- **Batch explanations** — only explain trades where `|weight_delta| > 0.02` (significant moves). Don't waste API calls on noise.
- **Fast mode** — when `explain_trades: false` in config (default for training), the module is a no-op.
- **Structured output** — LLM returns JSON with `explanation`, `confidence`, `agrees_with_trade` per stock.
- **Works with Ollama** — for offline backtesting, use local model to avoid API costs.

### Files to modify

1. **`config/config.yaml`** — add under `llm:`:
   ```yaml
   explainer:
     enabled: false               # Only enable for analysis runs
     min_delta_to_explain: 0.02   # Only explain weight changes > 2%
     model: "local"               # "local" for Ollama, "sonnet" for Claude
   ```

2. **`backtest/runner.py`** — in `_run_fold()`, after `test_env.step()`, call the explainer if enabled. Add explanation text to the trade log `info` dict.

3. **`llm/__init__.py`** — export TradeExplainer

### Tests to write

Create `tests/test_explainer.py`:

```python
"""Tests for post-hoc trade explainer."""
import numpy as np
import pytest

def test_explainer_disabled_is_noop():
    """When disabled in config, explain() returns None immediately."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["llm"]["explainer"]["enabled"] = False

    from llm.explainer import TradeExplainer
    tickers = config["market"]["tickers"]
    explainer = TradeExplainer(config, tickers)
    result = explainer.explain(
        weight_deltas=np.zeros(len(tickers)),
        alpha_direction=np.zeros(len(tickers)),
        alpha_conviction=np.zeros(len(tickers)),
        portfolio_weights=np.zeros(len(tickers)),
        date="2024-01-15",
    )
    assert result is None

def test_explainer_filters_small_deltas():
    """Only explains stocks with |delta| > min_delta_to_explain."""
    from llm.explainer import TradeExplainer, _filter_significant_trades
    tickers = ["A.NS", "B.NS", "C.NS"]
    deltas = np.array([0.05, 0.001, -0.03])
    significant = _filter_significant_trades(tickers, deltas, min_delta=0.02)
    assert "A.NS" in significant
    assert "B.NS" not in significant
    assert "C.NS" in significant

def test_explainer_prompt_structure():
    """Prompt includes alpha signals and weight deltas."""
    from llm.explainer import _build_explanation_prompt
    tickers = ["RELIANCE.NS"]
    deltas = {"RELIANCE.NS": 0.04}
    alpha_dirs = {"RELIANCE.NS": 0.7}
    alpha_convs = {"RELIANCE.NS": 0.85}
    weights = {"RELIANCE.NS": 0.15}

    prompt = _build_explanation_prompt(tickers, deltas, alpha_dirs, alpha_convs, weights, "2024-01-15")
    assert "RELIANCE.NS" in prompt
    assert "0.04" in prompt or "4%" in prompt
    assert "bullish" in prompt.lower() or "0.7" in prompt

def test_explanation_dataclass():
    """TradeExplanation has required fields."""
    from llm.explainer import TradeExplanation
    exp = TradeExplanation(
        date="2024-01-15",
        explanations={"RELIANCE.NS": "Increased due to strong earnings signal"},
        llm_agrees={"RELIANCE.NS": True},
        confidence={"RELIANCE.NS": 0.8},
    )
    assert exp.date == "2024-01-15"
    assert "RELIANCE.NS" in exp.explanations
    assert exp.llm_agrees["RELIANCE.NS"] is True
```

### Verification checklist

- [ ] `TradeExplainer` is a no-op when `explainer.enabled: false`
- [ ] Only significant trades (delta > threshold) get explained
- [ ] Prompt includes alpha signals, portfolio state, and date
- [ ] Works with both Anthropic and Ollama backends
- [ ] `backtest/runner.py` calls explainer in the evaluation loop
- [ ] Explanations appear in the trade log CSV
- [ ] All tests pass
- [ ] Training runs are unaffected (explainer disabled by default)

---

## Prompt 4: Adversarial Conviction Filter (Devil's Advocate)

### Context

The latent.alpha system has an alpha model that produces per-stock direction and conviction signals. Experiment B already uses a simple `conviction_gate` (threshold at 0.3) to zero out low-conviction signals. But this is a static filter — it doesn't consider qualitative factors.

Inspired by TradingAgents' adversarial debate mechanism, we want a smarter filter: before the SAC agent sees high-conviction alpha signals, run a quick "devil's advocate" check using an LLM to see if the bearish case invalidates the signal.

### Task

Create `alpha/adversarial_filter.py` — a module that:

1. Takes the alpha signal (direction + conviction per stock) from LightGBM/PatchTST
2. For stocks where conviction > threshold AND the agent is proposing a significant position change
3. Runs a quick LLM call asking: "Here is the bullish case for [STOCK]. What's the strongest counter-argument? Rate your counter-conviction 0–1."
4. If the LLM's counter-conviction is high (> 0.7), REDUCES the alpha conviction for that stock
5. This acts as an intelligent dampener on overconfident signals

### Architecture

```
AlphaSignal (from LightGBM) ──→ AdversarialFilter ──→ Filtered AlphaSignal
                                       │
                                       ▼
                               LLM "devil's advocate"
                               (only for high-conviction stocks)
```

Key design decisions:
- **Only runs on high-conviction stocks** (conviction > 0.7) — most stocks pass through unchanged. This keeps LLM calls to ~2-3 per rebalance, not 10.
- **Reduces conviction, never flips direction** — the filter is conservative. It says "maybe be less sure" not "do the opposite."
- **Cached by date+stock** — same stock on same date returns cached result.
- **Training bypass** — during SAC training, the filter is disabled (returns input unchanged). Only active during backtesting/live.
- **Configurable aggressiveness** — `adversarial_dampening` parameter controls how much the counter-argument reduces conviction (0.0 = no effect, 1.0 = full effect).

### Files to create

**`alpha/adversarial_filter.py`:**

The module should define:
- `AdversarialFilter` class with `filter(alpha_signal, date, headlines=None) → AlphaSignal`
- Uses the same LLM backend as the sentiment encoder (Anthropic or Ollama)
- Has `enabled` flag from config
- Returns input unchanged when disabled

### Files to modify

1. **`config/config.yaml`** — add under `alpha:`:
   ```yaml
   adversarial_filter:
     enabled: false               # Enable for backtest analysis
     conviction_threshold: 0.7    # Only challenge high-conviction signals
     counter_conviction_threshold: 0.7  # Reduce if counter > this
     dampening: 0.3               # How much to reduce conviction (0-1)
     model: "local"               # LLM backend
   ```

2. **`strategy/term_selector.py`** — in the `alpha_fn` closure (inside `_build_alpha_pipeline`), after getting the alpha signal, optionally run it through the adversarial filter before returning.

3. **`alpha/__init__.py`** — export AdversarialFilter

### Tests to write

Create `tests/test_adversarial_filter.py`:

```python
"""Tests for adversarial conviction filter."""
import numpy as np
import pytest

def test_filter_disabled_passthrough():
    """When disabled, filter returns input unchanged."""
    from alpha.adversarial_filter import AdversarialFilter
    from alpha.lightgbm_alpha import AlphaSignal
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["alpha"]["adversarial_filter"]["enabled"] = False

    filt = AdversarialFilter(config)
    signal = AlphaSignal(
        direction=np.array([0.8, -0.5, 0.3]),
        conviction=np.array([0.9, 0.7, 0.4]),
        raw_pred=np.array([0.02, -0.01, 0.005]),
    )
    result = filt.filter(signal, date="2024-01-15")
    np.testing.assert_array_equal(result.direction, signal.direction)
    np.testing.assert_array_equal(result.conviction, signal.conviction)

def test_filter_only_challenges_high_conviction():
    """Only stocks above conviction_threshold get challenged."""
    from alpha.adversarial_filter import AdversarialFilter, _select_stocks_to_challenge
    convictions = np.array([0.9, 0.5, 0.3, 0.8, 0.6])
    threshold = 0.7
    indices = _select_stocks_to_challenge(convictions, threshold)
    assert 0 in indices  # 0.9 > 0.7
    assert 3 in indices  # 0.8 > 0.7
    assert 1 not in indices
    assert 2 not in indices
    assert 4 not in indices

def test_filter_never_flips_direction():
    """Filter reduces conviction but never changes direction sign."""
    from alpha.adversarial_filter import AdversarialFilter
    from alpha.lightgbm_alpha import AlphaSignal
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["alpha"]["adversarial_filter"]["enabled"] = True

    filt = AdversarialFilter(config)
    signal = AlphaSignal(
        direction=np.array([0.8, -0.5]),
        conviction=np.array([0.9, 0.8]),
        raw_pred=np.array([0.02, -0.01]),
    )

    # Mock the LLM to return high counter-conviction
    filt._mock_counter_convictions = {0: 0.9, 1: 0.9}

    result = filt.filter(signal, date="2024-01-15")
    # Direction signs must be preserved
    assert np.sign(result.direction[0]) == np.sign(signal.direction[0])
    assert np.sign(result.direction[1]) == np.sign(signal.direction[1])
    # But conviction should be reduced
    assert result.conviction[0] < signal.conviction[0]
    assert result.conviction[1] < signal.conviction[1]

def test_dampening_math():
    """Verify conviction reduction formula."""
    from alpha.adversarial_filter import _apply_dampening
    original_conviction = 0.9
    counter_conviction = 0.8
    dampening = 0.3

    new_conv = _apply_dampening(original_conviction, counter_conviction, dampening)
    # new = original * (1 - dampening * counter_conviction)
    expected = 0.9 * (1 - 0.3 * 0.8)
    assert abs(new_conv - expected) < 1e-6
    assert new_conv > 0  # never negative
    assert new_conv <= original_conviction  # never increases

def test_filter_cache():
    """Same date+stock returns cached counter-conviction."""
    from alpha.adversarial_filter import AdversarialFilter
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    filt = AdversarialFilter(config)
    filt._cache[("2024-01-15", 0)] = 0.6
    assert filt._get_cached("2024-01-15", 0) == 0.6
    assert filt._get_cached("2024-01-15", 1) is None
```

### Verification checklist

- [ ] Filter is a no-op when `adversarial_filter.enabled: false`
- [ ] Only high-conviction stocks (> threshold) get challenged
- [ ] Direction is NEVER flipped — only conviction is reduced
- [ ] Dampening formula: `new_conv = conv * (1 - dampening * counter_conv)`
- [ ] Results cached by (date, stock_index)
- [ ] Works with both Anthropic and Ollama backends
- [ ] `term_selector.py` applies filter in the alpha_fn closure
- [ ] All tests pass
- [ ] Training runs unaffected (filter disabled by default)

---

## Running all tests

After implementing all 4 prompts, run the full test suite:

```bash
cd latent_alpha
python -m pytest tests/ -v --tb=short
```

Then verify training still works:

```bash
# Quick smoke test — 10K steps, should complete in <5 minutes
KMP_DUPLICATE_LIB_OK=TRUE python3 train.py --algo sac --risk balanced --term medium --timesteps 10000
```

## Implementation order

1. **Prompt 1** (Expanded Sentiment) — foundational, changes obs space
2. **Prompt 2** (Debate Signal) — adds new alpha channel
3. **Prompt 3** (Explainability) — logging only, no obs changes
4. **Prompt 4** (Adversarial Filter) — modifies alpha pipeline

Prompts 3 and 4 are independent of each other and can be done in parallel.
