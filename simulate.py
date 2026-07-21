"""
simulate.py
───────────
Live paper trade simulation using the trained agent.

Downloads the most recent market data, runs the agent in inference mode,
and prints a live terminal dashboard showing portfolio state, position
decisions, and key signals for each ticker.

Usage
-----
# Simulate with the balanced medium-term agent (default)
python simulate.py

# Simulate a specific profile / term
python simulate.py --risk aggressive --term short

# Run N steps and exit (default: run through all available recent data)
python simulate.py --steps 30

# Include live LLM news sentiment
python simulate.py --with-news
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

load_dotenv()
logging.basicConfig(level=logging.WARNING)   # quiet during live display
console = Console()

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Display helpers ───────────────────────────────────────────────────────────

def _pct_colour(value: float, good_positive: bool = True) -> Text:
    """Format a percentage value with colour."""
    sign = "+" if value >= 0 else ""
    text = f"{sign}{value:.2f}%"
    if good_positive:
        colour = "green" if value > 0 else ("red" if value < 0 else "dim")
    else:
        colour = "red" if value > 0 else ("green" if value < 0 else "dim")
    return Text(text, style=colour)


def _weight_bar(weight: float, width: int = 12) -> str:
    """Render a mini ASCII bar for portfolio weight."""
    filled = int(abs(weight) * width)
    bar    = "█" * filled + "░" * (width - filled)
    if weight > 0.01:
        return f"[green]+{bar}[/green]"
    elif weight < -0.01:
        return f"[red]-{bar}[/red]"
    return f"[dim] {bar}[/dim]"


def _action_label(weight: float) -> Text:
    """Convert a weight to a human-readable action label."""
    if weight > 0.10:
        return Text("▲ LONG",  style="bold green")
    elif weight > 0.01:
        return Text("△ long",  style="green")
    elif weight < -0.10:
        return Text("▼ SHORT", style="bold red")
    elif weight < -0.01:
        return Text("▽ short", style="red")
    else:
        return Text("● HOLD",  style="dim")


def build_portfolio_panel(
    step: int,
    date: str,
    portfolio_value: float,
    initial_capital: float,
    cash: float,
    risk_profile: str,
    term: str,
) -> Panel:
    pnl     = portfolio_value - initial_capital
    pnl_pct = pnl / initial_capital * 100
    sign    = "+" if pnl >= 0 else ""
    colour  = "green" if pnl >= 0 else "red"

    content = (
        f"[dim]Step[/dim]  {step:>5}     "
        f"[dim]Date[/dim]  {date}     "
        f"[dim]Profile[/dim]  [cyan]{risk_profile}[/cyan] / [cyan]{term}[/cyan]\n\n"
        f"[bold]Portfolio[/bold]  ₹{portfolio_value:>14,.0f}     "
        f"[bold]Cash[/bold]  ₹{cash:>12,.0f}     "
        f"[bold]P&L[/bold]  [{colour}]{sign}₹{abs(pnl):,.0f}  ({sign}{pnl_pct:.2f}%)[/{colour}]"
    )
    return Panel(content, title="[bold]latent.alpha[/bold]  paper portfolio", border_style="dim")


def build_positions_table(
    tickers: list,
    weights: list,
    prices: list,
    prev_weights: list,
) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="dim",
        padding=(0, 1),
    )
    table.add_column("Ticker",    style="bold", width=18)
    table.add_column("Action",    width=10)
    table.add_column("Weight",    width=16)
    table.add_column("Δ Weight",  width=10)
    table.add_column("Price (₹)", justify="right", width=12)

    for ticker, w, prev_w, price in zip(tickers, weights, prev_weights, prices):
        delta  = w - prev_w
        d_sign = "+" if delta >= 0 else ""
        d_col  = "green" if delta > 0.005 else ("red" if delta < -0.005 else "dim")
        short  = ticker.replace(".NS", "")

        table.add_row(
            short,
            _action_label(w),
            _weight_bar(w),
            Text(f"{d_sign}{delta:.3f}", style=d_col),
            f"{price:,.1f}",
        )
    return table


def build_signals_table(obs: np.ndarray, feature_names: list, n_stocks: int) -> Table:
    """Show the top 3 most influential features per selected stock."""
    table = Table(
        title="Top signals (first 3 stocks)",
        box=box.SIMPLE,
        show_header=True,
        header_style="dim",
        padding=(0, 1),
    )
    table.add_column("Ticker",  width=12)
    table.add_column("Feature", width=20)
    table.add_column("Value",   justify="right", width=10)

    n_features = len(feature_names)
    for stock_idx in range(min(3, n_stocks)):
        start = stock_idx * n_features
        row   = obs[start: start + n_features]
        top3  = np.argsort(np.abs(row))[-3:][::-1]
        for feat_idx in top3:
            val    = row[feat_idx]
            colour = "green" if val > 0 else ("red" if val < 0 else "dim")
            table.add_row(
                feature_names[0].split("_")[0] if stock_idx == 0 else "",
                feature_names[feat_idx],
                Text(f"{val:+.3f}", style=colour),
            )
    return table


# ── Main simulation loop ──────────────────────────────────────────────────────

def simulate(
    config: dict,
    risk_profile: str,
    term: str,
    n_steps: int,
    with_news: bool,
    step_delay: float,
) -> None:
    from agent.ppo_agent import LatentAlphaAgent
    from env.data_loader import NSEDataLoader
    from env.trading_env import NSETradingEnv

    # ── Load model ────────────────────────────────────────────────────────────
    agent = LatentAlphaAgent(config, risk_profile=risk_profile, term=term)
    if not agent._checkpoint_exists():
        console.print(
            f"[red]No trained model found for risk=[bold]{risk_profile}[/bold] "
            f"term=[bold]{term}[/bold][/red]\n"
            f"Train first: [bold]python train.py --risk {risk_profile} --term {term}[/bold]"
        )
        return
    agent.load()

    # ── Load recent data ──────────────────────────────────────────────────────
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=config["terms"][term]["lookback_days"] + 60)).strftime("%Y-%m-%d")

    console.print(f"[dim]Downloading recent data {start_date} → {end_date}...[/dim]")
    loader = NSEDataLoader(config)
    try:
        data = loader.load(start=start_date, end=end_date, term=term, use_cache=False)
        data = loader.align_dates(data)
    except Exception as e:
        console.print(f"[red]Data download failed: {e}[/red]")
        return

    feature_names = loader.get_feature_names()

    # ── LLM sentiment (optional) ──────────────────────────────────────────────
    sentiment_fn = None
    if with_news:
        try:
            from llm.news_fetcher import NewsFetcher
            from llm.verifier import NewsVerifier
            from llm.sentiment_encoder import SentimentEncoder
            console.print("[dim]Fetching news sentiment...[/dim]")
            fetcher   = NewsFetcher(config)
            verifier  = NewsVerifier(config["llm"]["min_sources_for_verification"])
            encoder   = SentimentEncoder(config, config["market"]["tickers"])
            articles  = fetcher.fetch_for_all_tickers(config["market"]["tickers"])
            verified  = {t: verifier.verify(a) for t, a in articles.items()}
            sentiment_fn = encoder.make_sentiment_fn(verified)
            console.print("[green]✓ News sentiment active[/green]")
        except Exception as e:
            console.print(f"[yellow]News sentiment unavailable: {e}[/yellow]")

    # ── Build environment ─────────────────────────────────────────────────────
    env = NSETradingEnv(
        data=data,
        feature_names=feature_names,
        config=config,
        risk_profile=risk_profile,
        term=term,
        sentiment_fn=sentiment_fn,
    )

    tickers       = env.tickers
    n_stocks      = env.n_stocks
    initial_cap   = env.initial_capital
    max_steps     = min(n_steps, env._n_steps - 1) if n_steps else env._n_steps - 1

    obs, info     = env.reset()
    prev_weights  = np.zeros(n_stocks)
    step          = 0
    done = truncated = False

    console.print(f"\n[bold]Starting simulation[/bold] — {max_steps} steps | "
                  f"[cyan]{risk_profile}[/cyan] / [cyan]{term}[/cyan] | "
                  f"{'news ON' if with_news else 'news OFF'}\n")

    with Live(console=console, refresh_per_second=4, screen=False) as live:
        while not (done or truncated) and step < max_steps:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            prices = env._get_prices(env._current_step).tolist()

            # ── Render ────────────────────────────────────────────────────────
            layout = Layout()
            layout.split_column(
                Layout(build_portfolio_panel(
                    step=step,
                    date=info["date"],
                    portfolio_value=info["portfolio_value"],
                    initial_capital=initial_cap,
                    cash=info["cash"],
                    risk_profile=risk_profile,
                    term=term,
                ), size=5),
                Layout(build_positions_table(
                    tickers=tickers,
                    weights=info["weights"],
                    prices=prices,
                    prev_weights=prev_weights.tolist(),
                ), name="positions"),
            )
            live.update(layout)

            prev_weights = np.array(info["weights"])
            step += 1
            time.sleep(step_delay)

    # ── Final summary ─────────────────────────────────────────────────────────
    console.print("\n")
    final_pnl     = info["portfolio_value"] - initial_cap
    final_pnl_pct = final_pnl / initial_cap * 100
    colour        = "green" if final_pnl >= 0 else "red"
    sign          = "+" if final_pnl >= 0 else ""

    console.print(Panel(
        f"[bold]Final portfolio:[/bold]  ₹{info['portfolio_value']:,.0f}\n"
        f"[bold]P&L:[/bold]             [{colour}]{sign}₹{abs(final_pnl):,.0f}  ({sign}{final_pnl_pct:.2f}%)[/{colour}]\n"
        f"[bold]Steps completed:[/bold] {step}",
        title="[bold]Simulation complete[/bold]",
        border_style="green" if final_pnl >= 0 else "red",
    ))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="latent.alpha — live paper trade simulation")
    parser.add_argument("--risk",       default="balanced",
                        choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--term",       default="medium",
                        choices=["short", "medium", "long"])
    parser.add_argument("--steps",      type=int, default=0,
                        help="Number of steps to simulate (0 = all available data)")
    parser.add_argument("--with-news",  action="store_true",
                        help="Fetch live news and compute LLM sentiment")
    parser.add_argument("--delay",      type=float, default=0.1,
                        help="Seconds between steps in the live display (default: 0.1)")
    args = parser.parse_args()

    config = load_config()
    simulate(
        config=config,
        risk_profile=args.risk,
        term=args.term,
        n_steps=args.steps,
        with_news=args.with_news,
        step_delay=args.delay,
    )


if __name__ == "__main__":
    main()
