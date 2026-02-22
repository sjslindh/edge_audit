import numpy as np


def expectancy_R(win_rate: float, avg_win_R: float, avg_loss_R: float, risk_per_trade: float) -> float:
    """
    Expected return per trade (decimal), assuming:
      - avg_win_R / avg_loss_R are expressed in R-multiples
      - each trade risks 'risk_per_trade' fraction of equity
    """
    loss_rate = 1.0 - win_rate
    exp_R = (win_rate * avg_win_R) - (loss_rate * avg_loss_R)
    return exp_R * risk_per_trade


def kelly_fraction_R(win_rate: float, avg_win_R: float, avg_loss_R: float) -> float:
    """
    Classical Kelly fraction (as fraction of bankroll to risk per trade),
    derived using b = avg_win_R / avg_loss_R for a binary outcome model.

    Can be negative if edge is negative.
    """
    if avg_loss_R <= 0:
        return np.nan
    b = avg_win_R / avg_loss_R
    q = 1.0 - win_rate
    return (b * win_rate - q) / b


def profit_factor_estimate_R(win_rate: float, avg_win_R: float, avg_loss_R: float) -> float:
    """
    Rough PF estimate from averages:
      PF ≈ (win_rate*avg_win_R) / ((1-win_rate)*avg_loss_R)
    """
    denom = (1.0 - win_rate) * avg_loss_R
    if denom <= 0:
        return np.nan
    return (win_rate * avg_win_R) / denom


def monte_carlo_equity_R(
    win_rate: float,
    avg_win_R: float,
    avg_loss_R: float,
    risk_per_trade: float,
    n_trades: int,
    start_capital: float,
    n_sims: int = 1000,
    seed: int = 42,
):
    """
    Monte Carlo equity simulation:

    Each trade:
      - win  -> +avg_win_R * risk_per_trade
      - loss -> -avg_loss_R * risk_per_trade

    risk_per_trade is a fraction of current equity (compounded).
    """
    rng = np.random.default_rng(seed)
    wins = rng.random((n_sims, n_trades)) < win_rate

    trade_returns = np.where(
        wins,
        avg_win_R * risk_per_trade,
        -avg_loss_R * risk_per_trade
    )

    equity = np.zeros((n_sims, n_trades + 1), dtype=float)
    equity[:, 0] = start_capital

    for t in range(n_trades):
        equity[:, t + 1] = equity[:, t] * (1.0 + trade_returns[:, t])

    peaks = np.maximum.accumulate(equity, axis=1)
    drawdowns = (equity - peaks) / peaks  # negative values
    max_dd = drawdowns.min(axis=1)        # most negative
    end_cap = equity[:, -1]

    stats = {
        "median_end": float(np.median(end_cap)),
        "p05_end": float(np.percentile(end_cap, 5)),
        "p95_end": float(np.percentile(end_cap, 95)),
        "prob_down_20": float(np.mean(end_cap <= start_capital * 0.8)),
        "prob_double": float(np.mean(end_cap >= start_capital * 2.0)),
        "median_max_dd": float(np.median(max_dd)),
        "prob_dd_30": float(np.mean(max_dd <= -0.30)),
        "prob_dd_50": float(np.mean(max_dd <= -0.50)),
    }

    return {"equity": equity, "max_dd": max_dd, "end_cap": end_cap, "stats": stats}