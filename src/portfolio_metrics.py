import pandas as pd

def calculate_portfolio_metrics(trades_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for trader_id, group in trades_df.groupby("trader_id"):
        group = group.sort_values("exit_time") if "exit_time" in group.columns else group.copy()
        returns = group["roi"].fillna(0)
        equity_curve = (1 + returns).cumprod()
        total_return = equity_curve.iloc[-1] - 1 if len(equity_curve) > 0 else 0
        volatility = returns.std()
        sharpe_ratio = returns.mean() / volatility if volatility and volatility != 0 else 0
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std()
        sortino_ratio = returns.mean() / downside_volatility if downside_volatility and downside_volatility != 0 else 0
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        results.append({"trader_id": trader_id, "total_return": total_return, "volatility": volatility, "sharpe_ratio": sharpe_ratio, "sortino_ratio": sortino_ratio, "max_drawdown": max_drawdown})
    return pd.DataFrame(results)