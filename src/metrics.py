import numpy as np
import pandas as pd

def safe_divide(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b

def classify_trader(row):
    win_ratio = row.get("win_ratio", 0)
    total_profit = row.get("total_profit", 0)
    risk_reward_ratio = row.get("risk_reward_ratio", np.nan)
    holding_time_ratio = row.get("win_loss_holding_time_ratio", np.nan)
    roi_ratio = row.get("win_loss_roi_ratio", np.nan)
    if total_profit > 0 and risk_reward_ratio >= 1 and roi_ratio >= 1:
        return "Profitable disciplined trader"
    if win_ratio >= 0.60 and total_profit <= 0:
        return "High win-rate but risky trader"
    if holding_time_ratio < 1 and total_profit <= 0:
        return "Loss-averse trader"
    if risk_reward_ratio < 1 and roi_ratio < 1:
        return "Poor risk-reward trader"
    return "Needs further review"

def calculate_trader_metrics(df: pd.DataFrame) -> pd.DataFrame:
    output = []
    for trader_id, group in df.groupby("trader_id"):
        wins = group[group["profit"] > 0]
        losses = group[group["profit"] < 0]
        total_trades = len(group)
        winning_trades = len(wins)
        losing_trades = len(losses)
        avg_profit_win = wins["profit"].mean() if winning_trades > 0 else 0
        avg_loss_abs = abs(losses["profit"].mean()) if losing_trades > 0 else 0
        avg_roi_win = wins["roi"].mean() if winning_trades > 0 else 0
        avg_roi_loss_abs = abs(losses["roi"].mean()) if losing_trades > 0 else 0
        avg_duration_win = wins["duration_minutes"].mean() if winning_trades > 0 else 0
        avg_duration_loss = losses["duration_minutes"].mean() if losing_trades > 0 else 0
        output.append({
            "trader_id": trader_id,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_ratio": safe_divide(winning_trades, total_trades),
            "total_profit": group["profit"].sum(),
            "average_profit_on_wins": avg_profit_win,
            "average_loss_on_losses": avg_loss_abs,
            "risk_reward_ratio": safe_divide(avg_profit_win, avg_loss_abs),
            "average_roi_on_wins": avg_roi_win,
            "average_roi_on_losses": avg_roi_loss_abs,
            "win_loss_roi_ratio": safe_divide(avg_roi_win, avg_roi_loss_abs),
            "average_winning_duration": avg_duration_win,
            "average_losing_duration": avg_duration_loss,
            "win_loss_holding_time_ratio": safe_divide(avg_duration_win, avg_duration_loss),
            "trade_frequency_score": total_trades,
            "profitable_flag": int(group["profit"].sum() > 0),
        })
    result = pd.DataFrame(output)
    result["behavioral_alpha_score"] = (
        result["risk_reward_ratio"].fillna(0).clip(0, 5) * 30
        + result["win_loss_roi_ratio"].fillna(0).clip(0, 5) * 25
        + result["win_loss_holding_time_ratio"].fillna(0).clip(0, 5) * 15
        + result["win_ratio"].fillna(0).clip(0, 1) * 30
    )
    result["behavioral_risk_score"] = (
        (1 - result["win_ratio"].fillna(0).clip(0, 1)) * 20
        + (1 / result["risk_reward_ratio"].fillna(0.1).clip(0.1, 10)) * 25
        + (1 / result["win_loss_roi_ratio"].fillna(0.1).clip(0.1, 10)) * 25
        + (1 / result["win_loss_holding_time_ratio"].fillna(0.1).clip(0.1, 10)) * 20
        + result["trade_frequency_score"].rank(pct=True) * 10
    ).clip(0, 100)
    result["risk_level"] = result["behavioral_risk_score"].apply(lambda x: "High Risk" if x >= 70 else "Medium Risk" if x >= 40 else "Low Risk")
    result["trader_class"] = result.apply(classify_trader, axis=1)
    return result.sort_values("behavioral_alpha_score", ascending=False)