from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)
ROOT = Path(__file__).resolve().parents[1]
BINANCE_FILE = ROOT / "data/raw/binance_btcusdt.csv"
OUTPUT_FILE = ROOT / "data/raw/binance_simulated_trades.csv"

def load_market_data():
    df = pd.read_csv(BINANCE_FILE)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["close_time"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "close"])
    df["market_return"] = df["close"].pct_change()
    df["volatility"] = df["market_return"].rolling(12).std().fillna(0)
    return df.reset_index(drop=True)

def simulate_trade(market, trader_id, trade_number, profile):
    start_idx = np.random.randint(0, len(market) - 25)
    if profile == "disciplined": holding_period, position_size, stop_loss, take_profit = np.random.randint(2,8), np.random.uniform(100,1000), -0.015, 0.025
    elif profile == "loss_averse": holding_period, position_size, stop_loss, take_profit = np.random.randint(8,24), np.random.uniform(100,1500), -0.08, 0.008
    elif profile == "revenge_trader": holding_period, position_size, stop_loss, take_profit = np.random.randint(3,15), np.random.uniform(500,3000), -0.05, 0.012
    elif profile == "greedy_trader": holding_period, position_size, stop_loss, take_profit = np.random.randint(6,20), np.random.uniform(200,2000), -0.04, 0.05
    else: holding_period, position_size, stop_loss, take_profit = np.random.randint(1,12), np.random.uniform(100,1200), -0.03, 0.02
    entry = market.loc[start_idx]
    entry_price = entry["close"]
    direction = np.random.choice(["long", "short"], p=[0.65, 0.35])
    exit_idx = min(start_idx + holding_period, len(market) - 1)
    for i in range(start_idx + 1, exit_idx + 1):
        current_price = market.loc[i, "close"]
        trade_return = (current_price - entry_price) / entry_price if direction == "long" else (entry_price - current_price) / entry_price
        if trade_return <= stop_loss or trade_return >= take_profit:
            exit_idx = i
            break
    exit_row = market.loc[exit_idx]
    exit_price = exit_row["close"]
    roi = (exit_price - entry_price) / entry_price if direction == "long" else (entry_price - exit_price) / entry_price
    profit = position_size * roi
    duration_minutes = (exit_row["close_time"] - entry["open_time"]).total_seconds() / 60
    return {"trader_id": trader_id, "trade_id": f"{trader_id}_{trade_number:04d}", "trade_type": np.random.choice(["single", "copy", "mirror"], p=[0.45, 0.10, 0.45]), "symbol": "BTCUSDT", "direction": direction, "entry_time": entry["open_time"], "exit_time": exit_row["close_time"], "entry_price": round(entry_price,2), "exit_price": round(exit_price,2), "investment": round(position_size,2), "profit": round(profit,2), "roi": round(roi,5), "duration_minutes": round(duration_minutes,2), "profile": profile}

def simulate_traders(n_traders=250):
    market = load_market_data()
    profiles = ["disciplined", "loss_averse", "revenge_trader", "greedy_trader", "random_trader"]
    profile_probs = [0.25, 0.30, 0.15, 0.15, 0.15]
    trades = []
    for trader_num in range(1, n_traders + 1):
        trader_id = f"BTR{trader_num:04d}"
        profile = np.random.choice(profiles, p=profile_probs)
        for trade_number in range(1, np.random.randint(30, 120) + 1):
            trades.append(simulate_trade(market, trader_id, trade_number, profile))
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(OUTPUT_FILE, index=False)
    print("Binance-based simulated trader dataset created successfully.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total trades: {len(trades_df):,}")
    print(trades_df.head())
    return trades_df

if __name__ == "__main__":
    simulate_traders()