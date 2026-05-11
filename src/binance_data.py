from binance.client import Client
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data/raw/binance_btcusdt.csv"
client = Client()

def fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=1000):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]
    df = pd.DataFrame(klines, columns=columns)
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df

if __name__ == "__main__":
    btc = fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=1000)
    btc.to_csv(OUTPUT_FILE, index=False)
    print("Binance BTCUSDT market data saved successfully.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(btc.head())