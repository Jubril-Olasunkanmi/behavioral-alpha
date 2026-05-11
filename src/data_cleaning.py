import pandas as pd

REQUIRED_COLUMNS = ["trader_id", "trade_id", "trade_type", "investment", "profit", "roi", "duration_minutes"]

def load_trade_data(file_path: str) -> pd.DataFrame:
    return clean_trade_data(pd.read_csv(file_path))

def clean_trade_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    for col in ["trader_id", "trade_id", "trade_type"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    for col in ["investment", "profit", "roi", "duration_minutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df["is_win"] = df["profit"] > 0
    df["is_loss"] = df["profit"] < 0
    return df