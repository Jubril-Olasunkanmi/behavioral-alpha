import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FEATURES = ["win_ratio", "risk_reward_ratio", "win_loss_roi_ratio", "win_loss_holding_time_ratio", "trade_frequency_score"]

def train_profitability_model(metrics_df: pd.DataFrame):
    df = metrics_df.copy().replace([float("inf"), -float("inf")], 0)
    df[FEATURES] = df[FEATURES].fillna(0)
    X = df[FEATURES]
    y = df["profitable_flag"]
    if y.nunique() < 2:
        return None, {"error": "Model needs both profitable and unprofitable traders."}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, {
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "precision": round(precision_score(y_test, preds, zero_division=0), 3),
        "recall": round(recall_score(y_test, preds, zero_division=0), 3),
        "f1_score": round(f1_score(y_test, preds, zero_division=0), 3),
    }

def add_model_predictions(model, metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    df[FEATURES] = df[FEATURES].replace([float("inf"), -float("inf")], 0).fillna(0)
    if model is None:
        df["ml_prediction"] = "Not available"
        df["profitability_probability"] = None
        return df
    df["profitability_probability"] = model.predict_proba(df[FEATURES])[:, 1]
    df["ml_prediction"] = df["profitability_probability"].apply(lambda x: "Likely profitable" if x >= 0.5 else "Likely unprofitable")
    return df