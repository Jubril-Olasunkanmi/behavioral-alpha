from src.data_cleaning import load_trade_data
from src.metrics import calculate_trader_metrics
from src.model import train_profitability_model, add_model_predictions
from src.portfolio_metrics import calculate_portfolio_metrics
from src.explainability import get_feature_importance

def main():
    trades = load_trade_data("data/raw/binance_simulated_trades.csv")
    behavioral_metrics = calculate_trader_metrics(trades)
    portfolio_metrics = calculate_portfolio_metrics(trades)
    metrics = behavioral_metrics.merge(portfolio_metrics, on="trader_id", how="left")
    model, scores = train_profitability_model(metrics)
    metrics_with_predictions = add_model_predictions(model, metrics)
    metrics_with_predictions.to_csv("data/processed/behavioral_alpha_phase2_metrics.csv", index=False)
    feature_importance = get_feature_importance(model)
    feature_importance.to_csv("data/processed/model_feature_importance.csv", index=False)
    print("Behavioral Alpha pipeline completed successfully.")
    print("Model scores:", scores)
    print("\nTop model drivers:")
    print(feature_importance)
    print(metrics_with_predictions.head(10))

if __name__ == "__main__":
    main()