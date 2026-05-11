import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_cleaning import clean_trade_data
from src.metrics import calculate_trader_metrics
from src.model import train_profitability_model, add_model_predictions
from src.portfolio_metrics import calculate_portfolio_metrics
from src.explainability import get_feature_importance

st.set_page_config(page_title="Behavioral Alpha", page_icon="⟁", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {background: linear-gradient(145deg, #161B22, #0E1117); padding: 20px; border-radius: 16px; border: 1px solid #30363D; text-align: center; box-shadow: 0 8px 22px rgba(0,0,0,0.25);}
.metric-title {color: #8B949E; font-size: 14px; margin-bottom: 6px;}
.metric-value {color: white; font-size: 30px; font-weight: 800;}
.section-title {font-size: 28px; font-weight: 800; margin-top: 20px; margin-bottom: 8px;}
.small-text {color: #8B949E; font-size: 15px; line-height: 1.5;}
.brand-box {background: linear-gradient(135deg, #111827, #0E1117); border: 1px solid #30363D; border-radius: 18px; padding: 28px; margin-bottom: 18px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='brand-box'>
    <h1 style='font-size:50px; font-weight:900; color:white; margin-bottom:0;'>⟁ Behavioral Alpha</h1>
    <p style='color:#8B949E; font-size:20px; margin-top:6px;'>AI-Powered Behavioral Finance Analytics Platform</p>
    <p class='small-text'>Quantifying trader psychology, risk discipline, profitability quality, and behavioral fragility using machine learning, portfolio analytics, and real Binance BTC market data.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("⟁ Behavioral Alpha")
st.sidebar.caption("Behavioral finance × ML × market simulation")
default_file = ROOT / "data/raw/binance_simulated_trades.csv"
uploaded_file = st.sidebar.file_uploader("Upload Trade Dataset", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.markdown("""### What this app evaluates
- Trader discipline
- Emotional risk
- Loss aversion
- Risk-reward quality
- Portfolio sustainability
- Profitability prediction
""")
raw_df = pd.read_csv(uploaded_file) if uploaded_file is not None else pd.read_csv(default_file)

try:
    trades = clean_trade_data(raw_df)
    behavioral_metrics = calculate_trader_metrics(trades)
    portfolio_metrics = calculate_portfolio_metrics(trades)
    metrics = behavioral_metrics.merge(portfolio_metrics, on="trader_id", how="left")
    model, model_scores = train_profitability_model(metrics)
    metrics = add_model_predictions(model, metrics)
    feature_importance = get_feature_importance(model)

    st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    vals=[("Total Trades",f"{len(trades):,}"),("Traders",f"{metrics['trader_id'].nunique()}"),("Avg Win Ratio",f"{metrics['win_ratio'].mean():.1%}"),("Profitable Traders",f"{metrics['profitable_flag'].mean():.1%}"),("Avg Risk Score",f"{metrics['behavioral_risk_score'].mean():.1f}")]
    for col,(title,value) in zip([c1,c2,c3,c4,c5],vals):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>{title}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='section-title'>BTC Market Trend</div>", unsafe_allow_html=True)
    market_file = ROOT / "data/raw/binance_btcusdt.csv"
    if market_file.exists():
        market_df = pd.read_csv(market_file)
        market_df["open_time"] = pd.to_datetime(market_df["open_time"])
        for col in ["open","high","low","close","volume"]:
            market_df[col] = pd.to_numeric(market_df[col], errors="coerce")
        fig = px.line(market_df, x="open_time", y="close", title="BTCUSDT Closing Price Trend", template="plotly_dark")
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)
        ca, cb = st.columns(2)
        with ca:
            candle = go.Figure(data=[go.Candlestick(x=market_df["open_time"], open=market_df["open"], high=market_df["high"], low=market_df["low"], close=market_df["close"])])
            candle.update_layout(title="BTCUSDT Candlestick View", template="plotly_dark", height=430, xaxis_rangeslider_visible=False)
            st.plotly_chart(candle, use_container_width=True)
        with cb:
            market_df["return"] = market_df["close"].pct_change()
            market_df["rolling_volatility"] = market_df["return"].rolling(24).std()
            fig = px.line(market_df, x="open_time", y="rolling_volatility", title="24-Hour Rolling Volatility", template="plotly_dark")
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
    if "error" in model_scores:
        st.warning(model_scores["error"])
    else:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy", model_scores["accuracy"]); m2.metric("Precision", model_scores["precision"]); m3.metric("Recall", model_scores["recall"]); m4.metric("F1 Score", model_scores["f1_score"])
    st.divider()

    st.markdown("<div class='section-title'>Model Explainability</div>", unsafe_allow_html=True)
    if not feature_importance.empty:
        fig = px.bar(feature_importance, x="importance", y="feature", orientation="h", title="Top Behavioral Drivers", template="plotly_dark")
        fig.update_layout(height=420, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.markdown("<div class='section-title'>Behavioral Risk Analysis</div>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(metrics, x="behavioral_risk_score", color="risk_level", nbins=40, title="Behavioral Risk Score Distribution", template="plotly_dark"), use_container_width=True)
    with col2:
        st.plotly_chart(px.scatter(metrics, x="behavioral_risk_score", y="total_profit", color="risk_level", hover_data=["trader_id","win_ratio","risk_reward_ratio"], title="Risk Score vs Profitability", template="plotly_dark"), use_container_width=True)
    st.divider()

    st.markdown("<div class='section-title'>Portfolio Analytics</div>", unsafe_allow_html=True)
    col3,col4 = st.columns(2)
    with col3:
        st.plotly_chart(px.scatter(metrics, x="sharpe_ratio", y="max_drawdown", color="risk_level", hover_data=["trader_id","total_return","total_profit"], title="Sharpe Ratio vs Maximum Drawdown", template="plotly_dark"), use_container_width=True)
    with col4:
        st.plotly_chart(px.scatter(metrics, x="total_return", y="volatility", color="ml_prediction", hover_data=["trader_id","risk_level"], title="Total Return vs Volatility", template="plotly_dark"), use_container_width=True)
    st.divider()

    st.markdown("<div class='section-title'>Behavioral Finance Insights</div>", unsafe_allow_html=True)
    col5,col6 = st.columns(2)
    with col5:
        st.plotly_chart(px.scatter(metrics, x="win_ratio", y="risk_reward_ratio", color="trader_class", hover_data=["trader_id","total_profit","behavioral_risk_score"], title="Win Ratio vs Risk-Reward Ratio", template="plotly_dark"), use_container_width=True)
    with col6:
        st.plotly_chart(px.scatter(metrics, x="win_ratio", y="total_profit", color="ml_prediction", hover_data=["trader_id","trader_class","risk_level"], title="High Win-Rate Traders Can Still Lose Money", template="plotly_dark"), use_container_width=True)
    st.divider()

    st.markdown("<div class='section-title'>Trader Rankings</div>", unsafe_allow_html=True)
    display_cols=["trader_id","behavioral_alpha_score","behavioral_risk_score","risk_level","win_ratio","total_profit","total_return","sharpe_ratio","sortino_ratio","max_drawdown","risk_reward_ratio","win_loss_roi_ratio","win_loss_holding_time_ratio","trader_class","ml_prediction","profitability_probability"]
    st.dataframe(metrics[display_cols], use_container_width=True, height=500)
    st.download_button("Download Metrics CSV", data=metrics.to_csv(index=False), file_name="behavioral_alpha_metrics.csv", mime="text/csv")
except Exception as e:
    st.error(f"Something went wrong: {e}")