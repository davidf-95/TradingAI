# MachineLearning.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import get_kline_data, get_all_usdt_pairs, get_suggested_coins
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import talib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image
import io
from NewsSources import NewsAggregator
import os
from pathlib import Path

# --- Setup API Keys ---
def setup_environment():
    """Initialize configuration and secrets"""
    # Create .streamlit directory if it doesn't exist
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    # Create secrets.toml template if it doesn't exist
    secrets_file = streamlit_dir / "secrets.toml"
    if not secrets_file.exists():
        with open(secrets_file, "w") as f:
            f.write("""# Add your API keys below
CRYPTOPANIC_API = \"your_api_key_here\"
# BINANCE_API = \"your_binance_key\"
# BINANCE_SECRET = \"your_binance_secret\"
""")

setup_environment()

# --- Configuration ---
MODEL_TYPES = {
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "LSTM (experimental)": "lstm"
}

# --- Feature Engineering ---
class FeatureEngineer:
    def __init__(self, data):
        self.data = data.copy()
        self.required_features = []
        
    def add_technical_indicators(self):
        close = self.data['close']
        self.data['rsi'] = talib.RSI(close, timeperiod=14)
        self.data['macd'], self.data['macd_signal'], _ = talib.MACD(close)
        self.data['bollinger_upper'], _, self.data['bollinger_lower'] = talib.BBANDS(close)
        self.data['atr'] = talib.ATR(self.data['high'], self.data['low'], close, timeperiod=14)
        self.required_features.extend(['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 'atr'])
        return self
    
    def add_volume_features(self):
        if 'volume' in self.data.columns:
            volume = self.data['volume']
            self.data['volume_ma'] = volume.rolling(5).mean()
            self.data['volume_pct'] = volume.pct_change()
            self.data['obv'] = talib.OBV(self.data['close'], volume)
            self.required_features.extend(['volume_ma', 'volume_pct', 'obv'])
        return self
    
    def add_time_features(self):
        self.data['hour'] = self.data.index.hour
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.required_features.extend(['hour', 'day_of_week', 'month'])
        return self
    
    def add_news_sentiment(self, sentiment_score: float):
        """Add news sentiment as a feature"""
        self.data['news_sentiment'] = sentiment_score
        self.required_features.append('news_sentiment')
        return self
    
    def prepare_for_training(self):
        """Returns features (X) and target (y) with consistent features"""
        df = self.data.dropna()
        self.required_features = [f for f in self.required_features if f in df.columns]
        return df[self.required_features], df['close']

# --- Model Training ---
def train_model(X, y, model_type="Random Forest"):
    if model_type == "lstm":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, X_reshaped.shape[2])),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        history = model.fit(
            X_reshaped, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        return model, history, scaler, X.columns.tolist()
    
    else:
        model = MODEL_TYPES[model_type](
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics.append({
                'mae': mean_absolute_error(y_test, preds),
                'r2': r2_score(y_test, preds)
            })
        
        avg_mae = np.mean([m['mae'] for m in metrics])
        avg_r2 = np.mean([m['r2'] for m in metrics])
        
        model.fit(X, y)
        return model, {'mae': avg_mae, 'r2': avg_r2}, None, X.columns.tolist()

# --- Prediction Generation ---
def generate_predictions(model, last_data, model_type, feature_names, scaler=None, steps=24):
    future_dates = [last_data.index[-1] + timedelta(hours=i) for i in range(1, steps+1)]
    future_predictions = []
    
    current_state = last_data[feature_names].iloc[-1:].copy()
    
    for _ in range(steps):
        if model_type == "lstm":
            scaled_data = scaler.transform(current_state)
            reshaped = np.reshape(scaled_data, (1, 1, scaled_data.shape[1]))
            pred = model.predict(reshaped, verbose=0)[0][0]
        else:
            pred = model.predict(current_state)[0]
        
        future_predictions.append(pred)
        
        # Update state for next prediction
        current_state.iloc[0] = update_features(current_state.iloc[0], pred, feature_names)
    
    return pd.Series(future_predictions, index=future_dates)

def update_features(row, new_price, feature_names):
    """Updates time-based features for next prediction step"""
    if 'hour' in feature_names:
        row['hour'] = (row['hour'] + 1) % 24
        if row['hour'] == 0 and 'day_of_week' in feature_names:
            row['day_of_week'] = (row['day_of_week'] + 1) % 7
    
    if 'price_change' in feature_names and 'close' in row:
        row['price_change'] = (new_price - row['close']) / row['close']
    
    if 'close' in feature_names:
        row['close'] = new_price
    
    return row

# --- Main Trading Page ---
def show_ai_prognosen_page():
   # st.set_page_config(layout="wide")
    st.title("🤖 Advanced AI Trading Suite")
    
    tab1, tab2, tab3 = st.tabs(["📈 Trading Signals", "📰 Market News", "⚙️ Settings"])
    
    with tab1:
        st.header("AI Price Predictions")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            symbol = st.selectbox(
                "Select Asset",
                get_all_usdt_pairs(),
                index=get_all_usdt_pairs().index("BTCUSDT"),
                key="asset_select"
            )
        with col2:
            model_type = st.selectbox(
                "Model Type",
                list(MODEL_TYPES.keys()),
                index=0,
                key="model_select"
            )
        
        # Get news sentiment
        news_df = NewsAggregator.get_all_news(
            selected_sources=['Binance', 'CryptoPanic', 'CoinGecko'],
            cryptopanic_key=st.secrets.get("CRYPTOPANIC_API", "")
        )
        avg_sentiment = news_df['sentiment'].mean() if not news_df.empty else 0.5
        
        if st.button("🔄 Generate Predictions", type="primary", key="predict_btn"):
            with st.spinner("Analyzing market data..."):
                try:
                    # 1. Get price data
                    price_data = get_kline_data(symbol, interval='1h', limit=500)
                    if price_data.empty:
                        st.error("No price data available")
                        return
                    
                    # 2. Feature engineering
                    engineer = FeatureEngineer(price_data)
                    (
                        engineer
                        .add_technical_indicators()
                        .add_volume_features()
                        .add_time_features()
                        .add_news_sentiment(avg_sentiment)
                    )
                    
                    X, y = engineer.prepare_for_training()
                    
                    if len(X) < 100:
                        st.warning("Insufficient data for reliable predictions")
                        return
                    
                    # 3. Train model
                    model, metrics, scaler, feature_names = train_model(X, y, model_type)
                    
                    # 4. Generate predictions
                    last_data = engineer.data[feature_names].iloc[-100:]
                    predictions = generate_predictions(
                        model, last_data, model_type, feature_names, scaler
                    )
                    
                    # 5. Display results
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=price_data.index,
                        y=price_data['close'],
                        name='Historical Prices',
                        line=dict(color='#1f77b4')
                    ))
                    fig.add_trace(go.Scatter(
                        x=predictions.index,
                        y=predictions,
                        name='AI Predictions',
                        line=dict(color='#ff7f0e', dash='dash')
                    ))
                    fig.update_layout(
                        title=f"{symbol} Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price (USDT)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Mean Absolute Error", f"{metrics['mae']:.4f}")
                    col2.metric("Model R² Score", f"{metrics['r2']:.2%}")
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        st.dataframe(importance_df.head(10))
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    
    with tab2:
        st.header("Latest Market News")
        if not news_df.empty:
            for _, row in news_df.head(15).iterrows():
                with st.expander(f"{row['source']}: {row['title']}"):
                    st.write(f"**Date:** {row['date']}")
                    st.write(f"**Sentiment:** {row['sentiment']:.0%}")
                    st.markdown(f"[Read more →]({row['url']})")
        else:
            st.warning("No news data available")
    
    with tab3:
        st.header("Settings")
        st.write("Configure your API keys and settings")
        
        st.subheader("API Configuration")
        with st.expander("CryptoPanic API Key"):
            st.write("Get your API key from [cryptopanic.com](https://cryptopanic.com/developers/api/)")
            st.text_input("Enter CryptoPanic API Key", 
                         value=st.secrets.get("CRYPTOPANIC_API", ""),
                         type="password",
                         key="cryptopanic_key_input")
        
        st.subheader("Model Settings")
        st.slider("Number of Estimators", 50, 300, 150, key="n_estimators")
        st.slider("Max Depth", 3, 20, 10, key="max_depth")
        st.checkbox("Include News Sentiment", True, key="use_news_sentiment")

if __name__ == "__main__":
    show_ai_prognosen_page()