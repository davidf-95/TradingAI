import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils import get_kline_data, get_all_usdt_pairs, get_suggested_coins
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import requests
import talib

# --- Hilfsfunktionen für zusätzliche Daten ---
@st.cache_data(ttl=3600)
def get_funding_rate(symbol):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {'symbol': symbol, 'limit': 30}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame(data)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df.set_index('fundingTime', inplace=True)
        df['fundingRate'] = df['fundingRate'].astype(float)
        return df[['fundingRate']]
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_sentiment_data(coin):
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    fake_sentiment = np.random.uniform(-1, 1, size=len(dates))
    return pd.DataFrame({'sentiment': fake_sentiment}, index=dates)

# --- Feature Engineering ---
def create_features(df, symbol):
    # Sicherstellen, dass benötigte Spalten vorhanden sind
    required_columns = ['close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Erforderliche Spalte '{col}' nicht in Daten gefunden")
    
    # Technische Indikatoren (funktionieren nur mit close)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
    df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(df['close'])
    
    # Volumen Features nur wenn vorhanden
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        df['volume_change'] = df['volume'].pct_change()
    else:
        st.warning("Volumendaten nicht verfügbar - Volumenfeatures werden übersprungen")
    
    # Zeit Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    return df.dropna()

# --- Modelltraining ---
def train_model(X, y):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    return model, mae, r2

# --- Prognoseerstellung ---
def make_predictions(model, last_data, steps=24):
    future_dates = [last_data.index[-1] + timedelta(hours=i) for i in range(1, steps+1)]
    future_features = []
    
    for i in range(steps):
        features = last_data.iloc[-1].copy()
        features['hour'] = (features['hour'] + i) % 24
        features['day_of_week'] = (features['day_of_week'] + (i // 24)) % 7
        future_features.append(features)
    
    future_df = pd.DataFrame(future_features, index=future_dates)
    predictions = model.predict(future_df)
    
    return pd.Series(predictions, index=future_dates)

# --- Hauptfunktion für die Seite ---
def show_ai_prognosen_page():
    st.header("🔮 AI Prognosen & Mustererkennung")
    st.markdown("""
    **Machine Learning Modell** das Kursprognosen auf Basis von:
    - Historischen Kursdaten
    - Technischen Indikatoren (RSI, MACD, Bollinger Bands)
    """)
    
    # --- Coin Auswahl ---
    suggested_coins = get_suggested_coins()
    col_coin, col_interval = st.columns([2, 1])
    
    symbol = col_coin.selectbox(
        "Coin auswählen",
        options=get_all_usdt_pairs(),
        index=get_all_usdt_pairs().index(suggested_coins[0]) if suggested_coins[0] in get_all_usdt_pairs() else 0
    )
    
    interval = col_interval.selectbox(
        "Zeitintervall",
        options=['1h', '4h', '1d'],
        index=0
    )
    
    if st.button("Prognose generieren", type="primary"):
        with st.spinner("Daten werden geladen und Modell trainiert..."):
            try:
                # 1. Kursdaten mit allen verfügbaren Spalten laden
                price_data = get_kline_data(symbol, interval=interval, limit=500)
                if price_data.empty:
                    st.error("Keine Preisdaten verfügbar")
                    return
                
                # Standard-Binance Kline Spalten: 
                # ['open', 'high', 'low', 'close', 'volume', 'close_time', ...]
                # Wir brauchen mindestens 'close'
                
                # Feature Engineering
                feature_df = create_features(price_data, symbol)
                
                # Zielvariable (nächster Kurswert)
                feature_df['target'] = feature_df['close'].shift(-1)
                feature_df = feature_df.dropna()
                
                if len(feature_df) < 100:
                    st.warning("Nicht genug Daten für zuverlässige Prognosen")
                    return
                
                # Modelltraining
                X = feature_df.drop(['target'], axis=1)
                y = feature_df['target']
                
                model, mae, r2 = train_model(X, y)
                
                # Prognose
                future_predictions = make_predictions(model, feature_df.drop('target', axis=1))
                
                # Visualisierung
                st.subheader(f"Prognose für {symbol}")
                
                col1, col2 = st.columns(2)
                col1.metric("Durchschnittlicher Fehler (MAE)", f"{mae:.4f}")
                col2.metric("Modellgüte (R²)", f"{r2:.2%}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=feature_df.index,
                    y=feature_df['close'],
                    name='Historische Daten',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=future_predictions.index,
                    y=future_predictions,
                    name='AI Prognose',
                    line=dict(color='green', dash='dash')
                ))
                fig.update_layout(
                    title=f"24-Perioden Prognose für {symbol}",
                    xaxis_title="Datum",
                    yaxis_title="Preis (USDT)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                st.subheader("Wichtigste Einflussfaktoren")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(feature_importance.head(10))
                
            except Exception as e:
                st.error(f"Fehler bei der Prognoseerstellung: {str(e)}")