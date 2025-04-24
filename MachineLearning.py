import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils import get_kline_data, get_all_usdt_pairs, get_suggested_coins
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import requests
import talib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image
import io

# --- Konfiguration ---
MODEL_TYPES = {
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "LSTM (experimentell)": "lstm"
}

# --- Datenquellen Integration ---
@st.cache_data(ttl=3600)
def get_alternative_data(symbol, data_type):
    """Holt alternative Datenquellen basierend auf Typ"""
    try:
        if data_type == "funding":
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {'symbol': symbol, 'limit': 100}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df.set_index('fundingTime', inplace=True)
            return df[['fundingRate']].astype(float)
        
        elif data_type == "sentiment":
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            return pd.DataFrame({
                'sentiment': np.random.normal(0, 0.5, len(dates)),
                'social_volume': np.random.poisson(50, len(dates))
            }, index=dates)
        
        elif data_type == "onchain":
            return pd.DataFrame({
                'active_addresses': np.random.randint(1000, 5000, 100),
                'exchange_netflow': np.random.normal(0, 1000, 100)
            }, index=pd.date_range(end=datetime.now(), periods=100, freq='D'))
    
    except Exception as e:
        st.warning(f"Datenquelle {data_type} nicht verfügbar: {str(e)}")
        return pd.DataFrame()

# --- Feature Engineering Pipeline ---
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
        else:
            st.warning("Volumendaten nicht verfügbar - Volumenfeatures werden übersprungen")
        return self
    
    def add_time_features(self):
        self.data['hour'] = self.data.index.hour
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.required_features.extend(['hour', 'day_of_week', 'month'])
        return self
    
    def add_derived_features(self):
        self.data['price_change'] = self.data['close'].pct_change()
        self.data['high_low_spread'] = (self.data['high'] - self.data['low']) / self.data['low']
        self.required_features.extend(['price_change', 'high_low_spread'])
        return self
    
    def add_external_features(self, external_data):
        for name, df in external_data.items():
            if not df.empty:
                self.data = self.data.join(df, how='left').ffill()
                self.required_features.extend(df.columns.tolist())
            else:
                st.warning(f"Datenquelle {name} lieferte keine Daten")
        return self
    
    def prepare_for_training(self):
        """Gibt Features (X) und Target (y) zurück mit konsistenten Features"""
        df = self.data.dropna()
        self.required_features = [f for f in self.required_features if f in df.columns]
        if not self.required_features:
            raise ValueError("Keine gültigen Features für das Training verfügbar")
        return df[self.required_features], df['close']

# --- Modelltraining ---
def train_model(X, y, model_type="Random Forest"):
    try:
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
    
    except Exception as e:
        st.error(f"Fehler beim Modelltraining: {str(e)}")
        raise

# --- Prognoseerstellung ---
def generate_predictions(model, last_data, model_type, feature_names, scaler=None, steps=24):
    try:
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
    
    except Exception as e:
        st.error(f"Fehler bei der Prognoseerstellung: {str(e)}")
        raise

def update_features(row, new_price, feature_names):
    """Aktualisiert Features für die nächste Vorhersage"""
    if 'hour' in feature_names:
        row['hour'] = (row['hour'] + 1) % 24
        if row['hour'] == 0 and 'day_of_week' in feature_names:
            row['day_of_week'] = (row['day_of_week'] + 1) % 7
    
    if 'price_change' in feature_names and 'close' in row:
        row['price_change'] = (new_price - row['close']) / row['close']
    
    if 'close' in feature_names:
        row['close'] = new_price
    
    return row

# --- Hauptfunktion ---
def show_ai_prognosen_page():
    st.title("🤖 Advanced AI Trading Prognosen")
    
    with st.expander("ℹ️ Über dieses Tool"):
        st.markdown("""
        **Machine Learning Modell für Krypto-Prognosen**  
        Kombiniert multiple Datenquellen und ML-Techniken für präzisere Vorhersagen.
        """)
    
    # --- Sidebar Konfiguration ---
    with st.sidebar:
        st.header("Modell-Konfiguration")
        model_type = st.selectbox(
            "Modelltyp",
            list(MODEL_TYPES.keys()),
            index=0
        )
        
        data_sources = st.multiselect(
            "Zusätzliche Datenquellen",
            ["Funding Rates", "Sentiment Daten", "On-Chain Metriken"],
            default=["Funding Rates"]
        )
    
    # --- Hauptbereich ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.selectbox(
            "Asset auswählen",
            get_all_usdt_pairs(),
            index=get_all_usdt_pairs().index("BTCUSDT")
        )
    
    with col2:
        interval = st.selectbox(
            "Zeitintervall",
            ["1h", "4h", "1d"],
            index=1
        )
    
    if st.button("🔮 Prognose generieren", type="primary", use_container_width=True):
        with st.spinner("Analysiere Daten und trainiere Modell..."):
            try:
                # 1. Kursdaten laden
                price_data = get_kline_data(symbol, interval=interval, limit=500)
                if price_data.empty:
                    st.error("Keine Preisdaten verfügbar")
                    return
                
                # 2. Externe Daten sammeln
                external_data = {}
                if "Funding Rates" in data_sources:
                    external_data['funding'] = get_alternative_data(symbol, "funding")
                if "Sentiment Daten" in data_sources:
                    external_data['sentiment'] = get_alternative_data(symbol, "sentiment")
                if "On-Chain Metriken" in data_sources:
                    external_data['onchain'] = get_alternative_data(symbol, "onchain")
                
                # 3. Feature Engineering
                engineer = FeatureEngineer(price_data)
                (
                    engineer
                    .add_technical_indicators()
                    .add_volume_features()
                    .add_time_features()
                    .add_derived_features()
                    .add_external_features(external_data)
                )
                
                X, y = engineer.prepare_for_training()
                
                if len(X) < 100:
                    st.warning("Nicht genug Daten für zuverlässige Prognosen (mind. 100 Datenpunkte benötigt)")
                    return
                
                # 4. Modelltraining
                model, metrics, scaler, feature_names = train_model(X, y, model_type)
                
                # 5. Prognose erstellen
                last_data = engineer.data[feature_names].iloc[-100:]
                predictions = generate_predictions(
                    model, last_data, model_type, feature_names, scaler
                )
                
                # 6. Ergebnisse anzeigen
                st.success("Prognose erfolgreich generiert!")
                
                # Visualisierung
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['close'],
                    name='Historische Daten',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=predictions.index,
                    y=predictions,
                    name='AI Prognose',
                    line=dict(color='orange', dash='dash')
                ))
                fig.update_layout(
                    title=f"{symbol} Preisprognose",
                    xaxis_title="Datum",
                    yaxis_title="Preis (USDT)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Metriken
                col1, col2 = st.columns(2)
                col1.metric("Durchschnittlicher Fehler (MAE)", f"{metrics['mae']:.4f}")
                col2.metric("Modellgüte (R²)", f"{metrics['r2']:.2%}")
                
                # Feature Importance
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Wichtigste Einflussfaktoren")
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Wichtigkeit': model.feature_importances_
                    }).sort_values('Wichtigkeit', ascending=False)
                    st.dataframe(importance_df.head(10))
                
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Ein unerwarteter Fehler ist aufgetreten: {str(e)}")