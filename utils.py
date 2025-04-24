import streamlit as st
import requests
import pandas as pd
import json
from streamlit_js_eval import streamlit_js_eval

# Binance API
BASE_URL = 'https://fapi.binance.com'

@st.cache_data(ttl=60)
def get_kline_data(symbol, interval='1h', limit=500):
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df  # Jetzt mit allen Spalten zurückgeben

@st.cache_data(ttl=3600)
def get_all_usdt_pairs():
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    return [s['symbol'] for s in data['symbols'] if s['symbol'].endswith('USDT')]

# Favoriten-Funktionen
FAVORITES_KEY = "favorites"

def initialize_favorites():
    if FAVORITES_KEY not in st.session_state:
        result = streamlit_js_eval(js_expressions="document.cookie", key="get_cookie")
        if result:
            cookies = dict(item.split("=", 1) for item in result.split("; ") if "=" in item)
            favorites_cookie = cookies.get("favorites", "")
            if favorites_cookie:
                try:
                    favorites_cookie = favorites_cookie.replace('%22', '"').replace('%5B', '[').replace('%5D', ']').replace('%2C', ',')
                    st.session_state[FAVORITES_KEY] = json.loads(favorites_cookie)
                except:
                    st.session_state[FAVORITES_KEY] = []
            else:
                st.session_state[FAVORITES_KEY] = []

def save_favorites_to_cookies(favorites):
    js_code = f"""document.cookie = "favorites={json.dumps(favorites)}; path=/; max-age=31536000";"""
    streamlit_js_eval(js_expressions=js_code, key="set_cookie")

def add_to_favorites(coin):
    favorites = st.session_state.get(FAVORITES_KEY, [])
    if coin not in favorites:
        favorites.append(coin)
        st.session_state[FAVORITES_KEY] = favorites
        save_favorites_to_cookies(favorites)

def remove_from_favorites(coin):
    favorites = st.session_state.get(FAVORITES_KEY, [])
    if coin in favorites:
        favorites.remove(coin)
        st.session_state[FAVORITES_KEY] = favorites
        save_favorites_to_cookies(favorites)

def is_favorite(coin):
    return coin in st.session_state.get(FAVORITES_KEY, [])

def get_suggested_coins():
    favorites = st.session_state.get(FAVORITES_KEY, [])
    popular = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
    return favorites + [c for c in popular if c not in favorites]