#  pip install streamlit pandas numpy matplotlib seaborn requests streamlit-js-eval
#   streamlit run app.py
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval
import json  # Import json

# --- Seitenkonfiguration ---
st.set_page_config(layout="wide")  # Dies muss der erste Streamlit-Befehl sein!

# Binance API-Endpunkt faür Kline-Daten
BASE_URL = 'https://fapi.binance.com'

# --- Hilfsfunktionen ---
@st.cache_data(ttl=60)
def get_kline_data(symbol, interval='1h', limit=500):
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df[['close']]

@st.cache_data(ttl=3600)
def get_all_usdt_pairs():
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    symbols = [symbol['symbol'] for symbol in data['symbols'] if symbol['symbol'].endswith('USDT')]
    return symbols



# --- Favoriten-Verwaltung mit localStorage ---
FAVORITES_KEY = "favorites"
DEFAULT_FAVORITES = []

def initialize_favorites():
    if FAVORITES_KEY not in st.session_state:
        # Cookie auslesen
        result = streamlit_js_eval(js_expressions="document.cookie", key="get_cookie")
        if result:
            cookies = dict(item.split("=", 1) for item in result.split("; ") if "=" in item)
            favorites_cookie = cookies.get("favorites")
            st.write(f" favorites_cookie: {favorites_cookie}")
            if favorites_cookie:
                try:
                    # Entschlüsselung der URL-komprimierten Zeichen
                    favorites_cookie = favorites_cookie.replace('%22', '"').replace('%5B', '[').replace('%5D', ']').replace('%2C', ',').replace('%20', ' ') 
                 #   st.write(f"try favorites_cookie: {favorites_cookie}")

                    # JSON-Parsing
                    favorites = json.loads(favorites_cookie)
                    if isinstance(favorites, list):
                        st.session_state[FAVORITES_KEY] = favorites
                    else:
                        st.session_state[FAVORITES_KEY] = DEFAULT_FAVORITES
                except json.JSONDecodeError as e:
                    st.warning(f"Fehler beim Laden der Favoriten aus dem Cookie: {e}")
                    st.session_state[FAVORITES_KEY] = DEFAULT_FAVORITES


def save_favorites_to_cookies(favorites):
    favorites_json = json.dumps(favorites)
    js_code = f"""
        const expires = new Date();
        expires.setFullYear(expires.getFullYear() + 1); // 1 Jahr gültig
        document.cookie = "favorites=" + encodeURIComponent('{favorites_json}') + "; path=/; expires=" + expires.toUTCString();
               //     setFrameHeight(document.documentElement.clientHeight)
    //             console.log("updateFavoritesFromPython aufgerufen mit:", newFavorites);

    """
    # Set the cookie using JavaScript
    # streamlit_js_eval verweist oder den Browser dazu bringt, die Seite neu zu laden.
    streamlit_js_eval(js_expressions=js_code, key="set_cookie")
    st.write(f"Cookie gespeichert: {favorites_json}")

def add_to_favorites(coin):
    favorites = st.session_state.get(FAVORITES_KEY, [])
    if coin not in favorites:
        favorites.append(coin)
        st.session_state[FAVORITES_KEY] = favorites
        st.write(f"{coin} zu Favoriten hinzugefügt. Neue Favoriten: {favorites}")
        save_favorites_to_cookies(favorites)
       # st.rerun()  # Update the page without a full reload

def remove_from_favorites(coin):
    favorites = st.session_state.get(FAVORITES_KEY, [])
    if coin in favorites:
        favorites.remove(coin)
        st.session_state[FAVORITES_KEY] = favorites
        st.write(f"{coin} von Favoriten entfernt. Neue Favoriten: {favorites}")
        save_favorites_to_cookies(favorites)
      #  st.rerun()  # Update the page without a full reload

def is_favorite(coin):
    return coin in st.session_state.get(FAVORITES_KEY, [])

def get_suggested_coins():
    # Get favorites from session state
    favorite_coins = st.session_state.get(FAVORITES_KEY, [])
    popular_coins = ['NILUSDT', 'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
    suggested = list(favorite_coins) + [coin for coin in popular_coins if coin not in favorite_coins]
    return suggested

# --- Sidebar für Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Gehe zu", ["Spread Analyse", "Korrelationsanalyse", "Favoriten"])

# --- Hauptteil der Anwendung ---
#initialize_favorites()

if page == "Favoriten":
    st.header("Favoriten Liste")
    initialize_favorites()
    new_favorite = st.text_input("Neuen Favoriten hinzufügen:")
    if st.button("➕ Hinzufügen"):
        new_favorite_upper = new_favorite.upper()
        if new_favorite_upper + "USDT" in get_all_usdt_pairs():
            add_to_favorites(new_favorite_upper + "USDT")
            new_favorite = ""  # Clear input field
        elif new_favorite_upper in get_all_usdt_pairs():
            add_to_favorites(new_favorite_upper)
            new_favorite = ""  # Clear input field
        elif new_favorite:
            st.warning(f"'{new_favorite_upper}' ist kein gültiges USDT-Paar oder existiert nicht.")

    if st.session_state.get(FAVORITES_KEY):
        for coin in sorted(st.session_state[FAVORITES_KEY]):
            col1, col2 = st.columns([3, 1])
            col1.write(coin)
            if col2.button("➖ Entfernen", key=f"remove_{coin}", on_click=remove_from_favorites, args=(coin,)):
                pass
    else:
        st.info("Deine Favoritenliste ist leer.")

elif page == "Spread Analyse":
    st.header("Spread Analyse")
    initialize_favorites()
    suggested_coins = get_suggested_coins()

    # Custom CSS für kleinere Buttons und kein Umbruch
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            font-size: 0.8em !important;
            padding: 0.2em 0.4em !important;
            margin-right: 0.1em !important;
            margin-bottom: 0.1em !important;
            word-break: normal !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_coin1, col_coin2, col_button = st.columns([2, 2, 1.5])

    # Logik zum Aktualisieren von st.session_state für Coin 1 BEVOR text_input erstellt wird
    if "suggested_coin1_clicked" in st.session_state and st.session_state["suggested_coin1_clicked"]:
        st.session_state["symbol1_spread"] = st.session_state["suggested_coin1_value"]
        st.session_state["suggested_coin1_clicked"] = False
        st.session_state["suggested_coin1_value"] = ""
        st.rerun()

    # Logik zum Aktualisieren von st.session_state für Coin 2 BEVOR text_input erstellt wird
    if "suggested_coin2_clicked" in st.session_state and st.session_state["suggested_coin2_clicked"]:
        st.session_state["symbol2_spread"] = st.session_state["suggested_coin2_value"]
        st.session_state["suggested_coin2_clicked"] = False
        st.session_state["suggested_coin2_value"] = ""
        st.rerun()

    symbol1 = col_coin1.text_input("Coin 1 (z.B. BABYUSDT)",
                                  value=st.session_state.get("symbol1_spread",
                                                            suggested_coins[0] if suggested_coins else ""),
                                  key="symbol1_spread")
    cols_suggest_1 = col_coin1.columns(min(5, len(suggested_coins)))
    for i, coin in enumerate(suggested_coins):
        if cols_suggest_1[i % len(cols_suggest_1)].button(coin, key=f"suggest1_button_{coin}"):
            st.session_state["suggested_coin1_clicked"] = True
            st.session_state["suggested_coin1_value"] = coin
            st.rerun()

    symbol2 = col_coin2.text_input("Coin 2 (z.B. NILUSDT)",
                                  value=st.session_state.get("symbol2_spread",
                                                            suggested_coins[1] if len(suggested_coins) > 1 else (
                                                                suggested_coins[0] if suggested_coins else "")),
                                  key="symbol2_spread")
    cols_suggest_2 = col_coin2.columns(min(5, len(suggested_coins)))
    for i, coin in enumerate(suggested_coins):
        if cols_suggest_2[i % len(cols_suggest_2)].button(coin, key=f"suggest2_button_{coin}"):
            st.session_state["suggested_coin2_clicked"] = True
            st.session_state["suggested_coin2_value"] = coin
            st.rerun()

    interval = st.sidebar.selectbox("Intervall für den Spread Plot", ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                                    index=4, key="interval_spread")

    if col_button.button("Spread Daten aktualisieren", use_container_width=True):
        try:
            df1 = get_kline_data(symbol1, interval=interval)
            df2 = get_kline_data(symbol2, interval=interval)
        except requests.exceptions.RequestException as e:
            st.error(f"Fehler beim Abrufen der Spread-Daten von Binance: e")
        else:
            if not df1.empty and not df2.empty:
                data = pd.merge(df1, df2, left_index=True, right_index=True,
                                  suffixes=(f'_{symbol1}', f'_{symbol2}'))
                data['spread'] = data[f'close_{symbol1}'] - data[f'close_{symbol2}']

                st.subheader(f"Spread zwischen {symbol1} und {symbol2}")

                col_fav1, col_sym1 = st.columns([0.1, 1])
                is_fav1 = is_favorite(symbol1)
                star_icon1 = "⭐" if is_fav1 else "☆"
                fav_button_key1 = f"fav_spread_{symbol1}"
                if col_fav1.button(star_icon1, key=fav_button_key1,
                                  help=f"{'Entferne' if is_fav1 else 'Füge hinzu'} {symbol1} zu Favoriten",
                                  use_container_width=True,
                                 # on_click=remove_from_favorites if is_fav1 else add_to_favorites,
                                  args=(symbol1,)):
                    pass
                col_sym1.write(f"**{symbol1}**")

                col_fav2, col_sym2 = st.columns([0.1, 1])
                is_fav2 = is_favorite(symbol2)
                star_icon2 = "⭐" if is_fav2 else "☆"
                fav_button_key2 = f"fav_spread_{symbol2}"
                if col_fav2.button(star_icon2, key=fav_button_key2,
                                  help=f"{'Entferne' if is_fav2 else 'Füge hinzu'} {symbol2} zu Favoriten",
                                  use_container_width=True,
                                  on_click=remove_from_favorites if is_fav2 else add_to_favorites,
                                  args=(symbol2,)):
                    pass
                col_sym2.write(f"**{symbol2}**")

                st.dataframe(data)

                spread_mean = data['spread'].mean()
                spread_std = data['spread'].std()
                data['corr'] = data[f'close_{symbol1}'].rolling(window=20).corr(data[f'close_{symbol2}'])
                data['signal'] = (data['corr'] > 0.8) & (abs(data['spread'] - spread_mean) > 2 * spread_std)

                fig_spread, ax_spread = plt.subplots(figsize=(14, 6))
                ax_spread.plot(data.index, data['spread'], label='Spread')
                ax_spread.axhline(spread_mean, color='gray', linestyle='--', label='Mean')
                ax_spread.axhline(spread_mean + 2 * spread_std, color='red', linestyle='--', label='+2 Std')
                ax_spread.axhline(spread_mean - 2 * spread_std, color='green', linestyle='--', label='–2 Std')
                ax_spread.scatter(data.index[data['signal']], data['spread'][data['signal']], color='purple',
                                    label='Signal', marker='o')
                ax_spread.set_title(f"Spread zwischen {symbol1} und {symbol2}")
                ax_spread.set_xlabel("Zeit")
                ax_spread.set_ylabel("Spread")
                ax_spread.legend()
                ax_spread.grid(True)
                st.pyplot(fig_spread)
                st.subheader("Spread Interpretation")
                st.markdown(f"- **Spread**: Preisunterschied zwischen {symbol1} und {symbol2}.")
                st.markdown("- **Graue Linie (Mean)**: Durchschnittlicher Spread.")
                st.markdown("- **Rote & grüne Linien (+/- 2 Std)**: Standardabweichungen.")
                st.markdown("- **Lila Punkte (Signal)**: Potentielle Mean Reversion Signale.")
                st.markdown("- **Beispielstrategie**: Short auf steigenden, Long auf fallenden Spread.")
            else:
                st.warning(
                    "Keine Daten für die ausgewählten Spread-Coins gefunden. Bitte überprüfe die Coin-Symbole und das Intervall.")

elif page == "Korrelationsanalyse":
    st.header("Korrelationsanalyse")
    initialize_favorites()
    suggested_coins = get_suggested_coins()

    # Custom CSS für kleinere Buttons und kein Umbruch
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            font-size: 0.8em !important;
            padding: 0.2em 0.4em !important;
            margin-right: 0.1em !important;
            margin-bottom: 0.1em !important;
            word-break: normal !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_coin, col_button_corr = st.columns([3, 1.5])

    # Logik zum Aktualisieren von st.session_state BEVOR text_input erstellt wird
    if "suggested_coin_corr_clicked" in st.session_state and st.session_state["suggested_coin_corr_clicked"]:
        st.session_state["coin_corr"] = st.session_state["suggested_coin_value_corr"]
        st.session_state["suggested_coin_corr_clicked"] = False
        st.session_state["suggested_coin_value_corr"] = ""
        st.rerun()

    coin_to_correlate = col_coin.text_input("Basiskoin für Korrelationsanalyse (z.B. BTCUSDT)",
                                           value=st.session_state.get("coin_corr",
                                                                     suggested_coins[0] if suggested_coins else ""),
                                           key="coin_corr")
    cols_suggest_corr = col_coin.columns(min(5, len(suggested_coins)))
    for i, coin in enumerate(suggested_coins):
        if cols_suggest_corr[i % len(cols_suggest_corr)].button(coin, key=f"suggest_corr_button_{coin}"):
            st.session_state["suggested_coin_corr_clicked"] = True
            st.session_state["suggested_coin_value_corr"] = coin
            st.rerun()

    interval = st.sidebar.selectbox("Intervall für die Korrelationsanalyse", ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                                    index=4, key="interval_corr")
    num_coins_corr = st.sidebar.slider("Anzahl der zu analysierenden Coins für Korrelation", min_value=5,
                                         max_value=100, value=20, step=5)

    if col_button_corr.button("Korrelationsdaten analysieren", use_container_width=True):
        all_usdt_pairs = get_all_usdt_pairs()
        if coin_to_correlate not in all_usdt_pairs:
            st.error(
                f"Der eingegebene Basis-Coin '{coin_to_correlate}' ist kein gültiges USDT-Paar auf Binance oder wurde nicht gefunden.")
        else:
            try:
                base_df = get_kline_data(coin_to_correlate, interval=interval)
                if base_df.empty:
                    st.warning(f"Keine Preisdaten für '{coin_to_correlate}' gefunden.")
                else:
                    correlation_data = {}
                    available_coins = [pair for pair in all_usdt_pairs if pair != coin_to_correlate]
                    num_analyze = min(num_coins_corr, len(available_coins))
                    analyzed_coins = available_coins[:num_analyze]
                    progress_bar = st.progress(0)
                    for i, other_coin in enumerate(analyzed_coins):
                        try:
                            other_df = get_kline_data(other_coin, interval=interval)
                            if not other_df.empty and base_df.index.equals(other_df.index):
                                merged_df = pd.merge(base_df, other_df, left_index=True, right_index=True,
                                                      suffixes=(f'_{coin_to_correlate}', f'_{other_coin}'))
                                correlation = merged_df[f'close_{coin_to_correlate}'].corr(
                                    merged_df[f'close_{other_coin}'])
                                correlation_data[other_coin] = correlation
                        except requests.exceptions.RequestException as e:
                            st.warning(f"Fehler beim Abrufen der Daten für {other_coin}: e")
                        progress_bar.progress((i + 1) / num_analyze)

                    if correlation_data:
                        correlation_series = pd.Series(correlation_data)
                        sorted_correlations = correlation_series.sort_values(ascending=False)

                        st.subheader(f"Korrelation mit {coin_to_correlate}")
                        st.dataframe(sorted_correlations)

                        # Heatmap-Visualisierung (begrenzt auf die Top-Korrelationen)
                        top_n = min(10, len(sorted_correlations))
                        top_correlated_coins = sorted_correlations.head(top_n).index.tolist()
                        all_coins_for_heatmap = [coin_to_correlate] + top_correlated_coins
                        heatmap_data = pd.DataFrame(index=all_coins_for_heatmap)

                        progress_bar_heatmap = st.progress(0)
                        for i, coin in enumerate(all_coins_for_heatmap):
                            try:
                                df = get_kline_data(coin, interval=interval)
                                if not df.empty and base_df.index.equals(df.index):
                                    heatmap_data[f'close_{coin}'] = df['close']
                                favorite_button_key = f"fav_corr_{coin}"
                                col_fav, col_coin_hm = st.columns([0.1, 1])
                                if col_fav.button("⭐" if is_favorite(coin) else "☆", key=favorite_button_key,
                                                  help=f"{'Entferne' if is_favorite(coin) else 'Füge hinzu'} {coin} zu Favoriten",
                                                  use_container_width=True,
                                                  on_click=remove_from_favorites if is_favorite(
                                                      coin) else add_to_favorites,
                                                  args=(coin,)):
                                    pass
                                col_coin_hm.write(coin)
                            except requests.exceptions.RequestException as e:
                                st.warning(f"Fehler beim Abrufen der Daten für {coin} für Heatmap: e")
                            progress_bar_heatmap.progress((i + 1) / len(all_coins_for_heatmap))

                        if len(heatmap_data.columns) > 1:
                            correlation_matrix = heatmap_data.corr()
                            fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
                            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                                        ax=ax_heatmap)
                            ax_heatmap.set_title(
                                f"Korrelationsmatrix (mit {coin_to_correlate} und den Top {top_n - 1} korrelierten Coins)")
                            st.pyplot(fig_heatmap)
                        else:
                            st.warning("Nicht genügend Daten für die Heatmap-Visualisierung.")

                    else:
                        st.info(f"Keine Korrelationen mit anderen Coins für '{coin_to_correlate}' gefunden.")
            except Exception as e:
                st.error(f"Ein unerwarteter Fehler ist aufgetreten: e")
