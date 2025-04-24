import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_kline_data, get_all_usdt_pairs, get_suggested_coins, is_favorite, add_to_favorites, remove_from_favorites, initialize_favorites

def show_spread_page():
    st.header("Spread Analyse")
    initialize_favorites()
    suggested_coins = get_suggested_coins()

    # Custom CSS für kleinere Buttons
    st.markdown("""<style>div.stButton > button:first-child {font-size: 0.8em !important; padding: 0.2em 0.4em !important;}</style>""", unsafe_allow_html=True)

    col_coin1, col_coin2, col_button = st.columns([2, 2, 1.5])

    # UI-Elemente für Coin-Auswahl
    symbol1 = col_coin1.text_input("Coin 1 (z.B. BABYUSDT)", value=st.session_state.get("symbol1_spread", suggested_coins[0] if suggested_coins else ""), key="symbol1_spread")
    symbol2 = col_coin2.text_input("Coin 2 (z.B. NILUSDT)", value=st.session_state.get("symbol2_spread", suggested_coins[1] if len(suggested_coins) > 1 else ""), key="symbol2_spread")

    # Buttons für vorgeschlagene Coins
    cols_suggest_1 = col_coin1.columns(min(5, len(suggested_coins)))
    for i, coin in enumerate(suggested_coins):
        if cols_suggest_1[i % len(cols_suggest_1)].button(coin, key=f"suggest1_button_{coin}"):
            st.session_state["symbol1_spread"] = coin
            st.rerun()

    cols_suggest_2 = col_coin2.columns(min(5, len(suggested_coins)))
    for i, coin in enumerate(suggested_coins):
        if cols_suggest_2[i % len(cols_suggest_2)].button(coin, key=f"suggest2_button_{coin}"):
            st.session_state["symbol2_spread"] = coin
            st.rerun()

    interval = st.sidebar.selectbox("Intervall", ['1m', '5m', '15m', '30m', '1h', '4h', '1d'], index=4)

    if col_button.button("Spread Daten aktualisieren", use_container_width=True):
        try:
            df1 = get_kline_data(symbol1, interval=interval)
            df2 = get_kline_data(symbol2, interval=interval)
            
            if not df1.empty and not df2.empty:
                data = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=(f'_{symbol1}', f'_{symbol2}'))
                data['spread'] = data[f'close_{symbol1}'] - data[f'close_{symbol2}']
                
                # Visualisierung und Analyse
                show_spread_analysis(data, symbol1, symbol2)
                
        except Exception as e:
            st.error(f"Fehler: {e}")

def show_spread_analysis(data, symbol1, symbol2):
    st.subheader(f"Spread zwischen {symbol1} und {symbol2}")
    
    # Favoriten-Buttons
    col_fav1, col_sym1 = st.columns([0.1, 1])
    if col_fav1.button("⭐" if is_favorite(symbol1) else "☆", key=f"fav_spread_{symbol1}", use_container_width=True):
        if is_favorite(symbol1):
            remove_from_favorites(symbol1)
        else:
            add_to_favorites(symbol1)
        st.rerun()
    col_sym1.write(f"**{symbol1}**")

    col_fav2, col_sym2 = st.columns([0.1, 1])
    if col_fav2.button("⭐" if is_favorite(symbol2) else "☆", key=f"fav_spread_{symbol2}", use_container_width=True):
        if is_favorite(symbol2):
            remove_from_favorites(symbol2)
        else:
            add_to_favorites(symbol2)
        st.rerun()
    col_sym2.write(f"**{symbol2}**")

    # Spread-Berechnungen
    spread_mean = data['spread'].mean()
    spread_std = data['spread'].std()
    data['corr'] = data[f'close_{symbol1}'].rolling(window=20).corr(data[f'close_{symbol2}'])
    data['signal'] = (data['corr'] > 0.8) & (abs(data['spread'] - spread_mean) > 2 * spread_std)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data.index, data['spread'], label='Spread')
    ax.axhline(spread_mean, color='gray', linestyle='--', label='Mean')
    ax.axhline(spread_mean + 2 * spread_std, color='red', linestyle='--', label='+2 Std')
    ax.axhline(spread_mean - 2 * spread_std, color='green', linestyle='--', label='–2 Std')
    ax.scatter(data.index[data['signal']], data['spread'][data['signal']], color='purple', label='Signal')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Interpretation
    st.subheader("Spread Interpretation")
    st.markdown(f"- **Spread**: Preisunterschied zwischen {symbol1} und {symbol2}")
    st.markdown("- **Graue Linie**: Durchschnittlicher Spread")
    st.markdown("- **Rote/grüne Linien**: Standardabweichungen")
    st.markdown("- **Lila Punkte**: Potentielle Mean Reversion Signale")